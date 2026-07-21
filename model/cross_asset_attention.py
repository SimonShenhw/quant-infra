"""
Cross-Asset GRU + Attention Model (v7.0) — the PRODUCTION model.

This is the architecture behind every result-producing pipeline since
v7: the v13 production checkpoint, the paper-trading tracks
(run_paper_daily.py), and the O2 magnitude-aware retrain
(tools/train_o2_production.py) all instantiate this class.

Architecture:
  1. Per-asset GRU encoder (shared weights): (B, A, T, F) → (B, A, D)
     Captures temporal dynamics within each asset.
  2. Cross-Asset Self-Attention: (B, A, D) → (B, A, D)
     Models lead-lag relationships between assets.
     Each asset attends to ALL other assets, learning inter-asset
     dependencies like BTC leading altcoins.
  3. Ranking head: (B, A, D) → (B, A) scores

WHY GRU + attention (not Transformer everywhere): the signal that
matters is CROSS-SECTIONAL — which assets beat which — so the budget
goes to the cross-asset attention stage; a shared GRU is a much cheaper
temporal encoder than a temporal Transformer, freeing VRAM/params for
the interaction layer (v7 decision, kept through v13).

KEY INVARIANT — ASSET IDENTITY IS POSITIONAL.  `asset_embed` is indexed
by POSITION along the asset axis, not by symbol: row i of the input must
always be the SAME asset, in the same sorted-universe order used at
training time.  Operational consequence: the universe order is frozen
per checkpoint; if one symbol is missing and the list silently shifts,
EVERY asset receives the wrong identity embedding and the model keeps
emitting confident, corrupted scores (REVIEW H-3).  This is exactly why
run_paper_daily.py hard-fails ("all-20-or-die") instead of skipping
missing symbols.

跨资产GRU + 注意力模型 (v7.0)——生产模型。v7 以来所有出结果管线
（v13 生产 checkpoint、模拟盘 run_paper_daily.py、O2 重训
tools/train_o2_production.py）都实例化本类。

架构:
  1. 每资产GRU编码器（共享权重）: (B, A, T, F) → (B, A, D)
     捕捉每个资产内部的时序动态。
  2. 跨资产自注意力: (B, A, D) → (B, A, D)
     建模资产间的领先-滞后关系。
     每个资产关注所有其他资产，学习资产间依赖（如BTC领先山寨币）。
  3. 排序头: (B, A, D) → (B, A) 分数

为什么 GRU + 注意力: 有价值的信号在横截面（谁强谁弱），预算应留给
跨资产注意力层；共享 GRU 做时序编码远比时序 Transformer 便宜（v7 决策，
沿用至 v13）。

关键不变量——资产身份按位置编码。`asset_embed` 按资产轴的下标索引而非
按 symbol：输入第 i 行必须永远是同一个资产、且与训练时的 sorted 宇宙
顺序一致。运维后果：宇宙顺序随 checkpoint 冻结；任一 symbol 缺失导致
列表整体平移时，所有资产都拿到错误身份嵌入，而模型照常输出自信的错误
分数（REVIEW H-3）——这正是 run_paper_daily.py 坚持"20 币缺一即
硬失败"的原因。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.cross_sectional import listmle_loss


class CrossAssetGRUAttention(nn.Module):
    """
    4D tensor: [Batch, Assets, Seq_len, Features]

    The Assets axis is ORDER-SENSITIVE: identity embeddings are assigned
    by position (see module KEY INVARIANT) — callers must present assets
    in the frozen training order.

    4D张量: [批次, 资产, 序列长度, 特征]

    资产轴对顺序敏感：身份嵌入按位置分配（见模块关键不变量），调用方
    必须按训练时冻结的顺序排列资产。
    """

    def __init__(
        self,
        n_factors: int = 10,
        d_model: int = 64,
        gru_layers: int = 2,
        n_cross_heads: int = 4,
        n_cross_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.25,
        seq_len: int = 24,
        max_assets: int = 20,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.seq_len: int = seq_len

        # --- Stage 1: Per-asset temporal encoder (shared GRU) --- / 阶段1: 每资产时序编码器（共享GRU）
        self.input_proj: nn.Linear = nn.Linear(n_factors, d_model)
        self.input_norm: nn.LayerNorm = nn.LayerNorm(d_model)
        self.gru: nn.GRU = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.temporal_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # --- Stage 2: Cross-Asset Self-Attention --- / 阶段2: 跨资产自注意力
        # Learnable asset embeddings — identity is keyed by POSITION i, not by
        # symbol; the symbol→row mapping lives only in the caller's frozen
        # sorted universe (module KEY INVARIANT / REVIEW H-3)
        # 可学习资产嵌入——身份按位置 i 绑定而非按 symbol；symbol→行的映射
        # 只存在于调用方冻结的 sorted 宇宙里（模块关键不变量 / REVIEW H-3）
        self.asset_embed: nn.Parameter = nn.Parameter(
            torch.randn(1, max_assets, d_model) * 0.02
        )
        cross_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_cross_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_attention: nn.TransformerEncoder = nn.TransformerEncoder(
            cross_layer, num_layers=n_cross_layers
        )

        # --- Stage 3: Ranking head --- / 阶段3: 排序头
        self.rank_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, A, T, F) → scores: (B, A)

        输入: (批次, 资产, 时间, 特征) → 输出: (批次, 资产) 排序分数
        """
        B, A, T, F_ = x.shape
        D: int = self.d_model

        # --- Stage 1: Temporal encoding --- / 阶段1: 时序编码
        # project: (B, A, T, F) → (B, A, T, D) / 投影
        h: Tensor = F.gelu(self.input_norm(self.input_proj(x)))

        # reshape for shared GRU: (B*A, T, D) — one weight set encodes every
        # asset's history; asset distinctions enter later via asset_embed
        # 重塑为共享GRU输入：一套权重编码所有资产的历史，资产差异稍后由
        # asset_embed 注入
        h = h.reshape(B * A, T, D)
        gru_out, _ = self.gru(h)  # (B*A, T, D)

        # take last hidden state → (B*A, D); the GRU recursion makes this a
        # causal summary of the full window / 取最后隐藏状态——GRU 递归使其
        # 成为整个窗口的因果摘要
        h = self.temporal_norm(gru_out[:, -1, :])

        # --- Stage 2: Cross-Asset Attention --- / 阶段2: 跨资产注意力
        # reshape: (B, A, D) / 重塑
        h = h.reshape(B, A, D)
        # add asset identity embeddings — POSITIONAL: row i gets embedding i,
        # so a shifted asset list silently corrupts every identity (H-3)
        # 添加资产身份嵌入——按位置：第 i 行拿第 i 个嵌入，列表平移会静默
        # 破坏所有身份（H-3）
        h = h + self.asset_embed[:, :A, :]
        # self-attention over asset dimension / 在资产维度上做自注意力
        h = self.cross_attention(h)  # (B, A, D)

        # --- Stage 3: Ranking --- / 阶段3: 排序
        scores: Tensor = self.rank_head(h).squeeze(-1)  # (B, A)
        return scores

    def get_attention_weights(self, x: Tensor) -> Tensor:
        """Extract cross-asset attention weights for analysis.

        提取跨资产注意力权重，用于分析。
        """
        B, A, T, F_ = x.shape
        D = self.d_model
        h = F.gelu(self.input_norm(self.input_proj(x)))
        h = h.reshape(B * A, T, D)
        gru_out, _ = self.gru(h)
        h = self.temporal_norm(gru_out[:, -1, :]).reshape(B, A, D)
        h = h + self.asset_embed[:, :A, :]
        # manually compute attention weights from first cross-attention layer / 手动从第一个跨资产注意力层计算权重
        layer = self.cross_attention.layers[0]
        # self_attn is the MultiheadAttention module / self_attn是MultiheadAttention模块
        _, weights = layer.self_attn(h, h, h, need_weights=True)
        return weights  # (B, A, A)
