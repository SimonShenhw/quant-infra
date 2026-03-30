"""
Cross-Asset GRU + Attention Model (v7.0).

Architecture:
  1. Per-asset GRU encoder (shared weights): (B, A, T, F) → (B, A, D)
     Captures temporal dynamics within each asset.
  2. Cross-Asset Self-Attention: (B, A, D) → (B, A, D)
     Models lead-lag relationships between assets.
     Each asset attends to ALL other assets, learning inter-asset
     dependencies like BTC leading altcoins.
  3. Ranking head: (B, A, D) → (B, A) scores

Key design: GRU is cheaper than Transformer for temporal encoding,
freeing VRAM budget for the cross-asset attention layer.

跨资产GRU + 注意力模型 (v7.0)。

架构:
  1. 每资产GRU编码器（共享权重）: (B, A, T, F) → (B, A, D)
     捕捉每个资产内部的时序动态。
  2. 跨资产自注意力: (B, A, D) → (B, A, D)
     建模资产间的领先-滞后关系。
     每个资产关注所有其他资产，学习资产间依赖（如BTC领先山寨币）。
  3. 排序头: (B, A, D) → (B, A) 分数

核心设计: GRU比Transformer更轻量，用于时序编码，
将显存预算留给跨资产注意力层。
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

    4D张量: [批次, 资产, 序列长度, 特征]
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
        # Learnable asset embeddings (captures asset identity) / 可学习资产嵌入（捕捉资产身份）
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

        # reshape for shared GRU: (B*A, T, D) / 重塑为共享GRU输入
        h = h.reshape(B * A, T, D)
        gru_out, _ = self.gru(h)  # (B*A, T, D)

        # take last hidden state → (B*A, D) / 取最后隐藏状态
        h = self.temporal_norm(gru_out[:, -1, :])

        # --- Stage 2: Cross-Asset Attention --- / 阶段2: 跨资产注意力
        # reshape: (B, A, D) / 重塑
        h = h.reshape(B, A, D)
        # add asset identity embeddings / 添加资产身份嵌入
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
