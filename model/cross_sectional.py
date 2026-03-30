"""
Cross-Sectional Multi-Asset Transformer with ListMLE Ranking Loss.

4D tensor architecture: [Batch, Assets, Seq_len, Features]
The model learns to RANK assets by relative performance, not predict
absolute returns. This avoids the MSE-on-returns trap entirely.

Key components:
  - Per-asset temporal encoder (shared weights)
  - Cross-asset attention (learns inter-asset dependencies)
  - ListMLE ranking loss (from Learning to Rank literature)

截面多资产Transformer，带ListMLE排序损失。

4D张量架构: [批次, 资产, 序列长度, 特征]
模型学习按相对收益对资产进行排序，而非预测绝对收益。
完全避免了"对收益率做MSE回归"的陷阱。

核心组件:
  - 每资产时序编码器（共享权重）
  - 跨资产注意力（学习资产间依赖关系）
  - ListMLE排序损失（来自Learning to Rank文献）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ListMLE Ranking Loss (Xia et al., 2008)
# ListMLE排序损失（Xia等，2008）
# ---------------------------------------------------------------------------

def listmle_loss(scores: Tensor, relevance: Tensor) -> Tensor:
    """
    ListMLE: Listwise ranking loss based on Plackett-Luce model.

    Given predicted scores and true relevance labels for a set of items,
    computes the negative log-likelihood of the permutation defined by
    sorting items by true relevance.

    ListMLE: 基于Plackett-Luce模型的列表级排序损失。
    给定预测分数和真实相关性标签，计算按真实相关性排序所定义排列的负对数似然。

    Parameters / 参数
    ----------
    scores : Tensor
        Predicted scores, shape (B, N) where N = number of assets.
        预测分数，形状 (B, N)，N为资产数。
    relevance : Tensor
        True relevance/returns, shape (B, N). Higher = better.
        真实相关性/收益率，形状 (B, N)。值越大越好。

    Returns / 返回
    -------
    Tensor
        Scalar loss.
        标量损失值。
    """
    # sort indices by true relevance (descending) / 按真实相关性降序排列索引
    _, sorted_idx = relevance.sort(dim=-1, descending=True)
    # gather predicted scores in true-relevance order / 按真实相关性顺序收集预测分数
    sorted_scores: Tensor = scores.gather(dim=-1, index=sorted_idx)

    # ListMLE: for each position i, compute log-softmax over remaining items
    # ListMLE: 对每个位置i，计算剩余项的log-softmax
    n: int = sorted_scores.size(1)
    # cumulative logsumexp from the end / 从末尾开始的累积logsumexp
    # log P(pi) = sum_{i=1}^{n} [s_{pi(i)} - logsumexp(s_{pi(i)}, ..., s_{pi(n)})]
    cumsums: Tensor = torch.logcumsumexp(sorted_scores.flip(dims=[1]), dim=1).flip(dims=[1])
    loss: Tensor = -(sorted_scores - cumsums).mean()
    return loss


# ---------------------------------------------------------------------------
# Cross-Sectional Transformer
# 截面Transformer
# ---------------------------------------------------------------------------

class CrossSectionalTransformer(nn.Module):
    """
    Multi-asset ranking model.

    Architecture:
      1. FactorProjection: (B, A, T, F) -> (B, A, T, D)
      2. TemporalEncoder: shared Transformer encoder over time axis
         (B*A, T, D) -> (B*A, D) via mean pooling
      3. CrossAssetAttention: Transformer over asset axis
         (B, A, D) -> (B, A, D)
      4. RankingHead: (B, A, D) -> (B, A) scores

    The output scores are used with ListMLE loss for ranking.

    多资产排序模型。

    架构:
      1. 因子投影: (B, A, T, F) -> (B, A, T, D)
      2. 时序编码器: 共享Transformer编码器处理时间轴
         (B*A, T, D) -> (B*A, D) 通过均值池化
      3. 跨资产注意力: Transformer处理资产轴
         (B, A, D) -> (B, A, D)
      4. 排序头: (B, A, D) -> (B, A) 分数

    输出分数与ListMLE损失配合用于排序。
    """

    def __init__(
        self,
        n_factors: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_temporal_layers: int = 2,
        n_cross_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 20,
        max_assets: int = 20,
    ) -> None:
        super().__init__()
        self.n_factors: int = n_factors
        self.d_model: int = d_model
        self.seq_len: int = seq_len

        # factor projection / 因子投影
        self.factor_proj: nn.Linear = nn.Linear(n_factors, d_model)
        self.factor_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # learnable positional encoding for time axis / 时间轴可学习位置编码
        self.time_pe: nn.Parameter = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # temporal encoder (shared across assets) / 时序编码器（跨资产共享）
        temp_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            temp_layer, num_layers=n_temporal_layers
        )

        # learnable asset positional encoding / 可学习资产位置编码
        self.asset_pe: nn.Parameter = nn.Parameter(
            torch.randn(1, max_assets, d_model) * 0.02
        )

        # cross-asset attention / 跨资产注意力
        cross_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            cross_layer, num_layers=n_cross_layers
        )

        # ranking head: project to scalar score per asset / 排序头: 每个资产投影为标量分数
        self.rank_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters / 参数
        ----------
        x : Tensor
            Shape (B, A, T, F) — batch, assets, time, features.
            形状 (B, A, T, F) — 批次、资产、时间、特征。

        Returns / 返回
        -------
        Tensor
            Shape (B, A) — ranking scores per asset.
            形状 (B, A) — 每个资产的排序分数。
        """
        B, A, T, _F = x.shape
        D: int = self.d_model

        # 1. Factor projection: (B, A, T, F) -> (B, A, T, D) / 因子投影
        h: Tensor = self.factor_norm(F.gelu(self.factor_proj(x)))

        # 2. Temporal encoding: reshape to (B*A, T, D) for shared encoder / 时序编码: 重塑为共享编码器输入
        h = h.reshape(B * A, T, D)
        h = h + self.time_pe[:, :T, :]
        h = self.temporal_encoder(h)  # (B*A, T, D)

        # pool over time -> (B*A, D) / 时间维度池化
        h = h.mean(dim=1)

        # 3. Reshape to (B, A, D) for cross-asset attention / 重塑为跨资产注意力输入
        h = h.reshape(B, A, D)
        h = h + self.asset_pe[:, :A, :]
        h = self.cross_encoder(h)  # (B, A, D)

        # 4. Ranking head → (B, A, 1) → (B, A) / 排序头
        scores: Tensor = self.rank_head(h).squeeze(-1)
        return scores
