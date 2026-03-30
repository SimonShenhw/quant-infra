"""
Cross-Sectional Multi-Asset Transformer with ListMLE Ranking Loss.

4D tensor architecture: [Batch, Assets, Seq_len, Features]
The model learns to RANK assets by relative performance, not predict
absolute returns. This avoids the MSE-on-returns trap entirely.

Key components:
  - Per-asset temporal encoder (shared weights)
  - Cross-asset attention (learns inter-asset dependencies)
  - ListMLE ranking loss (from Learning to Rank literature)
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
# ---------------------------------------------------------------------------

def listmle_loss(scores: Tensor, relevance: Tensor) -> Tensor:
    """
    ListMLE: Listwise ranking loss based on Plackett-Luce model.

    Given predicted scores and true relevance labels for a set of items,
    computes the negative log-likelihood of the permutation defined by
    sorting items by true relevance.

    Parameters
    ----------
    scores : Tensor
        Predicted scores, shape (B, N) where N = number of assets.
    relevance : Tensor
        True relevance/returns, shape (B, N). Higher = better.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    # sort indices by true relevance (descending)
    _, sorted_idx = relevance.sort(dim=-1, descending=True)
    # gather predicted scores in true-relevance order
    sorted_scores: Tensor = scores.gather(dim=-1, index=sorted_idx)

    # ListMLE: for each position i, compute log-softmax over remaining items
    n: int = sorted_scores.size(1)
    # cumulative logsumexp from the end
    # log P(pi) = sum_{i=1}^{n} [s_{pi(i)} - logsumexp(s_{pi(i)}, ..., s_{pi(n)})]
    cumsums: Tensor = torch.logcumsumexp(sorted_scores.flip(dims=[1]), dim=1).flip(dims=[1])
    loss: Tensor = -(sorted_scores - cumsums).mean()
    return loss


# ---------------------------------------------------------------------------
# Cross-Sectional Transformer
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

        # factor projection
        self.factor_proj: nn.Linear = nn.Linear(n_factors, d_model)
        self.factor_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # learnable positional encoding for time axis
        self.time_pe: nn.Parameter = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # temporal encoder (shared across assets)
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

        # learnable asset positional encoding
        self.asset_pe: nn.Parameter = nn.Parameter(
            torch.randn(1, max_assets, d_model) * 0.02
        )

        # cross-asset attention
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

        # ranking head: project to scalar score per asset
        self.rank_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (B, A, T, F) — batch, assets, time, features.

        Returns
        -------
        Tensor
            Shape (B, A) — ranking scores per asset.
        """
        B, A, T, _F = x.shape
        D: int = self.d_model

        # 1. Factor projection: (B, A, T, F) -> (B, A, T, D)
        h: Tensor = self.factor_norm(F.gelu(self.factor_proj(x)))

        # 2. Temporal encoding: reshape to (B*A, T, D) for shared encoder
        h = h.reshape(B * A, T, D)
        h = h + self.time_pe[:, :T, :]
        h = self.temporal_encoder(h)  # (B*A, T, D)

        # pool over time -> (B*A, D)
        h = h.mean(dim=1)

        # 3. Reshape to (B, A, D) for cross-asset attention
        h = h.reshape(B, A, D)
        h = h + self.asset_pe[:, :A, :]
        h = self.cross_encoder(h)  # (B, A, D)

        # 4. Ranking head → (B, A, 1) → (B, A)
        scores: Tensor = self.rank_head(h).squeeze(-1)
        return scores
