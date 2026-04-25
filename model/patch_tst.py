"""
PatchTST: A Time Series is Worth 64 Words (ICLR 2023).
PatchTST: 时间序列等于64个词 (ICLR 2023)。

Reference: arXiv 2211.14730
Key idea: split sequence into patches (like ViT for images), then Transformer.
Outperforms vanilla Transformer on financial time series by 10-15%.

核心思想：把时间序列切成 patch（类似 ViT 对图像），再过 Transformer。
在金融时序上比朴素 Transformer 强 10-15%。

Adapted to our 4D cross-asset architecture: [Batch, Assets, Seq, Features].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PatchTSTCrossAsset(nn.Module):
    """
    PatchTST adapted for cross-asset ranking.
    跨资产排序的 PatchTST 改编版。

    1. Split T bars into patches of size P → L = T/P patches
    2. Project each patch + factor independently → (B*A*F, L, D)
    3. Per-channel Transformer (channel-independent design from PatchTST)
    4. Reduce to (B, A, D) and apply cross-asset attention
    5. Ranking head → (B, A) scores
    """

    def __init__(
        self,
        n_factors: int = 13,
        d_model: int = 64,
        n_heads: int = 4,
        n_temp_layers: int = 2,
        n_cross_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.25,
        seq_len: int = 24,
        patch_size: int = 6,
        max_assets: int = 20,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.n_factors = n_factors

        # patch projection per channel / 每通道patch投影
        self.patch_proj = nn.Linear(patch_size, d_model)

        # patch positional encoding / patch位置编码
        self.patch_pe = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # temporal Transformer (channel-independent) / 时序Transformer（通道独立）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_enc = nn.TransformerEncoder(layer, num_layers=n_temp_layers)

        # cross-asset attention / 跨资产注意力
        self.asset_embed = nn.Parameter(torch.randn(1, max_assets, d_model) * 0.02)
        cross_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.cross_enc = nn.TransformerEncoder(cross_layer, num_layers=n_cross_layers)

        # factor mixing: weight each factor's contribution / 因子混合
        self.factor_mix = nn.Linear(n_factors, 1)

        # ranking head / 排序头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, A, T, F) → (B, A) scores
        """
        B, A, T, F_ = x.shape
        D = self.d_model
        P = self.patch_size
        L = self.n_patches

        # (B, A, T, F) → (B, A, F, T) for per-channel processing
        x = x.permute(0, 1, 3, 2)  # (B, A, F, T)
        # split into patches: (B, A, F, L, P)
        x = x.reshape(B, A, F_, L, P)
        # project: (B, A, F, L, D)
        x = self.patch_proj(x)
        # add positional encoding: broadcast over (B, A, F)
        x = x + self.patch_pe[None, None, None, :, :]

        # Per-channel Transformer: reshape to (B*A*F, L, D)
        x = x.reshape(B * A * F_, L, D)
        x = self.temporal_enc(x)  # (B*A*F, L, D)

        # pool over patches → (B*A*F, D)
        x = x.mean(dim=1)
        # reshape back → (B, A, F, D)
        x = x.reshape(B, A, F_, D)
        # mix factors: (B, A, F, D) → (B, A, D) via learned factor weighting
        x = x.permute(0, 1, 3, 2)  # (B, A, D, F)
        x = self.factor_mix(x).squeeze(-1)  # (B, A, D)

        # add asset embedding / 加资产嵌入
        x = x + self.asset_embed[:, :A, :]

        # cross-asset attention / 跨资产注意力
        x = self.cross_enc(x)  # (B, A, D)

        # rank head → (B, A) scores
        scores = self.head(x).squeeze(-1)
        return scores
