"""
Transformer-based Multi-Factor Time-Series Prediction Network.

Architecture derived from:
  - arXiv 2508.04975: Encoder(3)-Decoder(2), d_model=512, GELU, 8 heads
  - arXiv 2404.00424: Encoder-only, 16 heads, d_f=16, 6 layers
  - arXiv 2507.07107: 4-16 heads, 128-512 embedding, 10-60 seq len

We implement a flexible Encoder-Decoder Transformer with:
  - Learnable positional encoding for trading-calendar awareness
  - Multi-head self/cross-attention with CUDA-optimised operations
  - Configurable factor count as input channels
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding (learnable)
# ---------------------------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable position embeddings for financial time-series.
    Shape: (1, max_len, d_model) — broadcast over batch dim.
    """

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.pe: nn.Parameter = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        seq_len: int = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding (fixed, from original Transformer paper)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        pe: Tensor = torch.zeros(max_len, d_model)
        position: Tensor = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term: Tensor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len: int = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# ---------------------------------------------------------------------------
# Multi-Factor Input Projection
# ---------------------------------------------------------------------------

class FactorProjection(nn.Module):
    """
    Projects raw multi-factor features into d_model space.
    Applies LayerNorm after projection for training stability.
    """

    def __init__(self, n_factors: int, d_model: int) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(n_factors, d_model)
        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)
        self.activation: nn.GELU = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, n_factors)
        return self.activation(self.norm(self.linear(x)))


# ---------------------------------------------------------------------------
# Transformer Encoder-Decoder for Stock Prediction
# ---------------------------------------------------------------------------

class QuantTransformer(nn.Module):
    """
    Production Transformer for multi-factor quantitative prediction.

    Parameters
    ----------
    n_factors : int
        Number of input features per time-step (OHLCV + alpha factors).
    d_model : int
        Internal embedding dimension.
    n_heads : int
        Number of attention heads.
    n_encoder_layers : int
        Encoder depth.
    n_decoder_layers : int
        Decoder depth. Set to 0 for encoder-only mode.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate for regularisation.
    seq_len : int
        Encoder lookback window length.
    pred_len : int
        Decoder prediction horizon.
    n_classes : int
        Output dimension. 1 for regression, >1 for classification tiers.
    use_learnable_pe : bool
        Whether to use learnable or sinusoidal positional encoding.
    """

    def __init__(
        self,
        n_factors: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        seq_len: int = 20,
        pred_len: int = 1,
        n_classes: int = 1,
        use_learnable_pe: bool = True,
    ) -> None:
        super().__init__()
        self.n_factors: int = n_factors
        self.d_model: int = d_model
        self.seq_len: int = seq_len
        self.pred_len: int = pred_len
        self.n_classes: int = n_classes
        self._encoder_only: bool = n_decoder_layers == 0

        # input projection
        self.enc_projection: FactorProjection = FactorProjection(n_factors, d_model)

        # positional encoding
        pe_cls = LearnablePositionalEncoding if use_learnable_pe else SinusoidalPositionalEncoding
        self.enc_pos: nn.Module = pe_cls(d_model, max_len=max(seq_len, 256), dropout=dropout)

        # encoder
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

        # decoder (optional)
        if not self._encoder_only:
            self.dec_projection: FactorProjection = FactorProjection(n_factors, d_model)
            self.dec_pos: nn.Module = pe_cls(d_model, max_len=max(pred_len, 256), dropout=dropout)
            decoder_layer: nn.TransformerDecoderLayer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder: nn.TransformerDecoder = nn.TransformerDecoder(
                decoder_layer, num_layers=n_decoder_layers
            )

        # output head
        self.output_norm: nn.LayerNorm = nn.LayerNorm(d_model)
        self.output_head: nn.Linear = nn.Linear(d_model, n_classes)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> Tensor:
        mask: Tensor = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        src : Tensor
            Encoder input, shape (B, seq_len, n_factors).
        tgt : Tensor, optional
            Decoder input, shape (B, pred_len, n_factors).
            Ignored in encoder-only mode.

        Returns
        -------
        Tensor
            Predictions, shape (B, n_classes) for regression/classification.
        """
        # --- Encoder ---
        enc_emb: Tensor = self.enc_projection(src)  # (B, T, D)
        enc_emb = self.enc_pos(enc_emb)
        memory: Tensor = self.encoder(enc_emb)       # (B, T, D)

        if self._encoder_only:
            # pool over time axis → (B, D)
            pooled: Tensor = memory.mean(dim=1)
        else:
            # --- Decoder ---
            if tgt is None:
                # auto-create decoder input from last pred_len steps
                tgt = src[:, -self.pred_len:, :]
            dec_emb: Tensor = self.dec_projection(tgt)  # (B, P, D)
            dec_emb = self.dec_pos(dec_emb)
            tgt_mask: Tensor = self._generate_causal_mask(
                dec_emb.size(1), dec_emb.device
            )
            dec_out: Tensor = self.decoder(
                dec_emb, memory, tgt_mask=tgt_mask
            )  # (B, P, D)
            pooled = dec_out[:, -1, :]  # take last step → (B, D)

        out: Tensor = self.output_head(self.output_norm(pooled))  # (B, n_classes)
        return out


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_quant_transformer(
    n_factors: int = 8,
    preset: str = "medium",
    device: Optional[torch.device] = None,
) -> QuantTransformer:
    """
    Construct a QuantTransformer with sensible defaults.

    Presets
    -------
    small  : d=128,  4h, 2enc, 0dec  (encoder-only, fast)
    medium : d=256,  8h, 3enc, 2dec  (balanced, paper 2508.04975 inspired)
    large  : d=512, 16h, 6enc, 0dec  (encoder-only, paper 2404.00424 inspired)
    """
    configs = {
        "small": dict(d_model=128, n_heads=4, n_encoder_layers=2,
                       n_decoder_layers=0, d_ff=256, seq_len=20),
        "medium": dict(d_model=256, n_heads=8, n_encoder_layers=3,
                        n_decoder_layers=2, d_ff=512, seq_len=20),
        "large": dict(d_model=512, n_heads=16, n_encoder_layers=6,
                       n_decoder_layers=0, d_ff=1024, seq_len=60),
    }
    cfg = configs.get(preset, configs["medium"])
    model: QuantTransformer = QuantTransformer(n_factors=n_factors, **cfg)
    if device is not None:
        model = model.to(device)
    return model
