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

基于Transformer的多因子时序预测网络。

架构来源:
  - arXiv 2508.04975: Encoder(3)-Decoder(2), d_model=512, GELU, 8头
  - arXiv 2404.00424: 纯Encoder, 16头, d_f=16, 6层
  - arXiv 2507.07107: 4-16头, 128-512嵌入维度, 10-60序列长度

实现了灵活的Encoder-Decoder Transformer:
  - 可学习位置编码，适配交易日历
  - 多头自注意力/交叉注意力，CUDA优化
  - 可配置的因子数量作为输入通道
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
# 位置编码（可学习）
# ---------------------------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable position embeddings for financial time-series.
    Shape: (1, max_len, d_model) — broadcast over batch dim.

    可学习位置嵌入，用于金融时序数据。
    形状: (1, max_len, d_model) — 在batch维度上广播。
    """

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.pe: nn.Parameter = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) / 输入: (批次, 时间步, 维度)
        seq_len: int = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding (fixed, from original Transformer paper)
# 正弦位置编码（固定，源自原始Transformer论文）
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) / 增加batch维度
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len: int = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len, :])


# ---------------------------------------------------------------------------
# Multi-Factor Input Projection
# 多因子输入投影
# ---------------------------------------------------------------------------

class FactorProjection(nn.Module):
    """
    Projects raw multi-factor features into d_model space.
    Applies LayerNorm after projection for training stability.

    将原始多因子特征投影到d_model空间。
    投影后应用LayerNorm以稳定训练。
    """

    def __init__(self, n_factors: int, d_model: int) -> None:
        super().__init__()
        self.linear: nn.Linear = nn.Linear(n_factors, d_model)
        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)
        self.activation: nn.GELU = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, n_factors) / 输入: (批次, 时间步, 因子数)
        return self.activation(self.norm(self.linear(x)))


# ---------------------------------------------------------------------------
# Transformer Encoder-Decoder for Stock Prediction
# Transformer编码器-解码器，用于股票预测
# ---------------------------------------------------------------------------

class QuantTransformer(nn.Module):
    """
    Production Transformer for multi-factor quantitative prediction.

    生产级Transformer，用于多因子量化预测。

    Parameters / 参数
    ----------
    n_factors : int
        Number of input features per time-step (OHLCV + alpha factors).
        每个时间步的输入特征数（OHLCV + alpha因子）。
    d_model : int
        Internal embedding dimension.
        内部嵌入维度。
    n_heads : int
        Number of attention heads.
        注意力头数。
    n_encoder_layers : int
        Encoder depth.
        编码器层数。
    n_decoder_layers : int
        Decoder depth. Set to 0 for encoder-only mode.
        解码器层数。设为0则为纯编码器模式。
    d_ff : int
        Feed-forward hidden dimension.
        前馈网络隐藏层维度。
    dropout : float
        Dropout rate for regularisation.
        正则化Dropout比率。
    seq_len : int
        Encoder lookback window length.
        编码器回看窗口长度。
    pred_len : int
        Decoder prediction horizon.
        解码器预测时间跨度。
    n_classes : int
        Output dimension. 1 for regression, >1 for classification tiers.
        输出维度。1为回归，>1为分类层级。
    use_learnable_pe : bool
        Whether to use learnable or sinusoidal positional encoding.
        是否使用可学习位置编码（否则使用正弦编码）。
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

        # input projection / 输入投影
        self.enc_projection: FactorProjection = FactorProjection(n_factors, d_model)

        # positional encoding / 位置编码
        pe_cls = LearnablePositionalEncoding if use_learnable_pe else SinusoidalPositionalEncoding
        self.enc_pos: nn.Module = pe_cls(d_model, max_len=max(seq_len, 256), dropout=dropout)

        # encoder / 编码器
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

        # decoder (optional) / 解码器（可选）
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

        # output head / 输出头
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
        Parameters / 参数
        ----------
        src : Tensor
            Encoder input, shape (B, seq_len, n_factors).
            编码器输入，形状 (B, seq_len, n_factors)。
        tgt : Tensor, optional
            Decoder input, shape (B, pred_len, n_factors).
            Ignored in encoder-only mode.
            解码器输入，形状 (B, pred_len, n_factors)。纯编码器模式下忽略。

        Returns / 返回
        -------
        Tensor
            Predictions, shape (B, n_classes) for regression/classification.
            预测结果，形状 (B, n_classes)，用于回归/分类。
        """
        # --- Encoder --- / --- 编码器 ---
        enc_emb: Tensor = self.enc_projection(src)  # (B, T, D)
        enc_emb = self.enc_pos(enc_emb)
        memory: Tensor = self.encoder(enc_emb)       # (B, T, D)

        if self._encoder_only:
            # pool over time axis → (B, D) / 时间轴池化 → (B, D)
            pooled: Tensor = memory.mean(dim=1)
        else:
            # --- Decoder --- / --- 解码器 ---
            if tgt is None:
                # auto-create decoder input from last pred_len steps / 自动从最后pred_len步创建解码器输入
                tgt = src[:, -self.pred_len:, :]
            dec_emb: Tensor = self.dec_projection(tgt)  # (B, P, D)
            dec_emb = self.dec_pos(dec_emb)
            tgt_mask: Tensor = self._generate_causal_mask(
                dec_emb.size(1), dec_emb.device
            )
            dec_out: Tensor = self.decoder(
                dec_emb, memory, tgt_mask=tgt_mask
            )  # (B, P, D)
            pooled = dec_out[:, -1, :]  # take last step → (B, D) / 取最后一步 → (B, D)

        out: Tensor = self.output_head(self.output_norm(pooled))  # (B, n_classes)
        return out


# ---------------------------------------------------------------------------
# Convenience factory
# 便捷工厂函数
# ---------------------------------------------------------------------------

def build_quant_transformer(
    n_factors: int = 8,
    preset: str = "medium",
    device: Optional[torch.device] = None,
) -> QuantTransformer:
    """
    Construct a QuantTransformer with sensible defaults.

    使用合理默认值构建QuantTransformer。

    Presets / 预设配置
    -------
    small  : d=128,  4h, 2enc, 0dec  (encoder-only, fast)
               纯编码器，快速推理
    medium : d=256,  8h, 3enc, 2dec  (balanced, paper 2508.04975 inspired)
               均衡配置，参考论文2508.04975
    large  : d=512, 16h, 6enc, 0dec  (encoder-only, paper 2404.00424 inspired)
               纯编码器，参考论文2404.00424
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
