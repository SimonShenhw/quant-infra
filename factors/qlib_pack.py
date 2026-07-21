"""
Qlib-inspired factor pack — 8 high-quality factors from Microsoft Qlib's Alpha158.
受 Qlib 启发的因子包 — 8 个来自 Microsoft Qlib Alpha158 的高质量因子。

WHY this pack: candlestick geometry (kmid/klen/kup/klow), momentum (roc10,
max20_ratio), volume-price coherence (corr_pv) and realized volatility (std20)
are Alpha158's battle-tested primitives — cheap, causal, OHLCV-only.
为什么选这组：K 线几何（kmid/klen/kup/klow）、动量（roc10/max20_ratio）、
量价一致性（corr_pv）与已实现波动率（std20）是 Alpha158 久经检验的原语——
计算便宜、严格因果、只需 OHLCV。

IC evidence (2026-06-10 factor_analyzer rerun on TRUE 1h bars, 24h horizon):
std20 and klen are the two STRONGEST factors in the whole library
(|IC| 0.077 / 0.066, negative sign = low-volatility effect), and klow —
wrongly dropped in v12 on corrupted data — ranked #5. Lesson: re-derive factor
rankings after any data fix.
IC 证据（2026-06-10 在修复后的真 1h 数据上重跑 factor_analyzer，24h 周期）：
std20 与 klen 是全库最强两个因子（|IC| 0.077/0.066，负号=低波动效应）；
v12 在污染数据上错杀的 klow 实为第 5。教训：数据修复后必须重排因子。

Reference: github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
"""
from __future__ import annotations

import torch
from torch import Tensor

from factors.base import BaseFactor, register_factor
from model.features import compute_sma, compute_ema, _rolling_std


@register_factor
class KMid(BaseFactor):
    """Mid-price ratio: (close - open) / open. Bullish/bearish per-bar.
    K线中位比率：开盘到收盘的相对涨跌。"""
    name = "kmid"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        return (close - open_) / open_.clamp(min=1e-8)


@register_factor
class KLen(BaseFactor):
    """Bar range as fraction of open: (high-low)/open. Volatility proxy.
    #2 factor by |IC| on true 1h data (24h |IC| 0.066, low-vol effect).
    K线长度：(高-低)/开，波动率代理。真 1h 数据上 |IC| 第 2（24h 0.066，低波动效应）。"""
    name = "klen"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        return (high - low) / open_.clamp(min=1e-8)


@register_factor
class KUp(BaseFactor):
    """Upper shadow: (high - max(open,close)) / open. Selling pressure top.
    上影线：(高 - max(开,收))/开，上方卖压。"""
    name = "kup"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        upper = high - torch.maximum(open_, close)
        return upper / open_.clamp(min=1e-8)


@register_factor
class KLow(BaseFactor):
    """Lower shadow: (min(open,close) - low) / open. Buying pressure bottom.
    Wrongly dropped in v12 (corrupted-data artifact); #5 by |IC| on true 1h data.
    下影线：(min(开,收) - 低)/开，下方买压。v12 在污染数据上错杀；真 1h 数据 |IC| 第 5。"""
    name = "klow"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        lower = torch.minimum(open_, close) - low
        return lower / open_.clamp(min=1e-8)


@register_factor
class ROC10(BaseFactor):
    """Rate of change over 10 bars. Pure momentum signal.
    10根K线变化率，纯动量信号。"""
    name = "roc10"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        roc = torch.zeros_like(close)
        roc[10:] = close[10:] / close[:-10].clamp(min=1e-8) - 1.0
        return roc


@register_factor
class CORR(BaseFactor):
    """Rolling correlation between price and volume (10-bar). Volume-price coherence.
    价格与成交量的10根滚动相关性，量价一致性。"""
    name = "corr_pv"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        n, w = close.size(0), 10
        out = torch.zeros_like(close)
        log_v = torch.log(volume.clamp(min=1e-8))
        for i in range(w, n):
            p = close[i-w:i]
            v = log_v[i-w:i]
            pm, vm = p.mean(), v.mean()
            ps, vs = p.std().clamp(min=1e-8), v.std().clamp(min=1e-8)
            out[i] = ((p - pm) * (v - vm)).mean() / (ps * vs)
        return out


@register_factor
class STD20(BaseFactor):
    """Rolling 20-bar return volatility. Risk regime indicator.
    20根滚动收益率标准差，风险状态指示。"""
    name = "std20"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        ret = torch.zeros_like(close)
        ret[1:] = close[1:] / close[:-1].clamp(min=1e-8) - 1.0
        return _rolling_std(ret, 20)


@register_factor
class MAX20Ratio(BaseFactor):
    """Distance from 20-bar high. Resistance proximity indicator.
    距20根最高价的距离，阻力位接近度。"""
    name = "max20_ratio"
    def compute(self, open_, high, low, close, volume) -> Tensor:
        n, w = close.size(0), 20
        out = torch.zeros_like(close)
        for i in range(w, n):
            hi_max = high[i-w:i].max()
            out[i] = close[i] / hi_max.clamp(min=1e-8) - 1.0
        return out
