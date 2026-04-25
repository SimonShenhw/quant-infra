"""
Qlib-inspired factor pack — 8 high-quality factors from Microsoft Qlib's Alpha158.
受 Qlib 启发的因子包 — 8 个来自 Microsoft Qlib Alpha158 的高质量因子。

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
    K线长度：(高-低)/开，波动率代理。"""
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
    下影线：(min(开,收) - 低)/开，下方买压。"""
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
