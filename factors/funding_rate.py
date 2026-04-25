"""
Funding Rate Factor — 资金费率因子。

Perpetual futures funding rate: a crowd-pressure signal. Extreme positive
funding = longs crowded (reversal short); extreme negative = shorts crowded
(reversal long).

永续合约资金费率：拥挤度信号。极端正值=多头拥挤（反转做空）；
极端负值=空头拥挤（反转做多）。

Real data is loaded from funding_rates.db (populated by
data/funding_archive_downloader.py from data.binance.vision). The caller
passes a pre-aligned 1h funding tensor via `extras["funding"]`.

If real data is unavailable, falls back to an OHLCV-derived proxy:
  funding_proxy = (close - open) / ATR * volume_ratio
"""
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from factors.base import BaseFactor, register_factor
from model.features import compute_sma, compute_ema


@register_factor
class FundingRate(BaseFactor):
    name = "funding_rate"

    def compute(
        self,
        open_: Tensor, high: Tensor, low: Tensor,
        close: Tensor, volume: Tensor,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        # Real funding rate path (preferred) / 优先使用真实资金费率
        if extras is not None and "funding" in extras:
            f: Tensor = extras["funding"]
            if f.numel() == close.numel():
                return f.to(close.device, dtype=close.dtype)

        # Proxy fallback (OHLCV-only) / 后备 OHLCV proxy
        tr = high - low
        atr = compute_ema(tr, 14).clamp(min=1e-8)
        direction = (close - open_) / atr
        vol_sma = compute_sma(volume, 20).clamp(min=1e-8)
        vol_ratio = volume / vol_sma
        return direction * vol_ratio
