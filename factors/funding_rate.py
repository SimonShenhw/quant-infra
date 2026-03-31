"""
Funding Rate Factor — 资金费率因子。

Crypto-specific: perpetual contract funding rate is a natural reversal signal.
When funding is highly positive, longs are overcrowded → short signal.
When negative, shorts are overcrowded → long signal.

加密货币专属：永续合约资金费率是天然反转信号。
费率高正值 = 多头拥挤 → 做空信号；负值 = 空头拥挤 → 做多信号。

Since we only have spot OHLCV data, we approximate funding pressure using:
  funding_proxy = (close - open) / ATR * volume_ratio
This captures the directional pressure + leverage proxy.

由于仅有现货OHLCV数据，我们用以下方式近似资金费率压力：
  funding_proxy = (close - open) / ATR * volume_ratio
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_sma, compute_ema, _rolling_std
import torch
from torch import Tensor


@register_factor
class FundingRateProxy(BaseFactor):
    name = "funding_rate"

    def compute(
        self, open_: Tensor, high: Tensor, low: Tensor,
        close: Tensor, volume: Tensor,
    ) -> Tensor:
        # ATR proxy: rolling average of (high - low) / 真实波幅近似
        tr: Tensor = high - low
        atr: Tensor = compute_ema(tr, 14).clamp(min=1e-8)

        # directional pressure: (close - open) / ATR / 方向压力
        direction: Tensor = (close - open_) / atr

        # volume intensity: volume / SMA(volume, 20) / 成交量强度
        vol_sma: Tensor = compute_sma(volume, 20).clamp(min=1e-8)
        vol_ratio: Tensor = volume / vol_sma

        # combined: high direction + high volume = crowded trade / 组合：高方向+高量=拥挤交易
        funding_proxy: Tensor = direction * vol_ratio
        return funding_proxy
