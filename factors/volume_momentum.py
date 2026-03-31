"""
Volume Momentum Factor — 成交量动量因子。

Measures short-term volume acceleration: ratio of recent volume to
longer-term average. Spikes indicate institutional activity or
panic — both are predictive of short-term reversal.

衡量短期成交量加速度：近期成交量与长期均值的比率。
尖峰表明机构活动或恐慌——两者都能预测短期反转。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_sma
import torch
from torch import Tensor


@register_factor
class VolumeMomentum(BaseFactor):
    name = "volume_momentum"

    def compute(
        self, open_: Tensor, high: Tensor, low: Tensor,
        close: Tensor, volume: Tensor,
    ) -> Tensor:
        # short-term volume: SMA(6h) / 短期成交量
        vol_short: Tensor = compute_sma(volume, 6).clamp(min=1e-8)
        # long-term volume: SMA(48h) / 长期成交量
        vol_long: Tensor = compute_sma(volume, 48).clamp(min=1e-8)
        # ratio: >1 means volume accelerating / 比率>1表示量在放大
        return vol_short / vol_long - 1.0
