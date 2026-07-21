"""
Bollinger band position: where close sits inside the 20-bar band (0=lower, 1=upper).
布林带位置：收盘价在 20 根带内的相对位置（0=下轨，1=上轨）。

Economic hypothesis: mean-reversion pressure — price stretched to a band
extreme (relative to its own recent volatility) tends to revert.
经济假设：均值回归压力——价格相对自身近期波动被拉伸到带边缘时倾向回归。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_bollinger
from torch import Tensor

@register_factor
class BollingerPosition(BaseFactor):
    name = "bollinger"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        _, upper, lower = compute_bollinger(close, 20)
        return (close - lower) / (upper - lower).clamp(min=1e-8)
