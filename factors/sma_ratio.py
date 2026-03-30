from factors.base import BaseFactor, register_factor
from model.features import compute_sma
from torch import Tensor

@register_factor
class SMA5Ratio(BaseFactor):
    name = "sma5_ratio"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return close / compute_sma(close, 5).clamp(min=1e-8) - 1.0

@register_factor
class SMA20Ratio(BaseFactor):
    name = "sma20_ratio"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return close / compute_sma(close, 20).clamp(min=1e-8) - 1.0
