from factors.base import BaseFactor, register_factor
from model.features import compute_ema
from torch import Tensor

@register_factor
class EMA10Ratio(BaseFactor):
    name = "ema10_ratio"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return close / compute_ema(close, 10).clamp(min=1e-8) - 1.0
