from factors.base import BaseFactor, register_factor
from model.features import compute_bollinger
from torch import Tensor

@register_factor
class BollingerPosition(BaseFactor):
    name = "bollinger"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        _, upper, lower = compute_bollinger(close, 20)
        return (close - lower) / (upper - lower).clamp(min=1e-8)
