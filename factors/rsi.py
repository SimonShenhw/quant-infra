from factors.base import BaseFactor, register_factor
from model.features import compute_rsi
from torch import Tensor

@register_factor
class RSI(BaseFactor):
    name = "rsi"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return (compute_rsi(close, 14) - 50.0) / 50.0
