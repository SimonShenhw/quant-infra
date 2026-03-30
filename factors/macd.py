from factors.base import BaseFactor, register_factor
from model.features import compute_macd
from torch import Tensor

@register_factor
class MACD(BaseFactor):
    name = "macd"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        line, signal = compute_macd(close)
        return line - signal
