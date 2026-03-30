from factors.base import BaseFactor, register_factor
from model.features import compute_log_returns
from torch import Tensor

@register_factor
class LogReturn(BaseFactor):
    name = "log_return"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return compute_log_returns(close)
