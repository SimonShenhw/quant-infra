from factors.base import BaseFactor, register_factor
from model.obi_features import compute_trade_imbalance
from torch import Tensor

@register_factor
class TradeImbalance(BaseFactor):
    name = "trade_imbalance"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return compute_trade_imbalance(close, volume, window=10)
