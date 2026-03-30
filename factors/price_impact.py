from factors.base import BaseFactor, register_factor
from model.obi_features import compute_price_impact
from torch import Tensor

@register_factor
class PriceImpact(BaseFactor):
    name = "price_impact"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return compute_price_impact(close, volume, window=20)
