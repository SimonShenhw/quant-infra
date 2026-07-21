"""
EMA10 stretch: close relative to its 10-bar EMA, minus 1.
EMA10 拉伸度：收盘价相对 10 根 EMA 的偏离，减 1。

Economic hypothesis: price stretched away from its short EMA either continues
(momentum) or snaps back (mean reversion); the model learns which regime applies.
经济假设：价格偏离短期 EMA 后要么延续（动量）要么回抽（均值回归），
由模型学习当前属于哪种状态。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_ema
from torch import Tensor

@register_factor
class EMA10Ratio(BaseFactor):
    name = "ema10_ratio"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return close / compute_ema(close, 10).clamp(min=1e-8) - 1.0
