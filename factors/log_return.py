"""
1-bar log return: the base momentum/reversal primitive.
1 根 K 线对数收益率：最基础的动量/反转原语。

Economic hypothesis: last-bar return carries short-horizon autocorrelation
(sign and magnitude); log form makes returns additive across bars.
经济假设：上一根 bar 的收益携带短期自相关信息（方向与幅度）；
对数形式使收益跨 bar 可加。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_log_returns
from torch import Tensor

@register_factor
class LogReturn(BaseFactor):
    name = "log_return"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return compute_log_returns(close)
