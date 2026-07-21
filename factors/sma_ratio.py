"""
SMA Ratio Factors — 均线比率因子。

WHAT: relative gap between close and its own simple moving average,
close / SMA(n) - 1, at two windows (5 and 20 bars).
WHAT：收盘价相对自身简单均线的偏离度 close / SMA(n) - 1，两个窗口（5 与 20 根 bar）。

HYPOTHESIS: trend vs mean-reversion positioning. Price above its moving
average = trend-following pressure (momentum continuation); a large gap =
overextension (mean-reversion pull). The model, not this file, decides
which regime dominates — the factor just exposes the raw displacement.
Two windows give a fast and a slow read of the same quantity, and their
difference implicitly encodes the classic 5/20 crossover state.
经济假设：趋势 vs 均值回归的仓位含义。价格在均线上方 = 趋势延续压力；
偏离过大 = 过度伸展（回归拉力）。哪种状态占主导由模型判断——本因子只暴露
原始偏离量。快慢两个窗口读同一量，其差隐式编码了经典的 5/20 金叉/死叉状态。

Ratio (not difference) makes the value scale-free across assets: a $500
move means nothing without dividing by price level.
用比率而非差值使因子跨资产无量纲：不除以价位，500 美元的偏离毫无可比性。

Track record: both kept in v13's 19-factor set after the true-1h IC rerank.
战绩：真 1h IC 重排后两者均保留在 v13 的 19 因子集中。
"""
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
