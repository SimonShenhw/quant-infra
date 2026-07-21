"""
RSI Factor — 相对强弱指标因子。

WHAT: Wilder's 14-period Relative Strength Index, rescaled from [0, 100]
to [-1, 1] via (RSI - 50) / 50.
WHAT：Wilder 14 期相对强弱指标，经 (RSI - 50) / 50 从 [0, 100] 重标到 [-1, 1]。

HYPOTHESIS: mean-reversion pressure. RSI measures how one-sided recent
gains vs losses have been; extreme readings (overbought/oversold) proxy
short-term exhaustion of the marginal buyer/seller, so price tends to snap
back. Centering at 0 puts "no pressure" at zero BEFORE the registry's
shared rolling z-score, so the normalization measures deviation from
neutral rather than deviation from wherever RSI happened to drift.
经济假设：均值回归压力。RSI 衡量近期涨跌的单边程度；极端读数（超买/超卖）
代理边际买/卖方的短期衰竭，价格倾向回摆。先居中到 0 再进注册表的共享滚动
z-score，归一化度量的才是"偏离中性"而非"偏离 RSI 恰好漂到的位置"。

Track record: kept in v13's 19-factor set after the true-1h IC rerank
(REVIEW_2026-06-10 appendix).
战绩：真 1h IC 重排后保留在 v13 的 19 因子集中。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_rsi
from torch import Tensor

@register_factor
class RSI(BaseFactor):
    name = "rsi"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return (compute_rsi(close, 14) - 50.0) / 50.0
