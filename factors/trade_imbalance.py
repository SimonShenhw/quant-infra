"""
Trade Imbalance Factor — 成交失衡因子。

WHAT: tick-rule order-flow imbalance over a 10-bar window (delegates to
model/obi_features.compute_trade_imbalance): each bar's volume is signed
by the tick rule (close uptick = buy, downtick = sell, unchanged = carry
previous direction), then buy vs sell volume is netted.
WHAT：10 根 bar 窗口上的 tick 规则订单流失衡（委托给 model/obi_features）：
按 tick 规则给每根 bar 的成交量定号（收盘上涨=买、下跌=卖、不变=沿用前号），
再对买/卖量轧差。

HYPOTHESIS: order-flow pressure. Persistent net buying (selling) reveals
informed or forced flow that prices have not fully absorbed — a standard
microstructure predictor. True trade-side flags don't exist in OHLCV bars,
so the tick rule is the classic proxy (Lee-Ready style) — this is an
order-flow ESTIMATE from bar data, not real LOB imbalance; the real-LOB
version lives in the ws_daemon/OBI research path.
经济假设：订单流压力。持续的净买（卖）暴露价格尚未完全吸收的知情/被迫流，
是标准的微观结构预测量。OHLCV bar 没有真实买卖方向标记，tick 规则是经典
代理（Lee-Ready 思路）——这是从 bar 数据估计的订单流，不是真实 LOB 失衡；
真实 LOB 版本在 ws_daemon/OBI 研究线里。

Track record: kept in v13's 19-factor set after the true-1h IC rerank.
战绩：真 1h IC 重排后保留在 v13 的 19 因子集中。
"""
from factors.base import BaseFactor, register_factor
from model.obi_features import compute_trade_imbalance
from torch import Tensor

@register_factor
class TradeImbalance(BaseFactor):
    name = "trade_imbalance"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return compute_trade_imbalance(close, volume, window=10)
