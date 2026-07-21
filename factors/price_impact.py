"""
Amihud price impact: rolling |return| per unit volume (20-bar).
Amihud 价格冲击：滚动 20 根的 |收益| / 成交量。

Economic hypothesis: Amihud (2002) illiquidity premium — assets whose price
moves a lot per unit of volume are illiquid and demand higher expected returns;
in crypto it also flags thin books where flow pushes price.
经济假设：Amihud (2002) 非流动性溢价——单位成交量引起更大价格波动的资产
流动性差、要求更高预期收益；在 crypto 中还标记盘口薄、资金流易推动价格的币。

STATUS: ranked among the STRONG factors in the 2026-06-10 true-1h IC rerank
(alongside std20/klen); retained in v13's 19-factor set.
状态：2026-06-10 真 1h IC 重排中位列强因子（与 std20/klen 同档）；
保留在 v13 的 19 因子集合中。
"""
from factors.base import BaseFactor, register_factor
from model.obi_features import compute_price_impact
from torch import Tensor

@register_factor
class PriceImpact(BaseFactor):
    name = "price_impact"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        return compute_price_impact(close, volume, window=20)
