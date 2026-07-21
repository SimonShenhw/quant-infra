"""
MACD histogram (line minus signal): classic trend-following momentum.
MACD 柱状图（快线减信号线）：经典趋势跟随动量。

Economic hypothesis: a widening histogram signals accelerating trend.
经济假设：柱体展宽预示趋势加速。

STATUS: NOISE on true 1h data — the 2026-06-10 factor_analyzer rerun after the
timestamp fix measured |IC_1h|=0.0013 / |IC_24h|=0.0055; DROPPED from v13
(one of only two factors cut, with volume_zscore). Kept registered for
research comparisons.
状态：在真 1h 数据上为噪声——时间戳修复后 2026-06-10 重跑 factor_analyzer 测得
|IC_1h|=0.0013 / |IC_24h|=0.0055；v13 已剔除（仅有的两个被剔因子之一，另一个是
volume_zscore）。保留注册以供研究对照。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_macd
from torch import Tensor

@register_factor
class MACD(BaseFactor):
    name = "macd"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        line, signal = compute_macd(close)
        return line - signal
