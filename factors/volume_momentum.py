"""
Volume Momentum Factor — 成交量动量因子。

WHAT: short-term volume acceleration — SMA(volume, 6h) / SMA(volume, 48h) - 1.
WHAT：短期成交量加速度——SMA(6h) / SMA(48h) - 1。

HYPOTHESIS: participation shift. A rising short/long volume ratio means
the market is suddenly more crowded than its own recent baseline —
institutional entry or panic — and such spikes tend to precede short-term
reversal or regime change. Using two SMAs (not raw volume) makes the
signal self-normalizing across assets with wildly different base turnover.
经济假设：参与度迁移。短/长量比抬升意味着市场突然比自身近期基线更拥挤——
机构进场或恐慌——这类尖峰倾向预示短期反转或状态切换。用两条 SMA 而非
原始量，使信号在基础换手率相差悬殊的资产间自归一。

Track record — BORDERLINE, stated honestly: v11.1/v12's IC ranking
(computed on data later shown to be corrupted by the ms/µs timestamp bug)
dropped it as noise (|IC_1h| < 0.003). The 2026-06-10 true-1h rerank did
NOT confirm that verdict — it survives in v13's 19-factor set (only macd
and volume_zscore remain dropped). Treat it as weak but not dead.
战绩——边缘因子，如实交代：v11.1/v12 的 IC 排名（后来证实基于 ms/µs 时间戳
bug 污染的数据）把它当噪声剔除（|IC_1h| < 0.003）；2026-06-10 真 1h 重排
未确认该判决——它保留在 v13 的 19 因子集中（DROP 名单只剩 macd 与
volume_zscore）。定位：弱，但没死。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_sma
import torch
from torch import Tensor


@register_factor
class VolumeMomentum(BaseFactor):
    name = "volume_momentum"

    def compute(
        self, open_: Tensor, high: Tensor, low: Tensor,
        close: Tensor, volume: Tensor,
    ) -> Tensor:
        # short-term volume: SMA(6h) / 短期成交量
        vol_short: Tensor = compute_sma(volume, 6).clamp(min=1e-8)
        # long-term volume: SMA(48h) / 长期成交量
        vol_long: Tensor = compute_sma(volume, 48).clamp(min=1e-8)
        # ratio: >1 means volume accelerating / 比率>1表示量在放大
        return vol_short / vol_long - 1.0
