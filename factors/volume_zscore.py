"""
Volume Z-Score Factor — 成交量 Z-Score 因子。

WHAT: current volume vs its own trailing 20-bar mean, in units of the
trailing 20-bar standard deviation.
WHAT：当前成交量相对自身过去 20 根 bar 均值的偏离，以过去 20 根 bar 的
标准差为单位。

HYPOTHESIS: participation shift — abnormal volume (many-sigma bars) flags
news arrival or forced flow and should precede price adjustment.
经济假设：参与度迁移——异常放量（数个 σ 的 bar）标记消息到达或被迫成交，
理应先于价格调整。

Track record — NOISE, stated honestly: the 2026-06-10 true-1h IC rerank
(REVIEW_2026-06-10 appendix) found it ~zero at BOTH the 1h and 24h
horizons. It is in v13's DROP_FACTORS (run_v13_final.py) AND excluded
from the extended 18-factor 2021+ research set
(tools/research_extended_window.py). It had also been dropped by v12's
(corrupted-data) ranking — one of the few verdicts that survived the data
fix. The plugin stays registered so old checkpoints' `factor_names` still
resolve and the analyzer can re-test it after future data changes, but no
current result path trains on it.
战绩——噪声，如实交代：2026-06-10 真 1h IC 重排在 1h 与 24h 两个 horizon
上均约为零。它同时在 v13 的 DROP_FACTORS 和扩窗 18 因子研究集的剔除名单里；
v12（污染数据）的排名也剔过它——是少数在数据修复后仍成立的判决之一。
插件保留注册是为了旧 checkpoint 的 `factor_names` 仍可解析、未来数据变更后
analyzer 可复测，但当前任何结果路径都不用它训练。
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_sma, _rolling_std
from torch import Tensor

@register_factor
class VolumeZscore(BaseFactor):
    name = "volume_zscore"
    def compute(self, open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
        vol_mean = compute_sma(volume, 20)
        vol_std = _rolling_std(volume, 20).clamp(min=1e-8)
        return (volume - vol_mean) / vol_std
