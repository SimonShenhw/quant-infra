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
