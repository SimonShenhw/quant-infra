"""
BTC Dominance Factor — BTC主导地位因子。

When BTC outperforms the cross-sectional average, altcoins tend to underperform.
This factor measures each asset's relative strength vs BTC.

当BTC跑赢横截面均值时，山寨币倾向于跑输。
该因子衡量每个资产相对于BTC的强弱。

Since we compute per-asset, this factor captures:
  relative_strength = asset_return - asset_sma_return
  (deviation from own mean — assets reverting to their own mean)
"""
from factors.base import BaseFactor, register_factor
from model.features import compute_sma
import torch
from torch import Tensor


@register_factor
class RelativeStrength(BaseFactor):
    name = "btc_dominance"

    def compute(
        self, open_: Tensor, high: Tensor, low: Tensor,
        close: Tensor, volume: Tensor,
    ) -> Tensor:
        # 6-bar return (cumulative over ~6 hours) / 6bar累计收益
        ret6: Tensor = torch.zeros_like(close)
        ret6[6:] = close[6:] / close[:-6].clamp(min=1e-8) - 1.0

        # 24-bar rolling mean of returns / 24bar滚动均值
        ret6_sma: Tensor = compute_sma(ret6, 24)

        # deviation from own mean = relative strength / 偏离自身均值 = 相对强弱
        return ret6 - ret6_sma
