"""
"btc_dominance" — MISLEADING NAME: this is SELF relative strength, not BTC-anything.
"btc_dominance" —— 名字有误导性：这是相对自身的强弱，与 BTC 无关。

What it ACTUALLY computes (confirmed in the 2026-06-10 full review): each
asset's 6-bar return minus the 24-bar rolling mean of that return — i.e.
deviation from the asset's OWN recent momentum baseline. It never sees BTC,
the cross-section, or any other asset (compute() is per-asset by design).
实际计算内容（2026-06-10 全项目 review 确认）：每个资产的 6-bar 收益减去该收益的
24-bar 滚动均值——即偏离资产自身近期动量基线的程度。它从未接触 BTC、横截面
或任何其他资产（compute() 本就按单资产调用）。

Economic hypothesis: short-horizon momentum spikes above an asset's own
baseline tend to revert (self mean-reversion of momentum).
经济假设：短期动量冲高超出自身基线后倾向回落（动量的自我均值回归）。

The name is kept because trained checkpoints store `factor_names` — renaming
would break checkpoint loading. Documented honestly here instead.
名字保留是因为已训练 checkpoint 中存有 `factor_names`——改名会破坏 ckpt 加载，
故在此如实说明。
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

        # deviation from own rolling mean = self relative strength (NOT vs BTC)
        # 偏离自身滚动均值 = 相对自身的强弱（与 BTC 无关）
        return ret6 - ret6_sma
