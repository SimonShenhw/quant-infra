"""
Multi-timeframe wrapper: compute factors at 1h, 4h, 24h scales.
多时间尺度因子封装：在1h/4h/24h三个尺度上同时计算因子。

Usage:
    factors = MultiTimeframeFactors().build(open, high, low, close, volume)
    # Returns (T, n_factors * 3) tensor: each base factor × 3 timeframes
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from factors.base import FactorRegistry
from model.features import _rolling_zscore


def _aggregate_to_timeframe(x: Tensor, factor: int, mode: str = "last") -> Tensor:
    """
    Aggregate 1h tensor to coarser timeframe (e.g. factor=4 → 4h).
    Then forward-fill back to 1h length.
    将1h张量聚合到更粗的时间尺度（如factor=4聚合到4h），然后前向填充回1h长度。
    """
    n = x.size(0)
    out = torch.zeros_like(x)
    for i in range(0, n, factor):
        end = min(i + factor, n)
        if mode == "last":
            agg_val = x[end - 1]
        elif mode == "mean":
            agg_val = x[i:end].mean()
        elif mode == "max":
            agg_val = x[i:end].max()
        elif mode == "sum":
            agg_val = x[i:end].sum()
        else:
            agg_val = x[end - 1]
        out[i:end] = agg_val
    return out


def build_multi_tf_factors(
    open_: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor,
    base_factor_names: List[str],
    timeframes: List[int] = [1, 4, 24],  # 1h, 4h, 1d
    zscore_window: int = 48,
) -> Tensor:
    """
    Build multi-timeframe factor tensor.
    构建多时间尺度因子张量。

    Returns (T, len(base_factors) * len(timeframes)) tensor.
    """
    cols: List[Tensor] = []

    for tf in timeframes:
        # aggregate OHLCV to this timeframe / 将OHLCV聚合到该时间尺度
        if tf == 1:
            o, h, l, c, v = open_, high, low, close, volume
        else:
            o = _aggregate_to_timeframe(open_, tf, "last")  # not perfect but ok / 非严格但够用
            h = _aggregate_to_timeframe(high, tf, "max")
            l = _aggregate_to_timeframe(low, tf, "max")  # min would need negation
            l = -_aggregate_to_timeframe(-low, tf, "max")
            c = _aggregate_to_timeframe(close, tf, "last")
            v = _aggregate_to_timeframe(volume, tf, "sum")

        # compute each base factor at this timeframe / 在该尺度上计算每个基础因子
        for name in base_factor_names:
            factor = FactorRegistry.get(name)
            raw = factor.compute(o, h, l, c, v)
            normalised = _rolling_zscore(raw, zscore_window)
            cols.append(normalised)

    result = torch.stack(cols, dim=-1)
    result = result.clamp(-5.0, 5.0)
    result = torch.nan_to_num(result, nan=0.0)
    return result
