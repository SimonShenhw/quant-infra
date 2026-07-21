"""
Multi-timeframe wrapper: compute factors at 1h, 4h, 24h scales.
多时间尺度因子封装：在1h/4h/24h三个尺度上同时计算因子。

============================================================================
WARNING — KNOWN LOOK-AHEAD BUG, DO NOT USE AS-IS (REVIEW_2026-06-10 M-7)
警告 —— 已知前视 bug，禁止直接使用（REVIEW_2026-06-10 M-7）
============================================================================
_aggregate_to_timeframe writes each block's aggregate (last/max/sum over the
FULL block) back to every bar of the block, INCLUDING the block's first bars —
so a bar at the start of a 24h block sees that block's future close/high/
volume. Any factor built on top inherits look-ahead.
_aggregate_to_timeframe 把每个块的聚合值（整块的 last/max/sum）回填到块内
所有 bar，包括块首——24h 块开头的 bar 能看到该块未来的收盘/最高/成交量。
在此之上构建的任何因子都继承前视。

STATUS: NOT wired into any current pipeline (v13 / paper trading / research
scripts do not import this module) — dead code kept for reference. Before any
future use, the aggregates must be shifted so bar t only sees blocks that
CLOSED at or before t.
状态：未接入任何现行管线（v13/模拟盘/research 脚本均不 import 本模块），
属保留参考的 dead code。未来启用前必须先对聚合值做 shift，保证 bar t 只能
看到在 t 或之前已收盘的块。
============================================================================

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

    LOOK-AHEAD (M-7): `out[i:end] = agg_val` writes the block aggregate back to
    the block START — bars before the block closes see future values.
    前视（M-7）：`out[i:end] = agg_val` 把块聚合值回填到块首——
    块未收盘前的 bar 就能看到未来值。
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

    WARNING: inherits the M-7 look-ahead from _aggregate_to_timeframe for all
    tf > 1 columns. See module docstring before using.
    警告：所有 tf > 1 的列都继承 _aggregate_to_timeframe 的 M-7 前视。
    使用前先看模块 docstring。

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
