"""
Banded top-K portfolio construction — THE single shared implementation.
双阈值 banded top-K 组合构建——回测与模拟盘共享的唯一实现。

Novy-Marx & Velikov (RFS 2016) buy/hold spread: a long slot is entered only
when rank < enter_band and exited only when rank >= exit_band (symmetric for
shorts). Used by run_v13_final.py (backtest), run_paper_daily.py (paper),
and research scripts — the ENGINE_CROSSCHECK report attributed part of the
inter-engine divergence to duplicated implementations; this removes that class
of drift. Change only with tests/run_invariants.py green.
仅此一份实现；改动前必须先跑不变量测试。
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def banded_targets(
    scores: np.ndarray,
    long_set: Set[int], short_set: Set[int],
    k: int, enter_band: int, exit_band: int,
) -> Tuple[Set[int], Set[int]]:
    """Index-based core (backtest engines). / 索引版核心（回测用）。"""
    A = len(scores)
    order = np.argsort(-scores)             # best -> worst
    rank_of = np.empty(A, dtype=np.int64)
    rank_of[order] = np.arange(A)

    kept_l = {a for a in long_set if rank_of[a] < exit_band}
    kept_s = {a for a in short_set if rank_of[a] >= A - exit_band}

    new_l = set(kept_l)
    for a in order[:enter_band]:
        if len(new_l) >= k:
            break
        if a not in new_l and a not in kept_s:
            new_l.add(a)

    new_s = set(kept_s)
    for a in order[::-1][:enter_band]:
        if len(new_s) >= k:
            break
        if a not in new_s and a not in new_l:
            new_s.add(a)

    return new_l, new_s


def banded_update_symbols(
    score_dict: Dict[str, float],
    prev_long: List[str], prev_short: List[str],
    k: int, enter_band: int, exit_band: int,
) -> Tuple[List[str], List[str]]:
    """Symbol-based wrapper (paper trading). / 符号版封装（模拟盘用）。"""
    syms = sorted(score_dict.keys())
    idx = {s: i for i, s in enumerate(syms)}
    scores = np.asarray([score_dict[s] for s in syms])
    l0 = {idx[s] for s in prev_long if s in idx}
    s0 = {idx[s] for s in prev_short if s in idx}
    new_l, new_s = banded_targets(scores, l0, s0, k, enter_band, exit_band)
    return sorted(syms[a] for a in new_l), sorted(syms[a] for a in new_s)
