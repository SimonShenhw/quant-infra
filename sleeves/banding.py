"""
Banded top-K portfolio construction — THE single shared implementation.
双阈值 banded top-K 组合构建——回测与模拟盘共享的唯一实现。

WHAT — Novy-Marx & Velikov (RFS 2016) buy/hold spread: a long slot is entered
only when rank < enter_band and exited only when rank >= exit_band (symmetric
for shorts). The asymmetric thresholds create a no-trade band, and turnover
suppression is where the money is: on identical v13 OOS predictions and
costs, construction alone swings the backtest from -44.5% (plain hourly
top-1) to +32.6% (banded top-3) — see README results table.

WHY one shared module: this used to live as near-duplicate code in the
backtest engine and the paper script, and the ENGINE_CROSSCHECK report
attributed part of its ±1pp inter-engine divergence to exactly that
duplicated-implementation drift. run_v13_final.py (backtest),
run_paper_daily.py (paper), and the research scripts now all import THIS
function, so backtest and live can no longer disagree about what "banded"
means. Change only with tests/run_invariants.py green.
双阈值构成“持有带”，弱信号全靠压换手才能在成本下存活（-44.5% → +32.6%）；
回测与模拟盘曾各写一份、贡献了 ±1pp 引擎分歧——现在仅此一份实现；
改动前必须先跑不变量测试。
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np


def banded_targets(
    scores: np.ndarray,
    long_set: Set[int], short_set: Set[int],
    k: int, enter_band: int, exit_band: int,
) -> Tuple[Set[int], Set[int]]:
    """Index-based core (backtest engines): one banded update step.
    索引版核心（回测用）：单步 banded 更新。

    Hysteresis rules / 滞回规则:
      - HOLD an existing long slot while rank < exit_band (even if it left
        the top-k) — small rank wiggles must not trigger a trade;
      - fill vacancies only from rank < enter_band, best-ranked first, so a
        slot can sit empty rather than chase a mediocre name;
      - mirror-image from the bottom for shorts; on a contested name the
        long side (processed first) wins, a short never flips a held long.
    enter_band < exit_band is what makes this a buy/hold SPREAD rather than
    plain top-K. / 进场阈值严于离场阈值：排名小幅抖动不换仓，宁缺毋滥。"""
    A = len(scores)
    order = np.argsort(-scores)             # best -> worst
    rank_of = np.empty(A, dtype=np.int64)
    rank_of[order] = np.arange(A)

    # Incumbents survive while inside the (wider) exit band — the hold leg
    # of the spread. / 在场者只要仍在较宽的 exit band 内就续持。
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
    """Symbol-based wrapper (paper trading): maps score dicts to indices and
    back, returning SORTED symbol lists — the ledgers store them as JSON, so
    deterministic ordering keeps rows byte-stable across reruns.
    符号版封装（模拟盘用）：输出排序后的符号列表，保证账本行确定性。"""
    syms = sorted(score_dict.keys())
    idx = {s: i for i, s in enumerate(syms)}
    scores = np.asarray([score_dict[s] for s in syms])
    # A held symbol absent from score_dict silently vanishes WITHOUT booking
    # its exit — this is why the delisting SOP (ROADMAP G1) settles positions
    # manually in the ledger BEFORE editing the universe.
    # 持仓符号若缺席打分会“无声蒸发”（离场损益永不确认）：退市必须先按
    # G1 SOP 手工确认离场再改宇宙。
    l0 = {idx[s] for s in prev_long if s in idx}
    s0 = {idx[s] for s in prev_short if s in idx}
    new_l, new_s = banded_targets(scores, l0, s0, k, enter_band, exit_band)
    return sorted(syms[a] for a in new_l), sorted(syms[a] for a in new_s)
