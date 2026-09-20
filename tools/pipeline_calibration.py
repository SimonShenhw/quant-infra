"""
Pipeline calibration: what signal strength can this construction monetize?
管线标定：这套组合构造究竟需要多强的信号才能赚到钱？（只读诊断）
Usage: python tools/pipeline_calibration.py [--quick] [--seeds 5]

WHAT: feeds SYNTHETIC score matrices of KNOWN signal quality through the
REAL v13 backtest path (run_v13_final.run_backtest with the config-C
parameters: banded top-3, enter<3 / exit>=6, daily decisions, TWAP cost
model) and records, for each arm, the MEASURED cross-sectional rank IC
against the dollar outcome. Three arms:

  NULL      pure noise scores, zero information.
            Must NOT produce profit -- this is the machinery's null test.
  UNIFORM   s = a*z(y24) + sqrt(1-a^2)*eps, oracle accuracy spread evenly
            across the cross-section. Sweeping `a` traces the break-even
            curve: how much rank IC does this construction need?
  SMALLMOVE same construction, but oracle accuracy concentrated on the
            SMALL |y24| names and ~zero on the big movers -- a synthetic
            reproduction of the diagnosed v13 failure mode ("orders small
            pairs right, puts big movers on the wrong side",
            RESEARCH_2026-07-13_extended_window.md).

WHY THIS EXISTS: FALSIFICATION_2026-09-19.md section 6 lists, under "not
established", the fact that nothing in this project ever demonstrated the
converse direction -- that IF a real signal existed, the pipeline would
have found and monetized it. Every negative result so far is therefore
ambiguous between "no signal in the market" and "signal present, pipeline
cannot convert it". A planted-signal recovery test is what separates the
two, and it needs no alpha to run. (The same device is used by the
independent project github.com/Jareedd/qr-alpha-lab, which validates its
pipeline by recovering a planted signal and rejecting pure noise before
reporting that its 13 real-data trials all failed.)

HOW TO READ THE OUTPUT: the x-axis is the MEASURED rank IC at decision
bars, computed with the same Spearman used by the live-IC judge, so it is
directly comparable to v13's OOS ensemble rank IC of 0.064. The headline
number is the break-even IC -- the smallest measured IC at which the arm's
median total return crosses zero. If break-even sits far above 0.064, the
construction never had a chance with the signal the model actually
produced, and that is a statement about CONSTRUCTION, not about the market.

*** THESE SCORES USE FUTURE INFORMATION BY DESIGN. ***
They are built from the realized label y24. Nothing here is a tradeable
result or a strategy; it is a calibration of the construction's transfer
function from rank accuracy to dollars. Any number from this file that
escapes into a performance claim is a lie.
*** 本文件的分数按设计使用未来信息（由已实现的 y24 构造），不是策略、
不是可交易结果，只是"排名准确度 -> 美元"这条传递函数的标定。***

Not a selection trial -> deliberately NOT registered in trials.json: no
strategy candidate is being chosen here, and no output feeds a live system.
不是 selection trial，刻意不进 trials.json。

UNIVERSE IS PINNED, AND THAT MATTERS: run_v13_final.build_from_parquet
resolves its universe as sorted(lake_symbols)[:max_assets]. The lake has
grown since v13 trained (27 symbols now), so that expression SILENTLY
returns a different 20 coins today than it did then -- it now picks up
FET/FIL/PEPE/RENDER and drops SOL/SUI/UNI/XRP relative to the live basket.
This tool therefore pins the universe to the live basket's own first-day
symbol list and patches the loader accordingly. See the same hazard,
independently found, in qr-alpha-lab's survivorship case study (static
universe Sharpe +0.82 vs point-in-time -0.01).
⚠️ build_from_parquet 的宇宙随 lake 增长而静默漂移，故此处显式钉死为
live 篮子首日宇宙。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

import run_v13_final as V13                      # noqa: E402
from data.lake_loader import load_klines_multi as _real_loader  # noqa: E402
from paper_live_ic import spearman               # noqa: E402

DB = BASE / "paper_daily.db"
SEQ_LEN = 24
# v13 config C — the configuration that produced the +32.6% headline.
# 与 +32.6% 头条同配置。
CFG_C = dict(k=V13.BASKET_K, enter_band=V13.ENTER_BAND,
             exit_band=V13.EXIT_BAND, decision_every=V13.DECISION_EVERY,
             min_hold=0, use_vol_filter=False)
V13_OOS_IC = 0.064        # backtest OOS ensemble rank IC / 回测口径
V13_LIVE_IC = 0.0195      # what the deployed model ACTUALLY delivered live
                          # (96 in-window marks, t=+0.75 -> indistinguishable
                          # from zero; FALSIFICATION_2026-09-19.md section 2)
                          # 部署后实际交付的 live IC——这才是该比的那个数


def live_universe() -> list[str]:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT all_closes FROM basket_state ORDER BY date LIMIT 1").fetchone()
    conn.close()
    return sorted(json.loads(row[0]).keys())


def load_pinned(symbols: list[str]):
    """build_from_parquet with the universe pinned (see module docstring).
    Patching the loader keeps every one of build_from_parquet's alignment
    and label conventions intact — re-deriving them here would be exactly
    the duplicate-implementation drift this repo keeps getting burned by.
    钉死宇宙后复用 build_from_parquet，绝不另写一份对齐/标签逻辑。"""
    def _pinned(interval="5m", min_rows=10_000, **kw):
        raw = _real_loader(symbols=symbols, interval=interval,
                           min_rows=min_rows, **kw)
        missing = [s for s in symbols if s not in raw]
        if missing:
            raise RuntimeError(f"pinned symbols missing from lake: {missing}")
        return raw
    orig = V13.load_klines_multi
    V13.load_klines_multi = _pinned
    try:
        return V13.build_from_parquet(SEQ_LEN, len(symbols),
                                      torch.device("cpu"))
    finally:
        V13.load_klines_multi = orig


def zscore_rows(a: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score per row (demeaned, unit sd). 逐行横截面标准化。"""
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True)
    return (a - m) / np.maximum(s, 1e-12)


def pct_rank_rows(a: np.ndarray) -> np.ndarray:
    """Per-row percentile rank in [0,1]. 逐行百分位排名。"""
    order = np.argsort(a, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(a.shape[0])[:, None]
    ranks[rows, order] = np.arange(a.shape[1])[None, :]
    return ranks / max(a.shape[1] - 1, 1)


def make_scores(y24: np.ndarray, arm: str, a: float, rng) -> np.ndarray:
    """Synthetic scores of controlled quality. y24 is (n_samples, A).
    受控质量的合成分数。"""
    z = zscore_rows(y24)
    eps = rng.standard_normal(y24.shape)
    if arm == "NULL":
        return eps
    if arm == "UNIFORM":
        return a * z + math.sqrt(max(1.0 - a * a, 0.0)) * eps
    if arm == "SMALLMOVE":
        # accuracy weight 2 for the smallest |y24| name, ~0 for the largest
        # 权重：|y24| 最小者 ~2，最大者 ~0
        w = 2.0 * (1.0 - pct_rank_rows(np.abs(y24)))
        ai = np.clip(a * w, 0.0, 0.99)
        return ai * z + np.sqrt(np.maximum(1.0 - ai * ai, 0.0)) * eps
    raise ValueError(arm)


def measured_rank_ic(scores: np.ndarray, y24: np.ndarray,
                     decision_every: int) -> float:
    """Mean cross-sectional rank IC at DECISION bars only — the same
    quantity, same Spearman, as the registered live-IC judge, so the
    x-axis is comparable to v13's OOS 0.064.
    仅在决策bar上取横截面 rank IC，与注册判官同一算术，可与 0.064 直比。"""
    ics = []
    for t in range(0, scores.shape[0], decision_every):
        ics.append(spearman(list(scores[t]), list(y24[t])))
    return float(np.mean(ics))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true",
                    help="coarse grid, 2 seeds / 粗网格快跑")
    args = ap.parse_args()

    syms = live_universe()
    print("=" * 74)
    print("  PIPELINE CALIBRATION — planted-signal recovery through the")
    print("  REAL v13 backtest path (banded top-3, enter<3/exit>=6, TWAP costs)")
    print("  *** scores are built from the realized label: NOT a strategy ***")
    print("=" * 74)
    print(f"  universe PINNED to the live basket ({len(syms)} symbols)")

    X, y24_t, r1h_t, close_t, got, n_factors = load_pinned(syms)
    del X  # factors unused here; only labels/returns/closes matter / 只用标签与收益
    y24 = y24_t.cpu().numpy()
    n_samples, A = y24.shape
    years = n_samples / (24 * 365)
    print(f"  aligned window: {n_samples:,} hourly samples "
          f"({years:.2f} years), {A} assets")
    print(f"  config C: k={CFG_C['k']}, enter<{CFG_C['enter_band']}, "
          f"exit>={CFG_C['exit_band']}, every {CFG_C['decision_every']}h")

    grid = ([0.0, 0.05, 0.15, 0.30] if args.quick
            else [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.30, 0.45])
    n_seeds = 2 if args.quick else args.seeds
    valid_mask = torch.ones(n_samples, dtype=torch.bool)

    rows = []
    for arm in ("NULL", "UNIFORM", "SMALLMOVE"):
        levels = [0.0] if arm == "NULL" else [g for g in grid if g > 0]
        for a in levels:
            ics, rets_tot, sharpes = [], [], []
            for sd in range(n_seeds):
                rng = np.random.default_rng(1000 * sd + int(a * 1000))
                sc = make_scores(y24, arm, a, rng)
                ics.append(measured_rank_ic(sc, y24, CFG_C["decision_every"]))
                random.seed(V13.SEED)   # same execution randomness as main()
                res = V13.run_backtest(
                    f"{arm}_a{a}", torch.from_numpy(sc.astype(np.float32)),
                    valid_mask, r1h_t, close_t, SEQ_LEN,
                    CFG_C["k"], CFG_C["enter_band"], CFG_C["exit_band"],
                    CFG_C["decision_every"], CFG_C["min_hold"],
                    CFG_C["use_vol_filter"])
                rets_tot.append(res["total_return"])
                sharpes.append(res["sharpe"])
            rows.append(dict(arm=arm, a=a,
                             ic=float(np.mean(ics)),
                             ret_med=float(np.median(rets_tot)),
                             ret_lo=float(np.min(rets_tot)),
                             ret_hi=float(np.max(rets_tot)),
                             sharpe_med=float(np.median(sharpes))))
            r = rows[-1]
            print(f"    {arm:<10} a={a:<5} measured IC {r['ic']:+.4f}  "
                  f"Sharpe {r['sharpe_med']:+6.2f}  "
                  f"ret {r['ret_med']:+.1%}")

    print("\n" + "=" * 74)
    print(f"  {'arm':<10} {'measured IC':>12} {'median ret':>12} "
          f"{'median Sharpe':>14}")
    print("-" * 74)
    for r in rows:
        print(f"  {r['arm']:<10} {r['ic']:>+12.4f} {r['ret_med']:>+12.1%} "
              f"{r['sharpe_med']:>+14.2f}")

    def breakeven(arm: str):
        pts = sorted([(r["ic"], r["ret_med"]) for r in rows if r["arm"] == arm])
        for (i0, v0), (i1, v1) in zip(pts[:-1], pts[1:]):
            if v0 <= 0 < v1:      # linear interpolation on the crossing
                return i0 + (i1 - i0) * (-v0) / max(v1 - v0, 1e-12)
        return None

    print("\n  BREAK-EVEN rank IC (median total return crosses zero):")
    for arm in ("UNIFORM", "SMALLMOVE"):
        be = breakeven(arm)
        if be is None:
            print(f"    {arm:<10} not bracketed inside the tested grid")
        else:
            print(f"    {arm:<10} IC ~ {be:+.4f}"
                  f"   = {be / V13_OOS_IC:.2f}x the BACKTEST IC ({V13_OOS_IC:.3f})"
                  f"   = {be / V13_LIVE_IC:.2f}x the LIVE IC ({V13_LIVE_IC:.4f})")
    null = [r for r in rows if r["arm"] == "NULL"][0]
    print(f"\n  NULL arm (zero information): median ret {null['ret_med']:+.1%}, "
          f"Sharpe {null['sharpe_med']:+.2f}")
    print("    -> the machinery must NOT be profitable here; a positive NULL")
    print("       would invalidate every other number this repo has produced.")
    print()


if __name__ == "__main__":
    main()
