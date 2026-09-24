"""
w·y24 decomposition — Track A of the 2026-09-24 continuation plan.
w·y24 分解：为什么逐年为正的 rank IC 没有变成美元 P&L。（只读诊断）
Usage: python tools/wy24_decomposition.py

PRE-REGISTERED in PREREG_2026-09-24_wy24_decomposition.md (commit fbafd93),
committed BEFORE this script was first run. Read that file first: it fixes
the design, the stop rules, and three predictions with stated confidence.
本分析已预注册（fbafd93，先于首次运行提交），设计、停止规则与三条预测见该文件。

WHAT: the extended-window research reported rank IC positive in EVERY year
2021–2026 yet rank-weighted, 24h label-aligned gross P&L (w·y24) of
−0.0334%/day. Because the diagnostic uses RANK weights, the score-scale and
score-tail channels are excluded by construction; the shortfall between
the P&L the IC "should" produce and what was observed can only come from
(a) return-side tails or (b) dispersion timing. A 2×2 grid isolates them:

                       raw y (tails kept)      ranked y (tails removed)
  dispersion kept      A0 = w·y                A2' = w·q̃·σ_t
  dispersion removed   A1 = w·(y/σ_t)·σ̄        A2  = w·q̃·σ̄

A2 is the IC-implied P&L (rank-vs-rank, days equally weighted, at mean
dispersion). The shortfall A2 − A0 is split by Shapley averaging over both
removal orders. A3 regresses daily Spearman IC on σ_t directly.

Single-implementation reuse: build_window / WINDOWS (research_extended_
window), rank_weights / trailing_inv_vol / DECISION_EVERY (research_rank_
weighted), spearman (paper_live_ic). Only the Newey–West helper is local.
Diagnostic of an already-falsified system; not registered in trials.json.

中文：诊断用的是 rank 权重，分数尺度与分数尾部渠道在构造上已排除；
A2（IC 应得的 P&L）与 A0（实际）之差只能来自收益尾部 (a) 或离散度择时 (b)，
用 2×2 网格按两种移除顺序做 Shapley 分摊；A3 直接检验 IC 与离散度的关系。
全部复用既有单一实现，只有 Newey–West 是本地的。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from tools.research_extended_window import build_window, WINDOWS  # noqa: E402
from tools.research_rank_weighted import (  # noqa: E402
    rank_weights, trailing_inv_vol, DECISION_EVERY)
from tools.paper_live_ic import spearman  # noqa: E402

PREDS = BASE / "checkpoints" / "preds_ext_full.npy"
ARCHIVE = BASE / "checkpoints" / "research_rank_weighted.json"
OUT = BASE / "checkpoints" / "wy24_decomposition.json"
REPRO_TOL = 1e-8


def nw_t(x: np.ndarray, lag: int) -> tuple[float, float, float]:
    """(mean, Newey–West SE, t) with Bartlett weights.
    WHY HAC even though samples are non-overlapping: crypto P&L has
    volatility clustering, so adjacent days are not independent in
    variance; lag 5 plus a lag-10 robustness row is conservative here.
    样本不重叠，但加密 P&L 有波动率聚集，故仍用 HAC（lag 5，另报 lag 10）。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = x.mean()
    e = x - m
    var = (e @ e) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = float(np.sqrt(max(var, 0.0) / n))
    return float(m), se, float(m / se) if se > 0 else float("nan")


def nw_slope(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float]:
    """OLS slope of y on x with a Newey–West t-stat. / 带 NW 的回归斜率。"""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    n = len(y)
    Xu = X * u[:, None]
    S = Xu.T @ Xu / n
    for k in range(1, lag + 1):
        G = Xu[k:].T @ Xu[:-k] / n
        S += (1.0 - k / (lag + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    V = Q @ S @ Q / n
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1]))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 74)
    print("  w·y24 DECOMPOSITION (Track A) — pre-registered, commit fbafd93")
    print("=" * 74)

    scores = np.load(PREDS)
    n_pred = scores.shape[0]
    X, y24_t, r1h_t, close_t, bar_ms, n_factors = build_window(
        WINDOWS["FULL"], device)
    n_now = X.size(0)
    del X, close_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  preds rows {n_pred:,}; current lake yields {n_now:,} samples "
          f"-> using the first {n_pred:,} (lake grew after the cache was built)")
    if n_now < n_pred:
        raise SystemExit("STOP: current lake yields FEWER samples than the "
                         "cached predictions — alignment cannot be trusted")
    y24 = y24_t.cpu().numpy()[:n_pred]
    r1h = r1h_t.cpu().numpy()[:n_pred]
    bar_ms = np.asarray(bar_ms)[:n_pred]

    # ---- reproduction gate / 复现门槛 ------------------------------------
    arch = json.load(open(ARCHIVE))["diagnostic_w_y24"]
    inv_vol = trailing_inv_vol(r1h)
    idx = np.arange(0, n_pred, DECISION_EVERY)
    a0_rank = np.array([rank_weights(scores[t]) @ y24[t] for t in idx])
    a0_rv = np.array([rank_weights(scores[t], inv_vol[t]) @ y24[t] for t in idx])
    d_rank = abs(a0_rank.mean() - arch["rank"]["ALL"])
    d_rv = abs(a0_rv.mean() - arch["rank/vol"]["ALL"])
    print(f"\n  REPRODUCTION  rank     {a0_rank.mean():+.10f}  archived "
          f"{arch['rank']['ALL']:+.10f}  |diff| {d_rank:.2e}")
    print(f"                rank/vol {a0_rv.mean():+.10f}  archived "
          f"{arch['rank/vol']['ALL']:+.10f}  |diff| {d_rv:.2e}")
    if d_rank > REPRO_TOL or d_rv > REPRO_TOL:
        raise SystemExit("STOP (pre-registered): A0 does not reproduce the "
                         "archive — check input alignment before anything else")
    print("  -> reproduced within 1e-8; alignment of the first rows confirmed")

    # ---- 2x2 grid / 2×2 网格 --------------------------------------------
    Y = y24[idx]                                   # (D, A) decision-day labels
    W = np.array([rank_weights(scores[t]) for t in idx])
    sig = Y.std(axis=1)                            # cross-sectional σ_t, ddof=0
    sig_bar = sig.mean()
    Q = np.array([rank_weights(row) for row in Y]) # rank-centred y, Σ|q|=1
    Qt = Q / Q.std(axis=1, keepdims=True)          # unit cross-sectional sd

    cells = {
        "A0":  (W * Y).sum(1),                                  # raw, disp kept
        "A1":  (W * (Y / sig[:, None])).sum(1) * sig_bar,       # raw, disp removed
        "A2p": (W * Qt).sum(1) * sig,                           # ranked, disp kept
        "A2":  (W * Qt).sum(1) * sig_bar,                       # ranked, disp removed
    }
    A = {k: v.mean() for k, v in cells.items()}
    tails = 0.5 * ((A["A2"] - A["A1"]) + (A["A2p"] - A["A0"]))
    disp = 0.5 * ((A["A1"] - A["A0"]) + (A["A2"] - A["A2p"]))
    gap = A["A2"] - A["A0"]

    print(f"\n  decision days: {len(idx):,}   mean cross-sectional σ_t "
          f"{sig_bar:.4%}")
    print(f"\n  {'cell':<5} {'meaning':<34} {'mean %/day':>11} "
          f"{'NW t(5)':>8} {'NW t(10)':>9}")
    print("  " + "-" * 70)
    meaning = {"A0": "raw y, dispersion kept (observed)",
               "A1": "raw y, dispersion removed",
               "A2p": "ranked y, dispersion kept",
               "A2": "ranked y, dispersion removed (IC)"}
    stats = {}
    for k in ("A0", "A1", "A2p", "A2"):
        m, se, t5 = nw_t(cells[k], 5)
        _, _, t10 = nw_t(cells[k], 10)
        stats[k] = dict(mean=m, se=se, t5=t5, t10=t10)
        print(f"  {k:<5} {meaning[k]:<34} {m:>+11.4%} {t5:>+8.2f} {t10:>+9.2f}")

    print(f"\n  shortfall A2 - A0 = {gap:+.4%}/day  (what the IC should have "
          f"earned but did not)")
    print(f"    (a) return-side tails   {tails:+.4%}/day  "
          f"= {tails / gap:>6.1%} of shortfall")
    print(f"    (b) dispersion timing   {disp:+.4%}/day  "
          f"= {disp / gap:>6.1%} of shortfall")
    print(f"    order check: tails via A1->A2 {A['A2'] - A['A1']:+.4%}, "
          f"via A0->A2' {A['A2p'] - A['A0']:+.4%}")

    # ---- A3: IC vs dispersion / IC 与离散度 ---------------------------------
    ic = np.array([spearman(list(scores[t]), list(y24[t])) for t in idx])
    ic_m, ic_se, ic_t = nw_t(ic, 5)
    slope, slope_t = nw_slope(ic, sig, 5)
    corr = float(np.corrcoef(ic, sig)[0, 1])
    print(f"\n  daily Spearman IC: mean {ic_m:+.4f}  NW t(5) {ic_t:+.2f}")
    print(f"  A3  IC_t on σ_t: slope {slope:+.3f}  NW t(5) {slope_t:+.2f}  "
          f"corr {corr:+.3f}")

    # ---- per-year / 逐年 ------------------------------------------------
    yrs = ((bar_ms // 1000).astype("datetime64[s]").astype("datetime64[Y]")
           .astype(int) + 1970)[idx]
    print(f"\n  {'year':<6} {'days':>5} {'A0':>9} {'A1':>9} {'A2p':>9} "
          f"{'A2':>9} {'IC':>8} {'σ_t':>7}")
    per_year = {}
    for y in sorted(set(yrs.tolist())):
        mk = yrs == y
        row = {k: float(cells[k][mk].mean()) for k in cells}
        row.update(ic=float(ic[mk].mean()), sigma=float(sig[mk].mean()),
                   days=int(mk.sum()))
        per_year[str(y)] = row
        print(f"  {y:<6} {row['days']:>5} {row['A0']:>+9.4%} {row['A1']:>+9.4%} "
              f"{row['A2p']:>+9.4%} {row['A2']:>+9.4%} {row['ic']:>+8.4f} "
              f"{row['sigma']:>7.2%}")

    # ---- pre-registered readout / 预注册判读 ------------------------------
    p0 = A["A2"] > 0
    p1 = abs(stats["A0"]["t5"]) < 2
    p2 = tails > disp
    p3 = slope < 0
    print("\n  PRE-REGISTERED READOUT")
    print(f"    P0 (mechanical) A2 > 0 ..................... "
          f"{'HOLDS' if p0 else 'FAILS -> implementation bug, STOP'}")
    print(f"    P1 (~60%) |t(A0)| < 2, A0 not significant ... "
          f"{'CONFIRMED' if p1 else 'REFUTED'}  (t5 = {stats['A0']['t5']:+.2f})")
    print(f"    P2 (~55%) tails share > dispersion share .... "
          f"{'CONFIRMED' if p2 else 'REFUTED'}  "
          f"({tails / gap:.0%} vs {disp / gap:.0%})")
    print(f"    P3 (~60%) A3 slope < 0 ...................... "
          f"{'CONFIRMED' if p3 else 'REFUTED'}  (t = {slope_t:+.2f})")

    OUT.write_text(json.dumps(dict(
        prereg_commit="fbafd93", n_pred_rows=n_pred, n_lake_rows=n_now,
        decision_days=int(len(idx)), sigma_bar=float(sig_bar),
        reproduction=dict(rank=float(a0_rank.mean()),
                          rank_vol=float(a0_rv.mean()),
                          archived_rank=arch["rank"]["ALL"],
                          archived_rank_vol=arch["rank/vol"]["ALL"]),
        cells=stats, shortfall=float(gap), shapley_tails=float(tails),
        shapley_dispersion=float(disp),
        ic=dict(mean=ic_m, t5=ic_t), a3=dict(slope=slope, t5=slope_t, corr=corr),
        per_year=per_year,
        readout=dict(P0=bool(p0), P1=bool(p1), P2=bool(p2), P3=bool(p3)),
    ), indent=2), encoding="utf-8")
    print(f"\n  saved -> {OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()
