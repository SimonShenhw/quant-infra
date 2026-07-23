"""
Per-factor live-vs-backtest IC decay monitor (read-only diagnostic).
逐因子 live-vs-回测 IC 衰减监控（只读诊断）。
Usage: python tools/factor_decay_monitor.py
       [--bt-start 2024-09-01] [--bt-end 2026-03-30] [--live-start 2026-06-11]

WHAT: for every registered factor, the cross-sectional daily rank IC
(23:00 UTC mark -> next-mark ~24h return, the SAME cadence and label the
live gate judges on) computed over TWO windows from the SAME lake pipeline:
the v13 backtest window and the live paper window. Output: per-factor
bt_IC vs live_IC with SEs, the difference z-score, sign-flip flags, and a
factor-ordering survival summary (rank corr of bt vs live factor ICs).

WHY THIS TOOL (built 2026-07-23, ~7 weeks before the September gates): the
current provisional gate reading has v13 DEAD. If that holds, the decision
table sends us "back to the research desk" — and the FIRST question there
is which of three causes killed it: (a) the factors themselves decayed,
(b) factors fine but regime mismatch, (c) factors fine but construction /
monetization (the extended-window finding). This table separates (a) from
(b)/(c): factor-level ICs that collapsed out-of-window point at (a);
factor ICs intact while the basket bled point at (b)/(c). Building it now
means the September post-mortem starts from evidence, not archaeology.

WHY NOT a trials.json entry: this is a diagnostic of an ALREADY
pre-registered outcome, not a selection among strategy candidates —
nothing here feeds back into a live system or a reported Sharpe. If its
output later motivates new factor/model choices, THOSE experiments are
trials and must register; this readout is not.

Operationalization (pinned, single-implementation discipline):
  * factor values = FactorRegistry.build_tensor(...) — the z-scored,
    ±5σ-clamped series THE MODELS ACTUALLY CONSUME (zscore_window=48,
    trailing/causal; invariant t2 covers the builder). NOTE this differs
    from tools/factor_analyzer.py, which computes IC on RAW factor values
    at hourly cadence — its table answers "is there signal in the factor",
    this one answers "did the model's input stop working out-of-window".
  * 5m->1h aggregation imported from run_v13_final (factor_analyzer's own
    copy is a known duplicate; do not add a third).
  * rank IC = paper_live_ic.spearman — the registered live-IC judge's own
    correlation.
  * universe = the 20 symbols of basket_state's FIRST all_closes row (the
    live track's own universe), not sorted(lake)[:20] — the lake has since
    grown extra symbols and silent universe drift would poison the
    comparison.
  * daily mark = close of the 22:00-23:00 UTC bar (searchsorted, <=3h
    tolerance, >=72 bars warmup) — mirrors run_paper_daily
    mark_indices_for_date semantics.
  * MARK-SEMANTICS NOTE (verified 2026-07-23): the live system's SAME-DAY
    path marks at the RUN-TIME snapshot (today_idx = last fetched bar,
    run_paper_daily.py ~line 861 — an in-progress ~23:30 UTC bar when the
    19:30 ET task fires), while its BACKFILL path and THIS tool mark at
    the 23:00 UTC bar close. Cross-checking lake mark closes against
    basket_state.all_closes therefore shows ~40-500 bps differences —
    that is the ~30 min of price motion between the two mark times, NOT
    a data error. Each series is internally consistent (live scores and
    returns share the same snapshot instant; both windows here share the
    23:00 close), so within-series ICs are valid on both sides; only
    naive cross-source price comparisons look alarming. Run instants are
    recoverable from run_meta.run_utc if September needs them.

Data notes:
  * lake refreshed through 2026-06-30 (Vision monthly archives 2026-04..06
    pulled 2026-07-23); July extends when its monthly archive publishes
    (~Aug 1). NEVER write partial months into the lake — the downloader's
    skip-if-exists would freeze them incomplete forever.
  * data_lake 2026-03 is a partial month (ends 03-30 02:10, the initial
    pull date) — kept AS-IS because it is the bit-faithful history v13/O2
    trained on; the resulting ~44h hole sits between the two windows and
    touches neither.
  * funding_rates.db ends 2026-03-31 with NO live-window rows (the paper
    system persists carry SIGNALS in carry_state, not raw rates), so the
    funding_rate factor is bt-only here; live column prints n/a rather
    than a fake ~0 IC from a forward-filled constant. O2's 18-factor set
    excludes funding anyway.

中文：九月 v13 若确认死，回研究桌的第一问是死因三选一——因子衰减 /
regime 错配 / 构造变现。本表把第一种和后两种分开：live 窗口逐因子 IC
塌了指向衰减；因子 IC 完好而篮子仍亏指向 regime/构造（与扩窗结论同向）。
全部算术复用既有单一实现（build_tensor 的 z 空间=模型真实输入、
run_v13_final 聚合、paper_live_ic 的 spearman、basket_state 首日宇宙、
23:00 UTC 标记语义）；诊断不是 selection trial，不进 trials.json；
funding 因子 live 端无原始费率数据，明示 n/a 不造假零。

Sample-size honesty: at ~19 live marks the per-factor SE is ~0.065, so
only decays > ~0.13 clear 2σ — this run is a coarse early screen. Rerun
after each lake refresh; by the September gates (~80 live marks) the SE
halves and the table carries real weight.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))          # repo root: run_v13_final, factors, data
sys.path.insert(0, str(BASE / "tools"))  # sibling tools when run from anywhere

from factors.base import FactorRegistry          # noqa: E402
import factors  # noqa: F401,E402  (import populates the registry)
from data.lake_loader import load_klines_multi, klines_to_tensors  # noqa: E402
from run_v13_final import aggregate_5m_to_1h, DROP_FACTORS  # noqa: E402
from paper_live_ic import spearman               # noqa: E402
from factor_analyzer import load_funding         # noqa: E402

DB = BASE / "paper_daily.db"
O2_CKPT = BASE / "checkpoints" / "o2_production.pt"
MARK_HOUR_UTC = 23      # mark = close of the 22:00-23:00 UTC bar / 与模拟盘一致
MARK_TOL_MS = 3 * 3_600_000
WARMUP_BARS = 72        # run_paper_daily convention / 与模拟盘一致
MAX_GAP_DAYS = 3        # skip IC pairs spanning holes / 跨洞的日对不计

# O2's 18-factor set excludes funding by design; used as fallback when the
# ckpt is absent. O2 名单以 ckpt 内自带为准，缺 ckpt 时退回文档化的 drop 集。
DROP_FACTORS_EXT_FALLBACK = {"macd", "volume_zscore", "funding_rate"}


def live_universe() -> list[str]:
    """The 20 symbols of the live basket track's first state row.
    live 篮子首日宇宙——比 sorted(lake)[:20] 更抗 lake 扩容漂移。"""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    row = conn.execute(
        "SELECT all_closes FROM basket_state ORDER BY date LIMIT 1").fetchone()
    conn.close()
    return sorted(json.loads(row[0]).keys())


def o2_factor_names() -> set[str]:
    if O2_CKPT.exists():
        try:
            ckpt = torch.load(O2_CKPT, map_location="cpu", weights_only=False)
            names = ckpt.get("factor_names")
            if names:
                return set(names)
        except Exception:
            pass
    return set(FactorRegistry.list_factors()) - DROP_FACTORS_EXT_FALLBACK


def mark_index_map(times_ms: np.ndarray) -> dict[str, int]:
    """date -> aligned-bar index of that date's 23:00 UTC mark.
    Mirrors run_paper_daily.mark_indices_for_date: the mark bar OPENS at
    22:00 UTC; accept the nearest earlier bar within 3h; require warmup.
    """
    out: dict[str, int] = {}
    d0 = datetime.fromtimestamp(times_ms[0] / 1e3, tz=timezone.utc).date()
    d1 = datetime.fromtimestamp(times_ms[-1] / 1e3, tz=timezone.utc).date()
    d = d0
    while d <= d1:
        target = int(datetime(d.year, d.month, d.day, MARK_HOUR_UTC - 1,
                              tzinfo=timezone.utc).timestamp() * 1000)
        idx = int(np.searchsorted(times_ms, target, side="right")) - 1
        if idx >= WARMUP_BARS and target - times_ms[idx] <= MARK_TOL_MS:
            out[d.strftime("%Y-%m-%d")] = idx
        d += timedelta(days=1)
    return out


def window_ics(fmat: np.ndarray, close: np.ndarray, marks: dict[str, int],
               d_from: str, d_to: str) -> dict[int, list[float]]:
    """Per-factor list of daily cross-sectional rank ICs inside [d_from,d_to].
    fmat: (T, A, F) z-scored factor values; close: (T, A).
    """
    dates = sorted(d for d in marks if d_from <= d <= d_to)
    ics: dict[int, list[float]] = {f: [] for f in range(fmat.shape[2])}
    for da, db_ in zip(dates[:-1], dates[1:]):
        gap = (datetime.strptime(db_, "%Y-%m-%d")
               - datetime.strptime(da, "%Y-%m-%d")).days
        if gap > MAX_GAP_DAYS:
            continue
        i0, i1 = marks[da], marks[db_]
        rets = close[i1] / close[i0] - 1.0
        for f in range(fmat.shape[2]):
            vals = fmat[i0, :, f]
            if np.allclose(vals, vals[0]):   # degenerate cross-section / 退化截面
                continue
            ics[f].append(spearman(list(vals), list(rets)))
    return ics


def mean_se(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, float("nan"), 1
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt-start", default="2024-09-01")
    ap.add_argument("--bt-end", default="2026-03-30")
    ap.add_argument("--live-start", default="2026-06-11")
    args = ap.parse_args()

    syms = live_universe()
    print("=" * 78)
    print(f"  FACTOR DECAY MONITOR — z-scored (model-input) daily rank IC")
    print(f"  universe: {len(syms)} live-basket symbols; "
          f"windows bt=[{args.bt_start}..{args.bt_end}] "
          f"live=[{args.live_start}..lake end]")
    print("=" * 78)

    raw = load_klines_multi(symbols=syms, interval="5m", min_rows=40000)
    missing = [s for s in syms if s not in raw]
    if missing:
        print(f"  WARNING: not in lake / below min_rows: {missing}")
        syms = [s for s in syms if s in raw]

    agg = {s: aggregate_5m_to_1h(raw[s]) for s in syms}
    common = None
    for s in syms:
        ts = set(agg[s]["open_time"].to_list())
        common = ts if common is None else (common & ts)
    times = np.array(sorted(common), dtype=np.int64)
    lake_end = datetime.fromtimestamp(times[-1] / 1e3, tz=timezone.utc)
    print(f"  aligned 1h bars: {len(times)}  "
          f"(through {lake_end:%Y-%m-%d %H:%M} UTC)")

    names = FactorRegistry.list_factors()
    o2set = o2_factor_names()
    device = torch.device("cpu")

    fmat = np.zeros((len(times), len(syms), len(names)), dtype=np.float32)
    close = np.zeros((len(times), len(syms)), dtype=np.float64)
    for a, s in enumerate(syms):
        df = agg[s].filter(agg[s]["open_time"].is_in(times.tolist())) \
                   .sort("open_time")
        t = klines_to_tensors(df)
        extras = {"funding": load_funding(
            s, df["open_time"].to_numpy(), device)}
        ften = FactorRegistry.build_tensor(
            names, t["open"], t["high"], t["low"], t["close"], t["volume"],
            zscore_window=48, extras=extras)
        fmat[:, a, :] = ften.cpu().numpy()
        close[:, a] = t["close"].cpu().numpy()

    marks = mark_index_map(times)
    bt = window_ics(fmat, close, marks, args.bt_start, args.bt_end)
    lv = window_ics(fmat, close, marks, args.live_start, "9999-12-31")

    # funding factor has NO live raw-rate data — n/a, not a fake 0 (see
    # docstring). funding 因子 live 端无原始费率，明示 n/a。
    fund_idx = names.index("funding_rate") if "funding_rate" in names else -1

    print(f"\n  {'factor':<18} {'v13':>3} {'o2':>3} "
          f"{'bt_IC':>8} {'±SE':>6} {'n':>4}  "
          f"{'live_IC':>8} {'±SE':>6} {'n':>4}  {'Δ':>7} {'z':>6}  flag")
    print("-" * 78)
    rows = []
    for f, name in enumerate(names):
        bm, bse, bn = mean_se(bt[f])
        if f == fund_idx:
            lm, lse, ln = float("nan"), float("nan"), 0
        else:
            lm, lse, ln = mean_se(lv[f])
        rows.append((name, bm, bse, bn, lm, lse, ln))
    rows.sort(key=lambda r: -abs(r[1]) if not math.isnan(r[1]) else 0)

    bt_means, lv_means = [], []
    for name, bm, bse, bn, lm, lse, ln in rows:
        inv13 = "y" if name not in DROP_FACTORS else "-"
        ino2 = "y" if name in o2set else "-"
        if ln == 0:
            live_s = f"{'n/a':>8} {'':>6} {0:>4}"
            d_s, z_s, flag = f"{'—':>7}", f"{'—':>6}", ""
        else:
            delta = lm - bm
            z = delta / math.sqrt(bse ** 2 + lse ** 2) \
                if not math.isnan(bse) and not math.isnan(lse) else float("nan")
            flag = ""
            if abs(bm) >= 0.02 and (lm * bm) < 0:
                flag += "SIGN-FLIP "
            if not math.isnan(z) and abs(z) >= 2:
                flag += "z>2"
            live_s = f"{lm:>+8.4f} {lse:>6.4f} {ln:>4}"
            d_s, z_s = f"{delta:>+7.4f}", f"{z:>+6.2f}"
            bt_means.append(bm)
            lv_means.append(lm)
        print(f"  {name:<18} {inv13:>3} {ino2:>3} "
              f"{bm:>+8.4f} {bse:>6.4f} {bn:>4}  {live_s}  {d_s} {z_s}  {flag}")

    if len(bt_means) >= 5:
        order_corr = spearman(bt_means, lv_means)
        keep = sum(1 for b, l in zip(bt_means, lv_means)
                   if b * l > 0)
        # Power honesty: live per-factor SE (~0.06 at n=19) dwarfs the
        # spread of true bt ICs (~0.015), so even FULL persistence of the
        # ordering would show only a small observed corr. Print that
        # attenuation ceiling next to the observation, or the summary
        # line reads "ordering is dead" when it is merely unmeasurable.
        # 功效诚实：live 噪声远大于因子间真实 IC 差异，就算排序完全延续，
        # 观测 corr 也只有一个小上限——不并排打印就会误读成"排序已死"。
        var_bt = float(np.var(np.array(bt_means), ddof=1))
        noise_bt = float(np.mean([r[2] ** 2 for r in rows if r[6] > 0]))
        noise_lv = float(np.mean([r[5] ** 2 for r in rows if r[6] > 0]))
        sig = max(var_bt - noise_bt, 0.0)
        ceiling = (math.sqrt(sig / (sig + noise_bt))
                   * math.sqrt(sig / (sig + noise_lv))) if sig > 0 else 0.0
        print("-" * 78)
        print(f"  factor-ordering survival: rank corr(bt IC, live IC) = "
              f"{order_corr:+.3f} over {len(bt_means)} factors; "
              f"{keep}/{len(bt_means)} kept sign")
        print(f"  attenuation ceiling: even FULL ordering persistence would "
              f"only show ≈{ceiling:+.2f}")
        print(f"  at current noise — |observed| below ceiling is "
              f"UNINFORMATIVE, not a death verdict")
    n_live = max((r[6] for r in rows), default=0)
    det = 2 * 0.28 / math.sqrt(n_live) if n_live else float("nan")
    print(f"  detectability: at n={n_live} live marks only |ΔIC| > ~{det:.2f} "
          f"clears 2σ — coarse screen, rerun after each lake refresh")
    print(f"  model-level live IC (registered judge): "
          f"python tools/paper_live_ic.py")
    print()


if __name__ == "__main__":
    main()
