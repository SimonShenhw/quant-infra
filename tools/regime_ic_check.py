"""
Regime-conditional factor IC: does factor IC sign track market regime?
按 regime 拆分的逐因子 IC：因子 IC 的符号是否跟随市场状态翻转？（只读诊断）
Usage: python tools/regime_ic_check.py

WHAT: splits every daily mark into up/down regime by BTC close vs its own
trailing 200-day SMA (causal — the SMA excludes the current mark), then
computes each factor's cross-sectional daily rank IC within each regime,
and the z-score of the up-minus-down difference.

WHY IT EXISTS: tools/factor_decay_monitor.py (2026-09-18, n=81 live marks)
found the live window's factor ICs did not SHRINK, they FLIPPED SIGN, and
the flips looked one-directional (trend/vol factors negative -> positive)
while the market went from chop/bear (backtest window, EW -26.9%) to a
strong bull (live window, EW +47%). The obvious hypothesis was regime
mismatch rather than factor decay. This script was written to TEST that
hypothesis rather than assume it.

RESULT (2026-09-19, FALSIFICATION_2026-09-19.md section 4.3): the
hypothesis was REJECTED. Over 985 marks (2023-05..2026-08, up 614 / down
371) NO factor clears |z| > 2 (largest: std20 at -1.73), and the direction
is OPPOSITE to the hypothesis — std20 reads -0.0269 in up regimes (more
negative) vs +0.0032 in down, while the live bull window gave +0.0273. The
same regime label produces opposite signs, so this regime indicator does
not explain the live flips. Kept in the repo so that conclusion stays
reproducible, not because the hypothesis survived.

Diagnostic of an ALREADY-falsified system, not a selection among strategy
candidates -> deliberately NOT registered in trials.json. If its output
ever motivates a model change, THAT is a trial and must register.

中文：衰减监控发现 live 端因子 IC 不是变小而是变号，且方向像是随牛熊翻转
→ 提出 regime 错配假设 → 本脚本用于**检验**而非假定该假设 → **假设被否决**
（985 个记账日，无因子 |z|>2，且方向与假设相反：同一 regime 标签给出相反
符号）。留在仓库是为了让这个否定结论可复现。属已证伪系统的诊断，不算
selection trial，刻意不进 trials.json。

Window note: starts 2023-05 because the live 20-symbol universe contains
late-listed coins, so the all-symbol lake intersection cannot begin
earlier — the 2021 mania year is NOT covered.
"""
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

from factors.base import FactorRegistry
import factors  # noqa: F401
from data.lake_loader import load_klines_multi, klines_to_tensors
from run_v13_final import aggregate_5m_to_1h
from paper_live_ic import spearman
from factor_analyzer import load_funding
from factor_decay_monitor import live_universe, mark_index_map

syms = live_universe()
raw = load_klines_multi(symbols=syms, interval="5m", min_rows=40000)
syms = [s for s in syms if s in raw]
agg = {s: aggregate_5m_to_1h(raw[s]) for s in syms}
common = None
for s in syms:
    ts = set(agg[s]["open_time"].to_list())
    common = ts if common is None else (common & ts)
times = np.array(sorted(common), dtype=np.int64)

names = FactorRegistry.list_factors()
device = torch.device("cpu")
fmat = np.zeros((len(times), len(syms), len(names)), dtype=np.float32)
close = np.zeros((len(times), len(syms)), dtype=np.float64)
for a, s in enumerate(syms):
    df = agg[s].filter(agg[s]["open_time"].is_in(times.tolist())).sort("open_time")
    t = klines_to_tensors(df)
    extras = {"funding": load_funding(s, df["open_time"].to_numpy(), device)}
    ften = FactorRegistry.build_tensor(
        names, t["open"], t["high"], t["low"], t["close"], t["volume"],
        zscore_window=48, extras=extras)
    fmat[:, a, :] = ften.cpu().numpy()
    close[:, a] = t["close"].cpu().numpy()

from datetime import datetime, timezone
print(f"aligned 1h bars: {len(times)}  "
      f"{datetime.fromtimestamp(times[0]/1e3, tz=timezone.utc):%Y-%m-%d} -> "
      f"{datetime.fromtimestamp(times[-1]/1e3, tz=timezone.utc):%Y-%m-%d}")
print(f"symbols: {len(syms)}")

# BTC trailing 200-day SMA on daily marks (causal: uses marks up to t only)
btc = syms.index("BTCUSDT")
marks = mark_index_map(times)
mdates = sorted(marks)
mclose_btc = np.array([close[marks[d], btc] for d in mdates])
SMA = 200
regime = {}
for i, d in enumerate(mdates):
    if i < SMA:
        continue
    sma = mclose_btc[i - SMA:i].mean()      # trailing, excludes today
    regime[d] = "up" if mclose_btc[i] > sma else "down"

n_up = sum(1 for v in regime.values() if v == "up")
print(f"regime-labelled marks: {len(regime)} (up={n_up}, down={len(regime)-n_up})")

ics = {"up": {f: [] for f in range(len(names))},
       "down": {f: [] for f in range(len(names))}}
for da, db in zip(mdates[:-1], mdates[1:]):
    if da not in regime:
        continue
    gap = (datetime.strptime(db, "%Y-%m-%d") - datetime.strptime(da, "%Y-%m-%d")).days
    if gap > 3:
        continue
    i0, i1 = marks[da], marks[db]
    rets = close[i1] / close[i0] - 1.0
    r = regime[da]
    for f in range(len(names)):
        vals = fmat[i0, :, f]
        if np.allclose(vals, vals[0]):
            continue
        ics[r][f].append(spearman(list(vals), list(rets)))


def ms(xs):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), n
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, sd / np.sqrt(n), n


print(f"\n{'factor':<18} {'up_IC':>9} {'±SE':>7} {'n':>5}  "
      f"{'down_IC':>9} {'±SE':>7} {'n':>5}  {'up-down':>8} {'z':>6}")
print("-" * 82)
rows = []
for f, name in enumerate(names):
    um, use, un = ms(ics["up"][f])
    dm, dse, dn = ms(ics["down"][f])
    z = (um - dm) / np.sqrt(use ** 2 + dse ** 2) if un > 1 and dn > 1 else float("nan")
    rows.append((name, um, use, un, dm, dse, dn, z))
rows.sort(key=lambda r: -abs(r[7]) if not np.isnan(r[7]) else 0)
for name, um, use, un, dm, dse, dn, z in rows:
    print(f"{name:<18} {um:>+9.4f} {use:>7.4f} {un:>5}  "
          f"{dm:>+9.4f} {dse:>7.4f} {dn:>5}  {um-dm:>+8.4f} {z:>+6.2f}")
