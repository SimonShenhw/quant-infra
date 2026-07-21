"""
Extended-window retrain: does the cross-sectional alpha survive 2021-2026?
扩窗重训：横截面 alpha 能否穿越完整周期？(ROADMAP Phase 2.1)

HYPOTHESIS: v13's 19-month window (2024-09..2026-03) contains no sustained
alt-rally regime; the regime-IC study (research_regime_ic.py) acquitted the
short leg IN-window, so the remaining suspect is that the window itself is
unrepresentative. Retraining on 2021+ (bull mania, crash, chop, full cycle)
is the direct test — prioritized over any model change.

METHOD: two runs on the SAME 16-symbol universe (full 2021+ Binance
history; APT/ARB/OP/SUI excluded — listed later) and the SAME 18 factors
(funding_rate excluded: funding_rates.db only covers 2024-09+, keeping it
would split one input channel's semantics across the window):

  FULL: 2021-01 .. 2026-03  (~44K hourly bars — 2021 bull, 2022 crash,
        2023 chop, 2024-25 cycle)
  CTRL: 2024-09 .. 2026-03  (v13's window, same universe/factors —
        isolates the window effect from the universe/factor change)

Per run: CPCV (6,2) with purge 48 / embargo 48, 24h label, turnover-penalty
loss — the v13 recipe. Reports OOS rank IC overall / per year / per BTC-trend
regime, leg alphas, and the banded top-3 backtest. Fold ckpts go to
checkpoints/folds_ext_{full,ctrl} (folds_v13 untouched). Registers 2 trials.

VERDICT (2026-07-13, RESEARCH_2026-07-13_extended_window.md): the RANKING
signal is real and cycle-robust — CPCV 15/15 folds positive, OOS rank IC
+0.076, and positive in EVERY year 2021-2026 (0.041..0.092) and in both
trend regimes. But TAIL MONETIZATION FAILS: the same banded-top3 recipe
loses -64.3% (Sharpe -0.40, maxDD 75%) over the full cycle, and the CTRL
run is the sharpest evidence — same window as v13, merely dropping the 4
late-listed high-vol alts (APT/ARB/OP/SUI) + the funding factor flips
+32.6% to -13.6%: v13's headline profit was concentrated in tail trades on
those four names, a more concrete fragility than DSR 0.11. Final ruling on
the bear-artifact hypothesis: innocent at the IC level, guilty at the tail
level — tail shorts bleed systematically in mania years (2021 short alpha
-0.149%/24h), a historical rehearsal of the June-2026 live losses. Leads
directly to research_rank_weighted.py (can non-tail constructions monetize
the mid-book ranking?).

结论：排名信号是真的且穿越全周期——15/15 折为正，OOS IC +0.076，2021-26
每一年都为正；但尾部变现失败：同配方全周期 banded top3 -64.3%，且 CTRL
是最锋利的证据——同一窗口只去掉 4 个晚上市高波动山寨（APT/ARB/OP/SUI）
和 funding 因子，+32.6% 即翻 -13.6%，v13 的账面利润高度集中在那 4 个币的
尾部交易上（比 DSR 0.11 更具体的脆弱性）。熊市伪影假设终审：IC 层面无罪、
尾部层面成立——狂热年份尾部做空系统性亏损（2021 空腿 -0.149%/24h），正是
live 2026-06 亏损模式的历史彩排。下一步：research_rank_weighted.py 检验
非尾部构建能否变现中段排序。
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from data.lake_loader import load_klines_multi, klines_to_tensors
from engine.cpcv import generate_cpcv_splits
from factors.base import FactorRegistry
from model.cross_asset_attention import CrossAssetGRUAttention
from run_v13_final import (
    DualLoss, aggregate_5m_to_1h, train_fold_indexed, split_train_val,
    run_backtest, oos_rank_ic, LABEL_H, SEED,
)
from tools.research_regime_ic import rank_ic_rows
from tools.validation_stats import register_trial

import factors  # noqa: F401

SEQ_LEN = 24
# Fixed 16-symbol universe with full 2021+ Binance spot history. WHY fixed
# (no APT/ARB/OP/SUI): a dynamic universe would confound "window effect"
# with "listing effect", and the model's positional asset embedding cannot
# absorb mid-sample universe changes anyway. Dropping exactly these 4 is
# also what turns CTRL into the tail-concentration control experiment.
# 固定 16 币宇宙（2021 起全历史）。不含 APT/ARB/OP/SUI：动态宇宙会把窗口
# 效应与上市效应混在一起，且位置式资产嵌入不允许中途换宇宙；恰好也让
# CTRL 成为"利润是否集中在这 4 个币"的对照实验。
EXT_SYMBOLS = [
    "AAVEUSDT", "ADAUSDT", "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "DOTUSDT", "ETHUSDT", "INJUSDT", "LINKUSDT", "LTCUSDT",
    "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT",
]
DROP_FACTORS_EXT = {"macd", "volume_zscore", "funding_rate"}
WINDOWS = {
    "FULL": 1609459200000,   # 2021-01-01 UTC ms
    "CTRL": 1725148800000,   # 2024-09-01 UTC ms (v13 window)
}


def build_window(start_ms: int, device) -> Tuple[torch.Tensor, ...]:
    raw = load_klines_multi(symbols=EXT_SYMBOLS, interval="5m", min_rows=40000)
    assert len(raw) == len(EXT_SYMBOLS), f"missing symbols: got {len(raw)}"
    agg = {s: aggregate_5m_to_1h(raw[s]).filter(pl.col("open_time") >= start_ms)
           for s in EXT_SYMBOLS}
    common = None
    for s in EXT_SYMBOLS:
        ts = agg[s]["open_time"].to_numpy()
        common = ts if common is None else np.intersect1d(common, ts)
    aligned = {s: agg[s].filter(pl.col("open_time").is_in(common)).sort("open_time")
               for s in EXT_SYMBOLS}
    T = len(common)

    factor_names = [n for n in FactorRegistry.list_factors() if n not in DROP_FACTORS_EXT]
    close_mat = torch.zeros(T, len(EXT_SYMBOLS), device=device)
    fx = {}
    for j, s in enumerate(EXT_SYMBOLS):
        t = klines_to_tensors(aligned[s], device)
        close_mat[:, j] = t["close"]
        fx[s] = FactorRegistry.build_tensor(
            factor_names, t["open"], t["high"], t["low"], t["close"], t["volume"],
            zscore_window=48)

    n = T - SEQ_LEN - LABEL_H
    dec = torch.arange(n, device=device) + SEQ_LEN - 1
    y24 = close_mat[dec + LABEL_H] / close_mat[dec].clamp(min=1e-8) - 1.0
    r1h = close_mat[dec + 1] / close_mat[dec].clamp(min=1e-8) - 1.0
    X = torch.stack([torch.stack([fx[s][i:i + SEQ_LEN] for s in EXT_SYMBOLS], dim=0)
                     for i in range(n)])
    bar_ms = common[SEQ_LEN - 1: SEQ_LEN - 1 + n]  # decision-close timestamps
    print(f"  window>= {start_ms}: {T} bars -> {n} samples, "
          f"{len(factor_names)} factors, {len(EXT_SYMBOLS)} assets")
    return X, y24, r1h, close_mat, bar_ms, len(factor_names)


def run_cpcv_ext(X, y24, n_factors, fold_dir: str, device):
    n, A = X.size(0), X.size(1)
    splits = generate_cpcv_splits(n, n_groups=6, n_test_groups=2,
                                  purge_bars=SEQ_LEN + LABEL_H, embargo_bars=48)
    pred = torch.zeros(n, A, device=device)
    cnt = torch.zeros(n, device=device)
    corrs = []
    os.makedirs(fold_dir, exist_ok=True)
    for fi, (train_idx, test_idx) in enumerate(splits):
        tr, va = split_train_val(train_idx, SEQ_LEN)
        model = CrossAssetGRUAttention(
            n_factors=n_factors, d_model=128, gru_layers=2, n_cross_heads=4,
            n_cross_layers=3, d_ff=256, dropout=0.30, seq_len=SEQ_LEN,
            max_assets=A).to(device)
        loss_fn = DualLoss().to(device)
        t0 = time.time()
        model, corr = train_fold_indexed(model, loss_fn, X, y24, tr, va)
        corrs.append(corr)
        model.eval()
        t_idx = torch.from_numpy(test_idx).to(device)
        with torch.no_grad():
            for s in range(0, len(test_idx), 512):
                e = min(s + 512, len(test_idx))
                pred[t_idx[s:e]] += model(X[t_idx[s:e]])
                cnt[t_idx[s:e]] += 1
        torch.save({"state": model.state_dict(), "corr": corr,
                    "n_samples": n, "n_factors": n_factors},
                   f"{fold_dir}/fold_{fi:02d}.pt")
        print(f"    fold {fi+1}/{len(splits)}: val_corr={corr:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del model, loss_fn
        torch.cuda.empty_cache()
    valid = cnt > 0
    pred[valid] /= cnt[valid].unsqueeze(-1)
    return pred, valid, corrs


def report_buckets(pred, valid, y24, close_mat, bar_ms, btc_j, label=""):
    p = pred.cpu().numpy()
    y = y24.cpu().numpy()
    v = valid.cpu().numpy()
    n = p.shape[0]
    ic = rank_ic_rows(p, y)

    la = np.zeros(n)
    sa = np.zeros(n)
    for t in range(n):
        order = np.argsort(-p[t])
        ew = y[t].mean()
        la[t] = y[t][order[:3]].mean() - ew
        sa[t] = ew - y[t][order[-3:]].mean()

    btc = close_mat[:, btc_j].cpu().numpy()
    w = 200 * 24
    csum = np.cumsum(btc)
    sma = np.full(len(btc), np.nan)
    sma[w:] = (csum[w:] - csum[:-w]) / w
    dec = np.arange(n) + SEQ_LEN - 1
    has = ~np.isnan(sma[dec])
    up = has & (btc[dec] > sma[dec])

    years = (bar_ms // 1000).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    rows = [("ALL", v)]
    for yr in sorted(set(years)):
        rows.append((str(yr), v & (years == yr)))
    rows += [("trend200d_up", v & up), ("trend200d_down", v & has & ~up)]

    out = {}
    print(f"\n  [{label}] {'bucket':<16} {'n':>6} {'rank_IC':>8} "
          f"{'longA':>9} {'shortA':>9}")
    for name, m in rows:
        mm = m & ~np.isnan(ic)
        k = int(mm.sum())
        if k < 200:
            continue
        out[name] = {"n": k, "rank_ic": float(ic[mm].mean()),
                     "long_alpha": float(la[mm].mean()),
                     "short_alpha": float(sa[mm].mean())}
        r = out[name]
        print(f"  [{label}] {name:<16} {k:>6} {r['rank_ic']:>+8.4f} "
              f"{r['long_alpha']:>+9.4%} {r['short_alpha']:>+9.4%}")
    return out


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  EXTENDED-WINDOW RETRAIN (16 symbols, 18 factors, 24h label)")
    print("=" * 70)

    results = {}
    btc_j = EXT_SYMBOLS.index("BTCUSDT")
    for tag, start_ms in WINDOWS.items():
        print(f"\n[{tag}] building window ...", flush=True)
        X, y24, r1h, close_mat, bar_ms, n_factors = build_window(start_ms, device)
        print(f"[{tag}] CPCV training ...", flush=True)
        pred, valid, corrs = run_cpcv_ext(
            X, y24, n_factors, f"checkpoints/folds_ext_{tag.lower()}", device)
        ic24 = oos_rank_ic(pred, valid, y24)
        print(f"[{tag}] avg fold corr={sum(corrs)/len(corrs):.4f}, OOS IC={ic24:.4f}")

        buckets = report_buckets(pred, valid, y24, close_mat, bar_ms, btc_j, tag)

        random.seed(SEED)
        bt = run_backtest(f"{tag}_band_top3_daily", pred, valid, r1h, close_mat,
                          SEQ_LEN, k=3, enter_band=3, exit_band=6,
                          decision_every=24, min_hold=0, use_vol_filter=False)
        bt.pop("rets", None)
        print(f"[{tag}] banded backtest: return={bt['total_return']:+.2%} "
              f"sharpe={bt['sharpe']:.3f} maxDD={bt['max_drawdown']:.2%} "
              f"PSR={bt['psr']:.2f}")
        results[tag] = {"fold_corrs": corrs, "oos_ic24": ic24,
                        "buckets": buckets, "backtest": bt}
        register_trial(f"extended_window_2026-07-13/{tag}",
                       {"kind": "retrain", "symbols": len(EXT_SYMBOLS),
                        "factors": n_factors})
        del X, y24, r1h, close_mat, pred
        torch.cuda.empty_cache()

    with open("checkpoints/research_extended_window.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\n  saved: checkpoints/research_extended_window.json")


if __name__ == "__main__":
    main()
