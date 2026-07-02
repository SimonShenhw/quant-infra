"""
Offline research: long-only variants + funding-carry sleeve vs v13 long-short.
离线研究：多头变体 + funding carry sleeve，与 v13 多空对照。

Motivation (2026-07-01): after 20 live days the v13 basket is -7.5% with
live rank IC ~ 0; the bleeding is concentrated in the SHORT leg (squeezed
shorting strong coins), while the long leg holds up. Literature suggests
funding carry as a slow, low-turnover crypto signal (Crypto Carry, BIS WP
1087) — with the caveat that carry Sharpe decayed sharply in 2025
(arXiv:2510.14435). This script tests, on the SAME fixed data / cost model /
H-1-correct engine as run_v13_final.py:

  BENCH   equal-weight buy & hold of the 20-asset universe (no costs)
  A       v13 long-short banded top3 (baseline reproduction)
  A_fund  A + perp funding accrual on positions (v13 never accounted it)
  B       long-only banded top3 (gross 1.0 long)
  C       long banded top3 vs short EW-index hedge (market-neutral long tilt)
  D       funding-carry sleeve: banded top3/bottom3 by trailing 3d funding
          (long most-negative funding, short most-positive), incl. accrual
  E       50/50 daily combo of A and D returns (approximation: no netting)

OOS model predictions are reconstructed from checkpoints/folds_v13 with the
same CPCV splits (aborts if n_samples differs — split fingerprint check).

DISCLOSURE: every variant here is an additional research trial; the DSR
n_trials count grows accordingly. Nothing in this script touches the live
paper-trading experiment.
本脚本不触碰线上模拟盘实验；所有变体都会增加 DSR 的试验计数。
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from engine.cpcv import generate_cpcv_splits
from engine.twap_executor import TWAPExecutor
from model.cross_asset_attention import CrossAssetGRUAttention
from run_v13_final import (  # reuse the exact v13 pipeline / 复用v13管线
    build_from_parquet, _banded_targets, LABEL_H, SEED,
)
from tools.validation_stats import probabilistic_sharpe

SEQ_LEN = 24
N_SAMPLES_EXPECTED = 13105  # split fingerprint (M-12 guard) / split指纹校验
FUNDING_DB = str(BASE / "funding_rates.db")
CARRY_LOOKBACK = 72  # trailing 3d mean funding as carry signal / 3天均值作carry信号


def rebuild_oos_predictions(X: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    n_samples, n_assets = X.size(0), X.size(1)
    assert n_samples == N_SAMPLES_EXPECTED, (
        f"n_samples={n_samples} != {N_SAMPLES_EXPECTED}: data lake changed since "
        f"v13 training — fold/split mapping would be invalid (M-12). Retrain first.")
    splits = generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                                  purge_bars=SEQ_LEN + LABEL_H, embargo_bars=48)
    pred = torch.zeros(n_samples, n_assets, device=device)
    cnt = torch.zeros(n_samples, device=device)
    for fi, (_, test_idx) in enumerate(splits):
        ck = torch.load(f"checkpoints/folds_v13/fold_{fi:02d}.pt",
                        map_location=device, weights_only=True)
        model = CrossAssetGRUAttention(
            n_factors=X.size(3), d_model=128, gru_layers=2, n_cross_heads=4,
            n_cross_layers=3, d_ff=256, dropout=0.0, seq_len=SEQ_LEN,
            max_assets=n_assets).to(device)
        model.load_state_dict(ck["state"])
        model.eval()
        t_idx = torch.from_numpy(test_idx).to(device)
        with torch.no_grad():
            for s in range(0, len(test_idx), 512):
                e = min(s + 512, len(test_idx))
                pred[t_idx[s:e]] += model(X[t_idx[s:e]])
                cnt[t_idx[s:e]] += 1
        del model
        torch.cuda.empty_cache()
    valid = cnt > 0
    pred[valid] /= cnt[valid].unsqueeze(-1)
    return pred, valid


def load_funding_matrix(syms: List[str], close_mat: torch.Tensor,
                        bar_times_by_sym: Dict[str, np.ndarray]) -> np.ndarray:
    """(T, A) raw funding rate, 8h cadence forward-filled to 1h bars.
    原始资金费率矩阵（未标准化），8h前向填充到1h。"""
    T = close_mat.size(0)
    F = np.zeros((T, len(syms)), dtype=np.float64)
    conn = sqlite3.connect(f"file:{FUNDING_DB}?mode=ro", uri=True)
    for j, sym in enumerate(syms):
        rows = conn.execute(
            "SELECT ts_ms, rate FROM funding WHERE symbol=? ORDER BY ts_ms",
            (sym,)).fetchall()
        if not rows:
            continue
        ts = np.asarray([r[0] for r in rows], dtype=np.int64)
        rt = np.asarray([r[1] for r in rows], dtype=np.float64)
        bt = bar_times_by_sym[sym]
        idx = np.searchsorted(ts, bt, side="right") - 1
        ok = idx >= 0
        F[ok, j] = rt[idx[ok]]
    conn.close()
    return F


def run_variant(
    name: str,
    scores: np.ndarray, valid: np.ndarray,
    r1h: np.ndarray, close: np.ndarray, fund: Optional[np.ndarray],
    k: int = 3, enter: int = 3, exit_: int = 6, decision_every: int = 24,
    long_gross: float = 0.5, short_mode: str = "band",  # band | none | ew
    fund_accrual: bool = False,
) -> Dict:
    """Generic banded backtest, H-1-correct (positions effective next bar).
    通用banding回测，H-1修正口径（仓位次bar生效）。"""
    n, A = scores.shape
    T_full = close.shape[0]
    short_gross = 0.0 if short_mode == "none" else long_gross
    twap = TWAPExecutor(n_slices=4, favorable_reject_rate=0.60)
    equity = 1_000_000.0
    eq = [equity]
    w = np.zeros(A)
    if short_mode == "ew":
        pass  # hedge added on first entry / 对冲腿首次入场时建立
    long_set: Set[int] = set()
    short_set: Set[int] = set()
    total_cost = 0.0
    slot_trades = 0
    fund_pnl_cum = 0.0

    def slot_cost(asset: int, t: int, notional: float, side: str) -> float:
        di = t + SEQ_LEN - 1
        ep = close[di, asset]
        fu = [close[min(di + 1 + j, T_full - 1), asset] for j in range(4)]
        if ep <= 0:
            return 0.0
        _, cbps, _ = twap.execute_twap(side, notional, float(ep), fu)
        return notional * cbps / 10000.0

    hedge_on = False
    for t in range(n):
        pnl = float((w * r1h[t]).sum())
        if fund_accrual and fund is not None:
            fp = float(-(w * fund[t + SEQ_LEN - 1] / 8.0).sum())
            pnl += fp
            fund_pnl_cum += fp * equity
        cost_bar = 0.0

        if valid[t] and (t % decision_every == 0):
            if short_mode == "band":
                new_l, new_s = _banded_targets(scores[t], long_set, short_set,
                                               k, enter, exit_)
            else:
                new_l, _ = _banded_targets(scores[t], long_set, set(), k, enter, exit_)
                new_s = set()
            if new_l != long_set or new_s != short_set or (short_mode == "ew" and not hedge_on):
                slot_notional = equity * long_gross / k
                for a in (long_set - new_l):
                    cost_bar += slot_cost(a, t, slot_notional, "SELL")
                for a in (new_l - long_set):
                    cost_bar += slot_cost(a, t, slot_notional, "BUY")
                if short_mode == "band":
                    for a in (short_set - new_s):
                        cost_bar += slot_cost(a, t, slot_notional, "BUY")
                    for a in (new_s - short_set):
                        cost_bar += slot_cost(a, t, slot_notional, "SELL")
                slot_trades += (len(long_set - new_l) + len(new_l - long_set)
                                + len(short_set - new_s) + len(new_s - short_set))
                long_set, short_set = new_l, new_s
                w = np.zeros(A)
                for a in long_set:
                    w[a] = long_gross / k
                if short_mode == "band":
                    for a in short_set:
                        w[a] = -short_gross / k
                elif short_mode == "ew":
                    if not hedge_on:  # one-time hedge entry cost / 对冲腿一次性成本
                        cost_bar += slot_cost(0, t, equity * short_gross, "SELL")
                        hedge_on = True
                    w -= short_gross / A
        total_cost += cost_bar
        ret = pnl - cost_bar / max(equity, 1.0)
        ret = max(min(ret, 0.10), -0.10)
        equity *= (1.0 + ret)
        eq.append(equity)

    rets = np.diff(np.asarray(eq)) / np.asarray(eq)[:-1]
    m, sd = rets.mean(), rets.std()
    years = len(rets) / (24 * 365)
    peak, mdd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak)
    return {
        "name": name,
        "total_return": eq[-1] / eq[0] - 1,
        "sharpe": m / max(sd, 1e-12) * np.sqrt(24 * 365),
        "max_dd": mdd,
        "cost": total_cost,
        "slot_trades_py": slot_trades / max(years, 1e-9),
        "psr": probabilistic_sharpe(list(rets), 0.0),
        "fund_pnl": fund_pnl_cum,
        "rets": rets,
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  OFFLINE RESEARCH: long-only + funding carry vs v13 LS")
    print("=" * 70)

    X, y24, r1h_t, close_t, syms, n_factors = build_from_parquet(SEQ_LEN, 20, device)
    # bar timestamps per symbol for funding alignment (rebuild like pipeline)
    from data.lake_loader import load_klines_multi
    import polars as pl
    from run_v13_final import aggregate_5m_to_1h
    raw = load_klines_multi(interval="5m", min_rows=40000)
    agg = {s: aggregate_5m_to_1h(raw[s]) for s in syms}
    common = None
    for s in syms:
        ts = agg[s]["open_time"].to_numpy()
        common = ts if common is None else np.intersect1d(common, ts)
    bar_times = {s: agg[s].filter(pl.col("open_time").is_in(common))
                 .sort("open_time")["open_time"].to_numpy().astype(np.int64)
                 for s in syms}

    pred, valid_t = rebuild_oos_predictions(X, device)
    scores = pred.cpu().numpy()
    valid = valid_t.cpu().numpy()
    r1h = r1h_t.cpu().numpy()
    close = close_t.cpu().numpy()
    fund = load_funding_matrix(syms, close_t, bar_times)
    cov = (fund != 0).mean()
    print(f"  funding matrix coverage: {cov:.1%}")

    # carry signal: NEGATIVE trailing mean funding (long low/negative funding)
    # carry信号：3天均值funding取负（做多低/负funding，做空高funding）
    n = scores.shape[0]
    carry_sig = np.zeros_like(scores)
    fmat = fund  # (T, A)
    csum = np.cumsum(fmat, axis=0)
    for t in range(n):
        di = t + SEQ_LEN - 1
        lo = max(di - CARRY_LOOKBACK, 0)
        carry_sig[t] = -(csum[di] - csum[lo]) / max(di - lo, 1)
    carry_valid = np.ones(n, dtype=bool)
    carry_valid[:CARRY_LOOKBACK] = False

    results = []
    # benchmark: EW buy & hold (no costs) / 等权基准
    bench_rets = r1h.mean(axis=1)
    b_m, b_sd = bench_rets.mean(), bench_rets.std()
    b_eq = float(np.prod(1 + np.clip(bench_rets, -0.10, 0.10)))
    results.append({"name": "BENCH_ew_buyhold", "total_return": b_eq - 1,
                    "sharpe": b_m / max(b_sd, 1e-12) * np.sqrt(24 * 365),
                    "max_dd": float("nan"), "cost": 0.0, "slot_trades_py": 0,
                    "psr": probabilistic_sharpe(list(bench_rets), 0.0),
                    "fund_pnl": 0.0, "rets": bench_rets})

    variants = [
        dict(name="A_v13_LS_band", sc=scores, va=valid, short_mode="band", fa=False),
        dict(name="A_fund_LS_band+accrual", sc=scores, va=valid, short_mode="band", fa=True),
        dict(name="B_long_only_band", sc=scores, va=valid, short_mode="none",
             fa=False, long_gross=1.0),
        dict(name="C_long_band_vs_EW_hedge", sc=scores, va=valid, short_mode="ew", fa=False),
        dict(name="D_carry_sleeve", sc=carry_sig, va=carry_valid, short_mode="band", fa=True),
    ]
    for v in variants:
        random.seed(SEED)
        results.append(run_variant(
            v["name"], v["sc"], v["va"], r1h, close, fund,
            short_mode=v["short_mode"], fund_accrual=v["fa"],
            long_gross=v.get("long_gross", 0.5)))

    # E: 50/50 daily combo of A and D / 组合
    ra = results[1]["rets"]
    rd = results[-1]["rets"]
    rc = 0.5 * ra + 0.5 * rd
    m, sd = rc.mean(), rc.std()
    eqc = float(np.prod(1 + rc))
    peak, mdd = 1.0, 0.0
    acc = 1.0
    for x in rc:
        acc *= (1 + x)
        peak = max(peak, acc)
        mdd = max(mdd, (peak - acc) / peak)
    results.append({"name": "E_combo_50_A_50_D", "total_return": eqc - 1,
                    "sharpe": m / max(sd, 1e-12) * np.sqrt(24 * 365),
                    "max_dd": mdd, "cost": float("nan"), "slot_trades_py": 0,
                    "psr": probabilistic_sharpe(list(rc), 0.0),
                    "fund_pnl": 0.0, "rets": rc})

    print("\n" + "=" * 70)
    print(f"  {'variant':<28} {'return':>9} {'Sharpe':>7} {'maxDD':>7} "
          f"{'PSR':>5} {'slots/yr':>8} {'fundPnL':>10}")
    print("-" * 70)
    for r in results:
        print(f"  {r['name']:<28} {r['total_return']:>9.2%} {r['sharpe']:>7.3f} "
              f"{r['max_dd']:>7.2%} {r['psr']:>5.2f} {r['slot_trades_py']:>8.0f} "
              f"{r['fund_pnl']:>10,.0f}")

    # daily-return correlation matrix / 日收益相关矩阵
    print("\n  Daily-return correlations:")
    names = [r["name"] for r in results]
    daily = []
    for r in results:
        x = r["rets"]
        nb = len(x) // 24
        daily.append(x[:nb * 24].reshape(nb, 24).sum(axis=1))
    cm = np.corrcoef(np.vstack(daily))
    hdr = "        " + " ".join(f"{i:>6d}" for i in range(len(names)))
    print(hdr)
    for i, nm in enumerate(names):
        print(f"  [{i}] " + " ".join(f"{cm[i, j]:>+6.2f}" for j in range(len(names)))
              + f"  {nm}")

    out = {r["name"]: {k2: (v if not isinstance(v, np.ndarray) else None)
                       for k2, v in r.items() if k2 != "rets"} for r in results}
    out["_corr"] = {"names": names, "matrix": cm.tolist()}
    with open("checkpoints/research_carry_longonly.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\n  saved: checkpoints/research_carry_longonly.json")


if __name__ == "__main__":
    main()
