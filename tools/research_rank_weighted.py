"""
Rank-weighted full-book vs banded top-K on the 2021-2026 OOS predictions.
连续排名加权全簿 vs banded top-K（扩窗 OOS 预测，v14 变现层实验）。

Motivation (RESEARCH_2026-07-13_extended_window.md): the ranking signal is
robust across the full cycle (IC +0.076, every year positive) but top-3
tails lose -64% — the model's information lives in the MID-BOOK ranking,
not the extremes. Test the constructions the literature queued for exactly
this situation (Garleanu-Pedersen partial trading toward a rank-weighted
aim portfolio):

  A  banded top3/bottom3 (baseline, flat-cost re-run of the -64% strategy)
  B  rank-weighted full book: w_i propto (mean_rank - rank_i), gross 1.0
  C  B + GP partial trading tau=1/3  (move 1/3 toward the aim each day)
  D  B + GP partial trading tau=1/5

Same OOS predictions (rebuilt from checkpoints/folds_ext_full with the
n_samples fingerprint check), same daily decision cadence, same H-1 timing
(weights effective next bar), same FLAT 8 bps/side turnover cost for every
variant (the conservative cross-checked quote) + gross (cost-free) numbers
for decomposition. Registers 4 trials.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from engine.cpcv import generate_cpcv_splits
from model.cross_asset_attention import CrossAssetGRUAttention
from sleeves.banding import banded_targets
from tools.research_extended_window import build_window, WINDOWS, EXT_SYMBOLS, SEQ_LEN
from run_v13_final import LABEL_H, SEED
from tools.validation_stats import probabilistic_sharpe, register_trial

import factors  # noqa: F401

FOLD_DIR = "checkpoints/folds_ext_full"
COST_BPS = 8.0
DECISION_EVERY = 24


def rebuild_preds(X: torch.Tensor, device) -> torch.Tensor:
    n, A = X.size(0), X.size(1)
    splits = generate_cpcv_splits(n, n_groups=6, n_test_groups=2,
                                  purge_bars=SEQ_LEN + LABEL_H, embargo_bars=48)
    pred = torch.zeros(n, A, device=device)
    cnt = torch.zeros(n, device=device)
    for fi, (_, test_idx) in enumerate(splits):
        ck = torch.load(f"{FOLD_DIR}/fold_{fi:02d}.pt",
                        map_location=device, weights_only=True)
        assert int(ck["n_samples"]) == n, (
            f"fold {fi}: ckpt n_samples={ck['n_samples']} != {n} — "
            f"data window changed since training (fingerprint guard)")
        model = CrossAssetGRUAttention(
            n_factors=X.size(3), d_model=128, gru_layers=2, n_cross_heads=4,
            n_cross_layers=3, d_ff=256, dropout=0.0, seq_len=SEQ_LEN,
            max_assets=A).to(device)
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
    assert bool(valid.all()), "CPCV should cover every sample"
    return pred


def rank_weights(scores_row: np.ndarray, inv_vol: np.ndarray = None) -> np.ndarray:
    """w_i propto (mean_rank - rank_i) [/ vol_i], sum|w| = 1.
    排名去均值（可选按波动率缩放=每名字风险等权），毛敞口=1。"""
    A = len(scores_row)
    order = np.argsort(-scores_row)
    rank = np.empty(A)
    rank[order] = np.arange(A)
    w = (A - 1) / 2.0 - rank            # best -> +, worst -> -
    if inv_vol is not None:
        w = w * inv_vol
    return w / np.abs(w).sum()


def trailing_inv_vol(r1h: np.ndarray, window: int = 720) -> np.ndarray:
    """(n, A) causal inverse trailing vol, clipped. / 因果滚动波动率倒数。"""
    n, A = r1h.shape
    iv = np.ones((n, A))
    csum = np.cumsum(r1h, axis=0)
    csum2 = np.cumsum(r1h ** 2, axis=0)
    for t in range(window, n):
        m = (csum[t - 1] - csum[t - window - 1 if t > window else 0]) / window
        v = (csum2[t - 1] - csum2[t - window - 1 if t > window else 0]) / window - m ** 2
        sd = np.sqrt(np.maximum(v, 1e-12))
        iv[t] = 1.0 / np.maximum(sd, np.median(sd) * 0.25)  # clip extreme leverage / 限杠杆
    return iv


def band_weights(scores_row: np.ndarray, state: Dict) -> np.ndarray:
    l, s = banded_targets(scores_row, state["l"], state["s"], 3, 3, 6)
    state["l"], state["s"] = l, s
    A = len(scores_row)
    w = np.zeros(A)
    for a in l:
        w[a] = 0.5 / 3
    for a in s:
        w[a] = -0.5 / 3
    return w


def run_construction(name: str, scores: np.ndarray, r1h: np.ndarray,
                     bar_ms: np.ndarray, mode: str, tau: float = 1.0,
                     inv_vol: np.ndarray = None) -> Dict:
    """Flat-cost engine, H-1 timing (weights effective next bar).
    平价成本引擎，H-1 口径（权重次bar生效）。"""
    n, A = scores.shape
    w = np.zeros(A)
    equity = 1.0
    eq = [1.0]
    turnover = 0.0
    gross_pnl = 0.0
    state = {"l": set(), "s": set()}
    for t in range(n):
        pnl = float(w @ r1h[t])
        gross_pnl += pnl * equity
        cost = 0.0
        if t % DECISION_EVERY == 0:
            if mode == "band":
                target = band_weights(scores[t], state)
            else:
                iv = inv_vol[t] if inv_vol is not None else None
                target = rank_weights(scores[t], iv)
            new_w = (1 - tau) * w + tau * target
            turn = float(np.abs(new_w - w).sum())
            cost = turn * COST_BPS / 10000.0
            turnover += turn
            w = new_w  # earns from t+1 (pnl above used the old w) / 次bar生效
        ret = max(min(pnl - cost, 0.10), -0.10)
        equity *= 1.0 + ret
        eq.append(equity)

    rets = np.diff(np.asarray(eq)) / np.asarray(eq)[:-1]
    years_len = n / (24 * 365)
    m, sd = rets.mean(), rets.std()
    peak, mdd = 1.0, 0.0
    acc = 1.0
    for x in rets:
        acc *= 1 + x
        peak = max(peak, acc)
        mdd = max(mdd, (peak - acc) / peak)

    # per-year net Sharpe / 分年度净Sharpe
    yrs = (bar_ms // 1000).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    per_year = {}
    for y in sorted(set(yrs)):
        rr = rets[yrs == y]
        if len(rr) > 500:
            per_year[str(y)] = float(rr.mean() / max(rr.std(), 1e-12) * np.sqrt(24 * 365))

    return {"name": name,
            "total_return": float(acc - 1),
            "sharpe": float(m / max(sd, 1e-12) * np.sqrt(24 * 365)),
            "max_dd": float(mdd),
            "turnover_per_year": float(turnover / max(years_len, 1e-9)),
            "gross_return_sum": float(gross_pnl),
            "psr": probabilistic_sharpe(list(rets), 0.0),
            "per_year_sharpe": per_year}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  RANK-WEIGHTED FULL BOOK vs BANDED TOP-K (FULL window OOS)")
    print("=" * 70)

    X, y24_t, r1h_t, close_t, bar_ms, n_factors = build_window(WINDOWS["FULL"], device)
    cache = Path("checkpoints/preds_ext_full.npy")
    if cache.exists():
        scores = np.load(cache)
        assert scores.shape[0] == X.size(0), "preds cache stale — delete it"
        print(f"  loaded cached OOS preds {scores.shape}")
    else:
        pred = rebuild_preds(X, device)
        scores = pred.cpu().numpy()
        np.save(cache, scores)
    y24 = y24_t.cpu().numpy()
    r1h = r1h_t.cpu().numpy()
    del X, y24_t, close_t
    torch.cuda.empty_cache()

    inv_vol = trailing_inv_vol(r1h)

    # -- decisive diagnostic: 24h label-aligned GROSS alpha of each weighting,
    #    no path/timing effects. IC>0 with this negative = the rank info does
    #    not survive translation into return space (vol corruption).
    # -- 关键诊断：w·y24 的 24h 对齐毛alpha（无路径效应）。
    yrs = (bar_ms // 1000).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    print(f"\n  gross 24h-aligned alpha (w·y24, %/day, NO costs):")
    print(f"  {'weighting':<16}" + " ".join(f"{y:>8}" for y in sorted(set(yrs))) + f" {'ALL':>8}")
    diag = {}
    for wname, use_iv in (("rank", False), ("rank/vol", True)):
        daily = np.array([
            rank_weights(scores[t], inv_vol[t] if use_iv else None) @ y24[t]
            for t in range(0, scores.shape[0], DECISION_EVERY)])
        dyrs = yrs[::DECISION_EVERY][:len(daily)]
        row = {str(y): float(daily[dyrs == y].mean()) for y in sorted(set(dyrs))}
        row["ALL"] = float(daily.mean())
        diag[wname] = row
        print(f"  {wname:<16}" + " ".join(
            f"{row.get(str(y), float('nan')):>+8.4%}" for y in sorted(set(yrs)))
            + f" {row['ALL']:>+8.4%}")

    variants = [
        ("A_band_top3_flat8", "band", 1.0, False),
        ("B_rankw_full", "rank", 1.0, False),
        ("C_rankw_tau_1_3", "rank", 1 / 3, False),
        ("D_rankw_tau_1_5", "rank", 1 / 5, False),
        ("E_rankw_vol_full", "rank", 1.0, True),
        ("F_rankw_vol_tau_1_5", "rank", 1 / 5, True),
    ]
    results = []
    print(f"\n  {'variant':<20} {'return':>9} {'Sharpe':>7} {'maxDD':>7} "
          f"{'PSR':>5} {'turn/yr':>8}")
    print("-" * 62)
    for name, mode, tau, use_iv in variants:
        t0 = time.time()
        r = run_construction(name, scores, r1h, bar_ms, mode, tau,
                             inv_vol if use_iv else None)
        results.append(r)
        register_trial(f"rank_weighted_2026-07-13/{name}",
                       {"kind": "construction", "cost_bps": COST_BPS})
        print(f"  {name:<20} {r['total_return']:>9.2%} {r['sharpe']:>7.3f} "
              f"{r['max_dd']:>7.2%} {r['psr']:>5.2f} "
              f"{r['turnover_per_year']:>8.1f}  [{time.time()-t0:.0f}s]")
    print("\n  per-year net Sharpe:")
    hdr = "  " + f"{'variant':<20}" + " ".join(f"{y:>7}" for y in
          sorted(results[0]["per_year_sharpe"]))
    print(hdr)
    for r in results:
        print("  " + f"{r['name']:<20}" + " ".join(
            f"{r['per_year_sharpe'].get(y, float('nan')):>7.2f}"
            for y in sorted(results[0]["per_year_sharpe"])))

    with open("checkpoints/research_rank_weighted.json", "w") as f:
        json.dump({"variants": results, "diagnostic_w_y24": diag}, f, indent=2)
    print("\n  saved: checkpoints/research_rank_weighted.json")


if __name__ == "__main__":
    main()
