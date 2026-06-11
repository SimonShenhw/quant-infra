"""
crosscheck_v13_engines.py — Second-engine cross-check of the v13 config-C
backtest (banded top-3/bottom-3 long-short, daily decisions).

Motivation: arXiv:2603.20319 reports that high-turnover strategies can
diverge by up to 3.71pp across backtest engines once realistic costs are
added. This tool re-prices the SAME v13-C decision sequence in a fully
independent mark-to-market engine and quantifies the divergence in pp.
动机：高换手策略在不同回测引擎间（加入现实成本后）可分歧达 3.71pp，
故用独立第二引擎对同一决策序列复算并量化分歧。

vectorbt is NOT installed in this env (checked 2026-06-10) and installing
packages is forbidden, so ENGINE 2 is a hand-written pure-numpy engine.
本环境未装 vectorbt 且严禁改动环境，第二引擎为纯 numpy 手写实现。

Pipeline / 流程
---------------
 1. Rebuild the OOS prediction matrix from checkpoints/folds_v13 via the
    ORIGINAL data pipeline (run_v13_final.build_from_parquet) + fold-ckpt
    inference, identical to run_cpcv. Cached to
    checkpoints/v13_crosscheck/oos_cache.npz (CUDA needed first time only).
 2. ENGINE 1 = the original run_v13_final.run_backtest, untouched:
      a. original stochastic TWAP cost model, random.seed(42)
         -> must reproduce the +32.63% headline (validates step 1)
      b. fixed-fee variant: TWAPExecutor monkey-patched to a flat
         taker 4bps + slippage 4bps = 8 bps/side stub.
 3. ENGINE 2 = independent numpy engine (shares NO backtest code):
      unit-based positions, per-bar mark-to-market on the close matrix,
      fills at the first close AFTER the decision close (same effective
      timing as engine 1), flat 8 bps/side on ACTUAL traded notional,
      positions DRIFT between rebalances (engine 1 implicitly renormalizes
      weights to +/-1/(2k) every bar at zero cost). Two sizing modes:
        changed : trade only entering/exiting slots — the trades engine 1
                  actually charges for
        full    : re-target ALL slots at each rebalance event — what a
                  target-weight engine (e.g. vectorbt) would do
 4. Replica check: engine-1 semantics re-implemented in numpy must match
    2b to float precision -> proves the decision extraction is exact and
    any E1/E2 gap is engine mechanics, not strategy mismatch.
 5. Attribution: fee=0 runs split the divergence into compounding/drift
    (复利方式) vs cost accounting (成本计提); fill-timing is held equal by
    construction and verified via the replica.

Usage / 用法
------------
    python tools/crosscheck_v13_engines.py             # cache-aware
    python tools/crosscheck_v13_engines.py --rebuild   # force OOS rebuild
    python tools/crosscheck_v13_engines.py --fee-bps 8

Outputs (new files only; nothing existing is modified)
    checkpoints/v13_crosscheck/oos_cache.npz
    checkpoints/v13_crosscheck/equity_curves.csv
    checkpoints/v13_crosscheck/crosscheck_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "checkpoints" / "v13_crosscheck"
CACHE = OUTDIR / "oos_cache.npz"
SUMMARY_JSON = ROOT / "checkpoints" / "v13_backtest_summary.json"

# v13 config C / v13 配置 C
SEQ_LEN = 24
MAX_ASSETS = 20
K = 3
ENTER_BAND = 3
EXIT_BAND = 6
DECISION_EVERY = 24
START_EQUITY = 1_000_000.0
BARS_PER_YEAR = 24 * 365

# fixed-fee assumption / 固定费率假设: taker 4bps + slippage 4bps per side
DEFAULT_FEE_BPS = 4.0 + 4.0


# ============================================================================
# Step 1 — rebuild OOS predictions from fold checkpoints (cached)
# ============================================================================

def build_cache(force: bool) -> Dict[str, np.ndarray]:
    if CACHE.exists() and not force:
        z = np.load(CACHE)
        print(f"[Cache] Loaded {CACHE.name}: pred {z['pred'].shape}, "
              f"close {z['close'].shape}")
        return {k: z[k] for k in z.files}

    import torch
    import run_v13_final as v13
    from engine.cpcv import generate_cpcv_splits
    from model.cross_asset_attention import CrossAssetGRUAttention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Rebuild] device={device}")
    X, y24, r1h, close_mat, syms, n_factors = v13.build_from_parquet(
        SEQ_LEN, MAX_ASSETS, device)
    n_samples, n_assets = X.size(0), X.size(1)

    purge = SEQ_LEN + v13.LABEL_H  # must match run_cpcv / 与 run_cpcv 一致
    splits = generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                                  purge_bars=purge, embargo_bars=48)
    print(f"[Rebuild] {len(splits)} CPCV splits (purge={purge}, embargo=48)")

    pred = torch.zeros(n_samples, n_assets, device=device)
    cnt = torch.zeros(n_samples, device=device)
    for fi, (_, test_idx) in enumerate(splits):
        ckpt_path = ROOT / "checkpoints" / "folds_v13" / f"fold_{fi:02d}.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CrossAssetGRUAttention(
            n_factors=n_factors, d_model=128, gru_layers=2,
            n_cross_heads=4, n_cross_layers=3, d_ff=256,
            dropout=0.30, seq_len=SEQ_LEN, max_assets=n_assets,
        ).to(device)
        model.load_state_dict(ckpt["state"])
        model.eval()
        test_idx_t = torch.from_numpy(test_idx).to(device)
        with torch.no_grad():
            for ts in range(0, len(test_idx), 512):
                te = min(ts + 512, len(test_idx))
                chunk = test_idx_t[ts:te]
                pred[chunk] += model(X[chunk])
                cnt[chunk] += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  fold {fi + 1}/{len(splits)} predicted")

    valid = cnt > 0
    pred[valid] /= cnt[valid].unsqueeze(-1)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {
        "pred": pred.cpu().numpy(),
        "valid": valid.cpu().numpy(),
        "r1h": r1h.cpu().numpy(),
        "close": close_mat.cpu().numpy(),
        "syms": np.array(syms),
    }
    np.savez(CACHE, **out)
    print(f"[Rebuild] cached -> {CACHE}")
    return out


# ============================================================================
# Step 2 — ENGINE 1: the original run_backtest (TWAP / fixed-fee patched)
# ============================================================================

def _make_fixed_fee_twap(fee_bps: float):
    """Stub with TWAPExecutor's interface returning a flat cost.
    与 TWAPExecutor 同接口的固定费率桩。"""

    class _FixedFeeTWAP:
        def __init__(self, *args, **kwargs) -> None:
            self.total_slices = 0

        def execute_twap(self, side, target_notional, entry_price, future_closes):
            self.total_slices += max(len(future_closes), 1)
            return entry_price, fee_bps, 1.0

        def stats(self):
            return {"total_slices": self.total_slices, "maker_fill_pct": 0.0,
                    "adverse_fill_pct": 0.0, "taker_fill_pct": 1.0,
                    "reject_then_taker_pct": 0.0}

    return _FixedFeeTWAP


def run_engine1(cache: Dict[str, np.ndarray], fee_bps: float | None,
                name: str) -> Dict:
    """fee_bps=None -> original stochastic TWAP cost model (seeded).
    fee_bps=None 时用原始 TWAP 成本模型（按 main() 的方式播种）。"""
    import torch
    import run_v13_final as v13

    pred = torch.from_numpy(cache["pred"])
    valid = torch.from_numpy(cache["valid"])
    r1h = torch.from_numpy(cache["r1h"])
    close = torch.from_numpy(cache["close"])

    orig_twap = v13.TWAPExecutor
    try:
        if fee_bps is not None:
            v13.TWAPExecutor = _make_fixed_fee_twap(fee_bps)
        random.seed(v13.SEED)  # same as main(): reseed before each config
        res = v13.run_backtest(name, pred, valid, r1h, close, SEQ_LEN,
                               K, ENTER_BAND, EXIT_BAND, DECISION_EVERY,
                               min_hold=0, use_vol_filter=False)
    finally:
        v13.TWAPExecutor = orig_twap

    rets = np.asarray(res["rets"], dtype=np.float64)
    res["eq_curve"] = START_EQUITY * np.concatenate(
        [[1.0], np.cumprod(1.0 + rets)])
    res["max_abs_bar_ret"] = float(np.abs(rets).max())
    return res


# ============================================================================
# Step 3 — shared decision sequence (strategy is identical by construction)
# ============================================================================

def extract_decisions(scores: np.ndarray, valid: np.ndarray
                      ) -> Tuple[np.ndarray, List[Dict]]:
    """Replay the banded state machine ONCE; both engines consume the same
    weight schedule / trade events. Uses the original _banded_targets so the
    decisions are bit-identical to engine 1 (config C: min_hold=0, vol off).
    只回放一次 banded 状态机，两引擎共用同一决策序列。"""
    from run_v13_final import _banded_targets

    n, A = scores.shape
    W = np.zeros((n, A))
    events: List[Dict] = []
    long_set, short_set = set(), set()
    w = np.zeros(A)
    for t in range(n):
        if valid[t] and t % DECISION_EVERY == 0:
            new_l, new_s = _banded_targets(scores[t], long_set, short_set,
                                           K, ENTER_BAND, EXIT_BAND)
            if new_l != long_set or new_s != short_set:
                events.append({
                    "t": t,
                    "exit_l": sorted(long_set - new_l),
                    "enter_l": sorted(new_l - long_set),
                    "exit_s": sorted(short_set - new_s),
                    "enter_s": sorted(new_s - short_set),
                    "long": sorted(new_l), "short": sorted(new_s),
                })
                long_set, short_set = new_l, new_s
                w = np.zeros(A)
                w[list(long_set)] = 0.5 / K
                w[list(short_set)] = -0.5 / K
        W[t] = w
    return W, events


# ============================================================================
# Step 4 — replica of engine-1 semantics (decision-extraction exactness check)
# ============================================================================

def run_engine1_replica(W: np.ndarray, events: List[Dict], r1h: np.ndarray,
                        fee_rate: float) -> Dict:
    """Engine-1 accounting re-implemented on the extracted decisions:
    constant-weight compounding, cost on equity*0.5/K per changed slot at
    the decision bar, +/-10% clip. Must match engine-1-fixed to ~1e-12.
    引擎1记账方式在提取决策上的复刻，用于校验决策序列与引擎1完全一致。"""
    n, A = W.shape
    ev_by_t = {e["t"]: e for e in events}
    eq = START_EQUITY
    curve = [eq]
    total_cost = 0.0
    w_prev = np.zeros(A)
    for t in range(n):
        pnl = float((w_prev * r1h[t]).sum())
        cost = 0.0
        if t in ev_by_t:
            e = ev_by_t[t]
            n_slots = (len(e["exit_l"]) + len(e["enter_l"])
                       + len(e["exit_s"]) + len(e["enter_s"]))
            cost = n_slots * (eq * 0.5 / K) * fee_rate
            w_prev = W[t]
        total_cost += cost
        ret = pnl - cost / max(eq, 1.0)
        ret = max(min(ret, 0.10), -0.10)
        eq *= 1.0 + ret
        curve.append(eq)
    return _metrics("E2_replica_of_E1", np.asarray(curve), total_cost)


# ============================================================================
# Step 5 — ENGINE 2: independent unit-based mark-to-market engine
# ============================================================================

def run_engine2(close: np.ndarray, W: np.ndarray, events: List[Dict],
                fee_rate: float, retarget: str, name: str) -> Dict:
    """Positions held in UNITS, marked to market on every 1h close.
    Decision at sample t (close index t+SEQ_LEN-1) fills at close index
    t+SEQ_LEN — exactly when engine 1 starts crediting the new position.
    Costs hit the cash account at fill on ACTUAL traded notional. No clip,
    no per-bar renormalization: weights drift between rebalances.
    以持仓单位逐bar盯市；决策在下一根close成交（与引擎1生效时点一致）；
    成本按实际成交名义额在成交时从现金扣除；调仓间隙权重自然漂移。

    retarget='changed': trade only entering/exiting slots (engine 1 only
    charges these). retarget='full': re-target every slot to +/-1/(2K)
    at each rebalance event (target-weight-engine convention).
    """
    T, A = close.shape
    px_all = close.astype(np.float64)
    n = W.shape[0]
    ev_by_t = {e["t"]: e for e in events}

    cash = START_EQUITY
    units = np.zeros(A)
    curve = [cash]
    total_cost = 0.0
    total_traded = 0.0

    for t in range(n):
        ci = t + SEQ_LEN  # fill/mark close index for eq point t+1
        px = px_all[ci]
        if t in ev_by_t:
            e = ev_by_t[t]
            equity_now = cash + float(units @ px)
            delta = np.zeros(A)
            if retarget == "full":
                tgt_units = W[t] * equity_now / np.maximum(px, 1e-12)
                delta = tgt_units - units
            else:  # 'changed'
                slot_notional = equity_now * 0.5 / K
                for a in e["exit_l"] + e["exit_s"]:
                    delta[a] = -units[a]
                for a in e["enter_l"]:
                    delta[a] += slot_notional / max(px[a], 1e-12)
                for a in e["enter_s"]:
                    delta[a] -= slot_notional / max(px[a], 1e-12)
            traded = float(np.abs(delta) @ px)
            cost = traded * fee_rate
            cash -= float(delta @ px) + cost
            units = units + delta
            total_cost += cost
            total_traded += traded
        curve.append(cash + float(units @ px))

    out = _metrics(name, np.asarray(curve), total_cost)
    out["total_traded"] = total_traded
    return out


# ============================================================================
# Metrics + report
# ============================================================================

def _metrics(name: str, curve: np.ndarray, total_cost: float) -> Dict:
    from tools.validation_stats import probabilistic_sharpe

    rets = curve[1:] / curve[:-1] - 1.0
    avg, std = rets.mean(), rets.std()
    peak = np.maximum.accumulate(curve)
    return {
        "name": name,
        "total_return": float(curve[-1] / curve[0] - 1.0),
        "sharpe": float(avg / max(std, 1e-9) * BARS_PER_YEAR ** 0.5),
        "max_drawdown": float(((peak - curve) / peak).max()),
        "total_cost": float(total_cost),
        "final_equity": float(curve[-1]),
        "max_abs_bar_ret": float(np.abs(rets).max()),
        "psr": float(probabilistic_sharpe(rets.tolist(), 0.0)),
        "eq_curve": curve,
    }


def _gap_pp(a: Dict, b: Dict) -> float:
    return (a["total_return"] - b["total_return"]) * 100.0


def _curve_stats(a: Dict, b: Dict) -> Dict:
    ca, cb = a["eq_curve"], b["eq_curve"]
    log_gap = np.log(ca / cb)
    ra = ca[1:] / ca[:-1] - 1.0
    rb = cb[1:] / cb[:-1] - 1.0
    cc = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float("nan")
    return {
        "final_gap_pp": _gap_pp(a, b),
        "max_abs_log_gap": float(np.abs(log_gap).max()),
        "bar_ret_corr": cc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rebuild", action="store_true",
                    help="force OOS prediction rebuild (needs CUDA)")
    ap.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS,
                    help="flat per-side cost in bps (default 8 = taker 4 + slip 4)")
    args = ap.parse_args()
    fee_rate = args.fee_bps / 10000.0

    cache = build_cache(args.rebuild)
    scores = cache["pred"].astype(np.float64)
    valid = cache["valid"].astype(bool)
    r1h = cache["r1h"].astype(np.float64)
    close = cache["close"]

    # reference numbers from the original run / 原始跑批参考值
    ref = None
    if SUMMARY_JSON.exists():
        with open(SUMMARY_JSON) as f:
            for c in json.load(f)["configs"]:
                if c["name"] == "C_v13_band_top3_daily":
                    ref = c

    print("\n[Engine 1] original run_backtest ...")
    e1_twap = run_engine1(cache, fee_bps=None, name="E1_twap_original")
    e1_fix = run_engine1(cache, fee_bps=args.fee_bps, name="E1_fixed_fee")
    e1_free = run_engine1(cache, fee_bps=0.0, name="E1_zero_fee")

    print("[Engine 2] decision extraction + independent numpy engine ...")
    W, events = extract_decisions(scores, valid)
    n_slot_trades = sum(len(e["exit_l"]) + len(e["enter_l"])
                        + len(e["exit_s"]) + len(e["enter_s"]) for e in events)

    replica = run_engine1_replica(W, events, r1h, fee_rate)
    e2_chg = run_engine2(close, W, events, fee_rate, "changed", "E2_numpy_changed")
    e2_full = run_engine2(close, W, events, fee_rate, "full", "E2_numpy_full")
    e2_chg0 = run_engine2(close, W, events, 0.0, "changed", "E2_changed_zero_fee")
    e2_full0 = run_engine2(close, W, events, 0.0, "full", "E2_full_zero_fee")

    # ---- report / 报告 ----
    line = "=" * 74
    print("\n" + line)
    print("  v13-C ENGINE CROSS-CHECK  "
          f"(flat {args.fee_bps:.0f} bps/side = taker 4 + slippage 4)")
    print(line)

    if ref:
        d_rep = (e1_twap["total_return"] - ref["total_return"]) * 100.0
        print(f"  [Validation] E1 original TWAP rerun vs saved summary:")
        print(f"    total_return {e1_twap['total_return']:+.4%} vs "
              f"{ref['total_return']:+.4%}  (diff {d_rep:+.4f} pp)")
        print(f"    rebalances {e1_twap['rebalances']} vs {ref['rebalances']}, "
              f"slot_trades {e1_twap['slot_trades']} vs {ref['slot_trades']}")
    print(f"  [Validation] decision extraction: {len(events)} rebalance events, "
          f"{n_slot_trades} slot trades (engine 1: {e1_twap['rebalances']} / "
          f"{e1_twap['slot_trades']})")
    rep_gap = _gap_pp(replica, e1_fix)
    print(f"  [Validation] numpy replica of E1 semantics vs E1 fixed-fee: "
          f"{rep_gap:+.6f} pp (should be ~0)")

    rows = [e1_twap, e1_fix, replica, e2_chg, e2_full, e1_free, e2_chg0, e2_full0]
    print(f"\n  {'run':<26s}{'totRet':>10s}{'Sharpe':>9s}{'PSR':>8s}{'maxDD':>9s}"
          f"{'cost$':>12s}{'maxBarRet':>11s}")
    for r in rows:
        print(f"  {r['name']:<26s}{r['total_return']:>10.2%}{r['sharpe']:>9.3f}"
              f"{r.get('psr', float('nan')):>8.3f}"
              f"{r['max_drawdown']:>9.2%}{r['total_cost']:>12,.0f}"
              f"{r['max_abs_bar_ret']:>11.4f}")

    cs_chg = _curve_stats(e1_fix, e2_chg)
    cs_full = _curve_stats(e1_fix, e2_full)
    drift_chg = _gap_pp(e1_free, e2_chg0)
    drift_full = _gap_pp(e1_free, e2_full0)

    print(f"\n  --- divergence @ fixed {args.fee_bps:.0f} bps/side ---")
    print(f"  E1_fixed vs E2_changed : {cs_chg['final_gap_pp']:+.3f} pp "
          f"(max |log eq gap| {cs_chg['max_abs_log_gap']:.5f}, "
          f"bar-ret corr {cs_chg['bar_ret_corr']:.5f})")
    print(f"  E1_fixed vs E2_full    : {cs_full['final_gap_pp']:+.3f} pp "
          f"(max |log eq gap| {cs_full['max_abs_log_gap']:.5f}, "
          f"bar-ret corr {cs_full['bar_ret_corr']:.5f})")
    print(f"\n  --- attribution / 归因 ---")
    print(f"  compounding/drift only (fee=0)      : changed {drift_chg:+.3f} pp, "
          f"full {drift_full:+.3f} pp")
    print(f"  cost-accounting component (residual): changed "
          f"{cs_chg['final_gap_pp'] - drift_chg:+.3f} pp, "
          f"full {cs_full['final_gap_pp'] - drift_full:+.3f} pp")
    print(f"  cost dollars E1_fixed / E2_changed / E2_full : "
          f"{e1_fix['total_cost']:,.0f} / {e2_chg['total_cost']:,.0f} / "
          f"{e2_full['total_cost']:,.0f}")
    print(line)

    # ---- persist / 落盘 ----
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n_pts = len(e1_fix["eq_curve"])
    cols = {r["name"]: r["eq_curve"] for r in rows}
    csv_path = OUTDIR / "equity_curves.csv"
    with open(csv_path, "w") as f:
        f.write("idx,close_index," + ",".join(cols) + "\n")
        for i in range(n_pts):
            f.write(f"{i},{SEQ_LEN - 1 + i},"
                    + ",".join(f"{cols[c][i]:.4f}" for c in cols) + "\n")

    summary = {
        "fee_bps_per_side": args.fee_bps,
        "validation": {
            "e1_twap_rerun_vs_saved_pp": (
                (e1_twap["total_return"] - ref["total_return"]) * 100.0 if ref else None),
            "replica_vs_e1_fixed_pp": rep_gap,
            "rebalance_events": len(events),
            "slot_trades": n_slot_trades,
        },
        "runs": [{k: v for k, v in r.items() if k != "eq_curve"} for r in rows],
        "divergence": {
            "e1_fixed_vs_e2_changed": cs_chg,
            "e1_fixed_vs_e2_full": cs_full,
            "drift_only_pp": {"changed": drift_chg, "full": drift_full},
            "cost_component_pp": {
                "changed": cs_chg["final_gap_pp"] - drift_chg,
                "full": cs_full["final_gap_pp"] - drift_full,
            },
        },
    }
    json_path = OUTDIR / "crosscheck_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  saved: {csv_path}")
    print(f"  saved: {json_path}")


if __name__ == "__main__":
    main()
