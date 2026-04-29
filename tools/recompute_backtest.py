"""
recompute_backtest.py — Recompute backtest from saved fold checkpoints.

Avoids the ~80min retrain when ONLY the backtest config changes (TWAP
slippage, vol filter, min_hold). Loads checkpoints/<folds_dir>/fold_*.pt
and runs OOS inference + a configurable backtest with slippage.

只重算回测，不重训。改 slippage / vol / min_hold 时用本工具，免去 80min 训练。

Usage / 用法
------------
# v12 with realistic 5/2/0 bps slippage (default) + v12 native config
python tools/recompute_backtest.py checkpoints/folds_v12

# v11.1 with realistic slippage + v11.1 native config (no vol filter, min_hold 48)
python tools/recompute_backtest.py checkpoints/folds --min-hold 48 --vol-quantile 0

# Stress test: assume wide spreads (15 bps adverse)
python tools/recompute_backtest.py checkpoints/folds_v12 --slip-adverse 15
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.cpcv import generate_cpcv_splits
from engine.twap_executor import TWAPExecutor
from model.cross_asset_attention import CrossAssetGRUAttention
from run_v12_final import build_from_parquet


def load_fold_predictions(
    folds_dir: str, splits, X, n_factors: int, n_assets: int,
    seq_len: int, device: torch.device,
):
    """Load each fold ckpt, run OOS inference, aggregate to pred_matrix.
    加载每个 fold ckpt 在其 OOS 上推理，叠加到 pred_matrix。
    """
    n_samples = X.size(0)
    pred_matrix = torch.zeros(n_samples, n_assets, device=device)
    pred_count = torch.zeros(n_samples, device=device)
    fold_corrs = []

    for fi, (_, test_idx) in enumerate(splits):
        ckpt_path = os.path.join(folds_dir, f"fold_{fi:02d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] fold {fi}: ckpt missing ({ckpt_path})")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CrossAssetGRUAttention(
            n_factors=n_factors, d_model=128, gru_layers=2,
            n_cross_heads=4, n_cross_layers=3, d_ff=256,
            dropout=0.30, seq_len=seq_len, max_assets=n_assets,
        ).to(device)
        model.load_state_dict(ckpt["state"])
        model.eval()

        test_idx_t = torch.from_numpy(test_idx).to(device)
        with torch.no_grad():
            for ts in range(0, len(test_idx), 512):
                te = min(ts + 512, len(test_idx))
                chunk_idx = test_idx_t[ts:te]
                chunk_scores = model(X[chunk_idx])
                pred_matrix[chunk_idx] += chunk_scores
                pred_count[chunk_idx] += 1
        fold_corrs.append(float(ckpt.get("corr", float("nan"))))
        del model
        torch.cuda.empty_cache()
        if (fi + 1) % 5 == 0 or fi == 0:
            print(f"  Fold {fi+1}/{len(splits)} loaded + predicted")

    valid_mask = pred_count > 0
    pred_matrix[valid_mask] /= pred_count[valid_mask].unsqueeze(-1)
    valid_corrs = [c for c in fold_corrs if not np.isnan(c)]
    avg_corr = sum(valid_corrs) / max(len(valid_corrs), 1)
    return pred_matrix, valid_mask, avg_corr, fold_corrs


def run_backtest(
    pred_matrix, valid_mask, r1h, close_mat,
    seq_len: int, min_hold: int, vol_quantile: float,
    slip_adv: float, slip_tkr: float, slip_mkr: float,
):
    """Backtest loop with configurable TWAP slippage + optional vol filter.
    可配 slippage + vol filter 的回测循环。
    """
    n_samples = pred_matrix.size(0)

    # rolling 1d basket vol / 1d 滚动 basket 波动
    vol_window = 24
    basket_ret_np = r1h.mean(dim=1).cpu().numpy()
    rolling_vol = np.zeros(n_samples, dtype=np.float64)
    for i in range(vol_window, n_samples):
        rolling_vol[i] = basket_ret_np[i - vol_window:i].std()
    valid_vol_mask = rolling_vol > 0
    if vol_quantile > 0 and valid_vol_mask.any():
        vol_thresh = float(np.quantile(rolling_vol[valid_vol_mask], 1.0 - vol_quantile))
    else:
        vol_thresh = float("inf")  # disabled / 关闭

    twap = TWAPExecutor(
        n_slices=4, favorable_reject_rate=0.60,
        slippage_adverse_bps=slip_adv,
        slippage_taker_bps=slip_tkr,
        slippage_maker_bps=slip_mkr,
    )
    equity = 1_000_000.0
    eq_curve = [equity]
    total_cost = 0.0
    cl, cs = -1, -1
    hc = 0
    rebalances = 0
    hold_periods = []
    vol_skips = 0

    for t in range(n_samples):
        if not valid_mask[t]:
            eq_curve.append(equity); hc += 1; continue
        scores = pred_matrix[t]
        returns = r1h[t]
        need = cl < 0 or (hc >= min_hold and
               (scores.argmax().item() != cl or scores.argmin().item() != cs))
        high_vol_skip = (cl >= 0 and rolling_vol[t] > vol_thresh)
        if need and (hc >= min_hold or cl < 0) and not high_vol_skip:
            nl, ns = scores.argmax().item(), scores.argmin().item()
            cost_bar = 0.0
            legs = sum([cl != nl and cl >= 0, cs != ns and cs >= 0, cl != nl, cs != ns])
            if legs > 0:
                exec_bar = t + seq_len
                ci = min(exec_bar, close_mat.size(0) - 1)
                fi_list = [min(exec_bar + k, close_mat.size(0) - 1) for k in range(4)]
                ep = close_mat[ci, nl].item()
                fu = [close_mat[f, nl].item() for f in fi_list]
                if ep > 0 and fu:
                    _, cbps, _ = twap.execute_twap("BUY", equity * 0.5 / max(legs, 1), ep, fu)
                    cost_bar = equity * 0.5 * cbps / 10000.0 * legs
            total_cost += cost_bar
            if cl >= 0:
                hold_periods.append(hc)
            cl, cs = nl, ns
            hc = 0
            rebalances += 1
        else:
            cost_bar = 0.0
            if need and high_vol_skip:
                vol_skips += 1

        pr = 0.0
        if cl >= 0:
            pr += 0.5 * returns[cl].item()
        if cs >= 0:
            pr -= 0.5 * returns[cs].item()
        pr -= cost_bar / max(equity, 1.0)
        pr = max(min(pr, 0.10), -0.10)
        equity *= (1.0 + pr)
        eq_curve.append(equity)
        hc += 1

    rets = [(eq_curve[i] / eq_curve[i - 1]) - 1 for i in range(1, len(eq_curve))]
    avg = sum(rets) / len(rets) if rets else 0.0
    std = (sum((r - avg) ** 2 for r in rets) / len(rets)) ** 0.5 if rets else 1e-9
    sharpe = (avg / max(std, 1e-9)) * (24 * 365) ** 0.5
    peak = eq_curve[0]
    max_dd = 0.0
    for e in eq_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)

    return {
        "total_return": eq_curve[-1] / eq_curve[0] - 1,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "final_equity": eq_curve[-1], "total_cost": total_cost,
        "rebalances": rebalances, "vol_skips": vol_skips,
        "vol_thresh": vol_thresh,
        "avg_hold": sum(hold_periods) / max(len(hold_periods), 1),
        **twap.stats(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folds_dir", help="path to fold ckpt dir, e.g. checkpoints/folds_v12")
    p.add_argument("--min-hold", type=int, default=96, help="min bars to hold (default 96)")
    p.add_argument("--vol-quantile", type=float, default=0.15,
                   help="top-X quantile of basket vol to skip; 0 disables filter")
    p.add_argument("--slip-adverse", type=float, default=5.0,
                   help="bps charged on adverse fills (default 5)")
    p.add_argument("--slip-taker", type=float, default=2.0,
                   help="bps charged when chasing as taker (default 2)")
    p.add_argument("--slip-maker", type=float, default=0.0,
                   help="bps charged on lucky maker fills (default 0)")
    args = p.parse_args()

    if not os.path.isdir(args.folds_dir):
        print(f"FATAL: {args.folds_dir} not found")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Folds dir: {args.folds_dir}")
    print(f"Slippage: adverse={args.slip_adverse} taker={args.slip_taker} maker={args.slip_maker} bps")
    print(f"Backtest: min_hold={args.min_hold}h, vol_quantile={args.vol_quantile}")

    SEQ_LEN = 24
    MAX_ASSETS = 20
    X, y, r1h, close_mat, syms, n_factors = build_from_parquet(
        SEQ_LEN, MAX_ASSETS, device,
    )
    n_samples = X.size(0)
    n_assets = X.size(1)

    splits = generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                                   purge_bars=SEQ_LEN, embargo_bars=48)
    print(f"\n[CPCV] {len(splits)} splits, {n_factors} factors")

    pred_matrix, valid_mask, avg_corr, fold_corrs = load_fold_predictions(
        args.folds_dir, splits, X, n_factors, n_assets, SEQ_LEN, device,
    )
    print(f"\n[CPCV] avg fold corr (from ckpts): {avg_corr:.4f}")

    summary = run_backtest(
        pred_matrix, valid_mask, r1h, close_mat,
        seq_len=SEQ_LEN, min_hold=args.min_hold, vol_quantile=args.vol_quantile,
        slip_adv=args.slip_adverse, slip_tkr=args.slip_taker, slip_mkr=args.slip_maker,
    )

    print("\n" + "=" * 65)
    print(f"  BACKTEST WITH SLIPPAGE   ({args.folds_dir})")
    print("=" * 65)
    print(f"  {'Slippage adv/tkr/mkr (bps)':.<40s} {args.slip_adverse}/{args.slip_taker}/{args.slip_maker}")
    print(f"  {'Min hold (h)':.<40s} {args.min_hold}")
    print(f"  {'Vol filter quantile':.<40s} {args.vol_quantile}")
    print(f"  {'CPCV avg corr':.<40s} {avg_corr:.4f}")
    print(f"  {'--- PERFORMANCE ---':.<40s}")
    print(f"  {'Total Return':.<40s} {summary['total_return']:>10.4%}")
    print(f"  {'Sharpe':.<40s} {summary['sharpe']:>10.4f}")
    print(f"  {'Max DD':.<40s} {summary['max_drawdown']:>10.4%}")
    print(f"  {'Final Equity':.<40s} {summary['final_equity']:>14,.2f}")
    print(f"  {'Total Cost':.<40s} {summary['total_cost']:>14,.2f}")
    print(f"  {'Rebalances':.<40s} {summary['rebalances']}")
    print(f"  {'Vol-Filter Skips':.<40s} {summary['vol_skips']}")
    print(f"  {'Avg Hold (h)':.<40s} {summary['avg_hold']:.1f}")
    print(f"  {'Maker%':.<40s} {summary.get('maker_fill_pct', 0):.1%}")
    print(f"  {'Adverse%':.<40s} {summary.get('adverse_fill_pct', 0):.1%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
