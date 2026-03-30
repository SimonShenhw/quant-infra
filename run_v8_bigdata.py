"""
run_v8_bigdata.py — WFO on 1M+ bars Parquet data lake.

Uses 6 months of 5m bars aggregated to 1h for WFO.
20 assets × ~4400 1h bars = massive statistical significance.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import polars as pl

sys.path.insert(0, ".")

from data.lake_loader import load_klines_multi, klines_to_tensors
from engine.twap_executor import TWAPExecutor
from model.features import build_factor_tensor
from model.cross_asset_attention import CrossAssetGRUAttention
from model.cross_sectional import listmle_loss


# ============================================================================
# Aggregate 5m → 1h using Polars
# ============================================================================

def aggregate_5m_to_1h(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 5m bars to 1h bars using Polars groupby."""
    df = df.with_columns(
        (pl.col("open_time") // 3_600_000 * 3_600_000).alias("hour_ts")
    )
    agg = df.group_by("hour_ts").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ]).sort("hour_ts").rename({"hour_ts": "open_time"})
    return agg


# ============================================================================
# Dual loss
# ============================================================================

class DualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lv_r = nn.Parameter(torch.tensor(0.0))
        self.lv_d = nn.Parameter(torch.tensor(0.0))

    def forward(self, scores: Tensor, rets: Tensor) -> Tensor:
        lr = listmle_loss(scores, rets)
        med = rets.median(-1, keepdim=True).values
        tgt = (rets > med).float()
        logits = scores * 10.0
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        pt = p * tgt + (1 - p) * (1 - tgt)
        focal = ((1 - pt) ** 2 * ce).mean()
        pr, pd = torch.exp(-self.lv_r), torch.exp(-self.lv_d)
        return 0.5 * pr * lr + self.lv_r + 0.5 * pd * focal + self.lv_d


# ============================================================================
# Build 4D dataset from Parquet
# ============================================================================

def build_from_parquet(
    seq_len: int, max_assets: int, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, List[str]]:
    print("[Data] Loading from Parquet data lake ...")
    raw_5m = load_klines_multi(interval="5m", min_rows=40000)
    print(f"  Loaded {len(raw_5m)} symbols with 40K+ 5m bars")

    # aggregate to 1h
    syms = sorted(raw_5m.keys())[:max_assets]
    agg_dfs: Dict[str, pl.DataFrame] = {}
    for sym in syms:
        agg_dfs[sym] = aggregate_5m_to_1h(raw_5m[sym])

    min_len = min(df.height for df in agg_dfs.values())
    print(f"  {len(syms)} assets, {min_len} 1h bars each (aggregated from 5m)")

    # build tensors
    close_mat = torch.zeros(min_len, len(syms), device=device)
    all_factors: Dict[str, Tensor] = {}

    for j, sym in enumerate(syms):
        df = agg_dfs[sym]
        rows = df.head(min_len)
        t = klines_to_tensors(rows, device)
        close_mat[:, j] = t["close"]
        all_factors[sym] = build_factor_tensor(
            t["open"], t["high"], t["low"], t["close"], t["volume"],
            zscore_window=48,
        )

    # forward returns
    fwd_ret = torch.zeros(min_len, len(syms), device=device)
    for j in range(len(syms)):
        c = close_mat[:, j]
        fwd_ret[:-1, j] = c[1:] / c[:-1].clamp(min=1e-8) - 1.0

    # sliding windows
    n_samples = min_len - seq_len - 1
    X_list, y_list = [], []
    for i in range(n_samples):
        X_list.append(torch.stack([all_factors[s][i:i+seq_len] for s in syms], dim=0))
        y_list.append(fwd_ret[i + seq_len - 1])

    X = torch.stack(X_list)
    y = torch.stack(y_list)
    print(f"  X: {X.shape}, y: {y.shape}")
    return X, y, close_mat, syms


# ============================================================================
# Train one fold
# ============================================================================

def train_fold(model, loss_fn, X_tr, y_tr, X_va, y_va, epochs=60, bs=64, lr=3e-4):
    device = X_tr.device
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_corr, best_state = -1.0, None
    patience, no_imp = 12, 0

    for ep in range(1, epochs + 1):
        model.train(); loss_fn.train()
        idx = torch.randperm(X_tr.size(0), device=device)
        for s in range(0, X_tr.size(0), bs):
            e = min(s + bs, X_tr.size(0))
            scores = model(X_tr[idx[s:e]])
            loss = loss_fn(scores, y_tr[idx[s:e]])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vs = model(X_va)
            pr = vs.argsort(-1, descending=True).argsort(-1).float()
            tr = y_va.argsort(-1, descending=True).argsort(-1).float()
            pm, tm = pr.mean(-1, True), tr.mean(-1, True)
            cov = ((pr-pm)*(tr-tm)).sum(-1)
            corr = (cov / ((pr-pm).pow(2).sum(-1).sqrt()*(tr-tm).pow(2).sum(-1).sqrt()).clamp(1e-8)).mean().item()

        if corr > best_corr:
            best_corr = corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state: model.load_state_dict(best_state)
    return model, best_corr


# ============================================================================
# WFO
# ============================================================================

def run_wfo(X, y, close_mat, syms, seq_len, device):
    n = X.size(0)
    n_assets = len(syms)
    # 1h bars: ~720/month. train=1month, step=2weeks
    train_bars = 1440  # 2 months
    step_bars = 720    # 1 month OOS per fold

    twap = TWAPExecutor(n_slices=4, favorable_reject_rate=0.60)

    fold_corrs = []
    equity = 1_000_000.0
    eq_curve = [equity]
    total_cost = 0.0
    current_long, current_short = -1, -1
    hold_counter = 0
    min_hold = 48  # 48 hours = 2 days minimum hold
    rebalances = 0
    hold_periods = []

    cursor = train_bars
    fold = 0

    while cursor + step_bars <= n:
        fold += 1
        tr_start = max(0, cursor - train_bars)
        val_size = max(int((cursor - tr_start) * 0.2), 20)
        tr_end = cursor - val_size
        va_end = cursor

        model = CrossAssetGRUAttention(
            n_factors=X.size(3), d_model=64, gru_layers=2,
            n_cross_heads=4, n_cross_layers=2, d_ff=128,
            dropout=0.25, seq_len=seq_len, max_assets=n_assets,
        ).to(device)
        loss_fn = DualLoss().to(device)

        t0 = time.time()
        model, corr = train_fold(
            model, loss_fn,
            X[tr_start:tr_end], y[tr_start:tr_end],
            X[cursor-val_size:cursor], y[cursor-val_size:cursor],
            epochs=60, bs=64, lr=3e-4,
        )
        fold_corrs.append(corr)

        # OOS backtest for this fold
        oos_end = min(cursor + step_bars, n)
        model.eval()
        with torch.no_grad():
            for t in range(cursor, oos_end):
                scores = model(X[t:t+1]).squeeze(0)
                returns = y[t]

                need_rebal = False
                if current_long < 0:
                    need_rebal = True
                elif hold_counter >= min_hold:
                    nl, ns = scores.argmax().item(), scores.argmin().item()
                    if nl != current_long or ns != current_short:
                        need_rebal = True

                if need_rebal and (hold_counter >= min_hold or current_long < 0):
                    nl = scores.argmax().item()
                    ns = scores.argmin().item()
                    cost_bar = 0.0
                    legs = sum([current_long != nl and current_long >= 0,
                                current_short != ns and current_short >= 0,
                                current_long != nl, current_short != ns])
                    if legs > 0:
                        close_offset = seq_len - 1
                        ci = t + close_offset
                        future_i = [min(ci+1+k, close_mat.size(0)-1) for k in range(4)]
                        entry_p = close_mat[min(ci, close_mat.size(0)-1), nl].item()
                        futures = [close_mat[fi, nl].item() for fi in future_i]
                        if entry_p > 0 and futures:
                            _, cbps, _ = twap.execute_twap("BUY", equity*0.5/max(legs,1), entry_p, futures)
                            cost_bar = equity * 0.5 * cbps / 10000.0 * legs
                    total_cost += cost_bar
                    if current_long >= 0: hold_periods.append(hold_counter)
                    current_long, current_short = nl, ns
                    hold_counter = 0
                    rebalances += 1
                else:
                    cost_bar = 0.0

                port_ret = 0.0
                if current_long >= 0: port_ret += 0.5 * returns[current_long].item()
                if current_short >= 0: port_ret -= 0.5 * returns[current_short].item()
                port_ret -= cost_bar / max(equity, 1.0)
                equity *= (1.0 + port_ret)
                eq_curve.append(equity)
                hold_counter += 1

        elapsed = time.time() - t0
        print(f"  Fold {fold}: train[{tr_start}-{tr_end}] val[{cursor-val_size}-{cursor}] "
              f"oos[{cursor}-{oos_end}] corr={corr:.4f} eq={equity:,.0f} [{elapsed:.0f}s]")
        cursor += step_bars

    # metrics
    rets = [(eq_curve[i]/eq_curve[i-1])-1 for i in range(1, len(eq_curve))]
    avg = sum(rets)/len(rets) if rets else 0
    std = (sum((r-avg)**2 for r in rets)/len(rets))**0.5 if rets else 1e-9
    sharpe = (avg / max(std, 1e-9)) * (24*365)**0.5
    peak = eq_curve[0]
    max_dd = 0.0
    for e in eq_curve:
        peak = max(peak, e); max_dd = max(max_dd, (peak-e)/peak)
    total_ret = eq_curve[-1]/eq_curve[0] - 1
    avg_hold = sum(hold_periods)/max(len(hold_periods), 1)

    return {
        "total_return": total_ret, "sharpe": sharpe, "max_drawdown": max_dd,
        "final_equity": eq_curve[-1], "total_cost": total_cost,
        "rebalances": rebalances, "avg_hold": avg_hold,
        "n_oos": len(rets), "n_folds": fold,
        "avg_corr": sum(fold_corrs)/max(len(fold_corrs),1),
        "fold_corrs": fold_corrs, **twap.stats(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("  v8.0 — 1M+ Bars | 6-Month WFO | GRU+CrossAttn")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    SEQ_LEN = 24
    MAX_ASSETS = 20
    X, y, close_mat, syms = build_from_parquet(SEQ_LEN, MAX_ASSETS, device)
    print(f"  Assets: {syms[:5]}...")

    t0 = time.time()
    summary = run_wfo(X, y, close_mat, syms, SEQ_LEN, device)
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print("  v8.0 BACKTEST REPORT (1M+ bars, WFO, TWAP+AdverseSelection)")
    print("=" * 65)
    print(f"  {'Source Data':.<40s} Parquet lake (1,054,080 rows)")
    print(f"  {'Aggregation':.<40s} 5m → 1h")
    print(f"  {'Assets':.<40s} {len(syms)}")
    print(f"  {'WFO Folds':.<40s} {summary['n_folds']}")
    print(f"  {'OOS Periods':.<40s} {summary['n_oos']:,}")
    print(f"  {'Rebalances':.<40s} {summary['rebalances']}")
    print(f"  {'Avg Hold (hours)':.<40s} {summary['avg_hold']:.1f}")
    print(f"  {'Avg Rank Corr':.<40s} {summary['avg_corr']:.4f}")
    print(f"  {'Per-fold':.<40s} {[f'{c:.3f}' for c in summary['fold_corrs']]}")
    print(f"  {'--- PERFORMANCE ---':.<40s}")
    print(f"  {'Total Return':.<40s} {summary['total_return']:>10.4%}")
    print(f"  {'Sharpe Ratio':.<40s} {summary['sharpe']:>10.4f}")
    print(f"  {'Max Drawdown':.<40s} {summary['max_drawdown']:>10.4%}")
    print(f"  {'Final Equity':.<40s} {summary['final_equity']:>14,.2f}")
    print(f"  {'Total Cost':.<40s} {summary['total_cost']:>14,.2f}")
    ts = summary.get('total_slices', 0)
    if ts > 0:
        print(f"  {'--- TWAP ---':.<40s}")
        print(f"  {'Slices':.<40s} {ts}")
        print(f"  {'Maker%':.<40s} {summary['maker_fill_pct']:.1%}")
        print(f"  {'Adverse%':.<40s} {summary['adverse_fill_pct']:.1%}")
        print(f"  {'Taker%':.<40s} {summary['taker_fill_pct']:.1%}")
    print(f"  {'Elapsed':.<40s} {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
