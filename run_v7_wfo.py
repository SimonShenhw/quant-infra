"""
run_v7_wfo.py — Walk-Forward Optimization with Cross-Asset GRU+Attention.

v7.0:
  - Walk-Forward: 3-month train / 1-month val / 1-month step
  - CrossAssetGRUAttention: GRU temporal + self-attention across assets
  - ListMLE + Focal dual loss with uncertainty weighting
  - TWAP execution + adverse selection + holding lock
  - All OOS windows concatenated for final global Sharpe/Calmar
"""
from __future__ import annotations

import copy
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

sys.path.insert(0, ".")

from data.async_feed import DB_PATH, load_all_from_db
from engine.twap_executor import TWAPExecutor
from model.features import build_factor_tensor
from model.cross_asset_attention import CrossAssetGRUAttention
from model.cross_sectional import listmle_loss


# ============================================================================
# Dual Loss (compact)
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
# Data loading
# ============================================================================

def load_data(
    tf: str, min_bars: int, seq_len: int, max_assets: int, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, List[str], int]:
    """Returns (X, y, close_mat, assets, bars_per_month)."""
    raw = load_all_from_db(DB_PATH, tf, min_bars)
    syms = sorted(raw.keys(), key=lambda s: len(raw[s]), reverse=True)[:max_assets]
    min_len = min(len(raw[s]) for s in syms)
    print(f"  {len(syms)} assets, {min_len} bars ({tf})")

    close_mat = torch.zeros(min_len, len(syms), device=device)
    all_factors: Dict[str, Tensor] = {}
    for j, sym in enumerate(syms):
        bars = raw[sym][:min_len]
        c = torch.tensor([b[4] for b in bars], dtype=torch.float32, device=device)
        o = torch.tensor([b[1] for b in bars], dtype=torch.float32, device=device)
        h = torch.tensor([b[2] for b in bars], dtype=torch.float32, device=device)
        l = torch.tensor([b[3] for b in bars], dtype=torch.float32, device=device)
        v = torch.tensor([b[5] for b in bars], dtype=torch.float32, device=device)
        all_factors[sym] = build_factor_tensor(o, h, l, c, v, zscore_window=48)
        close_mat[:, j] = c

    fwd_ret = torch.zeros(min_len, len(syms), device=device)
    for j, sym in enumerate(syms):
        cc = close_mat[:, j]
        fwd_ret[:-1, j] = cc[1:] / cc[:-1].clamp(min=1e-8) - 1.0

    n_samples = min_len - seq_len - 1
    X_list, y_list = [], []
    for i in range(n_samples):
        X_list.append(torch.stack([all_factors[s][i:i+seq_len] for s in syms], dim=0))
        y_list.append(fwd_ret[i + seq_len - 1])

    X = torch.stack(X_list)
    y = torch.stack(y_list)

    # bars per month depends on timeframe
    bpm = {"1h": 720, "4h": 180, "5m": 8640}.get(tf, 720)
    print(f"  X: {X.shape}, y: {y.shape}, bars_per_month≈{bpm}")
    return X, y, close_mat, syms, bpm


# ============================================================================
# Single fold training (with gradient accumulation for VRAM safety)
# ============================================================================

def train_fold(
    model: CrossAssetGRUAttention,
    loss_fn: DualLoss,
    X_tr: Tensor, y_tr: Tensor,
    X_va: Tensor, y_va: Tensor,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 3e-4,
    accum_steps: int = 1,
) -> Tuple[CrossAssetGRUAttention, float]:
    """Train one fold. Returns (model, best_rank_corr)."""
    device = X_tr.device
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X_tr.size(0)

    best_corr: float = -1.0
    best_state: Optional[Dict] = None
    patience, no_imp = 10, 0

    for ep in range(1, epochs + 1):
        model.train(); loss_fn.train()
        idx = torch.randperm(n, device=device)
        opt.zero_grad()
        step_count: int = 0

        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            try:
                scores = model(X_tr[idx[s:e]])
                loss = loss_fn(scores, y_tr[idx[s:e]]) / accum_steps
                loss.backward()
            except RuntimeError as err:
                if "out of memory" in str(err).lower():
                    torch.cuda.empty_cache()
                    # halve batch and retry
                    half = max(1, (e - s) // 2)
                    scores = model(X_tr[idx[s:s+half]])
                    loss = loss_fn(scores, y_tr[idx[s:s+half]]) / accum_steps
                    loss.backward()
                else:
                    raise
            step_count += 1
            if step_count % accum_steps == 0:
                nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step(); opt.zero_grad()

        if step_count % accum_steps != 0:
            nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step(); opt.zero_grad()
        sched.step()

        # validate
        model.eval()
        with torch.no_grad():
            vs = model(X_va)
            pr = vs.argsort(-1, descending=True).argsort(-1).float()
            tr = y_va.argsort(-1, descending=True).argsort(-1).float()
            pm, tm = pr.mean(-1, True), tr.mean(-1, True)
            cov = ((pr - pm) * (tr - tm)).sum(-1)
            corr = (cov / ((pr-pm).pow(2).sum(-1).sqrt() *
                           (tr-tm).pow(2).sum(-1).sqrt()).clamp(1e-8)).mean().item()

        if corr > best_corr:
            best_corr = corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_corr


# ============================================================================
# Walk-Forward Optimization Engine
# ============================================================================

def walk_forward_optimization(
    X: Tensor, y: Tensor, close_mat: Tensor,
    assets: List[str],
    bars_per_month: int,
    seq_len: int,
    n_factors: int,
    max_assets: int,
    device: torch.device,
    train_months: int = 3,
    val_months: int = 1,
    step_months: int = 1,
) -> Dict[str, Any]:
    """
    Industrial Walk-Forward Optimization.

    For each window:
      1. Fresh model init
      2. Train on [t - train_window, t)
      3. Validate on [t, t + val_window)
      4. Predict OOS on [t + val_window, t + val_window + step_window)
      5. Record OOS predictions and returns
      6. Slide forward by step_window
    """
    train_bars: int = train_months * bars_per_month
    step_bars: int = step_months * bars_per_month
    # val is carved from end of train window (last 20%)
    val_frac: float = 0.2
    n_total: int = X.size(0)

    print(f"\n[WFO] train={train_bars} step={step_bars} total={n_total} "
          f"(val={val_frac:.0%} of train)")

    twap = TWAPExecutor(n_slices=4, favorable_reject_rate=0.60)

    all_oos_scores: List[Tensor] = []
    all_oos_returns: List[Tensor] = []
    all_oos_close: List[Tensor] = []
    fold_corrs: List[float] = []

    close_offset: int = seq_len - 1

    fold: int = 0
    cursor: int = train_bars

    while cursor + step_bars <= n_total:
        fold += 1
        tr_start: int = max(0, cursor - train_bars)
        tr_end: int = cursor
        # split train into train/val
        val_size: int = max(int((tr_end - tr_start) * val_frac), 10)
        actual_tr_end: int = tr_end - val_size
        va_start: int = actual_tr_end
        va_end: int = tr_end
        oos_start: int = cursor
        oos_end: int = min(cursor + step_bars, n_total)

        X_tr = X[tr_start:actual_tr_end]
        y_tr = y[tr_start:actual_tr_end]
        X_va = X[va_start:va_end]
        y_va = y[va_start:va_end]
        X_oos = X[oos_start:oos_end]
        y_oos = y[oos_start:oos_end]

        # fresh model each fold (no information leakage between windows)
        model = CrossAssetGRUAttention(
            n_factors=n_factors, d_model=64, gru_layers=2,
            n_cross_heads=4, n_cross_layers=2, d_ff=128,
            dropout=0.25, seq_len=seq_len, max_assets=max_assets,
        ).to(device)
        loss_fn = DualLoss().to(device)

        t0 = time.time()
        model, best_corr = train_fold(
            model, loss_fn, X_tr, y_tr, X_va, y_va,
            epochs=60, batch_size=32, lr=3e-4,
        )
        fold_corrs.append(best_corr)

        # OOS prediction
        model.eval()
        with torch.no_grad():
            oos_scores = model(X_oos)  # (N_oos, A)

        all_oos_scores.append(oos_scores.cpu())
        all_oos_returns.append(y_oos.cpu())
        # close matrix slice for TWAP
        c_start = oos_start + close_offset
        c_end = min(c_start + oos_scores.size(0) + 10, close_mat.size(0))
        all_oos_close.append(close_mat[c_start:c_end].cpu())

        elapsed = time.time() - t0
        print(f"  Fold {fold}: train[{tr_start}-{actual_tr_end}] val[{va_start}-{va_end}] "
              f"oos[{oos_start}-{oos_end}] corr={best_corr:.4f} [{elapsed:.1f}s]")

        cursor += step_bars

    print(f"\n[WFO] {fold} folds complete. Avg corr: {sum(fold_corrs)/max(len(fold_corrs),1):.4f}")

    # --- Concatenate all OOS and run backtest ---
    if not all_oos_scores:
        return {"sharpe": 0, "calmar": 0, "total_return": 0, "max_drawdown": 0,
                "final_equity": 1e6, "total_cost": 0, "rebalances": 0,
                "avg_hold_bars": 0, "n_oos_periods": 0, "n_folds": 0,
                "avg_rank_corr": 0, "fold_corrs": [], "error": "no OOS data",
                "total_slices": 0, "maker_fill_pct": 0, "adverse_fill_pct": 0,
                "taker_fill_pct": 0, "reject_then_taker_pct": 0}

    cat_scores = torch.cat(all_oos_scores, dim=0)
    cat_returns = torch.cat(all_oos_returns, dim=0)

    # build concatenated close matrix
    cat_close_parts: List[Tensor] = []
    offset = 0
    for sc, cl in zip(all_oos_scores, all_oos_close):
        cat_close_parts.append(cl[:sc.size(0) + 5])  # +5 for TWAP lookahead
    # can't trivially concat close with gaps, so simulate per-fold then concat equity

    # --- Portfolio backtest on concatenated OOS ---
    n_oos: int = cat_scores.size(0)
    n_assets: int = cat_scores.size(1)
    top_k: int = max(1, n_assets // 20)  # 5% filter

    equity: float = 1_000_000.0
    eq_curve: List[float] = [equity]
    total_cost: float = 0.0
    current_long, current_short = -1, -1
    hold_counter: int = 0
    min_hold: int = 6
    rebalances: int = 0
    hold_periods: List[int] = []

    # process per-fold OOS chunks with their close matrices
    global_t: int = 0
    for chunk_scores, chunk_returns, chunk_close in zip(all_oos_scores, all_oos_returns, all_oos_close):
        n_chunk = chunk_scores.size(0)
        for t in range(n_chunk):
            scores = chunk_scores[t]
            returns = chunk_returns[t]

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

                # TWAP execution cost
                cost_bar: float = 0.0
                future_range = list(range(t + 1, min(t + 5, n_chunk)))
                legs_changed = 0
                if current_long != nl:
                    legs_changed += 1
                    if current_long >= 0: legs_changed += 1
                if current_short != ns:
                    legs_changed += 1
                    if current_short >= 0: legs_changed += 1

                if future_range and legs_changed > 0 and t + 1 < chunk_close.size(0):
                    futures = [chunk_close[min(i, chunk_close.size(0)-1), nl].item()
                               for i in future_range]
                    entry_p = chunk_close[min(t, chunk_close.size(0)-1), nl].item()
                    if entry_p > 0 and futures:
                        _, cbps, _ = twap.execute_twap("BUY", equity * 0.5 / max(legs_changed, 1),
                                                        entry_p, futures)
                        cost_bar += equity * 0.5 * cbps / 10000.0 * legs_changed

                total_cost += cost_bar
                if current_long >= 0:
                    hold_periods.append(hold_counter)
                current_long, current_short = nl, ns
                hold_counter = 0
                rebalances += 1
            else:
                cost_bar = 0.0

            port_ret = 0.0
            if current_long >= 0:
                port_ret += 0.5 * returns[current_long].item()
            if current_short >= 0:
                port_ret -= 0.5 * returns[current_short].item()
            port_ret -= cost_bar / max(equity, 1.0)

            equity *= (1.0 + port_ret)
            eq_curve.append(equity)
            hold_counter += 1
            global_t += 1

    # --- Metrics ---
    rets = [(eq_curve[i]/eq_curve[i-1]) - 1 for i in range(1, len(eq_curve))]
    avg = sum(rets) / len(rets) if rets else 0
    std = (sum((r-avg)**2 for r in rets)/len(rets))**0.5 if rets else 1e-9
    # annualise: 1h bars → 24 bars/day × 365 days
    ann_factor = (24 * 365) ** 0.5
    sharpe = (avg / max(std, 1e-9)) * ann_factor

    peak = eq_curve[0]
    max_dd = 0.0
    for e in eq_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)

    total_ret = eq_curve[-1] / eq_curve[0] - 1
    ann_ret = total_ret * (365 * 6 / max(len(rets), 1))  # approximate
    calmar = ann_ret / max(max_dd, 1e-9)

    avg_hold = sum(hold_periods) / max(len(hold_periods), 1)

    return {
        "total_return": total_ret,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "final_equity": eq_curve[-1],
        "total_cost": total_cost,
        "rebalances": rebalances,
        "avg_hold_bars": avg_hold,
        "n_oos_periods": len(rets),
        "n_folds": fold,
        "avg_rank_corr": sum(fold_corrs) / max(len(fold_corrs), 1),
        "fold_corrs": fold_corrs,
        **twap.stats(),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("  QUANT INFRA v7.0 — Walk-Forward + Cross-Asset Attention")
    print("  27 Assets | 4h Bars | GRU+Attn | WFO 3m/1m/1m")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    SEQ_LEN = 24  # 24 × 1h = 1 day lookback
    MAX_ASSETS = 20

    X, y, close_mat, assets, bpm = load_data(
        "1h", min_bars=500, seq_len=SEQ_LEN,
        max_assets=MAX_ASSETS, device=device,
    )

    # 1h: 720 bars/month, use 1-week windows for more folds
    # train=1 week (168 bars), step=1 week, val=20% of train
    week_bars: int = 168  # 24*7
    summary = walk_forward_optimization(
        X, y, close_mat, assets,
        bars_per_month=week_bars,  # treat "month" as "week" for window sizing
        seq_len=SEQ_LEN,
        n_factors=X.size(3),
        max_assets=MAX_ASSETS,
        device=device,
        train_months=2,   # 2 "weeks" = 336 bars train
        val_months=0,     # embedded in train
        step_months=1,    # 1 "week" = 168 bars OOS
    )

    # --- Report ---
    print("\n" + "=" * 65)
    print("  v7.0 WALK-FORWARD BACKTEST REPORT")
    print("=" * 65)
    print(f"  {'Assets':.<40s} {len(assets)}")
    print(f"  {'Timeframe':.<40s} 4h")
    print(f"  {'WFO Folds':.<40s} {summary['n_folds']}")
    print(f"  {'OOS Periods (total)':.<40s} {summary['n_oos_periods']}")
    print(f"  {'Rebalances':.<40s} {summary['rebalances']}")
    print(f"  {'Avg Holding (bars × 4h)':.<40s} {summary['avg_hold_bars']:.1f}")
    print(f"  {'Avg Rank Correlation':.<40s} {summary['avg_rank_corr']:.4f}")
    print(f"  {'Per-fold Corrs':.<40s} {[f'{c:.3f}' for c in summary['fold_corrs']]}")
    print(f"  {'--- PERFORMANCE ---':.<40s}")
    print(f"  {'Total Return':.<40s} {summary['total_return']:>10.4%}")
    print(f"  {'Sharpe Ratio':.<40s} {summary['sharpe']:>10.4f}")
    print(f"  {'Calmar Ratio':.<40s} {summary['calmar']:>10.4f}")
    print(f"  {'Max Drawdown':.<40s} {summary['max_drawdown']:>10.4%}")
    print(f"  {'Final Equity':.<40s} {summary['final_equity']:>14.2f}")
    print(f"  {'Total Cost':.<40s} {summary['total_cost']:>14.2f}")
    print(f"  {'--- TWAP EXECUTION ---':.<40s}")
    ts = summary.get('total_slices', 0)
    print(f"  {'TWAP Slices':.<40s} {ts}")
    if ts > 0:
        print(f"  {'Maker%':.<40s} {summary['maker_fill_pct']:.1%}")
        print(f"  {'Adverse%':.<40s} {summary['adverse_fill_pct']:.1%}")
        print(f"  {'Taker%':.<40s} {summary['taker_fill_pct']:.1%}")
    print("=" * 65)
    print("\n[DONE] v7.0 complete.")


if __name__ == "__main__":
    main()
