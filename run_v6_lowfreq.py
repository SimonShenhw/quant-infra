"""
run_v6_lowfreq.py — Low-Frequency Cross-Sectional Pipeline.

v6.0 core changes:
  1. 1h bars (not 5m) — 12x lower frequency
  2. Holding period lock: min 6 bars (6 hours) before rebalance
  3. TWAP execution: 4-bar split instead of all-in limit order
  4. Top/Bottom 5% filter: 20 assets → only 1 long + 1 short
"""
from __future__ import annotations

import random
import sqlite3
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
from model.cross_sectional import CrossSectionalTransformer, listmle_loss


# ============================================================================
# Dual-Objective Loss (same as v5.0)
# ============================================================================

class DualLoss(nn.Module):
    def __init__(self, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.log_var_rank = nn.Parameter(torch.tensor(0.0))
        self.log_var_dir = nn.Parameter(torch.tensor(0.0))
        self.gamma = focal_gamma

    def forward(self, scores: Tensor, returns: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        loss_rank = listmle_loss(scores, returns)
        median = returns.median(dim=-1, keepdim=True).values
        targets = (returns > median).float()
        logits = scores * 10.0
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal = ((1.0 - p_t) ** self.gamma * ce).mean()

        pr = torch.exp(-self.log_var_rank)
        pd = torch.exp(-self.log_var_dir)
        total = 0.5 * pr * loss_rank + self.log_var_rank + 0.5 * pd * focal + self.log_var_dir
        return total, {"l_rank": loss_rank.item(), "l_focal": focal.item(),
                        "w_rank": pr.item(), "w_dir": pd.item()}


# ============================================================================
# Build 4D dataset from 1h bars
# ============================================================================

def build_dataset(
    db_path: str, timeframe: str, min_bars: int, seq_len: int,
    max_assets: int, device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, List[str]]:
    """Returns (X, y, close_matrix, asset_names)."""
    raw = load_all_from_db(db_path, timeframe, min_bars)
    syms = sorted(raw.keys(), key=lambda s: len(raw[s]), reverse=True)[:max_assets]
    min_len = min(len(raw[s]) for s in syms)
    print(f"  {len(syms)} assets, {min_len} bars each (timeframe={timeframe})")

    all_factors: Dict[str, Tensor] = {}
    close_mat = torch.zeros(min_len, len(syms), device=device)

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
        c = close_mat[:, j]
        fwd_ret[:-1, j] = c[1:] / c[:-1].clamp(min=1e-8) - 1.0

    n_samples = min_len - seq_len - 1
    X_list, y_list = [], []
    for i in range(n_samples):
        sample = torch.stack([all_factors[s][i:i+seq_len] for s in syms], dim=0)
        X_list.append(sample)
        y_list.append(fwd_ret[i + seq_len - 1])

    X = torch.stack(X_list)
    y = torch.stack(y_list)
    print(f"  X: {X.shape}, y: {y.shape}")
    return X, y, close_mat, syms


# ============================================================================
# Training
# ============================================================================

def train(model, loss_fn, X_tr, y_tr, X_va, y_va, epochs, bs, lr, device):
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_corr, best_state = -1.0, None
    patience, no_imp = 15, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train(); loss_fn.train()
        idx = torch.randperm(X_tr.size(0), device=device)
        el, nb = 0.0, 0
        for s in range(0, X_tr.size(0), bs):
            e = min(s + bs, X_tr.size(0))
            sc = model(X_tr[idx[s:e]])
            loss, _ = loss_fn(sc, y_tr[idx[s:e]])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0); opt.step()
            el += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vs = model(X_va)
            _, diag = loss_fn(vs, y_va)
            pr = vs.argsort(-1, descending=True).argsort(-1).float()
            tr = y_va.argsort(-1, descending=True).argsort(-1).float()
            pm, tm = pr.mean(-1, True), tr.mean(-1, True)
            cov = ((pr - pm) * (tr - tm)).sum(-1)
            corr = (cov / ((pr-pm).pow(2).sum(-1).sqrt() * (tr-tm).pow(2).sum(-1).sqrt()).clamp(1e-8)).mean().item()

        if corr > best_corr:
            best_corr = corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}: loss={el/nb:.4f} corr={corr:.4f} best={best_corr:.4f} "
                  f"w_r={diag['w_rank']:.1f} w_d={diag['w_dir']:.1f} [{time.time()-t0:.0f}s]")
        if no_imp >= patience:
            print(f"  Early stop at ep {ep}"); break

    if best_state: model.load_state_dict(best_state)
    print(f"  Done in {time.time()-t0:.0f}s, best_corr={best_corr:.4f}")
    return model


# ============================================================================
# Backtest: Holding Lock + TWAP + Top/Bottom 5%
# ============================================================================

def backtest_v6(
    model: CrossSectionalTransformer,
    X_oos: Tensor,
    y_oos: Tensor,
    close_mat: Tensor,  # aligned with X_oos
    assets: List[str],
    min_hold_bars: int = 6,
    twap_slices: int = 4,
    initial_cash: float = 1_000_000.0,
) -> Dict[str, Any]:
    """
    Low-frequency backtest with:
      - Top 1 / Bottom 1 (5% each of 20 assets)
      - Holding period lock: min_hold_bars
      - TWAP execution over twap_slices bars
    """
    twap = TWAPExecutor(
        n_slices=twap_slices,
        favorable_reject_rate=0.60,  # lower than v5 because TWAP averages
        taker_fee_bps=4.0,
        maker_fee_bps=1.0,
    )

    model.eval()
    n = X_oos.size(0)
    n_assets = X_oos.size(1)
    top_k: int = max(1, n_assets // 20)  # 5% → 1 asset for 20 assets

    equity: float = initial_cash
    eq_curve: List[float] = [equity]
    total_cost: float = 0.0

    # position state
    current_long: int = -1   # index of long asset
    current_short: int = -1  # index of short asset
    hold_counter: int = 0    # bars since last rebalance
    rebalance_count: int = 0
    hold_periods: List[int] = []

    with torch.no_grad():
        for t in range(n - twap_slices):
            scores = model(X_oos[t:t+1]).squeeze(0)  # (A,)
            returns = y_oos[t]  # (A,) actual returns this bar

            # --- Holding period lock ---
            need_rebalance: bool = False
            if current_long < 0 and current_short < 0:
                need_rebalance = True  # first entry
            elif hold_counter >= min_hold_bars:
                # check if top/bottom changed
                new_long = scores.argmax().item()
                new_short = scores.argmin().item()
                if new_long != current_long or new_short != current_short:
                    need_rebalance = True

            if need_rebalance and hold_counter >= min_hold_bars or (current_long < 0):
                new_long = scores.argmax().item()
                new_short = scores.argmin().item()

                # --- TWAP execution for position changes ---
                cost_this_bar: float = 0.0
                future_closes_range = list(range(t + 1, min(t + 1 + twap_slices, n)))

                # close old long if changed
                if current_long >= 0 and current_long != new_long:
                    futures = [close_mat[i, current_long].item() for i in future_closes_range]
                    entry_p = close_mat[t, current_long].item()
                    _, cost_bps, _ = twap.execute_twap("SELL", equity * 0.5, entry_p, futures)
                    cost_this_bar += equity * 0.5 * cost_bps / 10000.0

                # close old short if changed
                if current_short >= 0 and current_short != new_short:
                    futures = [close_mat[i, current_short].item() for i in future_closes_range]
                    entry_p = close_mat[t, current_short].item()
                    _, cost_bps, _ = twap.execute_twap("BUY", equity * 0.5, entry_p, futures)
                    cost_this_bar += equity * 0.5 * cost_bps / 10000.0

                # open new long
                if current_long != new_long:
                    futures = [close_mat[i, new_long].item() for i in future_closes_range]
                    entry_p = close_mat[t, new_long].item()
                    _, cost_bps, _ = twap.execute_twap("BUY", equity * 0.5, entry_p, futures)
                    cost_this_bar += equity * 0.5 * cost_bps / 10000.0

                # open new short
                if current_short != new_short:
                    futures = [close_mat[i, new_short].item() for i in future_closes_range]
                    entry_p = close_mat[t, new_short].item()
                    _, cost_bps, _ = twap.execute_twap("SELL", equity * 0.5, entry_p, futures)
                    cost_this_bar += equity * 0.5 * cost_bps / 10000.0

                total_cost += cost_this_bar

                if current_long >= 0:
                    hold_periods.append(hold_counter)

                current_long = new_long
                current_short = new_short
                hold_counter = 0
                rebalance_count += 1
            else:
                cost_this_bar = 0.0

            # --- Portfolio return ---
            port_ret: float = 0.0
            if current_long >= 0:
                port_ret += 0.5 * returns[current_long].item()
            if current_short >= 0:
                port_ret -= 0.5 * returns[current_short].item()
            port_ret -= cost_this_bar / max(equity, 1.0)

            equity *= (1.0 + port_ret)
            eq_curve.append(equity)
            hold_counter += 1

    # metrics
    rets = [(eq_curve[i]/eq_curve[i-1]) - 1 for i in range(1, len(eq_curve))]
    avg = sum(rets) / len(rets) if rets else 0
    std = (sum((r-avg)**2 for r in rets)/len(rets))**0.5 if rets else 1e-9
    # annualise: 1h bars, ~24 bars/day, ~252 trading days
    sharpe = (avg / max(std, 1e-9)) * (252 * 24)**0.5

    peak = eq_curve[0]
    max_dd = 0.0
    for e in eq_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)

    avg_hold = sum(hold_periods) / max(len(hold_periods), 1)

    return {
        "total_return": eq_curve[-1] / eq_curve[0] - 1,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": eq_curve[-1],
        "total_cost": total_cost,
        "rebalance_count": rebalance_count,
        "avg_hold_bars": avg_hold,
        "n_periods": len(rets),
        **twap.stats(),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("  QUANT INFRA v6.0 — Low-Frequency + TWAP + 5% Filter")
    print("  27 Assets | 1h Bars | Min 6h Hold | Top/Bot 1 Asset")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    SEQ_LEN = 24  # 24 hours lookback
    MAX_ASSETS = 20

    X, y, close_mat, assets = build_dataset(
        DB_PATH, "1h", min_bars=500, seq_len=SEQ_LEN,
        max_assets=MAX_ASSETS, device=device,
    )

    # align close_mat with X indices
    oos_close_start = int(X.size(0) * 0.8) + SEQ_LEN - 1

    n = X.size(0)
    tr_end = int(n * 0.75)
    va_end = int(n * 0.87)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]
    X_te, y_te = X[va_end:], y[va_end:]

    close_te = close_mat[oos_close_start:oos_close_start + X_te.size(0) + 10, :]

    print(f"  Split: train={X_tr.size(0)} val={X_va.size(0)} test={X_te.size(0)}")

    # Smaller model to reduce overfitting on limited data
    model = CrossSectionalTransformer(
        n_factors=X.size(3), d_model=64, n_heads=4,
        n_temporal_layers=2, n_cross_layers=1, d_ff=128,
        dropout=0.30, seq_len=SEQ_LEN, max_assets=MAX_ASSETS,
    ).to(device)
    loss_fn = DualLoss().to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"\n[Training] ...")
    model = train(model, loss_fn, X_tr, y_tr, X_va, y_va,
                  epochs=100, bs=32, lr=3e-4, device=device)

    print(f"\n[Backtest] OOS with TWAP + Holding Lock + 5% Filter ...")
    summary = backtest_v6(
        model, X_te, y_te, close_te, assets,
        min_hold_bars=6, twap_slices=4,
    )

    print("\n" + "=" * 65)
    print("  v6.0 BACKTEST REPORT")
    print("=" * 65)
    print(f"  {'Assets':.<40s} {len(assets)}")
    print(f"  {'Timeframe':.<40s} 1h")
    print(f"  {'OOS Periods':.<40s} {summary['n_periods']}")
    print(f"  {'Rebalance Count':.<40s} {summary['rebalance_count']}")
    print(f"  {'Avg Holding (bars=hours)':.<40s} {summary['avg_hold_bars']:.1f}")
    print(f"  {'--- PERFORMANCE ---':.<40s}")
    print(f"  {'Total Return':.<40s} {summary['total_return']:>10.4%}")
    print(f"  {'Sharpe Ratio':.<40s} {summary['sharpe']:>10.4f}")
    print(f"  {'Max Drawdown':.<40s} {summary['max_drawdown']:>10.4%}")
    print(f"  {'Final Equity':.<40s} {summary['final_equity']:>14.2f}")
    print(f"  {'Total Transaction Cost':.<40s} {summary['total_cost']:>14.2f}")
    print(f"  {'--- TWAP EXECUTION ---':.<40s}")
    print(f"  {'TWAP Total Slices':.<40s} {summary['total_slices']}")
    print(f"  {'Maker Fills':.<40s} {summary['maker_fill_pct']:.1%}")
    print(f"  {'Adverse Fills':.<40s} {summary['adverse_fill_pct']:.1%}")
    print(f"  {'Taker (rejected→market)':.<40s} {summary['taker_fill_pct']:.1%}")
    print("=" * 65)

    # comparison line
    cost_pct = summary['total_cost'] / 1_000_000.0
    print(f"\n  Cost as % of capital: {cost_pct:.2%}")
    print(f"  Cost per rebalance:   ${summary['total_cost']/max(summary['rebalance_count'],1):.2f}")

    print("\n[DONE] v6.0 complete.")


if __name__ == "__main__":
    main()
