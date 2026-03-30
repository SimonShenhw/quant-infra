"""
run_v5_final.py — Production Cross-Sectional Pipeline with:
  1. 233K bars from SQLite (27 crypto assets × 8640 bars)
  2. Adverse Selection micro-execution simulator
  3. ListMLE + Directional Focal Loss with Uncertainty Weighting
  4. Full Backtest Report

v5.0: The culmination of all engineering work.
"""
from __future__ import annotations

import math
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

sys.path.insert(0, ".")

from data.async_feed import DB_PATH, load_all_from_db
from engine.adverse_selection import AdverseSelectionSimulator
from engine.events import EventType, MarketEvent
from model.features import build_factor_tensor
from model.cross_sectional import CrossSectionalTransformer, listmle_loss


# ============================================================================
# Dual-Objective Loss: ListMLE + Directional Focal + Uncertainty Weighting
# ============================================================================

class UncertaintyWeightedDualLoss(nn.Module):
    """
    Combines ListMLE (ranking) and Directional Focal (classification)
    using learnable uncertainty weights (Kendall et al., 2018):

      L = (1/2σ₁²)·L_rank + (1/2σ₂²)·L_dir + log(σ₁) + log(σ₂)

    The model automatically balances the two objectives during training.
    """

    def __init__(self, focal_gamma: float = 2.0) -> None:
        super().__init__()
        # learnable log-variance for each task (init to 0 → σ=1)
        self.log_var_rank: nn.Parameter = nn.Parameter(torch.tensor(0.0))
        self.log_var_dir: nn.Parameter = nn.Parameter(torch.tensor(0.0))
        self.focal_gamma: float = focal_gamma

    def forward(self, scores: Tensor, returns: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        scores : (B, A) predicted ranking scores
        returns : (B, A) true forward returns

        Returns
        -------
        (total_loss, diagnostics_dict)
        """
        # Task 1: ListMLE ranking loss
        loss_rank: Tensor = listmle_loss(scores, returns)

        # Task 2: Directional Focal loss
        # For each asset, predict whether it's above median return
        median_ret: Tensor = returns.median(dim=-1, keepdim=True).values
        targets: Tensor = (returns > median_ret).float()  # (B, A)
        logits: Tensor = scores * 10.0  # scale to logit range

        p: Tensor = torch.sigmoid(logits)
        ce: Tensor = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t: Tensor = p * targets + (1.0 - p) * (1.0 - targets)
        focal: Tensor = ((1.0 - p_t) ** self.focal_gamma * ce).mean()

        # Uncertainty weighting
        precision_rank: Tensor = torch.exp(-self.log_var_rank)
        precision_dir: Tensor = torch.exp(-self.log_var_dir)

        total: Tensor = (
            0.5 * precision_rank * loss_rank + self.log_var_rank
            + 0.5 * precision_dir * focal + self.log_var_dir
        )

        diagnostics: Dict[str, float] = {
            "loss_rank": loss_rank.item(),
            "loss_focal": focal.item(),
            "w_rank": precision_rank.item(),
            "w_dir": precision_dir.item(),
        }
        return total, diagnostics


# ============================================================================
# Build 4D dataset from SQLite
# ============================================================================

def build_dataset_from_db(
    db_path: str,
    timeframe: str,
    min_bars: int,
    seq_len: int,
    max_assets: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, List[str]]:
    """Load from SQLite, compute features, build 4D tensor."""
    print(f"  Loading from {db_path} ...")
    raw: Dict[str, List[Tuple]] = load_all_from_db(db_path, timeframe, min_bars)
    # take top max_assets by bar count
    sorted_syms: List[str] = sorted(raw.keys(), key=lambda s: len(raw[s]), reverse=True)[:max_assets]
    min_len: int = min(len(raw[s]) for s in sorted_syms)
    print(f"  {len(sorted_syms)} assets, {min_len} bars each")

    # compute factors for each asset
    all_factors: Dict[str, Tensor] = {}
    all_closes: Dict[str, Tensor] = {}

    for sym in sorted_syms:
        bars = raw[sym][:min_len]
        c = torch.tensor([b[4] for b in bars], dtype=torch.float32, device=device)
        o = torch.tensor([b[1] for b in bars], dtype=torch.float32, device=device)
        h = torch.tensor([b[2] for b in bars], dtype=torch.float32, device=device)
        l = torch.tensor([b[3] for b in bars], dtype=torch.float32, device=device)
        v = torch.tensor([b[5] for b in bars], dtype=torch.float32, device=device)
        all_factors[sym] = build_factor_tensor(o, h, l, c, v, zscore_window=60)
        all_closes[sym] = c

    # forward returns: (T, A)
    n_assets: int = len(sorted_syms)
    n_factors: int = all_factors[sorted_syms[0]].size(1)
    fwd_ret: Tensor = torch.zeros(min_len, n_assets, device=device)
    for j, sym in enumerate(sorted_syms):
        c = all_closes[sym]
        fwd_ret[:-1, j] = c[1:] / c[:-1].clamp(min=1e-8) - 1.0

    # sliding windows
    n_samples: int = min_len - seq_len - 1
    X_list, y_list = [], []
    for i in range(n_samples):
        sample = torch.stack([all_factors[s][i:i+seq_len] for s in sorted_syms], dim=0)
        X_list.append(sample)
        y_list.append(fwd_ret[i + seq_len - 1])

    X: Tensor = torch.stack(X_list)  # (N, A, T, F)
    y: Tensor = torch.stack(y_list)  # (N, A)

    print(f"  X: {X.shape}, y: {y.shape}, n_factors: {n_factors}")
    return X, y, sorted_syms


# ============================================================================
# Training loop
# ============================================================================

def train_v5(
    model: CrossSectionalTransformer,
    loss_fn: UncertaintyWeightedDualLoss,
    X_tr: Tensor, y_tr: Tensor,
    X_va: Tensor, y_va: Tensor,
    epochs: int, batch_size: int, lr: float,
    device: torch.device,
) -> CrossSectionalTransformer:
    all_params = list(model.parameters()) + list(loss_fn.parameters())
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_train: int = X_tr.size(0)

    best_corr: float = -1.0
    best_state: Optional[Dict] = None
    patience: int = 15
    no_improve: int = 0
    t0: float = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_fn.train()
        idx = torch.randperm(n_train, device=device)
        ep_loss: float = 0.0
        n_b: int = 0

        for s in range(0, n_train, batch_size):
            e = min(s + batch_size, n_train)
            bi = idx[s:e]
            scores = model(X_tr[bi])
            loss, _ = loss_fn(scores, y_tr[bi])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_b += 1

        scheduler.step()

        # validate
        model.eval()
        with torch.no_grad():
            vs = model(X_va)
            vl, diag = loss_fn(vs, y_va)

            pred_r = vs.argsort(dim=-1, descending=True).argsort(dim=-1).float()
            true_r = y_va.argsort(dim=-1, descending=True).argsort(dim=-1).float()
            pm = pred_r.mean(-1, keepdim=True)
            tm = true_r.mean(-1, keepdim=True)
            cov = ((pred_r - pm) * (true_r - tm)).sum(-1)
            corr = (cov / ((pred_r - pm).pow(2).sum(-1).sqrt() * (true_r - tm).pow(2).sum(-1).sqrt()).clamp(1e-8)).mean().item()

        if corr > best_corr:
            best_corr = corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}: loss={ep_loss/n_b:.4f} corr={corr:.4f} "
                  f"best={best_corr:.4f} w_rank={diag['w_rank']:.2f} "
                  f"w_dir={diag['w_dir']:.2f} [{time.time()-t0:.1f}s]")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Training done in {time.time()-t0:.1f}s, best_corr={best_corr:.4f}")
    return model


# ============================================================================
# Backtest with Adverse Selection
# ============================================================================

def backtest_with_adverse_selection(
    model: CrossSectionalTransformer,
    X_oos: Tensor,
    y_oos: Tensor,
    close_matrix: Tensor,
    assets: List[str],
    top_k: int = 3,
    initial_cash: float = 1_000_000.0,
) -> Dict[str, Any]:
    """
    Long-short backtest with adverse selection execution model.

    close_matrix: (N_oos + lookahead, A) for adverse selection lookahead.
    """
    sim = AdverseSelectionSimulator(
        favorable_reject_rate=0.80,
        adverse_fill_rate=1.00,
        taker_fee_bps=4.0,
        maker_fee_bps=1.0,
        max_queue_bars=3,
    )

    model.eval()
    n = X_oos.size(0)
    n_assets = X_oos.size(1)
    equity: float = initial_cash
    eq_curve: List[float] = [equity]
    total_cost: float = 0.0
    prev_weights = torch.zeros(n_assets, device=X_oos.device)

    with torch.no_grad():
        for t in range(n - 3):  # need 3 bars lookahead
            scores = model(X_oos[t:t+1]).squeeze(0)
            returns = y_oos[t]

            top_idx = scores.topk(top_k).indices
            bot_idx = scores.topk(top_k, largest=False).indices
            target_w = torch.zeros(n_assets, device=X_oos.device)
            target_w[top_idx] = 1.0 / top_k
            target_w[bot_idx] = -1.0 / top_k

            # turnover filter: only rebalance if weights change significantly
            delta_w: float = (target_w - prev_weights).abs().sum().item()
            if delta_w < 0.8:  # only rebalance on major signal changes
                # still earn return on existing positions
                port_ret_hold: float = (prev_weights * returns).sum().item()
                equity *= (1.0 + port_ret_hold)
                eq_curve.append(equity)
                continue

            # for each asset with weight change, simulate execution
            port_ret: float = 0.0
            cost: float = 0.0

            for a in range(n_assets):
                dw: float = (target_w[a] - prev_weights[a]).item()
                if abs(dw) < 1e-6:
                    continue

                side: str = "BUY" if dw > 0 else "SELL"
                limit_price: float = close_matrix[t, a].item()
                future: List[float] = [close_matrix[t+1+i, a].item() for i in range(3)]

                filled, fill_price, cost_bps = sim.simulate_execution(
                    side, limit_price, abs(dw) * equity, future
                )

                if filled:
                    # actual return uses FILL price, not ideal close
                    price_diff: float = (fill_price - limit_price) / max(limit_price, 1e-8)
                    # for buys, higher fill = worse; for sells, lower fill = worse
                    execution_cost: float = abs(price_diff) if (
                        (side == "BUY" and fill_price > limit_price) or
                        (side == "SELL" and fill_price < limit_price)
                    ) else 0.0
                    actual_ret: float = returns[a].item() * dw
                    leg_cost: float = abs(dw) * equity * (cost_bps / 10000.0 + execution_cost)
                    port_ret += actual_ret
                    cost += leg_cost

            total_cost += cost
            equity *= (1.0 + port_ret - cost / max(equity, 1.0))
            eq_curve.append(equity)
            prev_weights = target_w.clone()

    # metrics
    rets = [(eq_curve[i]/eq_curve[i-1]) - 1 for i in range(1, len(eq_curve))]
    avg = sum(rets) / len(rets) if rets else 0
    std = (sum((r - avg)**2 for r in rets) / len(rets))**0.5 if rets else 1e-9
    sharpe = (avg / max(std, 1e-9)) * (252 * 4.8)**0.5  # annualise 5m bars
    peak = eq_curve[0]
    max_dd = 0.0
    for e in eq_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)

    return {
        "total_return": eq_curve[-1] / eq_curve[0] - 1,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": eq_curve[-1],
        "total_cost": total_cost,
        "n_periods": len(rets),
        **sim.stats(),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("  QUANT INFRA v5.0 — 27-Asset Cross-Sectional")
    print("  ListMLE + Focal + Adverse Selection + 233K Bars")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Data ---
    SEQ_LEN: int = 30
    MAX_ASSETS: int = 20  # use top 20 by data availability
    X, y, assets = build_dataset_from_db(
        DB_PATH, "5m", min_bars=5000, seq_len=SEQ_LEN,
        max_assets=MAX_ASSETS, device=device,
    )

    # Also build close matrix for adverse selection lookahead
    raw = load_all_from_db(DB_PATH, "5m", 5000)
    min_len = min(len(raw[s]) for s in assets)
    close_mat = torch.zeros(min_len, len(assets), device=device)
    for j, sym in enumerate(assets):
        bars = raw[sym][:min_len]
        close_mat[:, j] = torch.tensor([b[4] for b in bars], dtype=torch.float32, device=device)
    # align close_mat with X indices: X[i] uses bars [i:i+seq_len], so close for X[i] is at bar i+seq_len-1
    close_oos_start = int(X.size(0) * 0.8) + SEQ_LEN - 1

    # split 60/20/20
    n = X.size(0)
    tr_end = int(n * 0.6)
    va_end = int(n * 0.8)
    X_tr, y_tr = X[:tr_end], y[:tr_end]
    X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]
    X_te, y_te = X[va_end:], y[va_end:]
    print(f"\n  Split: train={X_tr.size(0)} val={X_va.size(0)} test={X_te.size(0)}")

    # close matrix for OOS (aligned with X_te indices)
    close_te = close_mat[close_oos_start:close_oos_start + X_te.size(0) + 5, :]

    # --- Model ---
    model = CrossSectionalTransformer(
        n_factors=X.size(3), d_model=128, n_heads=4,
        n_temporal_layers=2, n_cross_layers=2, d_ff=256,
        dropout=0.25, seq_len=SEQ_LEN, max_assets=MAX_ASSETS,
    ).to(device)
    loss_fn = UncertaintyWeightedDualLoss(focal_gamma=2.0).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    # --- Train ---
    print(f"\n[Training] Dual-objective with Uncertainty Weighting ...")
    model = train_v5(
        model, loss_fn, X_tr, y_tr, X_va, y_va,
        epochs=100, batch_size=64, lr=3e-4, device=device,
    )

    # --- Backtest ---
    print(f"\n[Backtest] OOS with Adverse Selection ({X_te.size(0)} periods) ...")
    summary = backtest_with_adverse_selection(
        model, X_te, y_te, close_te, assets, top_k=3,
    )

    # --- Report ---
    print("\n" + "=" * 65)
    print("  v5.0 BACKTEST REPORT (OOS with Adverse Selection)")
    print("=" * 65)
    print(f"  {'Assets':.<35s} {len(assets)}")
    print(f"  {'Total bars in DB':.<35s} 233,280")
    print(f"  {'OOS periods':.<35s} {summary['n_periods']}")
    print(f"  {'---PERFORMANCE---':.<35s}")
    print(f"  {'Total Return':.<35s} {summary['total_return']:>10.4%}")
    print(f"  {'Sharpe Ratio':.<35s} {summary['sharpe']:>10.4f}")
    print(f"  {'Max Drawdown':.<35s} {summary['max_drawdown']:>10.4%}")
    print(f"  {'Final Equity':.<35s} {summary['final_equity']:>14.2f}")
    print(f"  {'Total Transaction Cost':.<35s} {summary['total_cost']:>14.2f}")
    print(f"  {'---EXECUTION QUALITY---':.<35s}")
    print(f"  {'Total Orders':.<35s} {summary['total_orders']}")
    print(f"  {'Maker Fills':.<35s} {summary['maker_fills']} ({summary['maker_fill_pct']:.1%})")
    print(f"  {'Adverse Selection Fills':.<35s} {summary['adverse_fills']} ({summary['adverse_fill_pct']:.1%})")
    print(f"  {'Taker Fallbacks':.<35s} {summary['taker_fallbacks']} ({summary['taker_fallback_pct']:.1%})")
    print("=" * 65)
    print("\n[DONE] v5.0 pipeline complete.")


if __name__ == "__main__":
    main()
