"""
run_cross_sectional.py — Multi-Asset Cross-Sectional Ranking Pipeline.

v4.0: The model learns to RANK 10 crypto assets by relative performance,
then goes long the top-K and short the bottom-K.

Architecture:
  - 4D tensor: [Batch, 10_Assets, 20_bars, 10_factors]
  - CrossSectionalTransformer: temporal + cross-asset attention
  - ListMLE ranking loss
  - Long-Short portfolio: top 3 long, bottom 3 short
  - Crypto maker/taker cost model
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

sys.path.insert(0, ".")

from data.crypto_feed import fetch_multi_asset
from engine.events import EventType, MarketEvent
from model.features import build_factor_tensor
from model.cross_sectional import CrossSectionalTransformer, listmle_loss


# ============================================================================
# Data preparation: build 4D tensor from multi-asset bars
# ============================================================================

def build_4d_dataset(
    multi_bars: Dict[str, List[MarketEvent]],
    seq_len: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor, List[str], List[datetime]]:
    """
    Build 4D tensor [N_samples, A_assets, T_seq, F_factors] + labels.

    Labels: cross-sectional forward returns at each time step.
    For each timestep t, label[a] = return of asset a from t to t+1.

    Returns: (X, y, asset_names, timestamps)
    """
    asset_names: List[str] = sorted(multi_bars.keys())
    n_assets: int = len(asset_names)

    # find common time range (all assets must have same # of bars)
    min_len: int = min(len(multi_bars[s]) for s in asset_names)
    print(f"  -> {n_assets} assets, {min_len} bars each (trimmed to common range)")

    # build per-asset factor tensors
    all_factors: Dict[str, Tensor] = {}
    all_closes: Dict[str, Tensor] = {}
    timestamps: List[datetime] = [
        multi_bars[asset_names[0]][i].timestamp for i in range(min_len)
    ]

    for sym in asset_names:
        bars: List[MarketEvent] = multi_bars[sym][:min_len]
        closes: Tensor = torch.tensor([b.close for b in bars], dtype=torch.float32, device=device)
        opens: Tensor = torch.tensor([b.open for b in bars], dtype=torch.float32, device=device)
        highs: Tensor = torch.tensor([b.high for b in bars], dtype=torch.float32, device=device)
        lows: Tensor = torch.tensor([b.low for b in bars], dtype=torch.float32, device=device)
        vols: Tensor = torch.tensor([b.volume for b in bars], dtype=torch.float32, device=device)
        factors: Tensor = build_factor_tensor(opens, highs, lows, closes, vols, zscore_window=30)
        all_factors[sym] = factors  # (T, F)
        all_closes[sym] = closes

    # build forward returns matrix: (T, A)
    n_factors: int = all_factors[asset_names[0]].size(1)
    fwd_returns: Tensor = torch.zeros(min_len, n_assets, device=device)
    for j, sym in enumerate(asset_names):
        c: Tensor = all_closes[sym]
        fwd_returns[:-1, j] = c[1:] / c[:-1].clamp(min=1e-8) - 1.0

    # build sliding windows: X[i] = factors[i:i+seq_len] for all assets
    n_samples: int = min_len - seq_len - 1
    X_list: List[Tensor] = []
    y_list: List[Tensor] = []

    for i in range(n_samples):
        # (A, T, F)
        sample: List[Tensor] = []
        for sym in asset_names:
            sample.append(all_factors[sym][i:i + seq_len, :])  # (T, F)
        X_list.append(torch.stack(sample, dim=0))  # (A, T, F)
        y_list.append(fwd_returns[i + seq_len - 1, :])  # (A,)

    X: Tensor = torch.stack(X_list, dim=0)  # (N, A, T, F)
    y: Tensor = torch.stack(y_list, dim=0)  # (N, A)

    print(f"  -> X: {X.shape}, y: {y.shape}")
    print(f"  -> y cross-sectional std: {y.std(dim=1).mean():.6f}")

    return X, y, asset_names, timestamps


# ============================================================================
# Training with ListMLE
# ============================================================================

def train_ranking_model(
    model: CrossSectionalTransformer,
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
) -> CrossSectionalTransformer:
    """Train with ListMLE + early stopping on ranking metrics."""
    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_train: int = X_train.size(0)

    best_rank_corr: float = -1.0
    best_state: Optional[Dict] = None
    patience: int = 10
    no_improve: int = 0

    t0: float = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        idx: Tensor = torch.randperm(n_train, device=device)
        epoch_loss: float = 0.0
        n_batches: int = 0

        for start in range(0, n_train, batch_size):
            end: int = min(start + batch_size, n_train)
            bi: Tensor = idx[start:end]
            xb: Tensor = X_train[bi]
            yb: Tensor = y_train[bi]

            optimizer.zero_grad()
            scores: Tensor = model(xb)  # (B, A)
            loss: Tensor = listmle_loss(scores, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # evaluate: rank correlation (Spearman-like using Pearson on ranks)
        model.eval()
        with torch.no_grad():
            val_scores: Tensor = model(X_val)  # (V, A)
            val_loss: float = listmle_loss(val_scores, y_val).item()

            # rank correlation: for each sample, correlate predicted rank with true rank
            pred_ranks: Tensor = val_scores.argsort(dim=-1, descending=True).argsort(dim=-1).float()
            true_ranks: Tensor = y_val.argsort(dim=-1, descending=True).argsort(dim=-1).float()
            # Pearson correlation of ranks (per sample, then average)
            pr_mean: Tensor = pred_ranks.mean(dim=-1, keepdim=True)
            tr_mean: Tensor = true_ranks.mean(dim=-1, keepdim=True)
            cov: Tensor = ((pred_ranks - pr_mean) * (true_ranks - tr_mean)).sum(dim=-1)
            pr_std: Tensor = (pred_ranks - pr_mean).pow(2).sum(dim=-1).sqrt()
            tr_std: Tensor = (true_ranks - tr_mean).pow(2).sum(dim=-1).sqrt()
            corr: Tensor = cov / (pr_std * tr_std).clamp(min=1e-8)
            rank_corr: float = corr.mean().item()

            # top-3 vs bottom-3 return spread
            top3_ret: float = 0.0
            bot3_ret: float = 0.0
            for i in range(val_scores.size(0)):
                s: Tensor = val_scores[i]
                r: Tensor = y_val[i]
                top_idx: Tensor = s.topk(3).indices
                bot_idx: Tensor = s.topk(3, largest=False).indices
                top3_ret += r[top_idx].mean().item()
                bot3_ret += r[bot_idx].mean().item()
            n_v: int = val_scores.size(0)
            ls_spread: float = (top3_ret - bot3_ret) / n_v

        if rank_corr > best_rank_corr:
            best_rank_corr = rank_corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f} "
                  f"val_loss={val_loss:.4f} rank_corr={rank_corr:.4f} "
                  f"LS_spread={ls_spread*10000:.1f}bps best_corr={best_rank_corr:.4f} "
                  f"[{time.time()-t0:.1f}s]")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  -> Training complete in {time.time()-t0:.1f}s, best rank_corr={best_rank_corr:.4f}")
    return model


# ============================================================================
# Cross-Sectional Backtest (Long-Short)
# ============================================================================

def cross_sectional_backtest(
    model: CrossSectionalTransformer,
    X_oos: Tensor,
    y_oos: Tensor,
    asset_names: List[str],
    top_k: int = 3,
    initial_cash: float = 1_000_000.0,
    cost_bps: float = 4.0,
) -> Dict[str, float]:
    """
    Simple long-short backtest:
      - At each timestep, rank assets by model score
      - Go long top-K, short bottom-K (equal weight)
      - Pay taker fees on each rebalance
    """
    model.eval()
    n_samples: int = X_oos.size(0)
    n_assets: int = X_oos.size(1)
    equity: float = initial_cash
    equity_curve: List[float] = [equity]
    total_cost: float = 0.0
    turnover_total: float = 0.0

    prev_weights: Tensor = torch.zeros(n_assets, device=X_oos.device)

    with torch.no_grad():
        for t in range(n_samples):
            scores: Tensor = model(X_oos[t:t+1])  # (1, A)
            scores = scores.squeeze(0)  # (A,)
            returns: Tensor = y_oos[t]   # (A,) true forward returns

            # build target weights: +1/K for top-K, -1/K for bottom-K
            top_idx: Tensor = scores.topk(top_k).indices
            bot_idx: Tensor = scores.topk(top_k, largest=False).indices
            weights: Tensor = torch.zeros(n_assets, device=X_oos.device)
            weights[top_idx] = 1.0 / top_k
            weights[bot_idx] = -1.0 / top_k

            # turnover = sum of absolute weight changes
            turnover: float = (weights - prev_weights).abs().sum().item()
            turnover_total += turnover

            # cost = turnover * notional * cost_bps
            cost: float = turnover * equity * cost_bps / 10000.0
            total_cost += cost

            # portfolio return = sum(weight * return) - cost/equity
            port_ret: float = (weights * returns).sum().item() - cost / equity
            equity *= (1.0 + port_ret)
            equity_curve.append(equity)

            prev_weights = weights.clone()

    # compute metrics
    eq: List[float] = equity_curve
    total_return: float = eq[-1] / eq[0] - 1.0
    daily_rets: List[float] = [(eq[i] / eq[i-1]) - 1.0 for i in range(1, len(eq))]
    avg_ret: float = sum(daily_rets) / len(daily_rets) if daily_rets else 0.0
    std_ret: float = (sum((r - avg_ret)**2 for r in daily_rets) / len(daily_rets))**0.5 if daily_rets else 1e-9
    sharpe: float = (avg_ret / max(std_ret, 1e-9)) * (252 * 4) ** 0.5  # annualise for 15m bars (~4/hour * 6h)

    peak: float = eq[0]
    max_dd: float = 0.0
    for e in eq:
        peak = max(peak, e)
        dd: float = (peak - e) / peak
        max_dd = max(max_dd, dd)

    avg_turnover: float = turnover_total / max(n_samples, 1)

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": eq[-1],
        "total_cost": total_cost,
        "avg_turnover": avg_turnover,
        "n_rebalances": n_samples,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("  QUANT INFRA v4.0 — Multi-Asset Cross-Sectional Ranking")
    print("  10 Crypto Assets | ListMLE | Long-Short Portfolio")
    print("=" * 60)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # --- Step 1: Fetch multi-asset data ---
    print("\n[Step 1] Fetching 10 crypto assets from OKX ...")
    multi_bars: Dict[str, List[MarketEvent]] = fetch_multi_asset(
        timeframe="15m", limit=300
    )
    if len(multi_bars) < 5:
        print("ERROR: Need at least 5 assets")
        return

    # --- Step 2: Build 4D dataset ---
    print("\n[Step 2] Building 4D tensor dataset ...")
    SEQ_LEN: int = 20
    X, y, asset_names, timestamps = build_4d_dataset(multi_bars, SEQ_LEN, device)
    print(f"  -> Assets: {asset_names}")

    # --- Step 3: Train/Val/Test split (60/20/20) ---
    n: int = X.size(0)
    train_end: int = int(n * 0.6)
    val_end: int = int(n * 0.8)

    X_tr, y_tr = X[:train_end], y[:train_end]
    X_va, y_va = X[train_end:val_end], y[train_end:val_end]
    X_te, y_te = X[val_end:], y[val_end:]
    print(f"\n  -> Split: train={X_tr.size(0)}, val={X_va.size(0)}, test={X_te.size(0)}")

    # --- Step 4: Train ranking model ---
    print("\n[Step 3] Training CrossSectionalTransformer with ListMLE ...")
    n_assets: int = len(asset_names)
    model: CrossSectionalTransformer = CrossSectionalTransformer(
        n_factors=X.size(3),
        d_model=128,
        n_heads=4,
        n_temporal_layers=2,
        n_cross_layers=2,
        d_ff=256,
        dropout=0.1,
        seq_len=SEQ_LEN,
        max_assets=n_assets,
    ).to(device)
    print(f"  -> {sum(p.numel() for p in model.parameters()):,} params")

    model = train_ranking_model(
        model, X_tr, y_tr, X_va, y_va,
        epochs=80, batch_size=32, lr=3e-4, device=device,
    )

    # --- Step 5: OOS backtest ---
    print("\n[Step 4] Running OOS Long-Short backtest ...")
    summary: Dict[str, float] = cross_sectional_backtest(
        model, X_te, y_te, asset_names,
        top_k=3, initial_cash=1_000_000.0, cost_bps=4.0,
    )

    print("\n" + "=" * 60)
    print("  CROSS-SECTIONAL BACKTEST RESULTS (OOS)")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            if abs(v) > 100:
                print(f"  {k:<25s} {v:>15.2f}")
            else:
                print(f"  {k:<25s} {v:>15.6f}")
        else:
            print(f"  {k:<25s} {v:>15}")
    print("=" * 60)

    # --- Step 5b: Rank correlation analysis ---
    model.eval()
    with torch.no_grad():
        oos_scores: Tensor = model(X_te)
        pred_ranks = oos_scores.argsort(dim=-1, descending=True).argsort(dim=-1).float()
        true_ranks = y_te.argsort(dim=-1, descending=True).argsort(dim=-1).float()
        pr_mean = pred_ranks.mean(dim=-1, keepdim=True)
        tr_mean = true_ranks.mean(dim=-1, keepdim=True)
        cov = ((pred_ranks - pr_mean) * (true_ranks - tr_mean)).sum(dim=-1)
        pr_std = (pred_ranks - pr_mean).pow(2).sum(dim=-1).sqrt()
        tr_std = (true_ranks - tr_mean).pow(2).sum(dim=-1).sqrt()
        corr = cov / (pr_std * tr_std).clamp(min=1e-8)
        oos_rank_corr: float = corr.mean().item()

    print(f"\n  OOS Rank Correlation: {oos_rank_corr:.4f}")
    print(f"  Assets: {', '.join(asset_names)}")
    print("\n[DONE] v4.0 Cross-Sectional Pipeline Complete.")


if __name__ == "__main__":
    main()
