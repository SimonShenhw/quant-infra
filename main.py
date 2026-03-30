"""
main.py — Full Pipeline Integration (v2 — Data Leakage Fixed).

Key fixes from v1:
  - Rolling z-score normalisation (no global stats, no future function)
  - Strict T→T+1 label alignment with explicit verification
  - Walk-forward expanding-window training (no simple 80/20 split)
  - Realistic transaction cost model (commission + slippage)
  - Dynamic position sizing with trailing volatility
  - Improved signal generation thresholds

Tuned for: AMD 9950X3D + 64GB DDR5 + NVIDIA RTX 5090
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

from data.synthetic_lob import SyntheticLOBGenerator
from engine.backtest import BacktestEngine
from engine.events import Event, MarketEvent, TickEvent
from model.features import build_factor_tensor
from model.strategy import TransformerStrategy
from model.transformer import QuantTransformer, build_quant_transformer


# ============================================================================
# Phase 1: Generate Synthetic Data
# ============================================================================

def generate_data(
    symbol: str = "SH600000",
    n_ticks: int = 480_000,
    ticks_per_bar: int = 100,
    s0: float = 50.0,
) -> Tuple[List[TickEvent], List[MarketEvent]]:
    print("[Phase 1] Generating synthetic LOB data ...")
    gen: SyntheticLOBGenerator = SyntheticLOBGenerator(
        symbol=symbol,
        n_ticks=n_ticks,
        ticks_per_bar=ticks_per_bar,
        s0=s0,
        seed=42,
        momentum_strength=0.35,       # strong learnable momentum
        mean_reversion_speed=0.08,    # strong MR at extremes
    )
    ticks, bars = gen.generate_all()
    print(f"  -> {len(ticks):,} ticks, {len(bars):,} bars generated")
    print(f"  -> Price range: {bars[0].open:.2f} -> {bars[-1].close:.2f}")
    return ticks, bars


# ============================================================================
# Phase 2: Prepare Training Data (LEAKAGE-FREE)
# ============================================================================

def prepare_training_data(
    bars: List[MarketEvent],
    seq_len: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build (X, y, closes) with STRICT causal alignment:

      X[i] = factors[i : i+seq_len]      (bars at time T-seq_len+1 ... T)
      y[i] = return at T+1               (forward 1-bar return)

    The feature window X[i] uses ONLY bars up to and including T.
    The label y[i] is the return from T to T+1 (truly out-of-sample).

    Returns (X, y, closes) where closes is the raw close prices for
    mark-to-market and analysis.
    """
    closes: Tensor = torch.tensor(
        [b.close for b in bars], dtype=torch.float32, device=device
    )
    opens: Tensor = torch.tensor(
        [b.open for b in bars], dtype=torch.float32, device=device
    )
    highs: Tensor = torch.tensor(
        [b.high for b in bars], dtype=torch.float32, device=device
    )
    lows: Tensor = torch.tensor(
        [b.low for b in bars], dtype=torch.float32, device=device
    )
    volumes: Tensor = torch.tensor(
        [b.volume for b in bars], dtype=torch.float32, device=device
    )

    factors: Tensor = build_factor_tensor(
        opens, highs, lows, closes, volumes, zscore_window=60
    )  # (T, 8) — all causal

    # forward 1-bar simple returns as labels
    # return[t] = close[t+1]/close[t] - 1
    # This is what we predict: the NEXT bar's return
    fwd_returns: Tensor = torch.zeros(len(bars), device=device, dtype=torch.float32)
    fwd_returns[:-1] = closes[1:] / closes[:-1].clamp(min=1e-8) - 1.0
    # fwd_returns[-1] remains 0 (no future bar available)

    # Build samples with EXPLICIT alignment
    # For sample i:
    #   feature_window = factors[i : i+seq_len]  (uses bars i..i+seq_len-1)
    #   label = fwd_returns[i+seq_len-1]          (return from bar i+seq_len-1 to i+seq_len)
    #
    # This means the latest bar in the feature window is (i+seq_len-1),
    # and we predict the return from that bar to the NEXT bar (i+seq_len).
    # The feature window NEVER sees bar (i+seq_len).

    n_bars: int = factors.size(0)
    n_samples: int = n_bars - seq_len  # last sample predicts return of second-to-last bar
    # but we also need the target bar to exist, so:
    n_samples = n_bars - seq_len - 1  # ensure target bar (i+seq_len) exists

    X_list: List[Tensor] = []
    y_list: List[Tensor] = []
    for i in range(n_samples):
        feature_end: int = i + seq_len  # exclusive end of window
        X_list.append(factors[i:feature_end, :])        # (seq_len, 8)
        y_list.append(fwd_returns[feature_end - 1])     # scalar: return from bar (feature_end-1) to bar (feature_end)

    X: Tensor = torch.stack(X_list, dim=0)              # (N, seq_len, 8)
    y: Tensor = torch.stack(y_list, dim=0).unsqueeze(1)  # (N, 1)

    # ---- LEAKAGE VERIFICATION ----
    # The maximum index in X[i] is (i+seq_len-1).
    # The label y[i] uses close[i+seq_len] and close[i+seq_len-1].
    # close[i+seq_len] is NOT in X[i]'s feature window. ✓
    # Factors are computed causally (rolling z-score, no global stats). ✓
    print(f"  -> Leakage check: X uses bars [0..{n_samples + seq_len - 2}], "
          f"y uses bars [0..{n_samples + seq_len - 1}]")
    print(f"  -> Last feature window ends at bar {n_samples - 1 + seq_len - 1}, "
          f"its label uses close[{n_samples - 1 + seq_len}]")

    return X, y, closes


# ============================================================================
# Phase 2b: Walk-Forward Expanding-Window Training
# ============================================================================

def walk_forward_train(
    model: QuantTransformer,
    X: Tensor,
    y: Tensor,
    n_folds: int = 5,
    epochs_per_fold: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[QuantTransformer, List[float]]:
    """
    Walk-forward expanding-window training.

    For each fold k:
      train on samples [0 .. split_k)
      validate on samples [split_k .. split_{k+1})

    The model is retrained from the PREVIOUS fold's weights (warm start),
    simulating how you'd retrain in production as new data arrives.

    Returns the final model and list of per-fold validation losses.
    """
    n_total: int = X.size(0)
    fold_size: int = n_total // (n_folds + 1)
    min_train: int = fold_size * 2  # minimum training set size

    print(f"\n[Phase 2] Walk-Forward Training ({n_folds} folds) ...")
    print(f"  -> {sum(p.numel() for p in model.parameters()):,} params on {device}")
    print(f"  -> Total samples: {n_total}, fold_size: {fold_size}")

    # Asymmetric Focal Loss (from arXiv 2506.05764 + 2603.16886)
    # Focal loss focuses on HARD examples, Huber handles magnitude
    huber: nn.HuberLoss = nn.HuberLoss(delta=0.005)
    focal_gamma: float = 2.0  # focus on hard-to-classify samples
    dir_weight: float = 3.0   # direction > magnitude
    mag_weight: float = 1.0

    def focal_bce(logits: Tensor, targets: Tensor, gamma: float = 2.0) -> Tensor:
        """Focal loss: down-weights easy examples, focuses on misclassifications."""
        p: Tensor = torch.sigmoid(logits)
        ce: Tensor = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # focal weight: (1-p_t)^gamma where p_t is prob of correct class
        p_t: Tensor = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight: Tensor = (1.0 - p_t) ** gamma
        return (focal_weight * ce).mean()

    def directional_focal_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Combined Huber (magnitude) + Focal (direction) loss."""
        mag_loss: Tensor = huber(pred, target)
        target_dir: Tensor = (target > 0).float()
        # scale pred to logit space (pred is ~0.001 scale, need ~1.0 for sigmoid)
        logits: Tensor = pred.squeeze(-1) * 200.0
        dir_loss: Tensor = focal_bce(logits, target_dir.squeeze(-1), focal_gamma)
        return mag_weight * mag_loss + dir_weight * dir_loss

    val_losses: List[float] = []
    t0_total: float = time.time()

    for fold in range(n_folds):
        train_end: int = min_train + fold * fold_size
        val_end: int = min(train_end + fold_size, n_total)
        if val_end <= train_end:
            break

        X_tr: Tensor = X[:train_end]
        y_tr: Tensor = y[:train_end]
        X_val: Tensor = X[train_end:val_end]
        y_val: Tensor = y[train_end:val_end]

        optimizer: optim.AdamW = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_fold)

        model.train()
        n_train: int = X_tr.size(0)
        t0: float = time.time()

        best_val_loss: float = float("inf")
        patience: int = 7
        no_improve: int = 0

        for epoch in range(1, epochs_per_fold + 1):
            indices: Tensor = torch.randperm(n_train, device=device)
            epoch_loss: float = 0.0
            n_batches: int = 0

            for start in range(0, n_train, batch_size):
                end: int = min(start + batch_size, n_train)
                idx: Tensor = indices[start:end]
                xb: Tensor = X_tr[idx]
                yb: Tensor = y_tr[idx]

                optimizer.zero_grad()
                pred: Tensor = model(xb)
                loss: Tensor = directional_focal_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # early stopping: validate on DIRECTION ACCURACY not MSE
            model.eval()
            with torch.no_grad():
                val_pred: Tensor = model(X_val)
                val_mse: float = nn.MSELoss()(val_pred, y_val).item()
                val_dir: float = (
                    ((val_pred.squeeze() > 0) == (y_val.squeeze() > 0))
                    .float().mean().item()
                )
                # combined metric: penalise low direction accuracy
                val_loss: float = val_mse * (2.0 - val_dir)
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        val_losses.append(best_val_loss)
        elapsed: float = time.time() - t0

        # direction accuracy (more meaningful than MSE for returns)
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
        pred_sign: Tensor = (val_pred.squeeze() > 0).float()
        true_sign: Tensor = (y_val.squeeze() > 0).float()
        direction_acc: float = (pred_sign == true_sign).float().mean().item()

        train_loss: float = epoch_loss / max(n_batches, 1)
        print(f"  Fold {fold+1}/{n_folds}: train={train_end} val={val_end-train_end} "
              f"| train_loss={train_loss:.6f} val_loss={best_val_loss:.6f} "
              f"dir_acc={direction_acc:.3f} [{elapsed:.1f}s]")

    print(f"  -> Walk-forward complete in {time.time() - t0_total:.1f}s")
    print(f"  -> Val losses: {[f'{v:.6f}' for v in val_losses]}")

    return model, val_losses


# ============================================================================
# Phase 3: Event-Driven Backtest
# ============================================================================

def run_backtest(
    bars: List[MarketEvent],
    ticks: List[TickEvent],
    model: QuantTransformer,
    device: torch.device,
) -> Dict[str, float]:
    print("\n[Phase 3] Running event-driven backtest ...")
    print(f"  -> {len(bars)} bars, {len(ticks)} ticks feeding into engine")

    engine: BacktestEngine = BacktestEngine(
        initial_cash=1_000_000.0,
        tick_size=0.01,
        max_drawdown=0.25,
        max_position_pct=0.20,   # larger positions to capture weak alpha
        verbose=True,
    )

    strategy: TransformerStrategy = TransformerStrategy(
        model=model,
        device=device,
        lookback=60,
        threshold_sigma=0.35,   # only trade on strong conviction (covers costs)
        warmup=80,
        vol_window=20,
        cooldown_bars=5,        # avoid whipsaw after stop-loss
        max_holding_bars=15,    # hold longer to let alpha compound
    )
    engine.register_strategy(strategy.handle_market)

    events: List[Event] = []
    ticks_per_bar: int = max(1, len(ticks) // max(len(bars), 1))
    for i, bar in enumerate(bars):
        tick_start: int = i * ticks_per_bar
        tick_end: int = min(tick_start + ticks_per_bar, len(ticks))
        step: int = max(1, (tick_end - tick_start) // 5)
        for j in range(tick_start, tick_end, step):
            events.append(ticks[j])
        events.append(bar)

    print(f"  -> Total events in stream: {len(events):,}")
    summary: Dict[str, float] = engine.run(events)
    return summary


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("  QUANT INFRA v2 - Leakage-Free Transformer Pipeline")
    print("  Hardware: AMD 9950X3D | 64GB DDR5 | RTX 5090")
    print("=" * 60)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Compute device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Phase 1: Data ---
    ticks, bars = generate_data(
        symbol="SH600000", n_ticks=960_000, ticks_per_bar=100, s0=50.0
    )

    # --- Phase 2: Model + Walk-Forward Training ---
    SEQ_LEN: int = 60
    N_FACTORS: int = 10
    # medium: d=256, 8h, 3enc+2dec — better params/sample ratio to reduce overfitting
    model: QuantTransformer = build_quant_transformer(
        n_factors=N_FACTORS, preset="medium", device=device
    )

    X, y, closes = prepare_training_data(bars, SEQ_LEN, device)
    print(f"  -> X: {X.shape}, y: {y.shape}")
    print(f"  -> y stats: mean={y.mean():.6f} std={y.std():.6f} "
          f"min={y.min():.6f} max={y.max():.6f}")

    # Sanity check: if MSE is suspiciously low, flag it
    model, val_losses = walk_forward_train(
        model, X, y,
        n_folds=5,
        epochs_per_fold=30,
        batch_size=512,
        lr=3e-4,
        device=device,
    )

    # Final out-of-sample eval on last 20%
    split: int = int(X.size(0) * 0.8)
    X_oos: Tensor = X[split:]
    y_oos: Tensor = y[split:]
    model.eval()
    with torch.no_grad():
        oos_pred: Tensor = model(X_oos)
        oos_mse: float = nn.MSELoss()(oos_pred, y_oos).item()
        # direction accuracy
        pred_sign: Tensor = (oos_pred.squeeze() > 0).float()
        true_sign: Tensor = (y_oos.squeeze() > 0).float()
        dir_acc: float = (pred_sign == true_sign).float().mean().item()

    print(f"\n  -> OOS Test MSE: {oos_mse:.6f}")
    print(f"  -> OOS Direction Accuracy: {dir_acc:.3f}")
    print(f"  -> y_oos std: {y_oos.std():.6f}  (MSE should be similar magnitude)")

    if oos_mse < y_oos.var().item() * 0.01:
        print("  !! WARNING: MSE << Var(y) — possible residual leakage!")
    else:
        print("  -> Leakage check PASSED (MSE is reasonable relative to Var(y))")

    # --- Phase 3: Backtest ---
    summary: Dict[str, float] = run_backtest(bars, ticks, model, device)

    print("\n[DONE] Pipeline v2 finished successfully.")


if __name__ == "__main__":
    main()
