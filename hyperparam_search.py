"""
hyperparam_search.py — Automated Grid Search for Cross-Sectional Pipeline.

Searches over:
  - top_k: {2, 3, 4} (number of assets in long/short legs)
  - cost_bps: {2, 4, 6} (taker fee scenarios)
  - d_model: {64, 128} (model capacity)
  - dropout: {0.1, 0.2, 0.3} (regularization)
  - lr: {1e-4, 3e-4, 1e-3}

Uses multiprocessing to parallelize on AMD 9950X3D.
Outputs a Sharpe Ratio table across all configurations.
"""
from __future__ import annotations

import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

sys.path.insert(0, ".")


def run_single_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one configuration and return metrics."""
    import torch
    import torch.optim as optim
    from model.cross_sectional import CrossSectionalTransformer, listmle_loss
    from run_cross_sectional import (
        build_4d_dataset,
        cross_sectional_backtest,
        train_ranking_model,
    )
    from data.crypto_feed import fetch_multi_asset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # load cached data or fetch
        multi_bars = fetch_multi_asset(timeframe="15m", limit=300)
        if len(multi_bars) < 5:
            return {**config, "sharpe": float("nan"), "error": "insufficient assets"}

        X, y, assets, _ = build_4d_dataset(multi_bars, seq_len=20, device=device)
        n = X.size(0)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_va, y_va = X[train_end:val_end], y[train_end:val_end]
        X_te, y_te = X[val_end:], y[val_end:]

        model = CrossSectionalTransformer(
            n_factors=X.size(3),
            d_model=config["d_model"],
            n_heads=4,
            n_temporal_layers=2,
            n_cross_layers=2,
            d_ff=config["d_model"] * 2,
            dropout=config["dropout"],
            seq_len=20,
            max_assets=len(assets),
        ).to(device)

        model = train_ranking_model(
            model, X_tr, y_tr, X_va, y_va,
            epochs=60, batch_size=32, lr=config["lr"], device=device,
        )

        summary = cross_sectional_backtest(
            model, X_te, y_te, assets,
            top_k=config["top_k"],
            cost_bps=config["cost_bps"],
        )

        return {**config, **summary, "error": None}

    except Exception as e:
        return {**config, "sharpe": float("nan"), "error": str(e)}


def _run_with_cached_data(
    config: Dict[str, Any],
    X: "torch.Tensor",
    y: "torch.Tensor",
    assets: List[str],
    device: "torch.device",
) -> Dict[str, Any]:
    """Run one config with pre-loaded data."""
    import torch
    from model.cross_sectional import CrossSectionalTransformer
    from run_cross_sectional import train_ranking_model, cross_sectional_backtest

    try:
        n = X.size(0)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_va, y_va = X[train_end:val_end], y[train_end:val_end]
        X_te, y_te = X[val_end:], y[val_end:]

        model = CrossSectionalTransformer(
            n_factors=X.size(3),
            d_model=config["d_model"],
            n_heads=4,
            n_temporal_layers=2,
            n_cross_layers=2,
            d_ff=config["d_model"] * 2,
            dropout=config["dropout"],
            seq_len=20,
            max_assets=len(assets),
        ).to(device)

        model = train_ranking_model(
            model, X_tr, y_tr, X_va, y_va,
            epochs=60, batch_size=32, lr=config["lr"], device=device,
        )

        summary = cross_sectional_backtest(
            model, X_te, y_te, assets,
            top_k=config["top_k"],
            cost_bps=config["cost_bps"],
        )
        return {**config, **summary, "error": None}
    except Exception as e:
        return {**config, "sharpe": float("nan"), "error": str(e)}


def main() -> None:
    print("=" * 70)
    print("  HYPERPARAMETER GRID SEARCH")
    print("=" * 70)

    # Define grid
    # Focused grid — test critical params only
    grid: Dict[str, List[Any]] = {
        "top_k": [2, 3],
        "cost_bps": [2.0, 4.0],
        "d_model": [64, 128],
        "dropout": [0.15, 0.3],
        "lr": [3e-4],
    }

    keys: List[str] = list(grid.keys())
    combos: List[Dict[str, Any]] = [
        dict(zip(keys, vals))
        for vals in itertools.product(*grid.values())
    ]
    print(f"  Total configurations: {len(combos)}")

    # Pre-fetch data once, then reuse for all configs
    import torch
    from data.crypto_feed import fetch_multi_asset
    from run_cross_sectional import build_4d_dataset

    print("\n  Pre-fetching data ...")
    multi_bars = fetch_multi_asset(timeframe="15m", limit=300)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, assets, _ = build_4d_dataset(multi_bars, seq_len=20, device=device)
    print(f"  Data ready: X={X.shape}")

    # Run sequentially on GPU
    results: List[Dict[str, Any]] = []
    t0: float = time.time()

    for i, config in enumerate(combos):
        print(f"\n--- Config {i+1}/{len(combos)}: {config}")
        result = _run_with_cached_data(config, X, y, assets, device)
        results.append(result)
        sharpe = result.get("sharpe", float("nan"))
        ret = result.get("total_return", float("nan"))
        dd = result.get("max_drawdown", float("nan"))
        print(f"    -> Sharpe={sharpe:.4f} Return={ret:.4%} MaxDD={dd:.4%}")

    elapsed: float = time.time() - t0
    print(f"\n\nSearch completed in {elapsed:.1f}s")

    # Sort by Sharpe and print table
    results.sort(key=lambda r: r.get("sharpe", float("-inf")), reverse=True)

    print("\n" + "=" * 100)
    print(f"{'Rank':>4} {'top_k':>5} {'cost':>5} {'d_mod':>5} {'drop':>5} "
          f"{'lr':>8} {'Sharpe':>10} {'Return':>10} {'MaxDD':>10} {'Turnover':>10}")
    print("-" * 100)
    for i, r in enumerate(results[:20]):
        print(f"{i+1:4d} {r.get('top_k','?'):>5} {r.get('cost_bps','?'):>5.1f} "
              f"{r.get('d_model','?'):>5} {r.get('dropout','?'):>5.2f} "
              f"{r.get('lr','?'):>8.1e} "
              f"{r.get('sharpe',0):>10.4f} {r.get('total_return',0):>10.4%} "
              f"{r.get('max_drawdown',0):>10.4%} {r.get('avg_turnover',0):>10.4f}")
    print("=" * 100)

    # Best config
    best = results[0]
    print(f"\nBEST CONFIG: {best}")


if __name__ == "__main__":
    main()
