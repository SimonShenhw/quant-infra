"""
Task 3: Parquet Data Lake Loader.

Provides a unified interface to load data from the partitioned
Parquet data lake into Polars DataFrames or PyTorch tensors.

Data lake structure:
  data_lake/
    {SYMBOL}/
      {YEAR}/
        {MONTH}/
          klines_5m.parquet
          aggTrades.parquet
          realtime/
            trades_DD_HHMMSS.parquet
            depth_DD_HHMMSS.parquet
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import torch
from torch import Tensor

DATA_LAKE: str = str(Path(__file__).resolve().parent.parent / "data_lake")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_symbols(data_lake: str = DATA_LAKE) -> List[str]:
    """List all symbols with data in the lake."""
    if not os.path.exists(data_lake):
        return []
    return sorted([
        d for d in os.listdir(data_lake)
        if os.path.isdir(os.path.join(data_lake, d))
    ])


def list_partitions(
    symbol: str, data_lake: str = DATA_LAKE
) -> List[Tuple[int, int]]:
    """List (year, month) partitions for a symbol."""
    sym_dir: str = os.path.join(data_lake, symbol)
    if not os.path.exists(sym_dir):
        return []
    parts: List[Tuple[int, int]] = []
    for year_dir in sorted(os.listdir(sym_dir)):
        year_path: str = os.path.join(sym_dir, year_dir)
        if not os.path.isdir(year_path):
            continue
        for month_dir in sorted(os.listdir(year_path)):
            month_path: str = os.path.join(year_path, month_dir)
            if os.path.isdir(month_path):
                try:
                    parts.append((int(year_dir), int(month_dir)))
                except ValueError:
                    continue
    return parts


# ---------------------------------------------------------------------------
# Loading: Klines
# ---------------------------------------------------------------------------

def load_klines(
    symbol: str,
    interval: str = "5m",
    data_lake: str = DATA_LAKE,
) -> pl.DataFrame:
    """Load all kline parquet files for a symbol, sorted by time."""
    parts: List[Tuple[int, int]] = list_partitions(symbol, data_lake)
    frames: List[pl.DataFrame] = []
    for year, month in parts:
        path: str = os.path.join(
            data_lake, symbol, str(year), f"{month:02d}", f"klines_{interval}.parquet"
        )
        if os.path.exists(path):
            try:
                df: pl.DataFrame = pl.read_parquet(path)
                frames.append(df)
            except Exception:
                continue
    if not frames:
        return pl.DataFrame()
    # normalize schemas: keep only common OHLCV columns
    common_cols: list = ["open_time", "open", "high", "low", "close", "volume"]
    normalized: list = []
    for f in frames:
        cols_present = [c for c in common_cols if c in f.columns]
        if len(cols_present) == len(common_cols):
            normalized.append(f.select(common_cols))
    if not normalized:
        return pl.DataFrame()
    combined: pl.DataFrame = pl.concat(normalized)
    combined = combined.sort("open_time").unique(subset=["open_time"])
    return combined


def load_klines_multi(
    symbols: Optional[List[str]] = None,
    interval: str = "5m",
    min_rows: int = 10_000,
    data_lake: str = DATA_LAKE,
) -> Dict[str, pl.DataFrame]:
    """Load klines for multiple symbols, filtering by minimum rows."""
    if symbols is None:
        symbols = list_symbols(data_lake)
    result: Dict[str, pl.DataFrame] = {}
    for sym in symbols:
        df: pl.DataFrame = load_klines(sym, interval, data_lake)
        if df.height >= min_rows:
            result[sym] = df
    return result


# ---------------------------------------------------------------------------
# Loading: AggTrades
# ---------------------------------------------------------------------------

def load_aggtrades(
    symbol: str,
    data_lake: str = DATA_LAKE,
) -> pl.DataFrame:
    """Load all aggTrade parquet files for a symbol."""
    parts = list_partitions(symbol, data_lake)
    frames: List[pl.DataFrame] = []
    for year, month in parts:
        path = os.path.join(
            data_lake, symbol, str(year), f"{month:02d}", "aggTrades.parquet"
        )
        if os.path.exists(path):
            try:
                frames.append(pl.read_parquet(path))
            except Exception:
                continue
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sort("timestamp").unique(subset=["agg_trade_id"])


# ---------------------------------------------------------------------------
# Convert to PyTorch tensors (for model training)
# ---------------------------------------------------------------------------

def klines_to_tensors(
    df: pl.DataFrame,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tensor]:
    """
    Convert a klines DataFrame to OHLCV tensors.
    Returns {"open": Tensor, "high": Tensor, "low": Tensor, "close": Tensor, "volume": Tensor}
    """
    return {
        "open": torch.tensor(df["open"].to_numpy(), dtype=torch.float32, device=device),
        "high": torch.tensor(df["high"].to_numpy(), dtype=torch.float32, device=device),
        "low": torch.tensor(df["low"].to_numpy(), dtype=torch.float32, device=device),
        "close": torch.tensor(df["close"].to_numpy(), dtype=torch.float32, device=device),
        "volume": torch.tensor(df["volume"].to_numpy(), dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def lake_summary(data_lake: str = DATA_LAKE) -> Dict[str, Dict]:
    """Print summary of data lake contents."""
    symbols: List[str] = list_symbols(data_lake)
    summary: Dict[str, Dict] = {}
    total_size: int = 0

    for sym in symbols:
        sym_dir: str = os.path.join(data_lake, sym)
        files: List[Path] = list(Path(sym_dir).rglob("*.parquet"))
        size: int = sum(f.stat().st_size for f in files)
        total_size += size
        parts: List[Tuple[int, int]] = list_partitions(sym, data_lake)
        summary[sym] = {
            "partitions": len(parts),
            "files": len(files),
            "size_mb": size / 1e6,
        }

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Data Lake Summary")
    print("=" * 60)
    summary = lake_summary()
    total_files: int = 0
    total_size: float = 0
    for sym, info in sorted(summary.items()):
        print(f"  {sym:12s}  {info['partitions']:2d} partitions  "
              f"{info['files']:3d} files  {info['size_mb']:8.2f} MB")
        total_files += info["files"]
        total_size += info["size_mb"]
    print(f"\n  TOTAL: {len(summary)} symbols, {total_files} files, {total_size:.1f} MB")
