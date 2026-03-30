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

任务3：Parquet数据湖加载器。

提供统一接口，从分区Parquet数据湖加载数据到Polars DataFrame或PyTorch张量。

数据湖结构：
  data_lake/{SYMBOL}/{YEAR}/{MONTH}/ 下含klines、aggTrades及realtime子目录。
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
# Discovery / 数据发现
# ---------------------------------------------------------------------------

def list_symbols(data_lake: str = DATA_LAKE) -> List[str]:
    """
    List all symbols with data in the lake.
    列出数据湖中所有有数据的交易对。
    """
    if not os.path.exists(data_lake):
        return []
    return sorted([
        d for d in os.listdir(data_lake)
        if os.path.isdir(os.path.join(data_lake, d))
    ])


def list_partitions(
    symbol: str, data_lake: str = DATA_LAKE
) -> List[Tuple[int, int]]:
    """
    List (year, month) partitions for a symbol.
    列出某交易对的所有(年, 月)分区。
    """
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
# Loading: Klines / 加载K线数据
# ---------------------------------------------------------------------------

def load_klines(
    symbol: str,
    interval: str = "5m",
    data_lake: str = DATA_LAKE,
) -> pl.DataFrame:
    """
    Load all kline parquet files for a symbol, sorted by time.
    加载某交易对所有K线Parquet文件，按时间排序。
    """
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
    # normalize schemas: keep only common OHLCV columns / 统一schema：仅保留通用OHLCV列
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
    """
    Load klines for multiple symbols, filtering by minimum rows.
    加载多个交易对的K线数据，按最小行数过滤。
    """
    if symbols is None:
        symbols = list_symbols(data_lake)
    result: Dict[str, pl.DataFrame] = {}
    for sym in symbols:
        df: pl.DataFrame = load_klines(sym, interval, data_lake)
        if df.height >= min_rows:
            result[sym] = df
    return result


# ---------------------------------------------------------------------------
# Loading: AggTrades / 加载聚合成交数据
# ---------------------------------------------------------------------------

def load_aggtrades(
    symbol: str,
    data_lake: str = DATA_LAKE,
) -> pl.DataFrame:
    """
    Load all aggTrade parquet files for a symbol.
    加载某交易对所有aggTrade Parquet文件。
    """
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
# Convert to PyTorch tensors (for model training) / 转换为PyTorch张量（用于模型训练）
# ---------------------------------------------------------------------------

def klines_to_tensors(
    df: pl.DataFrame,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tensor]:
    """
    Convert a klines DataFrame to OHLCV tensors.
    Returns {"open": Tensor, "high": Tensor, "low": Tensor, "close": Tensor, "volume": Tensor}

    将K线DataFrame转换为OHLCV张量。
    返回 {"open": Tensor, "high": Tensor, ...} 字典。
    """
    return {
        "open": torch.tensor(df["open"].to_numpy(), dtype=torch.float32, device=device),
        "high": torch.tensor(df["high"].to_numpy(), dtype=torch.float32, device=device),
        "low": torch.tensor(df["low"].to_numpy(), dtype=torch.float32, device=device),
        "close": torch.tensor(df["close"].to_numpy(), dtype=torch.float32, device=device),
        "volume": torch.tensor(df["volume"].to_numpy(), dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Summary / 汇总
# ---------------------------------------------------------------------------

def lake_summary(data_lake: str = DATA_LAKE) -> Dict[str, Dict]:
    """
    Print summary of data lake contents.
    输出数据湖内容摘要。
    """
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
# CLI / 命令行入口
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
