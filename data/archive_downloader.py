"""
Task 1: Binance Public Data Archive Downloader.

Downloads historical AggTrades and Klines from data.binance.vision
(Binance's official S3-hosted public data archive).

URL pattern:
  https://data.binance.vision/data/spot/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY}-{MM}.zip
  https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY}-{MM}.zip

Features:
  - asyncio + aiohttp high-concurrency downloads
  - Semaphore-based rate limiting
  - Auto-extract ZIP → CSV → Polars → Parquet
  - Partitioned storage: data_lake/{asset}/{year}/{month}/
"""
from __future__ import annotations

import asyncio
import io
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp
import polars as pl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL: str = "https://data.binance.vision/data/spot/monthly"
DATA_LAKE: str = str(Path(__file__).resolve().parent.parent / "data_lake")

TOP_20_SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT", "AAVEUSDT",
]

# AggTrade CSV columns (per Binance documentation)
AGGTRADE_COLS: List[str] = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "timestamp", "is_buyer_maker", "is_best_match",
]
AGGTRADE_DTYPES: dict = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "timestamp": pl.Int64,
    "is_buyer_maker": pl.Boolean,
    "is_best_match": pl.Boolean,
}

KLINE_COLS: List[str] = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]

SEMAPHORE_LIMIT: int = 5
CHUNK_ROWS: int = 500_000  # process CSV in chunks to avoid OOM


# ---------------------------------------------------------------------------
# Download + extract + convert single archive
# ---------------------------------------------------------------------------

async def download_and_convert(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    url: str,
    symbol: str,
    year: int,
    month: int,
    data_type: str,  # "aggTrades" or "klines"
    interval: str = "5m",
) -> Tuple[str, int]:
    """
    Download one ZIP from Binance archive, extract CSV, convert to Parquet.
    Returns (parquet_path, n_rows).
    """
    # output path
    part_dir: str = os.path.join(DATA_LAKE, symbol, str(year), f"{month:02d}")
    os.makedirs(part_dir, exist_ok=True)
    out_name: str = f"{data_type}_{interval}.parquet" if data_type == "klines" else f"{data_type}.parquet"
    parquet_path: str = os.path.join(part_dir, out_name)

    # skip if already exists
    if os.path.exists(parquet_path):
        try:
            existing: pl.DataFrame = pl.read_parquet(parquet_path)
            return parquet_path, existing.height
        except Exception:
            pass  # corrupted, re-download

    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 404:
                    return "", 0
                if resp.status != 200:
                    return "", 0
                data: bytes = await resp.read()
        except Exception as e:
            print(f"    [DL ERROR] {url}: {e}")
            return "", 0

    # extract CSV from ZIP
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_names: List[str] = zf.namelist()
            if not csv_names:
                return "", 0
            csv_data: bytes = zf.read(csv_names[0])
    except Exception as e:
        print(f"    [ZIP ERROR] {url}: {e}")
        return "", 0

    # parse CSV with Polars (streaming for large files)
    try:
        if data_type == "aggTrades":
            df: pl.DataFrame = pl.read_csv(
                io.BytesIO(csv_data),
                has_header=False,
                new_columns=AGGTRADE_COLS,
                schema_overrides=AGGTRADE_DTYPES,
                n_threads=2,
            )
            # add derived columns
            df = df.with_columns([
                (pl.col("timestamp") * 1000).cast(pl.Datetime("us")).alias("datetime"),
                (pl.col("price") * pl.col("quantity")).alias("notional"),
            ])
        elif data_type == "klines":
            df = pl.read_csv(
                io.BytesIO(csv_data),
                has_header=False,
                new_columns=KLINE_COLS,
                n_threads=2,
            )
            # cast types
            for col in ["open", "high", "low", "close", "volume",
                        "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            df = df.with_columns([
                pl.col("open_time").cast(pl.Int64),
                pl.col("close_time").cast(pl.Int64),
                pl.col("count").cast(pl.Int64),
            ])
        else:
            return "", 0

        # write Parquet with snappy compression
        df.write_parquet(parquet_path, compression="snappy")
        return parquet_path, df.height

    except Exception as e:
        print(f"    [PARSE ERROR] {url}: {e}")
        return "", 0


# ---------------------------------------------------------------------------
# Orchestrator: download all symbols × months
# ---------------------------------------------------------------------------

async def download_archive(
    symbols: Optional[List[str]] = None,
    data_type: str = "klines",
    interval: str = "5m",
    months_back: int = 6,
) -> dict:
    """
    Download historical data for all symbols × months.
    Returns {symbol: total_rows}.
    """
    if symbols is None:
        symbols = TOP_20_SYMBOLS

    # generate (year, month) list going back N months
    now: datetime = datetime.now()
    year_months: List[Tuple[int, int]] = []
    for m_offset in range(1, months_back + 1):
        month: int = now.month - m_offset
        year: int = now.year
        while month <= 0:
            month += 12
            year -= 1
        year_months.append((year, month))

    print(f"[Archive] {len(symbols)} symbols × {len(year_months)} months "
          f"= {len(symbols) * len(year_months)} archives")
    print(f"[Archive] Type: {data_type}, Interval: {interval}")
    print(f"[Archive] Target: {DATA_LAKE}")

    sem: asyncio.Semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    results: dict = {s: 0 for s in symbols}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sym in symbols:
            for year, month in year_months:
                if data_type == "klines":
                    url = (f"{BASE_URL}/{data_type}/{sym}/{interval}/"
                           f"{sym}-{interval}-{year}-{month:02d}.zip")
                else:
                    url = (f"{BASE_URL}/{data_type}/{sym}/"
                           f"{sym}-{data_type}-{year}-{month:02d}.zip")
                tasks.append((sym, year, month, url))

        # process in batches to control memory
        batch_size: int = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            coros = [
                download_and_convert(
                    session, sem, url, sym, yr, mo, data_type, interval
                )
                for sym, yr, mo, url in batch
            ]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)

            for (sym, yr, mo, url), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"  [FAIL] {sym}/{yr}-{mo:02d}: {result}")
                    continue
                path, n_rows = result
                if n_rows > 0:
                    results[sym] += n_rows

            # progress
            done: int = min(i + batch_size, len(tasks))
            total_rows: int = sum(results.values())
            print(f"  [{done}/{len(tasks)}] {total_rows:,} rows total")

    # summary
    print(f"\n[Archive] Download complete:")
    for sym in sorted(results.keys()):
        if results[sym] > 0:
            print(f"  {sym}: {results[sym]:,} rows")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 65)
    print("  Binance Archive Downloader → Parquet Data Lake")
    print("=" * 65)
    import time
    t0 = time.time()

    # Download 5m klines for 6 months
    kline_results = await download_archive(
        data_type="klines", interval="5m", months_back=6
    )
    total_klines = sum(kline_results.values())
    print(f"\n  Total kline rows: {total_klines:,}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # Verify data lake structure
    lake = Path(DATA_LAKE)
    parquet_files = list(lake.rglob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    print(f"\n  Parquet files: {len(parquet_files)}")
    print(f"  Total size: {total_size / 1e6:.1f} MB")


if __name__ == "__main__":
    asyncio.run(main())
