"""
Concurrent data pipeline via CCXT sync + ThreadPoolExecutor.

Fetches 30+ crypto pairs × 1 month of 5m klines with pagination.
Stores in SQLite. Uses sync CCXT (which handles geo-restrictions internally).
"""
from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH: str = str(Path(__file__).resolve().parent.parent / "market_data.db")

SYMBOLS_30: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "UNI/USDT", "ATOM/USDT", "LTC/USDT", "FIL/USDT",
    "APT/USDT", "ARB/USDT", "OP/USDT", "NEAR/USDT", "SUI/USDT",
    "SEI/USDT", "INJ/USDT", "RUNE/USDT", "FET/USDT", "AAVE/USDT",
    "MKR/USDT", "PEPE/USDT", "WIF/USDT", "TIA/USDT", "RENDER/USDT",
    "SOL/USDT",  # dupe removed by set below
]

MS_5M: int = 5 * 60 * 1000
THREAD_WORKERS: int = 3


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS klines (
            symbol TEXT, timeframe TEXT, ts_ms INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timeframe, ts_ms))
    """)
    conn.commit()
    return conn


def count_bars(conn: sqlite3.Connection, symbol: str, tf: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM klines WHERE symbol=? AND timeframe=?", (symbol, tf)
    ).fetchone()[0]


def load_bars(conn: sqlite3.Connection, symbol: str, tf: str) -> List[Tuple]:
    return conn.execute(
        "SELECT ts_ms, open, high, low, close, volume "
        "FROM klines WHERE symbol=? AND timeframe=? ORDER BY ts_ms",
        (symbol, tf),
    ).fetchall()


def load_all_from_db(
    db_path: str = DB_PATH, timeframe: str = "5m", min_bars: int = 3000
) -> Dict[str, List[Tuple]]:
    conn = sqlite3.connect(db_path)
    syms = [row[0] for row in conn.execute(
        "SELECT symbol, COUNT(*) as c FROM klines WHERE timeframe=? "
        "GROUP BY symbol HAVING c >= ? ORDER BY c DESC", (timeframe, min_bars)
    ).fetchall()]
    result = {s: load_bars(conn, s, timeframe) for s in syms}
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Paginated fetcher (single symbol, single thread)
# ---------------------------------------------------------------------------

def _fetch_one_symbol(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    now_ms: int,
) -> List[List[Any]]:
    """Paginate forward for one symbol. Called from thread pool."""
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True, "timeout": 30000})
    ex.load_markets()

    if symbol not in ex.markets:
        return []

    all_candles: List[List[Any]] = []
    cursor: int = since_ms

    for _ in range(80):  # max pages
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=300)
        except Exception:
            time.sleep(2.0)
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=300)
            except Exception:
                break

        if not ohlcv:
            break

        all_candles.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        cursor = last_ts + MS_5M

        if cursor >= now_ms or len(ohlcv) < 50:
            break

        time.sleep(0.4)

    # deduplicate + sort
    seen: set = set()
    deduped = [c for c in all_candles if c[0] not in seen and not seen.add(c[0])]
    deduped.sort(key=lambda x: x[0])
    return deduped


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_all_symbols(
    symbols: Optional[List[str]] = None,
    timeframe: str = "5m",
    days_back: int = 30,
    db_path: str = DB_PATH,
    exchange_name: str = "okx",
) -> Dict[str, int]:
    if symbols is None:
        symbols = list(dict.fromkeys(SYMBOLS_30))  # dedupe preserving order

    conn = init_db(db_path)
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - days_back * 86400 * 1000
    results: Dict[str, int] = {}

    # check cache
    to_fetch: List[str] = []
    for sym in symbols:
        clean = sym.replace("/", "")
        existing = count_bars(conn, clean, timeframe)
        if existing >= 5000:
            results[clean] = existing
            print(f"  [CACHE] {clean}: {existing} bars")
        else:
            to_fetch.append(sym)

    print(f"  [FETCH] {len(to_fetch)} symbols to download ({exchange_name}) ...")

    # use ThreadPoolExecutor for concurrent I/O
    def _worker(sym: str) -> Tuple[str, List[List[Any]]]:
        candles = _fetch_one_symbol(exchange_name, sym, timeframe, since_ms, now_ms)
        return sym, candles

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as pool:
        futures = {pool.submit(_worker, sym): sym for sym in to_fetch}
        for future in as_completed(futures):
            sym = futures[future]
            clean = sym.replace("/", "")
            try:
                _, candles = future.result()
            except Exception as e:
                print(f"  [FAIL] {clean}: {e}")
                continue

            if not candles:
                print(f"  [FAIL] {clean}: no data")
                continue

            conn.executemany(
                "INSERT OR IGNORE INTO klines VALUES (?,?,?,?,?,?,?,?)",
                [(clean, timeframe, c[0], c[1], c[2], c[3], c[4], c[5]) for c in candles],
            )
            conn.commit()
            n = count_bars(conn, clean, timeframe)
            results[clean] = n
            print(f"  [OK] {clean}: +{len(candles)} new, {n} total")

    conn.close()
    total = sum(results.values())
    print(f"\n  TOTAL: {len(results)} symbols, {total:,} bars in SQLite")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  Data Pipeline — 30 Crypto × 1 Month × 5m")
    print("=" * 60)
    t0 = time.time()
    results = fetch_all_symbols(timeframe="5m", days_back=30)
    print(f"\n  Completed in {time.time() - t0:.1f}s")
    good = [s for s, n in results.items() if n >= 3000]
    print(f"  Symbols with 3000+ bars: {len(good)}/{len(results)}")
