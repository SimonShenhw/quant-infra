"""
Real funding rate downloader via Binance Vision public archive.
从 Binance Vision 公开归档下载真实资金费率。

Bypasses the fapi.binance.com HTTP 451 geo-block (US/some regions) by using
data.binance.vision static CSV archives — typically not region-blocked.

绕过 fapi.binance.com 的 HTTP 451 地理封锁（美国等地区），使用
data.binance.vision 的静态 CSV 归档 —— 通常不被区域封锁。

URL: https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY}-{MM}.zip
CSV cols: calc_time, funding_interval_hours, last_funding_rate
Cadence: every 8h (3 records per day per symbol).

Stores to funding_rates.db (same schema as data/funding_fetcher.py) so
load_funding() in that module keeps working.
"""
from __future__ import annotations

import io
import sqlite3
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

DB_PATH: str = str(Path(__file__).resolve().parent.parent / "funding_rates.db")
BASE_URL: str = "https://data.binance.vision/data/futures/um/monthly/fundingRate"

SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT", "AAVEUSDT",
]


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding (
            symbol TEXT, ts_ms INTEGER, rate REAL,
            PRIMARY KEY (symbol, ts_ms))
    """)
    conn.commit()
    return conn


def fetch_month(symbol: str, year: int, month: int) -> List[Tuple[int, float]]:
    """Returns [(ts_ms, rate), ...] for one (symbol, year, month). Empty on 404 or parse fail."""
    url = f"{BASE_URL}/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
    except urllib.error.HTTPError:
        return []
    except Exception:
        return []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            if not names:
                return []
            csv = zf.read(names[0]).decode("utf-8")
        lines = csv.strip().split("\n")
        if len(lines) < 2:
            return []
        out: List[Tuple[int, float]] = []
        # header may be "calc_time,funding_interval_hours,last_funding_rate"
        # or older variant; parse by position (ts in col 0, rate in col 2)
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                out.append((int(parts[0]), float(parts[2])))
            except ValueError:
                continue
        return out
    except Exception:
        return []


def generate_year_months(months_back: int) -> List[Tuple[int, int]]:
    now = datetime.now(timezone.utc)
    year_months: List[Tuple[int, int]] = []
    for offset in range(months_back):
        month = now.month - offset
        year = now.year
        while month <= 0:
            month += 12
            year -= 1
        year_months.append((year, month))
    return year_months


def fetch_all(months_back: int = 24) -> Dict[str, int]:
    conn = init_db()
    year_months = generate_year_months(months_back)
    tasks = [(s, y, m) for s in SYMBOLS for y, m in year_months]
    print(f"[Archive] {len(SYMBOLS)} symbols x {len(year_months)} months = {len(tasks)} files")

    results: Dict[str, int] = {s: 0 for s in SYMBOLS}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch_month, s, y, m): (s, y, m) for s, y, m in tasks}
        done = 0
        for f in as_completed(futs):
            sym, _, _ = futs[f]
            pairs = f.result()
            if pairs:
                conn.executemany(
                    "INSERT OR IGNORE INTO funding VALUES (?, ?, ?)",
                    [(sym, ts, r) for ts, r in pairs],
                )
                conn.commit()
                results[sym] += len(pairs)
            done += 1
            if done % 40 == 0 or done == len(tasks):
                total = sum(results.values())
                print(f"  [{done}/{len(tasks)}] {total:,} records")
    conn.close()
    return results


if __name__ == "__main__":
    print("Fetching real funding rates from data.binance.vision archive ...")
    t0 = time.time()
    results = fetch_all(months_back=24)
    total = sum(results.values())
    n_ok = len([s for s, n in results.items() if n > 0])
    print(f"\n[Done] {time.time()-t0:.1f}s, {total:,} records across {n_ok}/{len(results)} symbols")
    for sym in sorted(results.keys()):
        marker = "OK " if results[sym] > 0 else "---"
        print(f"  [{marker}] {sym:12s} {results[sym]:>6} records")
