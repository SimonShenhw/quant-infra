"""
Real funding rate fetcher from Binance Futures public API.
从 Binance 永续合约公开 API 获取真实资金费率。

Binance funding rate updates every 8h. Endpoint:
  https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000
No auth required. Returns list of {fundingTime, fundingRate, symbol}.

WHY funding matters: it is the crowding/carry signal behind the
funding_rate factor and the v13 carry sleeve — one of the few slow
(8h-cadence) signals in crypto. Getting the REAL series (instead of the
OHLCV proxy) was the v11.2 fix for the train/serve mismatch (REVIEW H-4).
为什么 funding 重要：它是 funding_rate 因子与 v13 carry sleeve 背后的
拥挤度/carry 信号，也是 crypto 少数"慢信号"（8 小时更新）之一。用真实序列
替代 OHLCV proxy 正是 v11.2 对训练/推理不一致（REVIEW H-4）的修复。

STATUS — honest scope note: this is the v11.2-era API fetcher, capped at
the API's last ~1000 records per symbol (~1 year at 8h cadence). The
current result path populates the SAME funding_rates.db (same schema) via
data/funding_archive_downloader.py from Binance Vision archives, which
provides the full history; run_v13_final.py and run_paper_daily.py read
the db / CCXT directly and do not import this module. Kept as a quick
no-dependency top-up tool for recent records.
状态——如实说明：这是 v11.2 时期的 API 抓取器，受 API 限制每币种只有最近
约 1000 条（按 8h 频率约一年）。当前结果路径由 data/funding_archive_downloader.py
从 Binance Vision 归档灌同一个 funding_rates.db（同 schema），覆盖完整历史；
run_v13_final.py 与 run_paper_daily.py 直接读库/走 CCXT，并不导入本模块。
保留用作零依赖的近期数据快速补抓工具。

资金费率每8小时更新一次，无需认证。
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

DB_PATH: str = str(Path(__file__).resolve().parent.parent / "funding_rates.db")
BASE_URL: str = "https://fapi.binance.com/fapi/v1/fundingRate"


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding (
            symbol TEXT, ts_ms INTEGER, rate REAL,
            PRIMARY KEY (symbol, ts_ms))
    """)
    conn.commit()
    return conn


def fetch_funding_one(symbol: str, limit: int = 1000) -> List[Tuple[int, float]]:
    """Returns list of (timestamp_ms, funding_rate). / 返回(时间戳, 资金费率)列表。"""
    url = f"{BASE_URL}?symbol={symbol}&limit={limit}"
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())
        return [(int(d["fundingTime"]), float(d["fundingRate"])) for d in data]
    except Exception as e:
        print(f"  [FAIL] {symbol}: {e}")
        return []


def fetch_funding_all(symbols: List[str]) -> Dict[str, int]:
    """Fetch funding rates for all symbols, store in SQLite. / 批量获取并入库。"""
    conn = init_db()
    results: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_funding_one, s): s for s in symbols}
        for f in as_completed(futures):
            sym = futures[f]
            rates = f.result()
            if rates:
                conn.executemany(
                    "INSERT OR IGNORE INTO funding VALUES (?, ?, ?)",
                    [(sym, ts, r) for ts, r in rates],
                )
                conn.commit()
                results[sym] = len(rates)
                print(f"  [OK] {sym}: {len(rates)} funding records")
    conn.close()
    return results


def load_funding(symbol: str) -> List[Tuple[int, float]]:
    """Load all funding records for a symbol. / 加载某币种全部资金费率。"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ts_ms, rate FROM funding WHERE symbol=? ORDER BY ts_ms", (symbol,)
    ).fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
        "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT",
        "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT", "AAVEUSDT",
    ]
    print(f"Fetching funding rates for {len(SYMBOLS)} symbols ...")
    t0 = time.time()
    results = fetch_funding_all(SYMBOLS)
    print(f"Done in {time.time()-t0:.1f}s, {sum(results.values()):,} records")
