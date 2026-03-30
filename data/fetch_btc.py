"""
Fetch real BTC/USDT historical data via CCXT (exchange-agnostic).

Falls back through multiple exchanges to handle geo-restrictions.
Saves 1m OHLCV data as CSV for out-of-sample testing.

通过CCXT获取真实BTC/USDT历史数据（交易所无关）。

依次尝试多个交易所以处理地域限制。
将1分钟OHLCV数据保存为CSV，用于样本外测试。
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import ccxt


def fetch_btc_klines(
    output_dir: str = ".",
    timeframe: str = "1m",
    limit: int = 2000,
) -> str:
    """
    Fetch BTC/USDT 1m klines from the first available exchange.
    Returns path to the saved CSV file.

    从首个可用交易所获取BTC/USDT 1分钟K线。
    返回已保存CSV文件的路径。
    """
    # Try multiple exchanges in order of reliability for China users / 按中国用户可靠性排序尝试多个交易所
    exchanges: List[str] = ["okx", "bybit", "gate", "mexc", "kucoin", "binance"]

    for name in exchanges:
        try:
            print(f"[FetchBTC] Trying {name} ...")
            exchange_class = getattr(ccxt, name)
            ex = exchange_class({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()

            symbol: str = "BTC/USDT"
            if symbol not in ex.markets:
                symbol = "BTC/USDT:USDT"
                if symbol not in ex.markets:
                    print(f"  -> {name}: BTC/USDT not found")
                    continue

            # fetch recent klines / 获取近期K线
            since: Optional[int] = None
            all_klines: List[Any] = []

            # paginate to get more data / 分页获取更多数据
            all_klines = []
            fetch_since: Optional[int] = None
            max_pages: int = max(1, limit // 300)
            for page in range(max_pages):
                ohlcv: List[Any] = ex.fetch_ohlcv(
                    symbol, timeframe, since=fetch_since, limit=min(limit, 300)
                )
                if not ohlcv:
                    break
                all_klines.extend(ohlcv)
                fetch_since = ohlcv[-1][0] + 1
                if len(ohlcv) < 100:
                    break
                time.sleep(0.5)  # rate limit / 速率限制

            if not all_klines:
                print(f"  -> {name}: no data returned")
                continue
            print(f"  -> {name}: fetched {len(all_klines)} bars")

            # save to CSV / 保存为CSV
            ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath: str = os.path.join(output_dir, f"btcusdt_{name}_{timeframe}_{ts}.csv")
            with open(filepath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
                for k in all_klines:
                    w.writerow([k[0], k[1], k[2], k[3], k[4], k[5]])

            print(f"  -> Saved to {filepath}")
            print(f"  -> Time range: {datetime.fromtimestamp(all_klines[0][0]/1000)} "
                  f"-> {datetime.fromtimestamp(all_klines[-1][0]/1000)}")
            print(f"  -> Price range: {all_klines[0][1]:.2f} -> {all_klines[-1][4]:.2f}")
            return filepath

        except Exception as e:
            print(f"  -> {name} failed: {e}")
            continue

    raise RuntimeError("All exchanges failed. Check network connectivity.")


if __name__ == "__main__":
    output_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."
    )
    filepath: str = fetch_btc_klines(output_dir=output_dir, timeframe="1m", limit=2000)
    print(f"\nDone: {filepath}")
