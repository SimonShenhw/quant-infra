"""
Binance WebSocket — Async BTC/USDT Tick + LOB Data Collector.

Connects to Binance spot market WebSocket streams and collects:
  - aggTrade: tick-by-tick aggregated trades
  - depth20@100ms: 20-level order book snapshots
  - kline_1m: 1-minute OHLCV bars

Saves data as CSV files in the project directory for OOS testing.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL: str = "btcusdt"
BASE_WS: str = "wss://stream.binance.com:9443"
STREAMS: str = f"{SYMBOL}@aggTrade/{SYMBOL}@depth20@100ms/{SYMBOL}@kline_1m"
COMBINED_URL: str = f"{BASE_WS}/stream?streams={STREAMS}"

# REST API fallback for historical klines
REST_BASE: str = "https://api.binance.com"
KLINES_URL: str = f"{REST_BASE}/api/v3/klines"


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------

class BinanceTickCollector:
    """
    Async collector for BTC/USDT market data from Binance.

    Usage:
        collector = BinanceTickCollector(output_dir="./data", duration_seconds=7200)
        asyncio.run(collector.run())
    """

    def __init__(
        self,
        output_dir: str = ".",
        duration_seconds: int = 7200,
    ) -> None:
        self._output_dir: str = output_dir
        self._duration: int = duration_seconds
        os.makedirs(output_dir, exist_ok=True)

        ts: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trades_file: str = os.path.join(output_dir, f"btcusdt_trades_{ts}.csv")
        self._depth_file: str = os.path.join(output_dir, f"btcusdt_depth_{ts}.csv")
        self._klines_file: str = os.path.join(output_dir, f"btcusdt_klines_{ts}.csv")

        self._trade_count: int = 0
        self._depth_count: int = 0
        self._kline_count: int = 0

    async def run(self) -> Dict[str, str]:
        """Run the WebSocket collector for the specified duration."""
        if not HAS_WEBSOCKETS:
            print("[BinanceWS] websockets not installed, using REST fallback")
            return await self._rest_fallback()

        print(f"[BinanceWS] Connecting to {COMBINED_URL}")
        print(f"[BinanceWS] Collecting for {self._duration}s ...")

        # open CSV writers
        trades_fp = open(self._trades_file, "w", newline="")
        depth_fp = open(self._depth_file, "w", newline="")
        klines_fp = open(self._klines_file, "w", newline="")

        tw = csv.writer(trades_fp)
        tw.writerow(["timestamp", "price", "quantity", "is_buyer_maker"])

        dw = csv.writer(depth_fp)
        depth_header: List[str] = ["timestamp"]
        for i in range(1, 21):
            depth_header.extend([f"bid_p{i}", f"bid_v{i}", f"ask_p{i}", f"ask_v{i}"])
        dw.writerow(depth_header)

        kw = csv.writer(klines_fp)
        kw.writerow(["timestamp", "open", "high", "low", "close", "volume"])

        t0: float = time.time()
        try:
            async with websockets.connect(COMBINED_URL, ping_interval=20) as ws:
                while time.time() - t0 < self._duration:
                    try:
                        msg: str = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue

                    data: Dict[str, Any] = json.loads(msg)
                    stream: str = data.get("stream", "")
                    payload: Dict[str, Any] = data.get("data", {})

                    if "aggTrade" in stream:
                        tw.writerow([
                            payload["T"],
                            payload["p"],
                            payload["q"],
                            payload["m"],
                        ])
                        self._trade_count += 1

                    elif "depth20" in stream:
                        row: List[Any] = [int(time.time() * 1000)]
                        bids: List[List[str]] = payload.get("bids", [])
                        asks: List[List[str]] = payload.get("asks", [])
                        for i in range(20):
                            bp = bids[i][0] if i < len(bids) else "0"
                            bv = bids[i][1] if i < len(bids) else "0"
                            ap = asks[i][0] if i < len(asks) else "0"
                            av = asks[i][1] if i < len(asks) else "0"
                            row.extend([bp, bv, ap, av])
                        dw.writerow(row)
                        self._depth_count += 1

                    elif "kline" in stream:
                        k: Dict[str, Any] = payload.get("k", {})
                        if k.get("x", False):
                            kw.writerow([
                                k["T"], k["o"], k["h"], k["l"], k["c"], k["v"],
                            ])
                            self._kline_count += 1

                    elapsed: float = time.time() - t0
                    if self._trade_count % 5000 == 0 and self._trade_count > 0:
                        print(f"  [{elapsed:.0f}s] trades={self._trade_count} "
                              f"depth={self._depth_count} klines={self._kline_count}")

        except Exception as e:
            print(f"[BinanceWS] Error: {e}")
            print("[BinanceWS] Falling back to REST API")
            return await self._rest_fallback()
        finally:
            trades_fp.close()
            depth_fp.close()
            klines_fp.close()

        print(f"[BinanceWS] Done: {self._trade_count} trades, "
              f"{self._depth_count} depth, {self._kline_count} klines")

        return {
            "trades": self._trades_file,
            "depth": self._depth_file,
            "klines": self._klines_file,
        }

    async def _rest_fallback(self) -> Dict[str, str]:
        """Fetch recent klines via REST API when WebSocket is unavailable."""
        print("[BinanceWS] Using REST API to fetch recent BTC/USDT 1m klines")

        if HAS_AIOHTTP:
            async with aiohttp.ClientSession() as session:
                params: Dict[str, Any] = {
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "limit": 120,  # 2 hours of 1m bars
                }
                async with session.get(KLINES_URL, params=params) as resp:
                    raw: List[Any] = await resp.json()
        else:
            import urllib.request
            import urllib.parse
            params_str: str = urllib.parse.urlencode({
                "symbol": "BTCUSDT",
                "interval": "1m",
                "limit": 120,
            })
            url: str = f"{KLINES_URL}?{params_str}"
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = json.loads(response.read())

        klines_fp = open(self._klines_file, "w", newline="")
        kw = csv.writer(klines_fp)
        kw.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for k in raw:
            kw.writerow([k[6], k[1], k[2], k[3], k[4], k[5]])
        klines_fp.close()

        print(f"[BinanceWS] Fetched {len(raw)} klines via REST")
        return {"trades": "", "depth": "", "klines": self._klines_file}


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    output_dir: str = os.path.dirname(os.path.abspath(__file__))
    collector: BinanceTickCollector = BinanceTickCollector(
        output_dir=output_dir,
        duration_seconds=30,  # quick test: 30 seconds
    )
    files: Dict[str, str] = await collector.run()
    for k, v in files.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
