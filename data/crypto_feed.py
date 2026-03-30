"""
Multi-asset Crypto data feed via CCXT.

Fetches historical OHLCV for multiple symbols in parallel,
returns a dict of {symbol: List[MarketEvent]}.
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from engine.events import EventType, MarketEvent

# Top crypto pairs by liquidity
DEFAULT_SYMBOLS: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT",
    "LINK/USDT", "DOT/USDT",
]


def fetch_multi_asset(
    symbols: Optional[List[str]] = None,
    timeframe: str = "15m",
    limit: int = 300,
    exchange_name: str = "okx",
) -> Dict[str, List[MarketEvent]]:
    """
    Fetch OHLCV for multiple symbols from a single exchange.
    Returns {symbol_clean: [MarketEvent, ...]}.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    exchanges_to_try: List[str] = [exchange_name, "bybit", "gate", "mexc"]
    ex: Any = None

    for name in exchanges_to_try:
        try:
            ex_cls = getattr(ccxt, name)
            ex = ex_cls({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()
            print(f"[CryptoFeed] Connected to {name}")
            break
        except Exception as e:
            print(f"[CryptoFeed] {name} failed: {e}")
            continue

    if ex is None:
        raise RuntimeError("All exchanges failed")

    result: Dict[str, List[MarketEvent]] = {}

    for sym in symbols:
        if sym not in ex.markets:
            # try futures variant
            alt: str = sym + ":USDT"
            if alt not in ex.markets:
                print(f"  -> {sym}: not found, skipping")
                continue
            sym = alt

        try:
            ohlcv: List[Any] = ex.fetch_ohlcv(sym, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 50:
                print(f"  -> {sym}: insufficient data ({len(ohlcv) if ohlcv else 0} bars)")
                continue

            clean_sym: str = sym.replace("/", "").replace(":", "_").upper()
            bars: List[MarketEvent] = []
            for k in ohlcv:
                ts: datetime = datetime.fromtimestamp(k[0] / 1000)
                bars.append(MarketEvent(
                    event_type=EventType.MARKET,
                    timestamp=ts,
                    symbol=clean_sym,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                ))
            result[clean_sym] = bars
            print(f"  -> {clean_sym}: {len(bars)} bars "
                  f"[{bars[0].close:.2f} -> {bars[-1].close:.2f}]")
            time.sleep(0.3)  # rate limit

        except Exception as e:
            print(f"  -> {sym}: error {e}")
            continue

    print(f"[CryptoFeed] Loaded {len(result)} assets, "
          f"{sum(len(v) for v in result.values())} total bars")
    return result


if __name__ == "__main__":
    data = fetch_multi_asset(timeframe="15m", limit=300)
    for sym, bars in data.items():
        print(f"  {sym}: {len(bars)} bars")
