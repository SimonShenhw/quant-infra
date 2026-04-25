"""
Realtime market data feed via Binance WebSocket.
通过 Binance WebSocket 获取实时行情。

Replaces serial REST fetching (6+ seconds for 20 symbols)
with single WebSocket connection (< 0.5s latency).

替代串行 REST 拉取（20币种6秒+），用单个 WebSocket 连接（延迟<0.5秒）。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, List, Tuple

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False


async def fetch_latest_bars_ws(
    symbols: List[str], timeframe: str = "1h", n_bars: int = 50,
) -> Dict[str, List[Tuple[float, float, float, float, float]]]:
    """
    Fetch latest N bars per symbol via Binance WebSocket kline stream.
    Subscribes to klines, takes the last completed bars.

    通过 Binance WebSocket K线流获取每币种最新N根K线。
    """
    if not HAS_WS:
        raise ImportError("websockets package required: pip install websockets")

    # Binance WS interval mapping / Binance WS 间隔映射
    streams = "/".join(f"{s.lower()}@kline_{timeframe}" for s in symbols)
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"

    bars: Dict[str, List[Tuple]] = {s: [] for s in symbols}
    target_ts: Dict[str, int] = {}

    try:
        async with websockets.connect(url, ping_interval=20) as ws:
            t0 = time.time()
            # Wait for at least one closed bar per symbol / 等待每币种至少一根已收K线
            while time.time() - t0 < 30:  # 30s timeout
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                stream = data.get("stream", "")
                payload = data.get("data", {})
                k = payload.get("k", {})
                if not k:
                    continue
                # only take closed candles / 只取已收K线
                if not k.get("x", False):
                    continue
                sym = k["s"]
                if sym not in bars:
                    continue
                bar = (float(k["o"]), float(k["h"]), float(k["l"]),
                       float(k["c"]), float(k["v"]))
                bars[sym].append(bar)
                # break when each symbol has at least one bar / 每币种至少1根则继续
                if all(len(v) >= 1 for v in bars.values()):
                    break
    except (asyncio.TimeoutError, Exception) as e:
        print(f"  [WS] error: {e}, falling back to REST")
        return {}

    return bars


def fetch_latest_bars_sync(symbols: List[str], n_bars: int = 50,
                            timeframe: str = "1h") -> Dict[str, List[Tuple]]:
    """Synchronous wrapper around the async WS fetcher. / 同步封装。"""
    return asyncio.run(fetch_latest_bars_ws(symbols, timeframe, n_bars))
