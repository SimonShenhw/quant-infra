"""
Task 2: WebSocket Daemon — Real-time LOB/Trade Stream Listener.

Async daemon that connects to Binance WebSocket streams and writes
incoming data to Parquet files in the data lake.

Features:
  - Hardcoded Ping/Pong heartbeat (20s interval)
  - Exponential backoff reconnection (1s → 2s → 4s → ... → 60s cap)
  - In-memory buffer → periodic Parquet flush (every N rows or M seconds)
  - Graceful shutdown on SIGINT/SIGTERM
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False

DATA_LAKE: str = str(Path(__file__).resolve().parent.parent / "data_lake")

# Binance combined stream
WS_BASE: str = "wss://stream.binance.com:9443"

# Buffer config
FLUSH_INTERVAL_SEC: float = 30.0
FLUSH_ROWS: int = 10_000
MAX_BACKOFF_SEC: float = 60.0


class WebSocketDaemon:
    """
    Async daemon that listens to Binance WebSocket streams,
    buffers data in memory, and flushes to Parquet periodically.
    """

    def __init__(
        self,
        symbols: List[str],
        streams: List[str],
        output_dir: str = DATA_LAKE,
    ) -> None:
        self._symbols: List[str] = [s.lower() for s in symbols]
        self._streams: List[str] = streams
        self._output_dir: str = output_dir
        self._running: bool = False

        # data buffers
        self._trade_buffer: List[Dict[str, Any]] = []
        self._depth_buffer: List[Dict[str, Any]] = []
        self._last_flush: float = time.time()

        # reconnection state
        self._backoff: float = 1.0
        self._connect_count: int = 0

    def _build_url(self) -> str:
        stream_names: List[str] = []
        for sym in self._symbols:
            for stream in self._streams:
                stream_names.append(f"{sym}@{stream}")
        return f"{WS_BASE}/stream?streams={'/'.join(stream_names)}"

    async def _flush_to_parquet(self) -> None:
        """Flush in-memory buffers to partitioned Parquet files."""
        now: datetime = datetime.now()
        year: str = str(now.year)
        month: str = f"{now.month:02d}"
        day: str = f"{now.day:02d}"
        ts_suffix: str = now.strftime("%H%M%S")

        if self._trade_buffer:
            df: pl.DataFrame = pl.DataFrame(self._trade_buffer)
            for sym in df["symbol"].unique().to_list():
                sym_df: pl.DataFrame = df.filter(pl.col("symbol") == sym)
                part_dir: str = os.path.join(
                    self._output_dir, sym.upper(), year, month, "realtime"
                )
                os.makedirs(part_dir, exist_ok=True)
                path: str = os.path.join(part_dir, f"trades_{day}_{ts_suffix}.parquet")
                sym_df.write_parquet(path, compression="snappy")
            n_trades: int = len(self._trade_buffer)
            self._trade_buffer.clear()
            print(f"  [FLUSH] {n_trades} trades → parquet")

        if self._depth_buffer:
            df = pl.DataFrame(self._depth_buffer)
            for sym in df["symbol"].unique().to_list():
                sym_df = df.filter(pl.col("symbol") == sym)
                part_dir = os.path.join(
                    self._output_dir, sym.upper(), year, month, "realtime"
                )
                os.makedirs(part_dir, exist_ok=True)
                path = os.path.join(part_dir, f"depth_{day}_{ts_suffix}.parquet")
                sym_df.write_parquet(path, compression="snappy")
            n_depth: int = len(self._depth_buffer)
            self._depth_buffer.clear()
            print(f"  [FLUSH] {n_depth} depth snapshots → parquet")

        self._last_flush = time.time()

    def _process_message(self, raw: str) -> None:
        """Parse WebSocket message and append to buffer."""
        data: Dict[str, Any] = json.loads(raw)
        stream: str = data.get("stream", "")
        payload: Dict[str, Any] = data.get("data", {})

        if "aggTrade" in stream or "trade" in stream:
            self._trade_buffer.append({
                "symbol": payload.get("s", ""),
                "timestamp": payload.get("T", 0),
                "price": float(payload.get("p", 0)),
                "quantity": float(payload.get("q", 0)),
                "is_buyer_maker": payload.get("m", False),
            })
        elif "depth" in stream:
            bids: List = payload.get("bids", [])[:5]
            asks: List = payload.get("asks", [])[:5]
            row: Dict[str, Any] = {
                "symbol": stream.split("@")[0].upper(),
                "timestamp": int(time.time() * 1000),
            }
            for i in range(5):
                bp = float(bids[i][0]) if i < len(bids) else 0.0
                bv = float(bids[i][1]) if i < len(bids) else 0.0
                ap = float(asks[i][0]) if i < len(asks) else 0.0
                av = float(asks[i][1]) if i < len(asks) else 0.0
                row[f"bid_p{i+1}"] = bp
                row[f"bid_v{i+1}"] = bv
                row[f"ask_p{i+1}"] = ap
                row[f"ask_v{i+1}"] = av
            self._depth_buffer.append(row)

    async def _connect_and_listen(self) -> None:
        """Single connection attempt with heartbeat."""
        url: str = self._build_url()
        self._connect_count += 1
        print(f"  [WS] Connecting (attempt #{self._connect_count}) ...")

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
            max_size=2 ** 22,  # 4MB max message
        ) as ws:
            self._backoff = 1.0  # reset backoff on success
            print(f"  [WS] Connected to {len(self._symbols)} streams")

            while self._running:
                try:
                    msg: str = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    self._process_message(msg)
                except asyncio.TimeoutError:
                    # no message in 30s, connection likely dead
                    break

                # periodic flush
                buf_size: int = len(self._trade_buffer) + len(self._depth_buffer)
                elapsed: float = time.time() - self._last_flush
                if buf_size >= FLUSH_ROWS or elapsed >= FLUSH_INTERVAL_SEC:
                    await self._flush_to_parquet()

    async def run(self, duration_seconds: float = 3600.0) -> None:
        """
        Main loop with exponential backoff reconnection.
        Runs for `duration_seconds` then gracefully shuts down.
        """
        if not HAS_WS:
            print("[WS] websockets not installed, daemon cannot start")
            return

        self._running = True
        t0: float = time.time()

        # setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: setattr(self, '_running', False))
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        print(f"[WS Daemon] Starting for {duration_seconds}s ...")

        while self._running and (time.time() - t0) < duration_seconds:
            try:
                await self._connect_and_listen()
            except websockets.exceptions.InvalidStatus as e:
                print(f"  [WS] Rejected: {e}")
            except Exception as e:
                print(f"  [WS] Error: {e}")

            if not self._running:
                break

            # exponential backoff
            print(f"  [WS] Reconnecting in {self._backoff:.1f}s ...")
            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, MAX_BACKOFF_SEC)

        # final flush
        await self._flush_to_parquet()
        print(f"[WS Daemon] Stopped. Total connects: {self._connect_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main() -> None:
    symbols: List[str] = ["btcusdt", "ethusdt", "solusdt"]
    daemon = WebSocketDaemon(
        symbols=symbols,
        streams=["aggTrade", "depth5@100ms"],
        output_dir=DATA_LAKE,
    )
    await daemon.run(duration_seconds=30)  # 30s test run


if __name__ == "__main__":
    asyncio.run(main())
