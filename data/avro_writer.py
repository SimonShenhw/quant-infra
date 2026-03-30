"""
Avro serialization for real-time streaming data.
Avro 实时流数据序列化。

Row-oriented format ideal for append-only streaming writes (vs Parquet for batch reads).
行式格式，适合追加写入的流式场景（Parquet 适合批量读取）。
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fastavro

# ---------------------------------------------------------------------------
# Schemas / 数据模式
# ---------------------------------------------------------------------------

TRADE_SCHEMA: Dict[str, Any] = {
    "type": "record",
    "name": "Trade",
    "fields": [
        {"name": "symbol", "type": "string"},
        {"name": "timestamp", "type": "long"},
        {"name": "price", "type": "double"},
        {"name": "quantity", "type": "double"},
        {"name": "is_buyer_maker", "type": "boolean"},
    ],
}

DEPTH_SCHEMA: Dict[str, Any] = {
    "type": "record",
    "name": "DepthSnapshot",
    "fields": [
        {"name": "symbol", "type": "string"},
        {"name": "timestamp", "type": "long"},
        *[{"name": f"bid_p{i}", "type": "double"} for i in range(1, 6)],
        *[{"name": f"bid_v{i}", "type": "double"} for i in range(1, 6)],
        *[{"name": f"ask_p{i}", "type": "double"} for i in range(1, 6)],
        *[{"name": f"ask_v{i}", "type": "double"} for i in range(1, 6)],
    ],
}

PARSED_TRADE = fastavro.parse_schema(TRADE_SCHEMA)
PARSED_DEPTH = fastavro.parse_schema(DEPTH_SCHEMA)


# ---------------------------------------------------------------------------
# Writer / 写入器
# ---------------------------------------------------------------------------

class AvroStreamWriter:
    """
    Append-only Avro writer for streaming data.
    追加写入的 Avro 流式数据写入器。

    Writes to: data_lake/{symbol}/{year}/{month}/realtime/{type}_{day}.avro
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir: str = base_dir
        self._trade_buffer: List[Dict] = []
        self._depth_buffer: List[Dict] = []

    def append_trade(self, record: Dict) -> None:
        """Buffer a trade record. / 缓冲一条成交记录。"""
        self._trade_buffer.append(record)

    def append_depth(self, record: Dict) -> None:
        """Buffer a depth snapshot. / 缓冲一条深度快照。"""
        self._depth_buffer.append(record)

    def flush(self) -> int:
        """Flush buffers to Avro files. Returns total records written. / 刷写缓冲区到Avro文件。"""
        total = 0
        now = datetime.now()

        if self._trade_buffer:
            path = self._make_path(self._trade_buffer[0].get("symbol", "UNKNOWN"), now, "trades")
            self._write_avro(path, PARSED_TRADE, self._trade_buffer)
            total += len(self._trade_buffer)
            self._trade_buffer.clear()

        if self._depth_buffer:
            path = self._make_path(self._depth_buffer[0].get("symbol", "UNKNOWN"), now, "depth")
            self._write_avro(path, PARSED_DEPTH, self._depth_buffer)
            total += len(self._depth_buffer)
            self._depth_buffer.clear()

        return total

    def _make_path(self, symbol: str, now: datetime, data_type: str) -> str:
        dir_path = os.path.join(
            self._base_dir, symbol.upper(),
            str(now.year), f"{now.month:02d}", "realtime",
        )
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"{data_type}_{now.day:02d}.avro")

    @staticmethod
    def _write_avro(path: str, schema: Dict, records: List[Dict]) -> None:
        """Append records to an Avro file (create if not exists). / 追加记录到Avro文件。"""
        # read existing if file exists / 如文件存在则读取已有内容
        existing: List[Dict] = []
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    existing = list(fastavro.reader(f))
            except Exception:
                pass

        with open(path, "wb") as f:
            fastavro.writer(f, schema, existing + records)
