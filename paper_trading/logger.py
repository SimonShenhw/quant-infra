"""
Paper Trading SQLite Logger.
模拟盘 SQLite 日志记录器。

Three tables: signals, fills, equity_snapshots.
三张表：信号、成交、权益快照。
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


DB_PATH: str = str(Path(__file__).resolve().parent.parent / "paper_trading.db")


class PaperTradeLogger:
    """SQLite logger for paper trading sessions. / 模拟盘会话的SQLite日志。"""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._conn: sqlite3.Connection = sqlite3.connect(db_path)
        self._init_tables()
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                session TEXT, timestamp INTEGER, symbol TEXT,
                direction TEXT, strength REAL, predicted_return REAL
            );
            CREATE TABLE IF NOT EXISTS fills (
                session TEXT, timestamp INTEGER, symbol TEXT,
                side TEXT, price REAL, quantity REAL,
                cost_bps REAL, fill_type TEXT
            );
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                session TEXT, timestamp INTEGER,
                equity REAL, cash REAL, unrealised_pnl REAL,
                realised_pnl REAL, drawdown REAL
            );
        """)
        self._conn.commit()

    def log_signal(self, symbol: str, direction: str, strength: float, pred_ret: float) -> None:
        self._conn.execute(
            "INSERT INTO signals VALUES (?,?,?,?,?,?)",
            (self._session_id, int(time.time()*1000), symbol, direction, strength, pred_ret),
        )
        self._conn.commit()

    def log_fill(self, symbol: str, side: str, price: float, qty: float,
                 cost_bps: float, fill_type: str) -> None:
        self._conn.execute(
            "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?)",
            (self._session_id, int(time.time()*1000), symbol, side, price, qty, cost_bps, fill_type),
        )
        self._conn.commit()

    def log_equity(self, equity: float, cash: float, unrealised: float,
                   realised: float, drawdown: float) -> None:
        self._conn.execute(
            "INSERT INTO equity_snapshots VALUES (?,?,?,?,?,?,?)",
            (self._session_id, int(time.time()*1000), equity, cash, unrealised, realised, drawdown),
        )
        self._conn.commit()

    def get_summary(self) -> Dict:
        """Get summary stats for current session. / 获取当前会话的汇总统计。"""
        n_signals = self._conn.execute(
            "SELECT COUNT(*) FROM signals WHERE session=?", (self._session_id,)
        ).fetchone()[0]
        n_fills = self._conn.execute(
            "SELECT COUNT(*) FROM fills WHERE session=?", (self._session_id,)
        ).fetchone()[0]
        latest = self._conn.execute(
            "SELECT equity, drawdown FROM equity_snapshots WHERE session=? ORDER BY timestamp DESC LIMIT 1",
            (self._session_id,),
        ).fetchone()
        return {
            "session": self._session_id,
            "signals": n_signals,
            "fills": n_fills,
            "equity": latest[0] if latest else 0,
            "drawdown": latest[1] if latest else 0,
        }

    def close(self) -> None:
        self._conn.close()
