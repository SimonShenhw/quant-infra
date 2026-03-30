"""
Paper Trading Engine — real-time inference + simulated execution.
模拟盘引擎 — 实时推理 + 模拟执行。

Connects to live exchange WebSocket, receives bars, runs model inference,
simulates execution (no real orders), logs everything to SQLite.
连接真实交易所WebSocket，接收K线，运行模型推理，模拟执行（不发真实订单），
全部记录到SQLite。
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from paper_trading.logger import PaperTradeLogger


class PaperTradingEngine:
    """
    Async paper trading engine.
    异步模拟盘引擎。

    Usage / 用法:
        engine = PaperTradingEngine(model, symbols, device)
        await engine.run(duration_seconds=3600)
    """

    def __init__(
        self,
        model: Any,
        symbols: List[str],
        device: torch.device,
        seq_len: int = 24,
        initial_cash: float = 100_000.0,
        min_hold_bars: int = 6,
        taker_fee_bps: float = 4.0,
    ) -> None:
        self._model = model
        self._symbols: List[str] = symbols
        self._device: torch.device = device
        self._seq_len: int = seq_len
        self._taker_bps: float = taker_fee_bps
        self._min_hold: int = min_hold_bars

        # portfolio state / 组合状态
        self._cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self._equity: float = initial_cash
        self._peak_equity: float = initial_cash
        self._current_long: int = -1
        self._current_short: int = -1
        self._hold_counter: int = 0
        self._total_cost: float = 0.0

        # bar buffers per symbol: {sym_idx: list of (o,h,l,c,v)} / 每个标的的K线缓冲
        self._bar_buffers: Dict[int, List[Tuple[float, ...]]] = defaultdict(list)

        self._logger: PaperTradeLogger = PaperTradeLogger()
        self._bar_count: int = 0

    def ingest_bars(self, bars: Dict[str, Tuple[float, float, float, float, float]]) -> Optional[Dict]:
        """
        Ingest one set of bars (one per symbol) and run inference.
        摄入一组K线（每个标的一根）并运行推理。

        Args:
            bars: {symbol: (open, high, low, close, volume)}
        Returns:
            action dict if signal generated, else None
        """
        self._bar_count += 1

        # buffer bars / 缓冲K线
        for i, sym in enumerate(self._symbols):
            if sym in bars:
                self._bar_buffers[i].append(bars[sym])

        # need seq_len bars to start / 需要seq_len根K线才能开始
        min_bars = min(len(self._bar_buffers[i]) for i in range(len(self._symbols)))
        if min_bars < self._seq_len + 30:  # warmup for z-score / z-score 预热
            return None

        # build factor tensor from buffers / 从缓冲区构建因子张量
        from model.features import build_factor_tensor
        all_factors: List[Tensor] = []
        all_closes: List[float] = []
        for i in range(len(self._symbols)):
            buf = self._bar_buffers[i]
            o = torch.tensor([b[0] for b in buf], dtype=torch.float32, device=self._device)
            h = torch.tensor([b[1] for b in buf], dtype=torch.float32, device=self._device)
            l = torch.tensor([b[2] for b in buf], dtype=torch.float32, device=self._device)
            c = torch.tensor([b[3] for b in buf], dtype=torch.float32, device=self._device)
            v = torch.tensor([b[4] for b in buf], dtype=torch.float32, device=self._device)
            factors = build_factor_tensor(o, h, l, c, v, zscore_window=48)
            all_factors.append(factors[-self._seq_len:])
            all_closes.append(buf[-1][3])  # last close

        # stack → (1, A, T, F) / 堆叠为4D张量
        x = torch.stack(all_factors, dim=0).unsqueeze(0)

        # inference / 推理
        self._model.eval()
        with torch.no_grad():
            scores = self._model(x).squeeze(0)  # (A,)

        # signal generation (same logic as backtest) / 信号生成
        self._hold_counter += 1
        need_rebal = False
        if self._current_long < 0:
            need_rebal = True
        elif self._hold_counter >= self._min_hold:
            nl, ns = scores.argmax().item(), scores.argmin().item()
            if nl != self._current_long or ns != self._current_short:
                need_rebal = True

        if not (need_rebal and (self._hold_counter >= self._min_hold or self._current_long < 0)):
            return None

        nl = scores.argmax().item()
        ns = scores.argmin().item()

        # log signal / 记录信号
        self._logger.log_signal(
            self._symbols[nl], "LONG", scores[nl].item(), scores[nl].item()
        )
        self._logger.log_signal(
            self._symbols[ns], "SHORT", scores[ns].item(), scores[ns].item()
        )

        # simulate execution cost / 模拟执行成本
        legs_changed = sum([
            self._current_long != nl and self._current_long >= 0,
            self._current_short != ns and self._current_short >= 0,
            self._current_long != nl,
            self._current_short != ns,
        ])
        cost = self._equity * 0.5 * self._taker_bps / 10000.0 * legs_changed
        self._total_cost += cost

        # log fills / 记录成交
        if self._current_long != nl:
            self._logger.log_fill(
                self._symbols[nl], "BUY", all_closes[nl], 0, self._taker_bps, "PAPER"
            )
        if self._current_short != ns:
            self._logger.log_fill(
                self._symbols[ns], "SELL", all_closes[ns], 0, self._taker_bps, "PAPER"
            )

        self._current_long = nl
        self._current_short = ns
        self._hold_counter = 0

        # log equity / 记录权益
        dd = (self._peak_equity - self._equity) / self._peak_equity if self._peak_equity > 0 else 0
        self._logger.log_equity(self._equity, self._cash, 0, 0, dd)

        return {
            "long": self._symbols[nl],
            "short": self._symbols[ns],
            "long_score": scores[nl].item(),
            "short_score": scores[ns].item(),
            "cost": cost,
            "equity": self._equity,
        }

    def update_pnl(self, returns: Dict[str, float]) -> None:
        """
        Update equity from actual returns (called after bar close).
        用实际收益更新权益（K线收盘后调用）。
        """
        port_ret = 0.0
        if self._current_long >= 0 and self._symbols[self._current_long] in returns:
            port_ret += 0.5 * returns[self._symbols[self._current_long]]
        if self._current_short >= 0 and self._symbols[self._current_short] in returns:
            port_ret -= 0.5 * returns[self._symbols[self._current_short]]
        self._equity *= (1.0 + port_ret)
        self._peak_equity = max(self._peak_equity, self._equity)

    def summary(self) -> Dict:
        """Get current session summary. / 获取当前会话汇总。"""
        return {
            **self._logger.get_summary(),
            "total_cost": self._total_cost,
            "total_return": self._equity / self._initial_cash - 1,
            "bar_count": self._bar_count,
        }

    def close(self) -> None:
        self._logger.close()
