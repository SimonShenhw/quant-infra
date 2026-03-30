"""
Execution Handler — converts SignalEvents into OrderEvents.

v2: Kelly Criterion position sizing based on model confidence.
     Position size = Kelly_fraction * equity / price
     Kelly_fraction = edge / odds = (p*b - q) / b
     where p = win_prob (from signal strength), b = avg_win/avg_loss

执行处理器 - 将 SignalEvent 转换为 OrderEvent。

v2: 基于模型置信度的凯利公式仓位管理。
     仓位大小 = Kelly系数 * 权益 / 价格
     Kelly系数 = 优势 / 赔率 = (p*b - q) / b
     其中 p = 胜率（来自信号强度），b = 平均盈利/平均亏损
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from engine.events import (
    Event,
    EventType,
    OrderEvent,
    OrderSide,
    OrderType,
    SignalDirection,
    SignalEvent,
)
from engine.portfolio import Portfolio
from engine.risk import RiskManager


class ExecutionHandler:
    """
    Kelly Criterion position sizing with half-Kelly safety margin.
    Tracks recent win/loss statistics to dynamically calibrate sizing.

    凯利公式仓位管理，使用半凯利安全系数。
    跟踪近期胜/负统计以动态校准仓位。
    """

    def __init__(
        self,
        portfolio: Portfolio,
        max_position_pct: float = 0.15,
        default_order_type: OrderType = OrderType.MARKET,
        risk_manager: Optional[RiskManager] = None,
        kelly_fraction: float = 0.5,
        min_win_rate: float = 0.52,
    ) -> None:
        self._portfolio: Portfolio = portfolio
        self._max_position_pct: float = max_position_pct
        self._default_order_type: OrderType = default_order_type
        self._risk_manager: Optional[RiskManager] = risk_manager
        self._kelly_fraction: float = kelly_fraction
        self._min_win_rate: float = min_win_rate
        self._latest_prices: Dict[str, float] = {}

        # track recent trade outcomes for dynamic Kelly / 跟踪近期交易结果用于动态凯利计算
        self._trade_results: Deque[float] = deque(maxlen=100)
        self._last_entry_price: Dict[str, float] = {}

    def update_prices(self, prices: Dict[str, float]) -> None:
        self._latest_prices.update(prices)

    def record_trade_result(self, pnl_pct: float) -> None:
        self._trade_results.append(pnl_pct)

    def _compute_kelly_size(self, signal_strength: float) -> float:
        """
        Compute Kelly-optimal position fraction.

        Uses running statistics of recent trades to estimate edge,
        then applies half-Kelly for safety.

        计算凯利最优仓位比例。

        使用近期交易的滚动统计估算优势，然后应用半凯利以确保安全。
        """
        if len(self._trade_results) < 10:
            # not enough data, use moderate fixed fraction / 数据不足，使用适中的固定比例
            return self._max_position_pct * 0.5 * signal_strength

        wins: List[float] = [r for r in self._trade_results if r > 0]
        losses: List[float] = [r for r in self._trade_results if r <= 0]

        if not wins or not losses:
            return self._max_position_pct * 0.3 * signal_strength

        win_rate: float = len(wins) / len(self._trade_results)
        avg_win: float = sum(wins) / len(wins)
        avg_loss: float = abs(sum(losses) / len(losses))

        if win_rate < self._min_win_rate:
            # edge too thin — minimal position / 优势太薄 - 最小仓位
            return self._max_position_pct * 0.1 * signal_strength

        # Kelly: f* = (p*b - q) / b  where b = avg_win/avg_loss / 凯利公式：f* = (p*b - q) / b
        b: float = avg_win / max(avg_loss, 1e-8)
        q: float = 1.0 - win_rate
        kelly_f: float = (win_rate * b - q) / max(b, 1e-8)
        kelly_f = max(kelly_f, 0.0)

        # apply safety fraction (half-Kelly) / 应用安全系数（半凯利）
        position_pct: float = kelly_f * self._kelly_fraction * signal_strength
        # cap at max / 上限截断
        position_pct = min(position_pct, self._max_position_pct)

        return position_pct

    def handle_signal(self, event: Event) -> Optional[List[Event]]:
        if not isinstance(event, SignalEvent):
            return None

        if self._risk_manager is not None and self._risk_manager.is_circuit_broken:
            return None

        price: float = self._latest_prices.get(event.symbol, 0.0)
        if price <= 0:
            return None

        # record PnL of closing trades for Kelly calibration / 记录平仓交易盈亏用于凯利校准
        current_pos_qty: float = 0.0
        pos_dict = self._portfolio.positions
        if event.symbol in pos_dict:
            current_pos_qty = pos_dict[event.symbol].quantity
            if event.symbol in self._last_entry_price and current_pos_qty != 0:
                entry_p: float = self._last_entry_price[event.symbol]
                if current_pos_qty > 0:
                    pnl_pct: float = (price / entry_p) - 1.0
                else:
                    pnl_pct = 1.0 - (price / entry_p)
                # only record when direction changes (i.e. closing a trade) / 仅在方向变化时记录（即平仓）
                is_closing: bool = (
                    (event.direction == SignalDirection.EXIT)
                    or (event.direction == SignalDirection.LONG and current_pos_qty < 0)
                    or (event.direction == SignalDirection.SHORT and current_pos_qty > 0)
                )
                if is_closing:
                    self.record_trade_result(pnl_pct)

        equity: float = self._portfolio.cash
        for pos in self._portfolio.positions.values():
            equity += abs(pos.quantity) * self._latest_prices.get(
                pos.symbol, pos.avg_cost
            )

        # Kelly-based position sizing / 基于凯利公式的仓位计算
        position_pct: float = self._compute_kelly_size(abs(event.strength))
        max_notional: float = equity * position_pct
        quantity: float = max_notional / price
        # minimum lot: 100 shares for stocks, 0.001 for crypto / 最小手数：股票100股，加密货币0.001
        min_lot: float = 100.0 if price < 1000.0 else 0.001
        lot_round: float = 100.0 if price < 1000.0 else 0.001
        if quantity < min_lot:
            return None
        quantity = round(quantity / lot_round) * lot_round

        side: OrderSide
        if event.direction == SignalDirection.LONG:
            if current_pos_qty < 0:
                quantity = abs(current_pos_qty) + quantity
            side = OrderSide.BUY
            self._last_entry_price[event.symbol] = price
        elif event.direction == SignalDirection.SHORT:
            if current_pos_qty > 0:
                quantity = current_pos_qty + quantity
            side = OrderSide.SELL
            self._last_entry_price[event.symbol] = price
        elif event.direction == SignalDirection.EXIT:
            if current_pos_qty > 0:
                side = OrderSide.SELL
                quantity = current_pos_qty
            elif current_pos_qty < 0:
                side = OrderSide.BUY
                quantity = abs(current_pos_qty)
            else:
                return None
        else:
            return None

        order: OrderEvent = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=event.timestamp,
            symbol=event.symbol,
            side=side,
            order_type=self._default_order_type,
            quantity=quantity,
        )
        return [order]
