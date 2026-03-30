"""
Portfolio and Position Management.

Tracks positions, P&L, and generates performance metrics.
Integrates with the EventBus via FillEvent handlers.

组合与持仓管理。

跟踪持仓、盈亏，并生成绩效指标。
通过 FillEvent 处理器与 EventBus 集成。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from engine.events import (
    Event,
    EventType,
    FillEvent,
    OrderSide,
)


@dataclass
class Position:
    """Single-instrument position with cost-basis tracking.
    单品种持仓，含成本基础跟踪。"""
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realised_pnl: float = 0.0
    total_commission: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_cost

    def update(self, side: OrderSide, fill_qty: float, fill_price: float, commission: float) -> float:
        """Apply a fill.  Returns realised P&L from this fill (if closing).
        应用一笔成交，返回已实现盈亏（如有平仓）。"""
        rpnl: float = 0.0
        self.total_commission += commission
        if side == OrderSide.BUY:
            if self.quantity >= 0:
                total_cost: float = self.avg_cost * self.quantity + fill_price * fill_qty
                self.quantity += fill_qty
                self.avg_cost = total_cost / self.quantity if self.quantity != 0 else 0.0
            else:
                close_qty: float = min(fill_qty, abs(self.quantity))
                rpnl = close_qty * (self.avg_cost - fill_price)
                self.quantity += fill_qty
                if self.quantity > 0:
                    self.avg_cost = fill_price
        else:
            if self.quantity <= 0:
                total_cost = abs(self.avg_cost * self.quantity) + fill_price * fill_qty
                self.quantity -= fill_qty
                self.avg_cost = total_cost / abs(self.quantity) if self.quantity != 0 else 0.0
            else:
                close_qty = min(fill_qty, self.quantity)
                rpnl = close_qty * (fill_price - self.avg_cost)
                self.quantity -= fill_qty
                if self.quantity < 0:
                    self.avg_cost = fill_price
        rpnl -= commission
        self.realised_pnl += rpnl
        return rpnl


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    realised_pnl: float
    unrealised_pnl: float


class Portfolio:
    """
    Manages cash, positions, and equity curve.  Subscribes to FillEvents.

    管理现金、持仓和权益曲线。订阅 FillEvent。
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self._initial_cash: float = initial_cash
        self._cash: float = initial_cash
        self._positions: Dict[str, Position] = {}
        self._equity_curve: List[PortfolioSnapshot] = []
        self._fill_log: List[FillEvent] = []

    # -- properties / 属性 ---------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    @property
    def equity_curve(self) -> List[PortfolioSnapshot]:
        return list(self._equity_curve)

    # -- fill handling / 成交处理 --------------------------------------------

    def handle_fill(self, event: Event) -> None:
        """EventBus handler for FILL events.
        EventBus 的 FILL 事件处理器。"""
        if not isinstance(event, FillEvent):
            return
        self._fill_log.append(event)

        pos: Position = self._positions.setdefault(
            event.symbol, Position(symbol=event.symbol)
        )
        pos.update(event.side, event.fill_quantity, event.fill_price, event.commission)

        if event.side == OrderSide.BUY:
            self._cash -= event.fill_price * event.fill_quantity + event.commission
        else:
            self._cash += event.fill_price * event.fill_quantity - event.commission

    # -- mark-to-market / 逐日盯市 ------------------------------------------

    def mark_to_market(
        self, timestamp: datetime, prices: Dict[str, float]
    ) -> PortfolioSnapshot:
        positions_value: float = 0.0
        unrealised: float = 0.0
        for sym, pos in self._positions.items():
            mkt_price: float = prices.get(sym, pos.avg_cost)
            positions_value += pos.quantity * mkt_price
            unrealised += pos.quantity * (mkt_price - pos.avg_cost)
        equity: float = self._cash + positions_value
        realised: float = sum(p.realised_pnl for p in self._positions.values())
        snap: PortfolioSnapshot = PortfolioSnapshot(
            timestamp=timestamp,
            equity=equity,
            cash=self._cash,
            positions_value=positions_value,
            realised_pnl=realised,
            unrealised_pnl=unrealised,
        )
        self._equity_curve.append(snap)
        return snap

    # -- performance / 绩效指标 ----------------------------------------------

    def summary(self) -> Dict[str, float]:
        if len(self._equity_curve) < 2:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

        equities: List[float] = [s.equity for s in self._equity_curve]
        total_return: float = (equities[-1] / equities[0]) - 1.0

        returns: List[float] = [
            (equities[i] / equities[i - 1]) - 1.0 for i in range(1, len(equities))
        ]
        avg_ret: float = sum(returns) / len(returns) if returns else 0.0
        std_ret: float = (
            (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5
            if returns
            else 1e-9
        )
        sharpe: float = (avg_ret / std_ret) * (252 ** 0.5) if std_ret > 1e-12 else 0.0

        peak: float = equities[0]
        max_dd: float = 0.0
        for eq in equities:
            peak = max(peak, eq)
            dd: float = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "final_equity": equities[-1],
            "num_fills": len(self._fill_log),
        }
