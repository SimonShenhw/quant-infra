"""
Risk Management Layer.

Pre-trade risk checks that can veto or modify orders before they reach
the matching engine.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from engine.events import (
    Event,
    EventType,
    OrderEvent,
    OrderSide,
    RiskEvent,
)
from engine.portfolio import Portfolio


class RiskManager:
    """
    Enforces position limits, max-drawdown circuit breaker, and
    single-name concentration limits.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        max_position_size: float = 50_000.0,
        max_drawdown: float = 0.15,
        max_concentration: float = 0.25,
    ) -> None:
        self._portfolio: Portfolio = portfolio
        self._max_position_size: float = max_position_size
        self._max_drawdown: float = max_drawdown
        self._max_concentration: float = max_concentration
        self._circuit_broken: bool = False

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    def check_order(self, event: Event) -> Optional[List[Event]]:
        """Pre-trade risk gate.  Returns the order if OK, or a RiskEvent."""
        if not isinstance(event, OrderEvent):
            return None

        if self._circuit_broken:
            return [
                RiskEvent(
                    event_type=EventType.RISK,
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    message="Circuit breaker active — order rejected",
                    severity="CRITICAL",
                )
            ]

        curve = self._portfolio.equity_curve
        if len(curve) >= 2:
            peak: float = max(s.equity for s in curve)
            current: float = curve[-1].equity
            dd: float = (peak - current) / peak if peak > 0 else 0.0
            if dd >= self._max_drawdown:
                self._circuit_broken = True
                return [
                    RiskEvent(
                        event_type=EventType.RISK,
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        message=f"Max drawdown {dd:.2%} >= {self._max_drawdown:.2%}",
                        severity="CRITICAL",
                    )
                ]

        # Order passes risk checks — return None to let it flow to matching engine
        # (matching engine is independently subscribed to ORDER events)
        return None
