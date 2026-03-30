"""
Main Backtest Orchestrator.

Wires together the EventBus, data feed, strategy, execution handler,
matching engine, portfolio, and risk manager into a single event loop.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from engine.events import (
    Event,
    EventBus,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    RiskEvent,
    SignalEvent,
    TickEvent,
)
from engine.execution import ExecutionHandler
from engine.order_book import MatchingEngine
from engine.portfolio import Portfolio, PortfolioSnapshot
from engine.risk import RiskManager


class BacktestEngine:
    """
    Production-grade event-driven backtester.

    Usage:
        engine = BacktestEngine(initial_cash=1_000_000)
        engine.register_strategy(my_strategy_handler)
        engine.run(tick_data)
        print(engine.portfolio.summary())
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        tick_size: float = 0.01,
        max_drawdown: float = 0.15,
        max_position_pct: float = 0.10,
        verbose: bool = True,
    ) -> None:
        self.bus: EventBus = EventBus()
        self.portfolio: Portfolio = Portfolio(initial_cash)
        self.matching: MatchingEngine = MatchingEngine(tick_size)
        self.risk: RiskManager = RiskManager(
            self.portfolio, max_drawdown=max_drawdown
        )
        self.execution: ExecutionHandler = ExecutionHandler(
            self.portfolio,
            max_position_pct=max_position_pct,
            risk_manager=self.risk,
        )
        self._strategy_handler: Optional[Callable[[Event], Optional[List[Event]]]] = None
        self._verbose: bool = verbose

        # wire event bus — risk is checked synchronously inside execution handler
        self.bus.subscribe(EventType.TICK, self._on_tick)
        self.bus.subscribe(EventType.MARKET, self._on_market)
        self.bus.subscribe(EventType.SIGNAL, self.execution.handle_signal)
        self.bus.subscribe(EventType.ORDER, self.matching.handle_order)
        self.bus.subscribe(EventType.FILL, self.portfolio.handle_fill)

    def register_strategy(
        self, handler: Callable[[Event], Optional[List[Event]]]
    ) -> None:
        self._strategy_handler = handler
        self.bus.subscribe(EventType.MARKET, handler)

    # -- internal handlers --------------------------------------------------

    def _on_tick(self, event: Event) -> Optional[List[Event]]:
        if not isinstance(event, TickEvent):
            return None
        self.matching.handle_tick(event)
        self.execution.update_prices({event.symbol: event.last_price})
        return None

    def _on_market(self, event: Event) -> Optional[List[Event]]:
        if not isinstance(event, MarketEvent):
            return None
        self.execution.update_prices({event.symbol: event.close})
        # Seed order book with synthetic liquidity from bar data
        # so market orders can fill (essential for bar-based backtesting)
        self.matching.seed_from_bar(
            event.symbol, event.close, event.volume, event.timestamp
        )
        return None

    # -- main loop ----------------------------------------------------------

    def run(self, events: List[Event]) -> Dict[str, float]:
        """Execute the full backtest over a sequence of pre-built events."""
        t0: float = time.time()
        total_processed: int = 0

        for i, event in enumerate(events):
            self.bus.publish(event)
            total_processed += self.bus.drain()

            if isinstance(event, (TickEvent, MarketEvent)):
                sym: str = event.symbol
                price: float = (
                    event.last_price
                    if isinstance(event, TickEvent)
                    else event.close
                )
                self.portfolio.mark_to_market(
                    event.timestamp, {sym: price}
                )

        elapsed: float = time.time() - t0

        summary: Dict[str, float] = self.portfolio.summary()
        summary["elapsed_seconds"] = round(elapsed, 3)
        summary["total_events_processed"] = total_processed

        if self._verbose:
            print("\n" + "=" * 60)
            print("  BACKTEST COMPLETE")
            print("=" * 60)
            for k, v in summary.items():
                if isinstance(v, float) and abs(v) < 1e6:
                    print(f"  {k:<30s} {v:>12.4f}")
                else:
                    print(f"  {k:<30s} {v:>12}")
            print("=" * 60)

        return summary
