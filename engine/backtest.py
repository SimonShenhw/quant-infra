"""
Main Backtest Orchestrator (EventBus demo stack).

WHAT — wires the EventBus, strategy, execution handler, matching engine,
portfolio, and risk manager into a single event loop:
TICK/MARKET -> strategy -> SIGNAL -> execution -> ORDER -> matching ->
FILL -> portfolio, with mark-to-market after every price event.

WHY this shape — pub/sub over typed events mirrors how a live trading
process is structured (components communicate only through the bus), so
the same strategy/execution code is reusable for paper trading, and each
component can be tested in isolation.

HONESTY NOTE — legacy/demonstration layer, NOT in the v11+
result-producing path: every published number comes from the simplified
~100-line long-short loops inside the run scripts (run_v13_final.py et
al.), not from this engine.  Known wiring gap kept as-is (M-9):
RiskManager.check_order is never subscribed to ORDER events nor invoked
by the ExecutionHandler, so the advertised drawdown circuit breaker
never fires here.  Documented, not fixed — see REVIEW_2026-06-10.md ①
and M-9.

回测主调度器（EventBus 演示栈）。

将 EventBus、策略、执行处理器、撮合引擎、组合管理和风控管理器整合到
单一事件循环中：TICK/MARKET → 策略 → SIGNAL → 执行 → ORDER → 撮合 →
FILL → 组合，每个价格事件后做逐日盯市。选择 pub/sub 是为了让组件只经
事件总线通信——同一套代码可复用于模拟盘，且各组件可独立测试。

诚实披露：遗留/演示层，不在 v11+ 出结果路径上——已发布结果全部出自
run 脚本内的简化多空循环。M-9：RiskManager.check_order 从未订阅 ORDER
事件、也未被执行处理器调用，回撤熔断在此引擎中从不触发。如实记录、
未修复——见 REVIEW_2026-06-10.md。
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
    Event-driven backtester (demonstration stack — see module HONESTY
    NOTE; the v11+ published results do not run through this class).

    Usage:
        engine = BacktestEngine(initial_cash=1_000_000)
        engine.register_strategy(my_strategy_handler)
        engine.run(tick_data)
        print(engine.portfolio.summary())

    事件驱动回测器（演示栈——见模块诚实披露；v11+ 已发布结果不经过本类）。

    用法：
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

        # wire event bus.  NOTE (M-9): RiskManager is passed into the execution
        # handler, but the handler only polls `is_circuit_broken` (which nothing
        # ever sets) — `check_order` is NOT subscribed to ORDER events here, so
        # the risk gate is effectively inert.  Documented, not fixed.
        # 连接事件总线。注意（M-9）：RiskManager 虽传入执行处理器，但处理器只
        # 轮询从未被置位的 `is_circuit_broken`——`check_order` 并未订阅 ORDER
        # 事件，风控关卡实际不生效。如实记录、未修复。
        self.bus.subscribe(EventType.TICK, self._on_tick)
        self.bus.subscribe(EventType.MARKET, self._on_market)
        self.bus.subscribe(EventType.SIGNAL, self.execution.handle_signal)
        self.bus.subscribe(EventType.ORDER, self.matching.handle_order)
        self.bus.subscribe(EventType.FILL, self.portfolio.handle_fill)

    def register_strategy(
        self, handler: Callable[[Event], Optional[List[Event]]]
    ) -> None:
        """Subscribe a strategy handler to MARKET events; any events it
        returns (typically SignalEvents) are re-published by the bus.
        将策略处理器订阅到 MARKET 事件；其返回的事件（通常是 SignalEvent）
        由总线再次发布。"""
        self._strategy_handler = handler
        self.bus.subscribe(EventType.MARKET, handler)

    # -- internal handlers / 内部处理器 ---------------------------------------

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
        # Seed order book with synthetic liquidity from bar data / 用K线数据向订单簿注入合成流动性
        # so market orders can fill (essential for bar-based backtesting) / 使市价单可以成交（K线回测必需）
        self.matching.seed_from_bar(
            event.symbol, event.close, event.volume, event.timestamp
        )
        return None

    # -- main loop / 主循环 --------------------------------------------------

    def run(self, events: List[Event]) -> Dict[str, float]:
        """Execute the full backtest over a sequence of pre-built events.
        在预构建的事件序列上执行完整回测。"""
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
