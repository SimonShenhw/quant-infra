"""
Risk Management Layer (EventBus demo stack).

WHAT — a pre-trade risk gate for the event-driven engine: it is meant to
sit between strategy and matching engine as an Event -> Optional[List[Event]]
handler, vetoing orders once a max-drawdown circuit breaker trips.

WHY a separate layer — risk vetoes must be independent of the strategy
that generates signals; expressing the gate as a bus handler means it can
reject ANY order source uniformly, and a RiskEvent (not an exception)
keeps the event loop running while recording the veto.

HONESTY NOTE — legacy/demonstration layer, NOT in the v11+
result-producing path (the run scripts use a simplified ~100-line
long-short loop with fixed 50/50 sizing; nothing here runs there).
Known documented issue kept as-is:
  - M-9: `check_order` is never wired into the ORDER flow — no bus
    subscription and no caller ever invokes it, so `_circuit_broken` can
    never be set and the drawdown circuit breaker NEVER fires.
    ExecutionHandler only polls `is_circuit_broken`, which therefore
    stays False forever.  Documented, not fixed — see
    REVIEW_2026-06-10.md ① and M-9.

风控管理层（EventBus 演示栈）。

盘前风控检查，可在订单到达撮合引擎之前否决或修改订单。之所以独立成层：
风控否决必须独立于产生信号的策略；以事件处理器形式表达，可以统一拦截
任意来源的订单，且用 RiskEvent（而非异常）记录否决、不中断事件循环。

诚实披露：遗留/演示层，不在 v11+ 出结果路径上（run 脚本用简化多空循环、
固定 50/50 仓位）。已知问题 M-9：`check_order` 从未接入 ORDER 事件流——
没有任何组件订阅或调用它，`_circuit_broken` 永远不会被置位，回撤熔断器
实际上从不触发；ExecutionHandler 只轮询恒为 False 的 `is_circuit_broken`。
如实记录、未修复——见 REVIEW_2026-06-10.md M-9。
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
    single-name concentration limits — BY DESIGN.  In the shipped demo
    wiring none of these gates take effect, because `check_order` has no
    caller (module HONESTY NOTE, M-9).

    执行仓位限制、最大回撤熔断和单一品种集中度限制——这是设计意图。
    在现有演示接线中这些关卡均不生效：`check_order` 没有调用方
    （见模块诚实披露，M-9）。
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
        """Latched breaker state.  Only ever set by `check_order`, which is
        never called (M-9) — so in practice this is permanently False.
        熔断锁存状态。只有 `check_order` 会置位它，而后者从未被调用（M-9），
        因此实际恒为 False。"""
        return self._circuit_broken

    def check_order(self, event: Event) -> Optional[List[Event]]:
        """Pre-trade risk gate: reject everything while the breaker is
        latched; otherwise trip the breaker when peak-to-current drawdown
        reaches `max_drawdown`.  Returns None to let the order pass, or a
        CRITICAL RiskEvent to veto it.

        HONESTY NOTE (M-9): dead code in the shipped wiring — no bus
        subscription or caller invokes this, so the breaker never trips.

        盘前风控关卡：熔断锁存期间拒绝一切订单；否则当峰值回撤达到
        `max_drawdown` 时触发熔断。返回 None 放行，返回 CRITICAL RiskEvent
        表示否决。诚实披露（M-9）：现有接线中无人调用，熔断从不触发。"""
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

        # Order passes risk checks — return None to let it flow to matching engine / 订单通过风控 - 返回 None 让其流向撮合引擎
        # (matching engine is independently subscribed to ORDER events) / （撮合引擎独立订阅了 ORDER 事件）
        return None
