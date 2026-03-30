"""
Event-Driven Architecture — Core Event Types and Event Bus.

All communication between engine components flows through typed Event objects
dispatched via a central EventBus.  This eliminates coupling between the
matching engine, portfolio, strategy, and execution layers.

事件驱动架构 - 核心事件类型与事件总线。

所有引擎组件之间的通信均通过类型化的 Event 对象经中央 EventBus 分发。
消除了撮合引擎、组合管理、策略与执行层之间的耦合。
"""
from __future__ import annotations

import enum
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations / 枚举类型
# ---------------------------------------------------------------------------

class EventType(enum.Enum):
    """Canonical set of events flowing through the backtest loop.
    回测循环中流转的标准事件集合。"""
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    TICK = "TICK"
    RISK = "RISK"


class OrderSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(enum.Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


class SignalDirection(enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


# ---------------------------------------------------------------------------
# Event Data-classes / 事件数据类
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Base event — every event carries a type, timestamp and unique id.
    基础事件 - 每个事件携带类型、时间戳和唯一 ID。"""
    event_type: EventType
    timestamp: datetime
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class TickEvent(Event):
    """Single LOB tick arriving from the data feed.
    来自数据源的单个订单簿 Tick。"""
    symbol: str = ""
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    last_price: float = 0.0
    last_volume: float = 0.0
    bid_levels: Optional[List[List[float]]] = None   # [[price, vol], ...] / [[价格, 量], ...]
    ask_levels: Optional[List[List[float]]] = None


@dataclass(frozen=True)
class MarketEvent(Event):
    """Aggregated OHLCV bar (built from ticks or supplied directly).
    聚合的 OHLCV K 线（由 Tick 构建或直接提供）。"""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


@dataclass(frozen=True)
class SignalEvent(Event):
    """Strategy output — a directional conviction with strength.
    策略输出 - 带强度的方向性信号。"""
    symbol: str = ""
    direction: SignalDirection = SignalDirection.LONG
    strength: float = 1.0
    predicted_return: float = 0.0


@dataclass(frozen=True)
class OrderEvent(Event):
    """Instruction to place an order into the matching engine.
    向撮合引擎下单的指令。"""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class FillEvent(Event):
    """Confirmation of (partial) execution from the matching engine.
    来自撮合引擎的（部分）成交确认。"""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    commission: float = 0.0
    order_id: str = ""
    slippage: float = 0.0


@dataclass(frozen=True)
class RiskEvent(Event):
    """Emitted by the risk manager when limits are breached.
    风控管理器在触发限制时发出。"""
    symbol: str = ""
    message: str = ""
    severity: str = "WARNING"


# ---------------------------------------------------------------------------
# Event Bus / 事件总线
# ---------------------------------------------------------------------------

EventHandler = Callable[[Event], Optional[List[Event]]]


class EventBus:
    """
    Central publish-subscribe bus.

    Components register handlers for specific EventTypes.  When an event is
    published, all subscribed handlers are invoked in registration order.
    Handlers may return new events which are enqueued for processing.

    中央发布-订阅总线。

    组件为特定 EventType 注册处理器。事件发布时，按注册顺序调用所有
    已订阅的处理器。处理器可返回新事件，这些事件会被加入队列继续处理。
    """

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._queue: List[Event] = []
        self._event_log: List[Event] = []

    # -- subscription / 订阅 ------------------------------------------------

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].remove(handler)

    # -- publishing / 发布 --------------------------------------------------

    def publish(self, event: Event) -> None:
        self._queue.append(event)

    def drain(self) -> int:
        """Process all queued events.  Returns total events processed.
        处理所有排队事件，返回已处理事件总数。"""
        processed: int = 0
        while self._queue:
            event = self._queue.pop(0)
            self._event_log.append(event)
            processed += 1
            for handler in self._handlers.get(event.event_type, []):
                result: Any = handler(event)
                if isinstance(result, list):
                    for e in result:
                        self._queue.append(e)
                elif isinstance(result, Event):
                    self._queue.append(result)
        return processed

    # -- introspection / 内省 -----------------------------------------------

    @property
    def event_log(self) -> List[Event]:
        return list(self._event_log)

    @property
    def pending_count(self) -> int:
        return len(self._queue)
