"""
Event-Driven Architecture — Core Event Types and Event Bus.

All communication between engine components flows through typed Event objects
dispatched via a central EventBus.  This eliminates coupling between the
matching engine, portfolio, strategy, and execution layers.
"""
from __future__ import annotations

import enum
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EventType(enum.Enum):
    """Canonical set of events flowing through the backtest loop."""
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
# Event Data-classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Base event — every event carries a type, timestamp and unique id."""
    event_type: EventType
    timestamp: datetime
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class TickEvent(Event):
    """Single LOB tick arriving from the data feed."""
    symbol: str = ""
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    last_price: float = 0.0
    last_volume: float = 0.0
    bid_levels: Optional[List[List[float]]] = None   # [[price, vol], ...]
    ask_levels: Optional[List[List[float]]] = None


@dataclass(frozen=True)
class MarketEvent(Event):
    """Aggregated OHLCV bar (built from ticks or supplied directly)."""
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


@dataclass(frozen=True)
class SignalEvent(Event):
    """Strategy output — a directional conviction with strength."""
    symbol: str = ""
    direction: SignalDirection = SignalDirection.LONG
    strength: float = 1.0
    predicted_return: float = 0.0


@dataclass(frozen=True)
class OrderEvent(Event):
    """Instruction to place an order into the matching engine."""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class FillEvent(Event):
    """Confirmation of (partial) execution from the matching engine."""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    commission: float = 0.0
    order_id: str = ""
    slippage: float = 0.0


@dataclass(frozen=True)
class RiskEvent(Event):
    """Emitted by the risk manager when limits are breached."""
    symbol: str = ""
    message: str = ""
    severity: str = "WARNING"


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

EventHandler = Callable[[Event], Optional[List[Event]]]


class EventBus:
    """
    Central publish-subscribe bus.

    Components register handlers for specific EventTypes.  When an event is
    published, all subscribed handlers are invoked in registration order.
    Handlers may return new events which are enqueued for processing.
    """

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._queue: List[Event] = []
        self._event_log: List[Event] = []

    # -- subscription -------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].remove(handler)

    # -- publishing ---------------------------------------------------------

    def publish(self, event: Event) -> None:
        self._queue.append(event)

    def drain(self) -> int:
        """Process all queued events.  Returns total events processed."""
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

    # -- introspection ------------------------------------------------------

    @property
    def event_log(self) -> List[Event]:
        return list(self._event_log)

    @property
    def pending_count(self) -> int:
        return len(self._queue)
