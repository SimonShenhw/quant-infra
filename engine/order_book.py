"""
Limit Order Book (LOB) and Matching Engine.

Implements a price-time priority matching engine with full order book
depth tracking.  Supports LIMIT and MARKET orders with partial fills.
"""
from __future__ import annotations

import heapq
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from engine.events import (
    Event,
    EventType,
    FillEvent,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    TickEvent,
)


# ---------------------------------------------------------------------------
# Internal order representation
# ---------------------------------------------------------------------------

@dataclass
class BookOrder:
    """Resting order inside the LOB."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    original_qty: float
    remaining_qty: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING

    @property
    def is_live(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)


# ---------------------------------------------------------------------------
# Single-side book (bid or ask)
# ---------------------------------------------------------------------------

class _HalfBook:
    """
    One side of the order book backed by a heap.

    For bids we use a max-heap (negate prices); for asks a min-heap.
    """

    def __init__(self, is_bid: bool) -> None:
        self._is_bid: bool = is_bid
        self._heap: List[Tuple[float, float, str]] = []  # (sort_key, ts, id)
        self._orders: Dict[str, BookOrder] = {}

    def insert(self, order: BookOrder) -> None:
        sort_price: float = -order.price if self._is_bid else order.price
        ts: float = order.timestamp.timestamp()
        heapq.heappush(self._heap, (sort_price, ts, order.order_id))
        self._orders[order.order_id] = order

    def peek(self) -> Optional[BookOrder]:
        while self._heap:
            _, _, oid = self._heap[0]
            order: Optional[BookOrder] = self._orders.get(oid)
            if order is not None and order.is_live:
                return order
            heapq.heappop(self._heap)
            self._orders.pop(oid, None)
        return None

    def pop(self) -> Optional[BookOrder]:
        while self._heap:
            _, _, oid = heapq.heappop(self._heap)
            order: Optional[BookOrder] = self._orders.get(oid)
            if order is not None and order.is_live:
                return order
            self._orders.pop(oid, None)
        return None

    def cancel(self, order_id: str) -> bool:
        order: Optional[BookOrder] = self._orders.get(order_id)
        if order is not None and order.is_live:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    def reset(self) -> None:
        """Clear all resting orders."""
        self._heap.clear()
        self._orders.clear()

    def depth(self, levels: int = 5) -> List[Tuple[float, float]]:
        """Return [(price, total_qty)] aggregated by price, best first."""
        agg: Dict[float, float] = defaultdict(float)
        for order in self._orders.values():
            if order.is_live:
                agg[order.price] += order.remaining_qty
        sorted_prices: List[float] = sorted(
            agg.keys(), reverse=self._is_bid
        )
        return [(p, agg[p]) for p in sorted_prices[:levels]]


# ---------------------------------------------------------------------------
# Order Book per symbol
# ---------------------------------------------------------------------------

class OrderBook:
    """Full two-sided limit order book for a single instrument."""

    def __init__(self, symbol: str, tick_size: float = 0.01) -> None:
        self.symbol: str = symbol
        self.tick_size: float = tick_size
        self.bids: _HalfBook = _HalfBook(is_bid=True)
        self.asks: _HalfBook = _HalfBook(is_bid=False)
        self.last_trade_price: float = 0.0
        self.last_trade_qty: float = 0.0
        self._trade_log: List[Dict[str, object]] = []

    # -- accessors ----------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        top: Optional[BookOrder] = self.bids.peek()
        return top.price if top else None

    @property
    def best_ask(self) -> Optional[float]:
        top: Optional[BookOrder] = self.asks.peek()
        return top.price if top else None

    @property
    def mid_price(self) -> Optional[float]:
        bb: Optional[float] = self.best_bid
        ba: Optional[float] = self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        bb: Optional[float] = self.best_bid
        ba: Optional[float] = self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None

    @property
    def trade_log(self) -> List[Dict[str, object]]:
        return list(self._trade_log)

    # -- matching -----------------------------------------------------------

    def submit_order(
        self, order: BookOrder
    ) -> List[FillEvent]:
        """Match an incoming order against resting liquidity."""
        fills: List[FillEvent] = []
        if order.order_type == OrderType.MARKET:
            fills = self._match_market(order)
        elif order.order_type == OrderType.LIMIT:
            fills = self._match_limit(order)
        return fills

    def _match_market(self, order: BookOrder) -> List[FillEvent]:
        contra: _HalfBook = self.asks if order.side == OrderSide.BUY else self.bids
        return self._sweep(order, contra, price_limit=None)

    def _match_limit(self, order: BookOrder) -> List[FillEvent]:
        if order.side == OrderSide.BUY:
            contra: _HalfBook = self.asks
            price_limit: Optional[float] = order.price
        else:
            contra = self.bids
            price_limit = order.price
        fills: List[FillEvent] = self._sweep(order, contra, price_limit)
        if order.remaining_qty > 0 and order.is_live:
            book_side: _HalfBook = (
                self.bids if order.side == OrderSide.BUY else self.asks
            )
            book_side.insert(order)
        return fills

    def _sweep(
        self,
        aggressor: BookOrder,
        contra: _HalfBook,
        price_limit: Optional[float],
    ) -> List[FillEvent]:
        fills: List[FillEvent] = []
        while aggressor.remaining_qty > 0:
            resting: Optional[BookOrder] = contra.peek()
            if resting is None:
                break
            if price_limit is not None:
                if aggressor.side == OrderSide.BUY and resting.price > price_limit:
                    break
                if aggressor.side == OrderSide.SELL and resting.price < price_limit:
                    break

            match_qty: float = min(aggressor.remaining_qty, resting.remaining_qty)
            match_price: float = resting.price

            aggressor.remaining_qty -= match_qty
            resting.remaining_qty -= match_qty

            if resting.remaining_qty <= 0:
                resting.status = OrderStatus.FILLED
                contra.pop()
            else:
                resting.status = OrderStatus.PARTIAL

            if aggressor.remaining_qty <= 0:
                aggressor.status = OrderStatus.FILLED
            else:
                aggressor.status = OrderStatus.PARTIAL

            self.last_trade_price = match_price
            self.last_trade_qty = match_qty
            self._trade_log.append(
                {
                    "price": match_price,
                    "qty": match_qty,
                    "aggressor_id": aggressor.order_id,
                    "resting_id": resting.order_id,
                    "timestamp": aggressor.timestamp,
                }
            )

            # Adaptive transaction costs by asset class
            notional: float = match_price * match_qty
            commission: float
            total_slippage: float

            if match_price > 500.0:
                # CRYPTO: Taker fee 4bps (Binance/OKX VIP0), no stamp tax
                commission = notional * 0.0004
                # slippage: tighter for liquid crypto
                impact_bps: float = 2.0 * (match_qty * match_price / 100000.0) ** 0.5
                total_slippage = notional * impact_bps / 10000.0
            else:
                # A-SHARE: commission 2.5bps + stamp tax 5bps sell + exchange 0.5bps
                broker_fee: float = max(notional * 0.00025, 5.0)
                stamp_tax: float = notional * 0.0005 if aggressor.side == OrderSide.SELL else 0.0
                commission = broker_fee + stamp_tax + notional * 0.00005
                impact_bps = 10.0 * (match_qty / 10000.0) ** 0.5
                total_slippage = match_price * impact_bps / 10000.0 * match_qty

            fill: FillEvent = FillEvent(
                event_type=EventType.FILL,
                timestamp=aggressor.timestamp,
                symbol=aggressor.symbol,
                side=aggressor.side,
                fill_price=match_price,
                fill_quantity=match_qty,
                commission=commission + total_slippage,
                order_id=aggressor.order_id,
                slippage=total_slippage,
            )
            fills.append(fill)
        return fills


# ---------------------------------------------------------------------------
# Matching Engine — multi-symbol wrapper
# ---------------------------------------------------------------------------

class MatchingEngine:
    """
    Routes OrderEvents to per-symbol OrderBooks, returns FillEvents.
    Designed to plug directly into the EventBus.
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self._books: Dict[str, OrderBook] = {}
        self._tick_size: float = tick_size

    def get_or_create_book(self, symbol: str) -> OrderBook:
        if symbol not in self._books:
            self._books[symbol] = OrderBook(symbol, self._tick_size)
        return self._books[symbol]

    def handle_order(self, event: Event) -> Optional[List[Event]]:
        """EventBus handler for ORDER events."""
        if not isinstance(event, OrderEvent):
            return None
        book: OrderBook = self.get_or_create_book(event.symbol)
        internal_order: BookOrder = BookOrder(
            order_id=event.order_id,
            symbol=event.symbol,
            side=event.side,
            order_type=event.order_type,
            price=event.limit_price if event.limit_price is not None else 0.0,
            original_qty=event.quantity,
            remaining_qty=event.quantity,
            timestamp=event.timestamp,
        )
        fills: List[FillEvent] = book.submit_order(internal_order)
        return fills if fills else None

    def handle_tick(self, event: Event) -> None:
        """Seed book liquidity from TickEvents (LOB depth snapshot replaces old state)."""
        if not isinstance(event, TickEvent):
            return
        book: OrderBook = self.get_or_create_book(event.symbol)
        # reset book — each tick is a full LOB snapshot, not incremental
        book.bids.reset()
        book.asks.reset()
        if event.bid_levels:
            for price, vol in event.bid_levels:
                lo: BookOrder = BookOrder(
                    order_id=uuid.uuid4().hex[:12],
                    symbol=event.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=price,
                    original_qty=vol,
                    remaining_qty=vol,
                    timestamp=event.timestamp,
                )
                book.bids.insert(lo)
        if event.ask_levels:
            for price, vol in event.ask_levels:
                lo = BookOrder(
                    order_id=uuid.uuid4().hex[:12],
                    symbol=event.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=price,
                    original_qty=vol,
                    remaining_qty=vol,
                    timestamp=event.timestamp,
                )
                book.asks.insert(lo)

    def seed_from_bar(
        self,
        symbol: str,
        close_price: float,
        volume: float,
        timestamp: "datetime",
    ) -> None:
        """
        Seed synthetic liquidity from an OHLCV bar.
        Creates bid/ask levels around the close price so market orders can fill.
        Essential for bar-based backtesting without tick data.
        """
        book: OrderBook = self.get_or_create_book(symbol)
        book.bids.reset()
        book.asks.reset()

        spread: float = close_price * 0.0002  # 2bps spread
        available_qty: float = max(volume * 0.1, 1.0)  # 10% of bar volume

        for i in range(1, 6):
            bid_price: float = round(close_price - spread * i, 2)
            ask_price: float = round(close_price + spread * i, 2)
            level_qty: float = available_qty / (i * 2)

            bid_order: BookOrder = BookOrder(
                order_id=uuid.uuid4().hex[:12],
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                original_qty=level_qty,
                remaining_qty=level_qty,
                timestamp=timestamp,
            )
            ask_order: BookOrder = BookOrder(
                order_id=uuid.uuid4().hex[:12],
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                original_qty=level_qty,
                remaining_qty=level_qty,
                timestamp=timestamp,
            )
            book.bids.insert(bid_order)
            book.asks.insert(ask_order)

    @property
    def books(self) -> Dict[str, OrderBook]:
        return dict(self._books)
