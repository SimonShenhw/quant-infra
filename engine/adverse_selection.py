"""
Adverse Selection Micro-Execution Simulator.

Models the reality of limit-order execution in a live market:
  - Queue delay: orders don't fill instantly, must wait in queue
  - Adverse selection: if price moves IN your favor after placing
    a limit order, you almost certainly WON'T get filled (the
    informed flow ate the other side). If price moves AGAINST you,
    you fill 100% and immediately face an unrealised loss.

Rules (per the user specification):
  1. Limit order placed at time T
  2. Check next 3 bars [T+1, T+2, T+3]:
     - If price moves favorably (in direction of your trade): 80% REJECT
       (you're behind informed traders in the queue)
     - If price moves adversely (against your trade): 100% FILL
       (you're providing liquidity to informed flow — adverse selection)
  3. Fill price = your limit price (no improvement)
  4. Taker fallback: if not filled after 3 bars, convert to taker order
     with full taker fee
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class PendingLimitOrder:
    """A limit order waiting in the queue."""
    symbol: str
    side: str         # "BUY" or "SELL"
    price: float
    quantity: float
    placed_bar: int   # bar index when placed
    max_wait: int = 3 # max bars to wait


class AdverseSelectionSimulator:
    """
    Simulates realistic limit-order execution with adverse selection.

    For cross-sectional backtests: called at each rebalance to determine
    actual fill prices and reject rates.
    """

    def __init__(
        self,
        favorable_reject_rate: float = 0.80,
        adverse_fill_rate: float = 1.00,
        taker_fee_bps: float = 4.0,
        maker_fee_bps: float = 1.0,
        max_queue_bars: int = 3,
    ) -> None:
        self._fav_reject: float = favorable_reject_rate
        self._adv_fill: float = adverse_fill_rate
        self._taker_bps: float = taker_fee_bps
        self._maker_bps: float = maker_fee_bps
        self._max_wait: int = max_queue_bars

        # stats
        self.total_orders: int = 0
        self.maker_fills: int = 0
        self.adverse_fills: int = 0
        self.taker_fallbacks: int = 0
        self.rejected: int = 0

    def simulate_execution(
        self,
        side: str,
        limit_price: float,
        quantity: float,
        future_closes: List[float],
    ) -> Tuple[bool, float, float]:
        """
        Simulate execution of a limit order given future price bars.

        Parameters
        ----------
        side : str
            "BUY" or "SELL"
        limit_price : float
            The limit price of the order.
        quantity : float
            Order quantity.
        future_closes : List[float]
            Close prices of the next max_queue_bars bars.

        Returns
        -------
        (filled: bool, fill_price: float, cost_bps: float)
        """
        self.total_orders += 1

        if len(future_closes) == 0:
            # no future data, force taker
            self.taker_fallbacks += 1
            return True, limit_price, self._taker_bps

        # determine if price moved favorably or adversely
        for i, close in enumerate(future_closes[:self._max_wait]):
            if side == "BUY":
                favorable: bool = close < limit_price  # price dropped (good for buyer)
                adverse: bool = close > limit_price     # price rose (bad for buyer)
            else:
                favorable = close > limit_price  # price rose (good for seller)
                adverse = close < limit_price     # price dropped (bad for seller)

            if favorable:
                # favorable move → informed traders are on same side
                # high probability of NOT getting filled (queue position)
                if random.random() < self._fav_reject:
                    # rejected — you're behind in queue
                    self.rejected += 1
                    # fall back to taker after max_wait
                    if i == len(future_closes[:self._max_wait]) - 1:
                        self.taker_fallbacks += 1
                        return True, future_closes[i], self._taker_bps
                    continue
                else:
                    # lucky fill at limit price (20% chance)
                    self.maker_fills += 1
                    return True, limit_price, self._maker_bps

            elif adverse:
                # adverse move → you're the one providing liquidity to informed flow
                # 100% fill, but at YOUR limit price (immediate unrealised loss)
                self.adverse_fills += 1
                return True, limit_price, self._maker_bps

        # exhausted queue window without clear signal → taker fallback
        self.taker_fallbacks += 1
        last_price: float = future_closes[min(self._max_wait - 1, len(future_closes) - 1)]
        return True, last_price, self._taker_bps

    def stats(self) -> Dict[str, float]:
        total: int = max(self.total_orders, 1)
        return {
            "total_orders": self.total_orders,
            "maker_fills": self.maker_fills,
            "adverse_fills": self.adverse_fills,
            "taker_fallbacks": self.taker_fallbacks,
            "rejected": self.rejected,
            "adverse_fill_pct": self.adverse_fills / total,
            "maker_fill_pct": self.maker_fills / total,
            "taker_fallback_pct": self.taker_fallbacks / total,
        }
