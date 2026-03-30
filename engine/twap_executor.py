"""
TWAP (Time-Weighted Average Price) Execution Layer.

Instead of placing a single all-in limit order (which triggers 100%
adverse selection on unfavorable moves), TWAP splits large orders into
N equal slices executed over N consecutive bars.

Each slice is independently subject to adverse selection, but:
  - Smaller per-slice impact
  - Averaging effect reduces worst-case fill price
  - Some slices fill favorably even when the aggregate direction is adverse
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple


class TWAPExecutor:
    """
    Simulates TWAP execution with adverse selection on each slice.

    Parameters
    ----------
    n_slices : int
        Number of bars over which to spread execution.
    favorable_reject_rate : float
        Probability of rejection when price moves favorably (per slice).
    taker_fee_bps : float
        Fee when a slice falls back to taker execution.
    maker_fee_bps : float
        Fee when a slice fills as maker.
    """

    def __init__(
        self,
        n_slices: int = 4,
        favorable_reject_rate: float = 0.60,
        taker_fee_bps: float = 4.0,
        maker_fee_bps: float = 1.0,
    ) -> None:
        self._n_slices: int = n_slices
        self._fav_reject: float = favorable_reject_rate
        self._taker_bps: float = taker_fee_bps
        self._maker_bps: float = maker_fee_bps

        # stats
        self.total_slices: int = 0
        self.maker_fills: int = 0
        self.adverse_fills: int = 0
        self.taker_fills: int = 0
        self.rejected_slices: int = 0

    def execute_twap(
        self,
        side: str,
        target_notional: float,
        entry_price: float,
        future_closes: List[float],
    ) -> Tuple[float, float, float]:
        """
        Execute a TWAP order over n_slices bars.

        Parameters
        ----------
        side : "BUY" or "SELL"
        target_notional : total notional to execute
        entry_price : price at signal time
        future_closes : close prices for the next n_slices bars

        Returns
        -------
        (avg_fill_price, total_cost_bps, fill_rate)
        """
        n_bars: int = min(self._n_slices, len(future_closes))
        if n_bars == 0:
            return entry_price, self._taker_bps, 0.0

        slice_notional: float = target_notional / n_bars
        filled_notional: float = 0.0
        total_cost: float = 0.0
        weighted_price: float = 0.0
        total_attempted: float = 0.0

        for i in range(n_bars):
            self.total_slices += 1
            bar_close: float = future_closes[i]
            limit_price: float = entry_price  # place at original signal price

            # determine favorable vs adverse for this slice
            if side == "BUY":
                favorable: bool = bar_close < limit_price
            else:
                favorable = bar_close > limit_price

            total_attempted += slice_notional

            if favorable:
                # favorable move — high reject probability
                if random.random() < self._fav_reject:
                    self.rejected_slices += 1
                    # use taker at current bar close (worse price, but filled)
                    filled_notional += slice_notional
                    weighted_price += slice_notional * bar_close
                    total_cost += slice_notional * self._taker_bps / 10000.0
                    self.taker_fills += 1
                else:
                    # lucky maker fill at limit
                    filled_notional += slice_notional
                    weighted_price += slice_notional * limit_price
                    total_cost += slice_notional * self._maker_bps / 10000.0
                    self.maker_fills += 1
            else:
                # adverse move — guaranteed fill at limit price
                filled_notional += slice_notional
                weighted_price += slice_notional * limit_price
                total_cost += slice_notional * self._maker_bps / 10000.0
                self.adverse_fills += 1

        avg_price: float = weighted_price / max(filled_notional, 1e-8)
        cost_bps: float = (total_cost / max(filled_notional, 1e-8)) * 10000.0
        fill_rate: float = filled_notional / max(total_attempted, 1e-8)

        return avg_price, cost_bps, fill_rate

    def stats(self) -> Dict[str, float]:
        t: int = max(self.total_slices, 1)
        return {
            "total_slices": self.total_slices,
            "maker_fill_pct": self.maker_fills / t,
            "adverse_fill_pct": self.adverse_fills / t,
            "taker_fill_pct": self.taker_fills / t,
            "reject_then_taker_pct": self.rejected_slices / t,
        }
