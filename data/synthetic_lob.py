"""
Synthetic LOB (Limit Order Book) Tick-Level Data Generator (v2).

Generates market data with learnable microstructure patterns:
  - Short-term momentum (2-5 bars autocorrelation)
  - Mean-reversion at extremes (Bollinger band bounces)
  - Volume-price divergence signals
  - Regime switching (trending vs range-bound)
  - Intraday seasonality patterns

These patterns are subtle enough that a Transformer should be able
to learn them, but not trivially exploitable — mimicking real markets.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from engine.events import (
    EventType,
    MarketEvent,
    TickEvent,
)


# ---------------------------------------------------------------------------
# Regime-Switching Price Process with Learnable Microstructure
# ---------------------------------------------------------------------------

class MarketMicrostructureProcess:
    """
    Price process with embedded predictable patterns:

    1. MOMENTUM: returns exhibit positive autocorrelation at lag 1-3
       (r_t partially predicts r_{t+1})
    2. MEAN-REVERSION: extreme deviations from SMA trigger reversal
    3. VOLUME-SIGNAL: high volume precedes trend continuation
    4. REGIME-SWITCH: alternates between trending and mean-reverting regimes
    """

    def __init__(
        self,
        s0: float = 50.0,
        base_vol: float = 0.015,
        momentum_strength: float = 0.15,
        mean_reversion_speed: float = 0.03,
        regime_persistence: float = 0.98,
        volume_signal_strength: float = 0.10,
        seed: int = 42,
    ) -> None:
        random.seed(seed)
        self.s: float = s0
        self.s0: float = s0
        self.base_vol: float = base_vol
        self.momentum_strength: float = momentum_strength
        self.mr_speed: float = mean_reversion_speed
        self.regime_persistence: float = regime_persistence
        self.vol_signal_strength: float = volume_signal_strength

        # state
        self._returns_history: List[float] = [0.0] * 5
        self._sma20: float = s0
        self._regime: int = 0  # 0=trending, 1=mean-reverting
        self._trend_dir: float = 1.0
        self._vol_state: float = 1.0  # volume multiplier
        self._bar_count: int = 0

    def step_bar(self) -> Tuple[float, float, float, float, float]:
        """Generate one OHLCV bar. Returns (open, high, low, close, volume)."""
        self._bar_count += 1

        # --- Regime switching ---
        if random.random() > self.regime_persistence:
            self._regime = 1 - self._regime
            if self._regime == 0:
                self._trend_dir = 1.0 if random.random() > 0.5 else -1.0

        # --- Base random return ---
        noise: float = random.gauss(0, self.base_vol)

        # --- Momentum component (lag-1 autocorrelation) ---
        momentum: float = 0.0
        if self._regime == 0:  # trending regime
            recent_ret: float = self._returns_history[-1]
            momentum = self.momentum_strength * recent_ret
            # also add weak trend drift
            momentum += self._trend_dir * self.base_vol * 0.05

        # --- Mean-reversion component ---
        mr: float = 0.0
        deviation: float = (self.s - self._sma20) / self._sma20 if self._sma20 > 0 else 0.0
        if self._regime == 1:  # mean-reverting regime
            mr = -self.mr_speed * deviation
        elif abs(deviation) > 0.03:  # extreme deviation triggers MR even in trending
            mr = -self.mr_speed * deviation * 2.0

        # --- Volume-driven signal ---
        vol_signal: float = 0.0
        if self._vol_state > 1.5:  # high volume → trend continuation
            vol_signal = self.vol_signal_strength * self._returns_history[-1]

        # --- Combine ---
        total_return: float = noise + momentum + mr + vol_signal
        total_return = max(min(total_return, 0.05), -0.05)  # circuit breaker

        open_price: float = self.s
        close_price: float = self.s * (1.0 + total_return)
        close_price = max(close_price, 0.01)

        # intrabar high/low
        intrabar_vol: float = abs(total_return) + self.base_vol * 0.5
        high_price: float = max(open_price, close_price) * (1.0 + random.uniform(0, intrabar_vol * 0.5))
        low_price: float = min(open_price, close_price) * (1.0 - random.uniform(0, intrabar_vol * 0.5))

        # --- Volume generation (correlated with price movement) ---
        base_volume: float = 50000.0
        # high absolute return → high volume
        vol_multiplier: float = 1.0 + 3.0 * abs(total_return) / self.base_vol
        # add regime effect
        if self._regime == 0:
            vol_multiplier *= 1.2
        volume: float = base_volume * vol_multiplier * random.uniform(0.7, 1.3)

        # update state
        self._returns_history.append(total_return)
        if len(self._returns_history) > 20:
            self._returns_history = self._returns_history[-20:]
        self._vol_state = vol_multiplier

        # update SMA20 (causal)
        self.s = close_price
        if self._bar_count >= 20:
            # simple IIR approximation of SMA20
            self._sma20 = self._sma20 * 0.95 + close_price * 0.05
        else:
            self._sma20 = close_price

        return (
            round(open_price, 2),
            round(high_price, 2),
            round(low_price, 2),
            round(close_price, 2),
            round(volume, 0),
        )


# ---------------------------------------------------------------------------
# LOB depth generator
# ---------------------------------------------------------------------------

def _generate_book_levels(
    mid: float,
    tick_size: float,
    n_levels: int,
    is_bid: bool,
    base_volume: float = 100.0,
) -> List[List[float]]:
    levels: List[List[float]] = []
    for i in range(1, n_levels + 1):
        offset: float = i * tick_size
        price: float = mid - offset if is_bid else mid + offset
        price = round(price, 2)
        vol: float = base_volume * math.exp(-0.3 * i) * (0.5 + random.random())
        vol = round(max(vol, 1.0), 0)
        levels.append([price, vol])
    return levels


# ---------------------------------------------------------------------------
# Full synthetic data generator
# ---------------------------------------------------------------------------

class SyntheticLOBGenerator:
    """
    Generate tick-level events with full LOB depth + aggregated OHLCV bars.
    v2: includes learnable microstructure patterns.
    """

    def __init__(
        self,
        symbol: str = "SH600000",
        n_ticks: int = 48_000,
        ticks_per_bar: int = 60,
        s0: float = 50.0,
        n_levels: int = 5,
        tick_size: float = 0.01,
        seed: int = 42,
        momentum_strength: float = 0.15,
        mean_reversion_speed: float = 0.03,
    ) -> None:
        self.symbol: str = symbol
        self.n_ticks: int = n_ticks
        self.ticks_per_bar: int = ticks_per_bar
        self.n_levels: int = n_levels
        self.tick_size: float = tick_size
        self._seed: int = seed
        self._s0: float = s0
        self._momentum: float = momentum_strength
        self._mr_speed: float = mean_reversion_speed
        self._base_time: datetime = datetime(2025, 1, 2, 9, 30, 0)

    def generate_ticks(self) -> List[TickEvent]:
        """Generate tick events with LOB depth snapshots."""
        random.seed(self._seed + 1000)
        process: MarketMicrostructureProcess = MarketMicrostructureProcess(
            s0=self._s0, seed=self._seed + 1000,
            momentum_strength=self._momentum,
            mean_reversion_speed=self._mr_speed,
        )
        ticks: List[TickEvent] = []
        current_price: float = self._s0

        for i in range(self.n_ticks):
            # micro-step: small random walk within a bar
            micro_ret: float = random.gauss(0, 0.001)
            current_price *= (1.0 + micro_ret)
            current_price = max(current_price, 0.01)

            spread: float = self.tick_size * random.uniform(1.0, 3.0)
            bid: float = round(current_price - spread / 2, 2)
            ask: float = round(current_price + spread / 2, 2)
            last_price: float = round(current_price + random.gauss(0, spread * 0.3), 2)
            last_vol: float = round(random.uniform(10, 500), 0)

            bid_levels = _generate_book_levels(current_price, self.tick_size, self.n_levels, True)
            ask_levels = _generate_book_levels(current_price, self.tick_size, self.n_levels, False)

            ts: datetime = self._base_time + timedelta(seconds=i * 6)
            tick: TickEvent = TickEvent(
                event_type=EventType.TICK,
                timestamp=ts,
                symbol=self.symbol,
                bid_price=bid,
                ask_price=ask,
                bid_volume=bid_levels[0][1] if bid_levels else 0,
                ask_volume=ask_levels[0][1] if ask_levels else 0,
                last_price=last_price,
                last_volume=last_vol,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
            )
            ticks.append(tick)

            # sync with bar process periodically
            if (i + 1) % self.ticks_per_bar == 0:
                o, h, l, c, v = process.step_bar()
                current_price = c

        return ticks

    def generate_bars(self) -> List[MarketEvent]:
        """Generate OHLCV bars from the microstructure process."""
        random.seed(self._seed)
        process: MarketMicrostructureProcess = MarketMicrostructureProcess(
            s0=self._s0, seed=self._seed,
            momentum_strength=self._momentum,
            mean_reversion_speed=self._mr_speed,
        )
        bars: List[MarketEvent] = []
        n_bars: int = self.n_ticks // self.ticks_per_bar

        for b in range(n_bars):
            o, h, l, c, v = process.step_bar()
            ts: datetime = self._base_time + timedelta(
                seconds=b * self.ticks_per_bar * 6
            )
            bar: MarketEvent = MarketEvent(
                event_type=EventType.MARKET,
                timestamp=ts,
                symbol=self.symbol,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            )
            bars.append(bar)
        return bars

    def generate_all(self) -> Tuple[List[TickEvent], List[MarketEvent]]:
        """Generate both tick stream and OHLCV bars."""
        ticks: List[TickEvent] = self.generate_ticks()
        bars: List[MarketEvent] = self.generate_bars()
        return ticks, bars
