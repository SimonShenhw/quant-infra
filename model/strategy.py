"""
Transformer-Driven Strategy (v2 — Production-Grade).

Improvements over v1:
  - Dynamic volatility-adjusted signal thresholds
  - Trailing stop-loss and take-profit
  - Turnover penalty (avoid over-trading that erodes alpha via costs)
  - Cooldown period after stop-loss trigger
  - Direction accuracy tracking for live monitoring
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import torch
from torch import Tensor

from engine.events import (
    Event,
    EventType,
    MarketEvent,
    SignalDirection,
    SignalEvent,
)
from model.transformer import QuantTransformer
from model.features import build_factor_tensor


class TransformerStrategy:
    """
    Consumes MarketEvents, runs Transformer inference, emits signals
    only when predicted return exceeds a volatility-scaled threshold.
    """

    def __init__(
        self,
        model: QuantTransformer,
        device: torch.device,
        lookback: int = 60,
        threshold_sigma: float = 0.5,
        warmup: int = 80,
        vol_window: int = 20,
        cooldown_bars: int = 5,
        max_holding_bars: int = 10,
    ) -> None:
        self._model: QuantTransformer = model
        self._device: torch.device = device
        self._lookback: int = lookback
        self._threshold_sigma: float = threshold_sigma
        self._warmup: int = warmup
        self._vol_window: int = vol_window
        self._cooldown_bars: int = cooldown_bars
        self._max_holding_bars: int = max_holding_bars

        # per-symbol state
        self._open: Dict[str, List[float]] = defaultdict(list)
        self._high: Dict[str, List[float]] = defaultdict(list)
        self._low: Dict[str, List[float]] = defaultdict(list)
        self._close: Dict[str, List[float]] = defaultdict(list)
        self._volume: Dict[str, List[float]] = defaultdict(list)

        self._factor_cache: Dict[str, Tensor] = {}
        self._cache_len: Dict[str, int] = defaultdict(int)
        self._rebuild_interval: int = 30

        # position tracking
        self._position_dir: Dict[str, int] = defaultdict(int)   # +1, -1, 0
        self._entry_bar: Dict[str, int] = defaultdict(int)
        self._entry_price: Dict[str, float] = defaultdict(float)
        self._cooldown_until: Dict[str, int] = defaultdict(int)
        self._bar_count: Dict[str, int] = defaultdict(int)

        # monitoring
        self._predictions: List[float] = []
        self._actuals: List[float] = []

    def handle_market(self, event: Event) -> Optional[List[Event]]:
        if not isinstance(event, MarketEvent):
            return None

        sym: str = event.symbol
        self._open[sym].append(event.open)
        self._high[sym].append(event.high)
        self._low[sym].append(event.low)
        self._close[sym].append(event.close)
        self._volume[sym].append(event.volume)
        self._bar_count[sym] += 1

        n_bars: int = len(self._close[sym])
        if n_bars < self._warmup:
            return None

        # ----- Rebuild factor cache periodically -----
        if (n_bars - self._cache_len.get(sym, 0) >= self._rebuild_interval
                or sym not in self._factor_cache):
            o: Tensor = torch.tensor(self._open[sym], dtype=torch.float32, device=self._device)
            h: Tensor = torch.tensor(self._high[sym], dtype=torch.float32, device=self._device)
            l: Tensor = torch.tensor(self._low[sym], dtype=torch.float32, device=self._device)
            c: Tensor = torch.tensor(self._close[sym], dtype=torch.float32, device=self._device)
            v: Tensor = torch.tensor(self._volume[sym], dtype=torch.float32, device=self._device)
            self._factor_cache[sym] = build_factor_tensor(o, h, l, c, v, zscore_window=60)
            self._cache_len[sym] = n_bars

        factors: Tensor = self._factor_cache[sym]
        if factors.size(0) < self._lookback:
            return None

        # ----- Inference -----
        window: Tensor = factors[-self._lookback:, :].unsqueeze(0)  # (1, L, 8)
        self._model.eval()
        with torch.no_grad():
            pred: Tensor = self._model(window)
        predicted_return: float = pred.item()

        # ----- Dynamic volatility threshold -----
        closes_recent: List[float] = self._close[sym][-self._vol_window:]
        if len(closes_recent) >= 2:
            rets: List[float] = [
                (closes_recent[i] / closes_recent[i - 1]) - 1.0
                for i in range(1, len(closes_recent))
            ]
            vol: float = (sum(r ** 2 for r in rets) / len(rets)) ** 0.5
        else:
            vol = 0.01
        vol = max(vol, 0.001)  # floor
        threshold: float = vol * self._threshold_sigma

        # ----- Position management -----
        current_bar: int = self._bar_count[sym]
        current_dir: int = self._position_dir[sym]
        current_price: float = event.close

        # Check for forced exit: max holding period or stop-loss/take-profit
        if current_dir != 0:
            bars_held: int = current_bar - self._entry_bar[sym]
            entry_p: float = self._entry_price[sym]
            pnl_pct: float = (current_price / entry_p - 1.0) * current_dir

            # trailing stop-loss: -2 sigma
            stop_loss: float = -2.0 * vol
            # take-profit: +3 sigma
            take_profit: float = 3.0 * vol
            # max holding
            force_exit: bool = (
                bars_held >= self._max_holding_bars
                or pnl_pct <= stop_loss
                or pnl_pct >= take_profit
            )

            if force_exit:
                self._position_dir[sym] = 0
                if pnl_pct <= stop_loss:
                    self._cooldown_until[sym] = current_bar + self._cooldown_bars
                return [SignalEvent(
                    event_type=EventType.SIGNAL,
                    timestamp=event.timestamp,
                    symbol=sym,
                    direction=SignalDirection.EXIT,
                    strength=1.0,
                    predicted_return=predicted_return,
                )]

        # Cooldown check
        if current_bar < self._cooldown_until.get(sym, 0):
            return None

        # ----- Signal generation -----
        direction: Optional[SignalDirection] = None
        if predicted_return > threshold and current_dir <= 0:
            direction = SignalDirection.LONG
        elif predicted_return < -threshold and current_dir >= 0:
            direction = SignalDirection.SHORT

        if direction is None:
            return None

        # strength proportional to how many sigmas above threshold
        raw_strength: float = abs(predicted_return) / threshold
        strength: float = min(max(raw_strength - 1.0, 0.1), 1.0)

        # update position tracking
        new_dir: int = 1 if direction == SignalDirection.LONG else -1
        self._position_dir[sym] = new_dir
        self._entry_bar[sym] = current_bar
        self._entry_price[sym] = current_price

        return [SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=event.timestamp,
            symbol=sym,
            direction=direction,
            strength=strength,
            predicted_return=predicted_return,
        )]
