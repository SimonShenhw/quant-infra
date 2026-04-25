"""
Adaptive position sizing — RL-inspired Kelly with regime awareness.
自适应仓位管理 — RL 启发的 Kelly + 状态感知。

Not full RL (avoid the complexity). Instead:
  - Track recent prediction quality (rolling win rate, avg PnL)
  - Adjust position size based on confidence + recent performance
  - Reduce size after drawdown (conservative)
  - Increase after streaks (momentum in your own equity)

不是完整RL，避免复杂性。改用：
  - 跟踪最近预测质量（滚动胜率、平均PnL）
  - 根据置信度+近期表现调整仓位
  - 回撤后降仓位（保守）
  - 连胜后加仓位（自身权益的动量）
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional


class AdaptivePositionSizer:
    """
    Adaptive sizing based on rolling Kelly + drawdown control.
    基于滚动 Kelly + 回撤控制的自适应仓位。
    """

    def __init__(
        self,
        base_size: float = 0.5,        # baseline notional fraction / 基线仓位比例
        max_size: float = 1.0,         # cap / 上限
        min_size: float = 0.1,         # floor / 下限
        kelly_window: int = 30,        # bars to estimate win rate / 滚动窗口
        kelly_safety: float = 0.5,     # half-Kelly / 半凯利
        max_drawdown_thresh: float = 0.10,  # cut size if dd > this / 触发降仓的回撤阈值
    ) -> None:
        self._base = base_size
        self._max = max_size
        self._min = min_size
        self._win_buffer: Deque[float] = deque(maxlen=kelly_window)
        self._kelly_safety = kelly_safety
        self._dd_thresh = max_drawdown_thresh

        # equity tracking / 权益跟踪
        self._peak_eq: float = 0.0
        self._cur_eq: float = 0.0

    def update_pnl(self, trade_pnl: float, current_equity: float) -> None:
        """Record trade outcome + equity. / 记录交易结果+权益。"""
        self._win_buffer.append(trade_pnl)
        self._cur_eq = current_equity
        if current_equity > self._peak_eq:
            self._peak_eq = current_equity

    def get_size(self, signal_confidence: float = 1.0) -> float:
        """
        Return position size as notional fraction.
        返回仓位的名义比例。

        signal_confidence: 0-1, scales further (1 = full conf, 0 = skip).
        """
        # not enough data → use base size / 数据不足时用基线仓位
        if len(self._win_buffer) < 10:
            return min(self._max, max(self._min, self._base * signal_confidence))

        wins = [p for p in self._win_buffer if p > 0]
        losses = [p for p in self._win_buffer if p < 0]
        n_wins = len(wins)
        n_total = len(self._win_buffer)
        win_rate = n_wins / max(n_total, 1)

        if not wins or not losses:
            kelly = 0.0
        else:
            avg_win = sum(wins) / len(wins)
            avg_loss = abs(sum(losses) / len(losses))
            b = avg_win / max(avg_loss, 1e-9)
            # Kelly: f* = (p*b - q) / b
            kelly = (win_rate * b - (1 - win_rate)) / max(b, 1e-9)
            kelly = max(0.0, kelly) * self._kelly_safety

        # base + kelly adjustment / 基线 + Kelly 调整
        size = self._base * (1.0 + kelly)

        # drawdown reduction / 回撤降仓
        if self._peak_eq > 0:
            dd = (self._peak_eq - self._cur_eq) / self._peak_eq
            if dd > self._dd_thresh:
                # linear scale-down: at dd=2*thresh, size→min / 线性降仓
                scale = max(0.0, 1.0 - (dd - self._dd_thresh) / self._dd_thresh)
                size = self._min + (size - self._min) * scale

        size *= signal_confidence
        return min(self._max, max(self._min, size))

    def stats(self) -> dict:
        if not self._win_buffer:
            return {"win_rate": 0.0, "n_trades": 0}
        wins = [p for p in self._win_buffer if p > 0]
        return {
            "win_rate": len(wins) / max(len(self._win_buffer), 1),
            "n_trades": len(self._win_buffer),
            "avg_pnl": sum(self._win_buffer) / max(len(self._win_buffer), 1),
            "peak_equity": self._peak_eq,
            "current_equity": self._cur_eq,
        }
