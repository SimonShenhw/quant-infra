"""
TWAP (Time-Weighted Average Price) Execution Layer.

Instead of placing a single all-in limit order (which triggers 100%
adverse selection on unfavorable moves), TWAP splits large orders into
N equal slices executed over N consecutive bars.

Each slice is independently subject to adverse selection, but:
  - Smaller per-slice impact
  - Averaging effect reduces worst-case fill price
  - Some slices fill favorably even when the aggregate direction is adverse

TWAP（时间加权平均价格）执行层。

不一次性下全部限价单（不利变动时触发100%逆向选择），而是将大单拆分为
N 个等额切片，在连续 N 根 K 线上分批执行。

每个切片独立受逆向选择影响，但：
  - 单笔冲击更小
  - 均价效应降低最差成交价
  - 即使总体方向不利，部分切片仍可能有利成交
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

    模拟每个切片均受逆向选择影响的 TWAP 执行。

    参数
    ----------
    n_slices : int
        分批执行的K线数。
    favorable_reject_rate : float
        价格有利变动时每个切片的拒绝概率。
    taker_fee_bps : float
        切片回退为 Taker 时的费率。
    maker_fee_bps : float
        切片作为 Maker 成交时的费率。
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

        # stats / 统计
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

        在 n_slices 根K线上执行 TWAP 订单。

        参数
        ----------
        side : "BUY" 或 "SELL"
        target_notional : 待执行的总名义金额
        entry_price : 信号触发时的价格
        future_closes : 后续 n_slices 根K线的收盘价

        返回
        -------
        (平均成交价, 总费用基点, 成交率)
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
            limit_price: float = entry_price  # place at original signal price / 以原始信号价格挂单

            # determine favorable vs adverse for this slice / 判断本切片是有利还是不利
            if side == "BUY":
                favorable: bool = bar_close < limit_price
            else:
                favorable = bar_close > limit_price

            total_attempted += slice_notional

            if favorable:
                # favorable move — high reject probability / 有利变动 - 高拒绝概率
                if random.random() < self._fav_reject:
                    self.rejected_slices += 1
                    # use taker at current bar close (worse price, but filled) / 以当前K线收盘价 Taker 成交（价格更差，但保证成交）
                    filled_notional += slice_notional
                    weighted_price += slice_notional * bar_close
                    total_cost += slice_notional * self._taker_bps / 10000.0
                    self.taker_fills += 1
                else:
                    # lucky maker fill at limit / 幸运地以限价 Maker 成交
                    filled_notional += slice_notional
                    weighted_price += slice_notional * limit_price
                    total_cost += slice_notional * self._maker_bps / 10000.0
                    self.maker_fills += 1
            else:
                # adverse move — guaranteed fill at limit price / 不利变动 - 以限价保证成交
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
