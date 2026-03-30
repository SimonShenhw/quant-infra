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

逆向选择微观执行模拟器。

模拟实盘中限价单执行的现实情况：
  - 排队延迟：订单不会立即成交，需在队列中等待
  - 逆向选择：挂限价单后若价格朝有利方向移动，几乎不会成交
    （知情交易者已吃掉对手方）；若价格朝不利方向移动，则100%成交
    并立即面临浮亏。

规则：
  1. 在 T 时刻挂限价单
  2. 检查后续 3 根 K 线 [T+1, T+2, T+3]：
     - 价格有利变动：80% 概率被拒（排队靠后）
     - 价格不利变动：100% 成交（为知情流提供流动性 - 逆向选择）
  3. 成交价 = 限价（无价格改善）
  4. Taker 兜底：3 根 K 线后仍未成交，则转为 Taker 单并支付全额 Taker 费用
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class PendingLimitOrder:
    """A limit order waiting in the queue.
    在队列中等待的限价单。"""
    symbol: str
    side: str         # "BUY" or "SELL" / "买入"或"卖出"
    price: float
    quantity: float
    placed_bar: int   # bar index when placed / 下单时的K线索引
    max_wait: int = 3 # max bars to wait / 最大等待K线数


class AdverseSelectionSimulator:
    """
    Simulates realistic limit-order execution with adverse selection.

    For cross-sectional backtests: called at each rebalance to determine
    actual fill prices and reject rates.

    模拟含逆向选择的真实限价单执行。

    用于截面回测：在每次调仓时调用，确定实际成交价和拒绝率。
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

        # stats / 统计
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

        给定未来价格K线，模拟限价单执行。

        参数
        ----------
        side : str
            "BUY" 或 "SELL"
        limit_price : float
            限价单价格。
        quantity : float
            订单数量。
        future_closes : List[float]
            后续 max_queue_bars 根K线的收盘价。

        返回
        -------
        (是否成交, 成交价, 费用基点)
        """
        self.total_orders += 1

        if len(future_closes) == 0:
            # no future data, force taker / 无未来数据，强制 Taker 成交
            self.taker_fallbacks += 1
            return True, limit_price, self._taker_bps

        # determine if price moved favorably or adversely / 判断价格是有利还是不利变动
        for i, close in enumerate(future_closes[:self._max_wait]):
            if side == "BUY":
                favorable: bool = close < limit_price  # price dropped (good for buyer) / 价格下跌（对买方有利）
                adverse: bool = close > limit_price     # price rose (bad for buyer) / 价格上涨（对买方不利）
            else:
                favorable = close > limit_price  # price rose (good for seller) / 价格上涨（对卖方有利）
                adverse = close < limit_price     # price dropped (bad for seller) / 价格下跌（对卖方不利）

            if favorable:
                # favorable move → informed traders are on same side / 有利变动 → 知情交易者在同侧
                # high probability of NOT getting filled (queue position) / 不成交概率高（排队位置）
                if random.random() < self._fav_reject:
                    # rejected — you're behind in queue / 被拒 - 排队靠后
                    self.rejected += 1
                    # fall back to taker after max_wait / 等待期满后转为 Taker
                    if i == len(future_closes[:self._max_wait]) - 1:
                        self.taker_fallbacks += 1
                        return True, future_closes[i], self._taker_bps
                    continue
                else:
                    # lucky fill at limit price (20% chance) / 幸运地以限价成交（20%概率）
                    self.maker_fills += 1
                    return True, limit_price, self._maker_bps

            elif adverse:
                # adverse move → you're the one providing liquidity to informed flow / 不利变动 → 你在为知情流提供流动性
                # 100% fill, but at YOUR limit price (immediate unrealised loss) / 100%成交，但以你的限价（立即面临浮亏）
                self.adverse_fills += 1
                return True, limit_price, self._maker_bps

        # exhausted queue window without clear signal → taker fallback / 排队窗口耗尽无明确信号 → Taker 兜底
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
