"""
Numba JIT-accelerated backtest loop.
Numba JIT 加速回测循环。

Speedup: ~50x over pure Python loop on 100K+ bar backtests.
For hyperparameter sweeps where you run thousands of backtests.

加速比：纯Python在10万+bar回测上约50倍。
适用于跑成千上万次回测的超参数搜索。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(f=None, **kwargs):
        if f is None: return lambda x: x
        return f


@njit(cache=True)
def backtest_long_short_jit(
    pred_matrix: np.ndarray,    # (T, A) predicted scores
    returns_1h: np.ndarray,     # (T, A) actual 1h returns
    min_hold_bars: int = 48,
    cost_bps: float = 4.0,
    return_clamp: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Vectorized long-short backtest with min hold + cost + return clamp.
    向量化的多空回测，含最小持仓+成本+收益率限制。

    Returns
    -------
    eq_curve : (T+1,) equity curve / 权益曲线
    rebalances_arr : (T,) 1 if rebalance else 0
    n_rebalances : int
    """
    T, A = pred_matrix.shape
    eq_curve = np.zeros(T + 1, dtype=np.float64)
    eq_curve[0] = 1_000_000.0
    rebalances_arr = np.zeros(T, dtype=np.int32)

    cl = -1
    cs = -1
    hc = 0
    rebalances = 0
    equity = 1_000_000.0

    for t in range(T):
        # find top/bottom asset / 找top/bot资产
        scores = pred_matrix[t]
        nl = 0
        ns = 0
        max_s = scores[0]
        min_s = scores[0]
        for j in range(1, A):
            if scores[j] > max_s:
                max_s = scores[j]
                nl = j
            if scores[j] < min_s:
                min_s = scores[j]
                ns = j

        # rebalance check / 换仓判断
        need = (cl < 0) or (hc >= min_hold_bars and (nl != cl or ns != cs))
        cost_bar = 0.0
        if need and (hc >= min_hold_bars or cl < 0):
            legs = 0
            if cl != nl and cl >= 0: legs += 1
            if cs != ns and cs >= 0: legs += 1
            if cl != nl: legs += 1
            if cs != ns: legs += 1
            if legs > 0:
                cost_bar = equity * 0.5 * (cost_bps / 10000.0) * legs
            cl = nl
            cs = ns
            hc = 0
            rebalances += 1
            rebalances_arr[t] = 1

        # PnL / 盈亏
        pr = 0.0
        if cl >= 0: pr += 0.5 * returns_1h[t, cl]
        if cs >= 0: pr -= 0.5 * returns_1h[t, cs]
        pr -= cost_bar / max(equity, 1.0)

        # clamp / 限幅
        if pr > return_clamp: pr = return_clamp
        if pr < -return_clamp: pr = -return_clamp

        equity *= (1.0 + pr)
        eq_curve[t + 1] = equity
        hc += 1

    return eq_curve, rebalances_arr, rebalances


def run_backtest(
    pred_matrix: np.ndarray, returns_1h: np.ndarray,
    min_hold_bars: int = 48, cost_bps: float = 4.0, return_clamp: float = 0.10,
) -> Dict[str, float]:
    """High-level wrapper returning standard metrics. / 顶层封装，返回标准指标。"""
    eq, _, n_reb = backtest_long_short_jit(
        pred_matrix, returns_1h, min_hold_bars, cost_bps, return_clamp,
    )
    rets = (eq[1:] / eq[:-1]) - 1.0
    avg = rets.mean() if len(rets) > 0 else 0
    std = rets.std() if len(rets) > 1 else 1e-9
    sharpe = (avg / max(std, 1e-9)) * np.sqrt(24 * 365)
    peak = np.maximum.accumulate(eq)
    max_dd = ((peak - eq) / peak).max()
    return {
        "total_return": eq[-1] / eq[0] - 1.0,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "rebalances": int(n_reb),
        "final_equity": float(eq[-1]),
    }
