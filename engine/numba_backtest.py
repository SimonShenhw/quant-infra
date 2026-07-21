"""
Numba JIT-accelerated backtest loop.
Numba JIT 加速回测循环。

WHAT — a single @njit long-short loop (top-1 long / bottom-1 short with a
min-hold gate, flat per-leg bps cost, per-bar return clamp) plus a thin
wrapper that turns the equity curve into standard metrics.

WHY numba — hyperparameter sweeps run thousands of backtests; ~50x
speedup over the pure-Python loop on 100K+ bar inputs makes that
tractable.  The `njit` fallback shim below keeps the module importable
(the loop just runs slow) when numba is not installed.

Speedup: ~50x over pure Python loop on 100K+ bar backtests.
加速比：纯Python在10万+bar回测上约50倍。
适用于跑成千上万次回测的超参数搜索。

HONESTY NOTE — NOT in the current result path: nothing in the repo
imports this module.  It is a v11.2-era utility frozen with v11
semantics: hourly top1/bottom1 + min_hold (the construction v13 showed
loses -44.5% net), PnL accrued on the rebalance bar itself (predates the
H-1 next-bar fix used by v13), and sqrt(24*365) annualization that
presumes true 1h bars (wrong on the pre-fix ms/µs-corrupted data).  The
v13 published numbers come from the loop inside run_v13_final.py.
Documented, not fixed — see REVIEW_2026-06-10.md ① and H-1.

诚实披露：当前仓库无任何模块导入本文件，不在 v11+ 出结果路径上。它冻结
的是 v11 语义：每小时 top1/bottom1 + min_hold（v13 已证明该构建净亏
44.5%）、换仓当根 bar 即计 PnL（早于 v13 采用的 H-1 次 bar 生效修复）、
sqrt(24*365) 年化假设真 1h bar（对修复前被污染的数据不成立）。v13 的
发布结果出自 run_v13_final.py 内的循环。如实记录、未修复——见
REVIEW_2026-06-10.md。
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
    JIT long-short backtest with min hold + cost + return clamp.
    Numba constraint: no numpy fancy indexing / dicts — hence the manual
    argmax/argmin scan and integer state (cl/cs = current long/short asset
    index, hc = bars held, -1 = no position).

    JIT 多空回测，含最小持仓+成本+收益率限制。受 numba 限制不用高级索引/
    字典，因此手写 argmax/argmin 扫描与整数状态（cl/cs=当前多/空头资产
    下标，hc=已持有 bar 数，-1=空仓）。

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

        # rebalance check: first entry, or hold expired AND target changed
        # 换仓判断：首次建仓，或最小持有期已满且目标资产变化
        need = (cl < 0) or (hc >= min_hold_bars and (nl != cl or ns != cs))
        cost_bar = 0.0
        if need and (hc >= min_hold_bars or cl < 0):
            # legs = closes of old positions + opens of new ones; each leg
            # trades half the book, charged at cost_bps of its notional
            # legs = 平旧仓 + 开新仓的腿数；每条腿动用半仓，按名义额收费
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

        # PnL — accrued on the SAME bar the rebalance happens (pre-H-1
        # semantics: the model's predicted bar is captured instantly; v13's
        # loop defers to the next bar instead — see module HONESTY NOTE)
        # 盈亏——换仓当根 bar 即计入（H-1 修复前语义：白拿被预测的那根 bar；
        # v13 循环改为次 bar 生效，见模块诚实披露）
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
    """High-level wrapper returning standard metrics.  Sharpe is annualized
    with sqrt(24*365), i.e. it ASSUMES true 1h bars (see module HONESTY NOTE).
    顶层封装，返回标准指标。Sharpe 用 sqrt(24*365) 年化——隐含假设输入是
    真 1h bar（见模块诚实披露）。"""
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
