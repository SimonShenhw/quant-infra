"""
Feature Engineering — causal factor primitives + the legacy v2 pipeline.

ALL normalisation is strictly causal (rolling-window only, no future info).
No global statistics. No look-ahead padding.

WHY the paranoia about causality — v1's very first data-leakage bug was a
GLOBAL z-score: normalising each factor with full-sample mean/std leaked
future statistics into every training sample and produced a fake MSE of
1e-6 (README v2 history).  Every primitive here is therefore built to use
only data at or before t:
  - compute_sma / _rolling_std: conv1d LEFT-padded with the series' FIRST
    value — padding with a past-known constant instead of zeros/replicate
    avoids both future leakage and artificial scale jumps at the boundary;
  - compute_ema: plain recursive scan, causal by construction;
  - _rolling_std: single-pass E[x²] − E[x]² (clamped ≥ 0 against
    floating-point negatives) instead of a windowed two-pass;
  - _rolling_zscore: trailing-window mean/std only.

Causality is ENFORCED, not just promised: invariant test t2
(tests/run_invariants.py) scrambles all data after t and asserts factors
at ≤ t are bit-identical.  Run it before touching anything here.

Status — two lives in one file: the primitives (compute_sma, compute_ema,
_rolling_std, _rolling_zscore, compute_rsi, ...) ARE in the current
result path, imported by the factors/ plugin library that feeds v13 and
the paper tracks.  `build_factor_tensor` (the fixed 10-factor pipeline at
the bottom) is the LEGACY v2-era path — used by main.py, run_btc_oos.py,
run_v5–v10 and the legacy paper_trading engine; v11+ builds factors via
FactorRegistry instead.

特征工程——因果因子原语 + 遗留 v2 管线。

所有标准化严格因果（仅使用滚动窗口，无未来信息）。
无全局统计量。无前瞻性填充。

为什么对因果性如此偏执——v1 的第一个数据泄露 bug 就是全局 z-score：
用全样本均值/标准差归一化把未来统计量漏进每个训练样本，造出 1e-6 的
假 MSE（README v2 版本史）。因此每个原语都只用 t 及之前的数据：
conv SMA 用序列首值做左填充（已知的过去常数，避免泄露也避免边界尺度
跳变）；EMA 是递归扫描；滚动标准差用单遍 E[x²]−E[x]²（clamp≥0 防浮点
负数）；z-score 只用尾部窗口。因果性由不变量测试 t2
（tests/run_invariants.py）强制执行：打乱 t 之后的数据，断言 ≤t 的
因子逐位不变——改动本文件前必跑。

状态——一个文件两种身份：原语函数在当前出结果路径上（factors/ 插件库
导入并供 v13 与模拟盘使用）；`build_factor_tensor`（文件底部的固定
10 因子管线）是遗留 v2 路径，v11+ 改经 FactorRegistry 构建因子。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Causal rolling primitives (only use past data)
# 因果滚动原语（仅使用历史数据）
# ---------------------------------------------------------------------------

def compute_returns(close: Tensor) -> Tensor:
    """Simple returns: (close_t - close_{t-1}) / close_{t-1}.
    Index 0 is set to 0.0 (no look-ahead).

    简单收益率。索引0设为0.0（无前瞻）。
    """
    ret: Tensor = torch.zeros_like(close)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1].clamp(min=1e-8)
    return ret


def compute_log_returns(close: Tensor) -> Tensor:
    """Log returns: ln(close_t / close_{t-1}).  Index 0 = 0.0.

    对数收益率。索引0 = 0.0。
    """
    ret: Tensor = torch.zeros_like(close)
    ret[1:] = torch.log(close[1:] / close[:-1].clamp(min=1e-8))
    return ret


def compute_sma(series: Tensor, window: int) -> Tensor:
    """Simple Moving Average — causal only (left-pad with first value).

    Implementation note: conv1d with a uniform kernel after left-padding
    `window-1` copies of series[0].  The first-value pad (vs zero pad)
    keeps early outputs on the series' own scale, so downstream ratios
    like close/sma stay ≈1 instead of exploding during warm-up; the pad
    value is known at t=0, so no future information enters.

    简单移动平均 — 仅因果（左填充首值）。实现：左补 window-1 个 series[0]
    后做均匀核 conv1d。用首值（而非零）填充使暖机期输出保持在序列自身
    量级，下游 close/sma 之类的比率约为 1 而不会爆炸；填充值在 t=0 已知，
    不引入未来信息。
    """
    kernel: Tensor = torch.ones(
        1, 1, window, device=series.device, dtype=series.dtype
    ) / window
    # pad left with the first element (no future leak, no replicate of boundary) / 左填充首元素（无未来泄漏）
    first_val: Tensor = series[0].expand(window - 1)
    padded_series: Tensor = torch.cat([first_val, series])
    x: Tensor = padded_series.unsqueeze(0).unsqueeze(0)
    result: Tensor = torch.nn.functional.conv1d(x, kernel).squeeze(0).squeeze(0)
    return result


def _rolling_std(series: Tensor, window: int) -> Tensor:
    """Causal rolling standard deviation using conv1d (E[x^2] - E[x]^2).

    One-pass variance identity: two uniform convolutions give rolling
    E[x] and E[x²]; var = E[x²] − E[x]² can go slightly negative from
    floating-point cancellation, hence the clamp(min=0) before sqrt.
    Same first-value left-pad as compute_sma keeps it causal.

    因果滚动标准差，使用conv1d实现（E[x^2] - E[x]^2）。
    单遍方差恒等式：两次均匀卷积得到滚动 E[x] 与 E[x²]；浮点相消可能
    使 var 轻微为负，故 sqrt 前 clamp(min=0)。左填充首值方式与
    compute_sma 相同，保持因果。
    """
    first_val: Tensor = series[0].expand(window - 1)
    padded: Tensor = torch.cat([first_val, series])
    x: Tensor = padded.unsqueeze(0).unsqueeze(0)
    kernel: Tensor = torch.ones(
        1, 1, window, device=series.device, dtype=series.dtype
    ) / window
    mean: Tensor = torch.nn.functional.conv1d(x, kernel)
    mean_sq: Tensor = torch.nn.functional.conv1d(x ** 2, kernel)
    var: Tensor = (mean_sq - mean ** 2).clamp(min=0.0)
    return var.squeeze(0).squeeze(0).sqrt()


def compute_ema(series: Tensor, span: int) -> Tensor:
    """Exponential Moving Average — strictly causal recursive scan.

    ema[t] = alpha*x[t] + (1-alpha)*ema[t-1], alpha = 2/(span+1), seeded
    with x[0].  The recursion is inherently sequential, so it runs on CPU
    in a Python loop (acceptable: computed once per series, then cached).

    指数移动平均 — 严格因果递归扫描。alpha = 2/(span+1)，以 x[0] 起始。
    递归天然串行，故在 CPU 上循环计算（可接受：每条序列只算一次并缓存）。
    """
    alpha: float = 2.0 / (span + 1)
    s: Tensor = series.detach().cpu()
    ema: Tensor = torch.empty_like(s)
    ema[0] = s[0]
    for i in range(1, s.size(0)):
        ema[i] = alpha * s[i] + (1.0 - alpha) * ema[i - 1]
    return ema.to(series.device)


def compute_rsi(close: Tensor, period: int = 14) -> Tensor:
    """Relative Strength Index — causal.

    相对强弱指标 — 因果计算。
    """
    delta: Tensor = torch.zeros_like(close)
    delta[1:] = close[1:] - close[:-1]
    gain: Tensor = torch.clamp(delta, min=0.0)
    loss: Tensor = torch.clamp(-delta, min=0.0)
    avg_gain: Tensor = compute_ema(gain, period * 2 - 1)  # Wilder smoothing / Wilder平滑
    avg_loss: Tensor = compute_ema(loss, period * 2 - 1)
    rs: Tensor = avg_gain / avg_loss.clamp(min=1e-8)
    rsi: Tensor = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    close: Tensor, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[Tensor, Tensor]:
    """MACD line and signal line — causal EMA only.

    MACD线和信号线 — 仅使用因果EMA。
    """
    ema_fast: Tensor = compute_ema(close, fast)
    ema_slow: Tensor = compute_ema(close, slow)
    macd_line: Tensor = ema_fast - ema_slow
    signal_line: Tensor = compute_ema(macd_line, signal)
    return macd_line, signal_line


def compute_bollinger(
    close: Tensor, window: int = 20, num_std: float = 2.0
) -> Tuple[Tensor, Tensor, Tensor]:
    """Bollinger Bands — causal rolling mean & std.

    布林带 — 因果滚动均值和标准差。
    """
    mid: Tensor = compute_sma(close, window)
    std: Tensor = _rolling_std(close, window).clamp(min=1e-8)
    upper: Tensor = mid + num_std * std
    lower: Tensor = mid - num_std * std
    return mid, upper, lower


# ---------------------------------------------------------------------------
# Causal rolling z-score normalisation
# 因果滚动z-score标准化
# ---------------------------------------------------------------------------

def _rolling_zscore(series: Tensor, window: int) -> Tensor:
    """Z-score normalisation using ONLY the trailing `window` bars.
    No future information leaks. First `window` bars use expanding window.

    仅使用尾部`window`根K线的z-score标准化。
    无未来信息泄漏。前`window`根K线使用扩展窗口。
    """
    mean: Tensor = compute_sma(series, window)
    std: Tensor = _rolling_std(series, window).clamp(min=1e-8)
    return (series - mean) / std


# ---------------------------------------------------------------------------
# Factor construction
# 因子构建
# ---------------------------------------------------------------------------

def build_factor_tensor(
    open_: Tensor,
    high: Tensor,
    low: Tensor,
    close: Tensor,
    volume: Tensor,
    zscore_window: int = 60,
) -> Tensor:
    """
    Construct the full factor matrix from OHLCV.
    ALL normalisation is causal (rolling z-score, no global stats).

    Returns shape (T, n_factors) where n_factors=8:
      [log_return, sma5_ratio, sma20_ratio, ema10_ratio,
       rsi14, macd_norm, bb_position, volume_zscore]

    Key difference from v1: every factor is expressed as a RATIO or
    already-bounded indicator, then z-scored with a trailing window.
    No global mean/std — no future function.

    从OHLCV构建完整因子矩阵。
    所有标准化均为因果（滚动z-score，无全局统计量）。

    返回形状 (T, n_factors)，n_factors=8:
      [对数收益率, sma5比率, sma20比率, ema10比率,
       rsi14, macd标准化, 布林带位置, 成交量z-score]

    与v1的关键区别: 每个因子表示为比率或已有界指标，
    然后用尾部窗口做z-score。无全局均值/标准差。
    """
    # --- raw factors (all causal) --- / 原始因子（均为因果）
    log_ret: Tensor = compute_log_returns(close)

    # price ratios (dimensionless, avoids absolute-price leakage) / 价格比率（无量纲，避免绝对价格泄漏）
    sma5: Tensor = compute_sma(close, 5)
    sma5_ratio: Tensor = close / sma5.clamp(min=1e-8) - 1.0

    sma20: Tensor = compute_sma(close, 20)
    sma20_ratio: Tensor = close / sma20.clamp(min=1e-8) - 1.0

    ema10: Tensor = compute_ema(close, 10)
    ema10_ratio: Tensor = close / ema10.clamp(min=1e-8) - 1.0

    rsi: Tensor = compute_rsi(close, 14)
    rsi_norm: Tensor = (rsi - 50.0) / 50.0  # already bounded [-1, 1] / 已有界 [-1, 1]

    macd_line, signal_line = compute_macd(close)
    macd_diff: Tensor = macd_line - signal_line

    _, upper, lower = compute_bollinger(close, 20)
    bb_width: Tensor = (upper - lower).clamp(min=1e-8)
    bb_pos: Tensor = (close - lower) / bb_width  # bounded ~[0, 1] / 有界 ~[0, 1]

    vol_mean: Tensor = compute_sma(volume, 20)
    vol_std: Tensor = _rolling_std(volume, 20).clamp(min=1e-8)
    vol_zscore: Tensor = (volume - vol_mean) / vol_std

    # --- OBI microstructure features (from arXiv 2506.05764) --- / OBI微观结构特征
    from model.obi_features import compute_trade_imbalance, compute_price_impact
    trade_imb: Tensor = compute_trade_imbalance(close, volume, window=10)
    price_imp: Tensor = compute_price_impact(close, volume, window=20)

    # --- rolling z-score normalisation (strictly causal) --- / 滚动z-score标准化（严格因果）
    raw_factors: List[Tensor] = [
        log_ret, sma5_ratio, sma20_ratio, ema10_ratio,
        rsi_norm, macd_diff, bb_pos, vol_zscore,
        trade_imb, price_imp,
    ]
    normalised: List[Tensor] = []
    for f in raw_factors:
        normalised.append(_rolling_zscore(f, zscore_window))

    result: Tensor = torch.stack(normalised, dim=-1)  # (T, 8)

    # clamp extreme outliers to prevent gradient explosion / 截断极端异常值以防梯度爆炸
    result = result.clamp(-5.0, 5.0)

    # replace any NaN (from early bars) with 0 / 将NaN（来自早期K线）替换为0
    result = torch.nan_to_num(result, nan=0.0)

    return result
