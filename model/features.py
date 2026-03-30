"""
Feature Engineering — computes technical factors from OHLCV data.

ALL normalisation is strictly causal (rolling-window only, no future info).
No global statistics. No look-ahead padding.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Causal rolling primitives (only use past data)
# ---------------------------------------------------------------------------

def compute_returns(close: Tensor) -> Tensor:
    """Simple returns: (close_t - close_{t-1}) / close_{t-1}.
    Index 0 is set to 0.0 (no look-ahead)."""
    ret: Tensor = torch.zeros_like(close)
    ret[1:] = (close[1:] - close[:-1]) / close[:-1].clamp(min=1e-8)
    return ret


def compute_log_returns(close: Tensor) -> Tensor:
    """Log returns: ln(close_t / close_{t-1}).  Index 0 = 0.0."""
    ret: Tensor = torch.zeros_like(close)
    ret[1:] = torch.log(close[1:] / close[:-1].clamp(min=1e-8))
    return ret


def compute_sma(series: Tensor, window: int) -> Tensor:
    """Simple Moving Average — causal only (left-pad with first value)."""
    kernel: Tensor = torch.ones(
        1, 1, window, device=series.device, dtype=series.dtype
    ) / window
    # pad left with the first element (no future leak, no replicate of boundary)
    first_val: Tensor = series[0].expand(window - 1)
    padded_series: Tensor = torch.cat([first_val, series])
    x: Tensor = padded_series.unsqueeze(0).unsqueeze(0)
    result: Tensor = torch.nn.functional.conv1d(x, kernel).squeeze(0).squeeze(0)
    return result


def _rolling_std(series: Tensor, window: int) -> Tensor:
    """Causal rolling standard deviation using conv1d (E[x^2] - E[x]^2)."""
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
    """Exponential Moving Average — strictly causal recursive scan."""
    alpha: float = 2.0 / (span + 1)
    s: Tensor = series.detach().cpu()
    ema: Tensor = torch.empty_like(s)
    ema[0] = s[0]
    for i in range(1, s.size(0)):
        ema[i] = alpha * s[i] + (1.0 - alpha) * ema[i - 1]
    return ema.to(series.device)


def compute_rsi(close: Tensor, period: int = 14) -> Tensor:
    """Relative Strength Index — causal."""
    delta: Tensor = torch.zeros_like(close)
    delta[1:] = close[1:] - close[:-1]
    gain: Tensor = torch.clamp(delta, min=0.0)
    loss: Tensor = torch.clamp(-delta, min=0.0)
    avg_gain: Tensor = compute_ema(gain, period * 2 - 1)  # Wilder smoothing
    avg_loss: Tensor = compute_ema(loss, period * 2 - 1)
    rs: Tensor = avg_gain / avg_loss.clamp(min=1e-8)
    rsi: Tensor = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    close: Tensor, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[Tensor, Tensor]:
    """MACD line and signal line — causal EMA only."""
    ema_fast: Tensor = compute_ema(close, fast)
    ema_slow: Tensor = compute_ema(close, slow)
    macd_line: Tensor = ema_fast - ema_slow
    signal_line: Tensor = compute_ema(macd_line, signal)
    return macd_line, signal_line


def compute_bollinger(
    close: Tensor, window: int = 20, num_std: float = 2.0
) -> Tuple[Tensor, Tensor, Tensor]:
    """Bollinger Bands — causal rolling mean & std."""
    mid: Tensor = compute_sma(close, window)
    std: Tensor = _rolling_std(close, window).clamp(min=1e-8)
    upper: Tensor = mid + num_std * std
    lower: Tensor = mid - num_std * std
    return mid, upper, lower


# ---------------------------------------------------------------------------
# Causal rolling z-score normalisation
# ---------------------------------------------------------------------------

def _rolling_zscore(series: Tensor, window: int) -> Tensor:
    """Z-score normalisation using ONLY the trailing `window` bars.
    No future information leaks. First `window` bars use expanding window."""
    mean: Tensor = compute_sma(series, window)
    std: Tensor = _rolling_std(series, window).clamp(min=1e-8)
    return (series - mean) / std


# ---------------------------------------------------------------------------
# Factor construction
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
    """
    # --- raw factors (all causal) ---
    log_ret: Tensor = compute_log_returns(close)

    # price ratios (dimensionless, avoids absolute-price leakage)
    sma5: Tensor = compute_sma(close, 5)
    sma5_ratio: Tensor = close / sma5.clamp(min=1e-8) - 1.0

    sma20: Tensor = compute_sma(close, 20)
    sma20_ratio: Tensor = close / sma20.clamp(min=1e-8) - 1.0

    ema10: Tensor = compute_ema(close, 10)
    ema10_ratio: Tensor = close / ema10.clamp(min=1e-8) - 1.0

    rsi: Tensor = compute_rsi(close, 14)
    rsi_norm: Tensor = (rsi - 50.0) / 50.0  # already bounded [-1, 1]

    macd_line, signal_line = compute_macd(close)
    macd_diff: Tensor = macd_line - signal_line

    _, upper, lower = compute_bollinger(close, 20)
    bb_width: Tensor = (upper - lower).clamp(min=1e-8)
    bb_pos: Tensor = (close - lower) / bb_width  # bounded ~[0, 1]

    vol_mean: Tensor = compute_sma(volume, 20)
    vol_std: Tensor = _rolling_std(volume, 20).clamp(min=1e-8)
    vol_zscore: Tensor = (volume - vol_mean) / vol_std

    # --- OBI microstructure features (from arXiv 2506.05764) ---
    from model.obi_features import compute_trade_imbalance, compute_price_impact
    trade_imb: Tensor = compute_trade_imbalance(close, volume, window=10)
    price_imp: Tensor = compute_price_impact(close, volume, window=20)

    # --- rolling z-score normalisation (strictly causal) ---
    raw_factors: List[Tensor] = [
        log_ret, sma5_ratio, sma20_ratio, ema10_ratio,
        rsi_norm, macd_diff, bb_pos, vol_zscore,
        trade_imb, price_imp,
    ]
    normalised: List[Tensor] = []
    for f in raw_factors:
        normalised.append(_rolling_zscore(f, zscore_window))

    result: Tensor = torch.stack(normalised, dim=-1)  # (T, 8)

    # clamp extreme outliers to prevent gradient explosion
    result = result.clamp(-5.0, 5.0)

    # replace any NaN (from early bars) with 0
    result = torch.nan_to_num(result, nan=0.0)

    return result
