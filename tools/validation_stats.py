"""
Backtest validation statistics: PSR and DSR.
回测验证统计量：PSR（概率夏普比率）与 DSR（紧缩夏普比率）。

References:
  - Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (PSR), 2012
  - Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", JPM 2014

PSR answers: given non-normal returns and finite sample, what is the
probability that the TRUE Sharpe exceeds a benchmark SR*?
DSR sets SR* to the expected maximum Sharpe under N independent trials
(selection bias correction), so DSR < 0.95 means the observed Sharpe is
not distinguishable from the best of N random strategies.

PSR 回答：给定非正态收益与有限样本，真实 Sharpe 超过基准 SR* 的概率。
DSR 将 SR* 设为 N 次独立试验下的期望最大 Sharpe（修正选择偏差）。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

EULER_GAMMA: float = 0.5772156649015329

# ---------------------------------------------------------------------------
# Trial registry: every research configuration ever tried increments the DSR
# n_trials count. HISTORICAL_BASE covers v1-v13 iteration + pre-registry
# research (hyperparameter sweeps, factor sets, strategy variants) — a
# deliberately conservative (high) manual estimate frozen at registry launch.
# trial 登记表：每个试过的研究配置都会增加 DSR 的 n_trials。
# HISTORICAL_BASE 为登记表启用前的人工保守估计，启用后不再改动。
# ---------------------------------------------------------------------------

TRIALS_PATH = Path(__file__).resolve().parent.parent / "trials.json"
HISTORICAL_BASE: int = 56


def _load_trials() -> dict:
    if TRIALS_PATH.exists():
        with open(TRIALS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"historical_base": HISTORICAL_BASE, "trials": []}


def register_trial(name: str, meta: Optional[dict] = None) -> None:
    """Append a trial (idempotent by name). / 登记一次试验（按名幂等）。"""
    data = _load_trials()
    if any(t["name"] == name for t in data["trials"]):
        return
    data["trials"].append({
        "name": name,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta": meta or {},
    })
    with open(TRIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def dsr_n_trials() -> int:
    """Total trials for DSR: historical base + registered. / DSR 用的总试验数。"""
    data = _load_trials()
    return int(data.get("historical_base", HISTORICAL_BASE)) + len(data["trials"])


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF approximation (|err| < 1.15e-9).
    Acklam 逆正态 CDF 近似。"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _moments(rets: Sequence[float]):
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    std = math.sqrt(max(var, 1e-18))
    skew = sum(((r - mean) / std) ** 3 for r in rets) / n
    kurt = sum(((r - mean) / std) ** 4 for r in rets) / n  # raw kurtosis (normal=3)
    return n, mean, std, skew, kurt


def probabilistic_sharpe(rets: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """
    PSR = P[true SR > sr_benchmark], using per-period (NOT annualized) Sharpe.
    PSR：真实 Sharpe 超过基准的概率（按单期 Sharpe 计算，非年化）。
    """
    n, mean, std, skew, kurt = _moments(rets)
    sr = mean / max(std, 1e-18)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return norm_cdf(z)


def expected_max_sharpe(var_sr_trials: float, n_trials: int) -> float:
    """
    E[max SR] across n_trials with per-trial SR variance var_sr_trials,
    under the null that all true SRs are 0 (Bailey-Lopez de Prado 2014).
    N 次试验、真实 SR 全为 0 的零假设下的期望最大（虚假）Sharpe。
    """
    if n_trials <= 1 or var_sr_trials <= 0:
        return 0.0
    return math.sqrt(var_sr_trials) * (
        (1 - EULER_GAMMA) * norm_ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(
    rets: Sequence[float],
    sr_trials: List[float],
    n_trials: int,
) -> dict:
    """
    DSR = PSR with benchmark set to E[max SR] over n_trials.
    `sr_trials`: per-period Sharpe of each tried configuration (used for V[SR]).
    `n_trials`: TOTAL number of configurations ever tried (count honestly —
    every hyperparameter/factor-set/strategy variant across versions).

    DSR：把基准 SR* 设为 n_trials 次试验下的期望最大 Sharpe 后的 PSR。
    n_trials 必须如实计入历史上所有试过的配置。
    """
    k = len(sr_trials)
    if k >= 2:
        m = sum(sr_trials) / k
        var_sr = sum((s - m) ** 2 for s in sr_trials) / (k - 1)
    else:
        var_sr = 0.0
    sr_star = expected_max_sharpe(var_sr, n_trials)
    dsr = probabilistic_sharpe(rets, sr_star)
    n, mean, std, _, _ = _moments(rets)
    return {
        "sr_per_period": mean / max(std, 1e-18),
        "sr_star_expected_max": sr_star,
        "var_sr_across_trials": var_sr,
        "n_trials_assumed": n_trials,
        "psr_vs_zero": probabilistic_sharpe(rets, 0.0),
        "dsr": dsr,
    }
