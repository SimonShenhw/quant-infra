"""
Regime-conditioned IC analysis of the v13 OOS predictions (ROADMAP Phase 2.2).
v13 OOS 预测的 regime 条件 IC 分析。

HYPOTHESIS (from RESEARCH_2026-07-02.md): v13's backtest alpha is
short-leg concentrated and BEAR-REGIME dependent — i.e. the short leg only
worked because the 2024-09..2026-03 window was mostly a falling market, and
the live June-2026 bleeding (shorting strong coins in an up-market) is that
artifact surfacing. If true, short-leg alpha should collapse in the
up-trend bucket.

METHOD: bucket every decision point by
  - TREND: BTC close vs its causal 200d (4800-bar) SMA  -> up / down
    (also 100d for coverage on the shorter history)
  - VOL:   30d realized vol of the EW basket -> full-sample terciles
    (diagnostic bucketing, NOT a tradeable rule — noted in output)
and report per bucket: OOS ensemble rank IC vs the 24h label, LONG-leg
alpha (model top-3 minus EW basket, 24h fwd), SHORT-leg alpha (EW basket
minus model bottom-3). WHY leg alphas and not just IC: IC measures the
whole ranking; the baskets only trade the tails, so the legs can die while
IC survives.

VERDICT (2026-07-13, appendix of RESEARCH_2026-07-02.md): hypothesis
REJECTED within this window — short-leg alpha in the UP-regime (+0.138%/24h)
is NOT lower than in the down-regime (+0.095%), and rank IC is positive in
every bucket (ALL +0.064). In-window, the short alpha is fine in up-markets.
That leaves only two candidate explanations for the live divergence: (1) the
specific June-July 2026 regime (broad alt rally) has NO representative
sample inside the 19-month window — which made the 2021+ extended-window
retrain (research_extended_window.py) the key evidence line, and indeed the
tail-level regime dependence only became visible there; (2) unmodeled live
frictions. Note IC_t already deflates effective N by 24 for the overlapping
24h labels.

结论：「空头腿 alpha = 熊市伪影」假设在本窗口内不成立——上涨 regime 的
空头 alpha（+0.138%/24h）不低于下跌 regime（+0.095%），各桶 IC 全为正。
live 亏损的解释只剩：① 2026 年 6-7 月的山寨普涨结构在 19 个月窗口内没有
代表性样本（由此把扩窗重训定为关键证据线，后来尾部层面的 regime 依赖
确实只在 2021 入窗后才现形）；② 回测未建模的 live 摩擦。IC_t 已按 24h
标签重叠把有效样本数缩减 24 倍。

Registers itself in trials.json (one trial).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from run_v13_final import build_from_parquet
from tools.research_carry_longonly import rebuild_oos_predictions

SEQ_LEN = 24
SMA_TREND_BARS = {"200d": 200 * 24, "100d": 100 * 24}
VOL_WINDOW = 30 * 24


def rank_ic_rows(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    pr = (-pred).argsort(axis=1).argsort(axis=1).astype(float)
    tr = (-tgt).argsort(axis=1).argsort(axis=1).astype(float)
    pm = pr.mean(axis=1, keepdims=True)
    tm = tr.mean(axis=1, keepdims=True)
    cov = ((pr - pm) * (tr - tm)).sum(axis=1)
    den = np.sqrt(((pr - pm) ** 2).sum(axis=1) * ((tr - tm) ** 2).sum(axis=1))
    return cov / np.maximum(den, 1e-12)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  REGIME-CONDITIONED IC (v13 OOS predictions)")
    print("=" * 70)

    X, y24_t, r1h_t, close_t, syms, _ = build_from_parquet(SEQ_LEN, 20, device)
    pred_t, valid_t = rebuild_oos_predictions(X, device)
    pred = pred_t.cpu().numpy()
    y24 = y24_t.cpu().numpy()
    r1h = r1h_t.cpu().numpy()
    close = close_t.cpu().numpy()
    n = pred.shape[0]
    btc_j = syms.index("BTCUSDT")
    dec = np.arange(n) + SEQ_LEN - 1  # decision close index / 决策时点

    ic = rank_ic_rows(pred, y24)

    # leg alphas at each decision point (24h horizon) / 各时点多空腿alpha
    ew24 = np.zeros(n)
    long_alpha = np.zeros(n)
    short_alpha = np.zeros(n)
    for t in range(n):
        fwd = y24[t]
        ew = fwd.mean()
        order = np.argsort(-pred[t])
        long_alpha[t] = fwd[order[:3]].mean() - ew
        short_alpha[t] = ew - fwd[order[-3:]].mean()
        ew24[t] = ew

    # trend buckets / 趋势分桶
    btc = close[:, btc_j]
    csum = np.cumsum(btc)
    buckets = {}
    for name, w in SMA_TREND_BARS.items():
        sma = np.full(len(btc), np.nan)
        sma[w:] = (csum[w:] - csum[:-w]) / w
        up = btc[dec] > sma[dec]
        has = ~np.isnan(sma[dec])
        buckets[f"trend{name}_up"] = has & up
        buckets[f"trend{name}_down"] = has & ~up

    # vol terciles (full-sample, diagnostic only) / 波动三分位（诊断用）
    br = r1h.mean(axis=1)
    rv = np.full(n, np.nan)
    for t in range(VOL_WINDOW, n):
        rv[t] = br[t - VOL_WINDOW:t].std()
    q1, q2 = np.nanquantile(rv, [1 / 3, 2 / 3])
    buckets["vol_low"] = rv < q1
    buckets["vol_mid"] = (rv >= q1) & (rv < q2)
    buckets["vol_high"] = rv >= q2

    valid = valid_t.cpu().numpy()
    print(f"\n  {'bucket':<18} {'n':>6} {'rank_IC':>8} {'IC_t':>6} "
          f"{'longA_24h':>10} {'shortA_24h':>10} {'EW_24h':>8}")
    print("-" * 70)
    out = {}
    rows = [("ALL", valid)] + [(k, valid & m) for k, m in buckets.items()]
    for name, mask in rows:
        m = mask & ~np.isnan(ic)
        k = int(m.sum())
        if k < 100:
            continue
        # ~24h overlap per decision → effective N deflated by 24. WHY: adjacent
        # hourly samples share ~23/24 of their 24h label window, so treating
        # them as independent would inflate the t-stat by ~sqrt(24) (~4.9x).
        # 相邻小时样本的 24h 标签窗口重叠 ~23/24，按独立样本算 t 值会虚高
        # ~4.9 倍，故有效 N 除以 24。
        ic_t = ic[m].mean() / max(ic[m].std(), 1e-12) * np.sqrt(k / 24)
        out[name] = {
            "n": k, "rank_ic": float(ic[m].mean()), "ic_t_eff": float(ic_t),
            "long_alpha_24h": float(long_alpha[m].mean()),
            "short_alpha_24h": float(short_alpha[m].mean()),
            "ew_24h": float(ew24[m].mean()),
        }
        r = out[name]
        print(f"  {name:<18} {k:>6} {r['rank_ic']:>+8.4f} {r['ic_t_eff']:>6.1f} "
              f"{r['long_alpha_24h']:>+10.4%} {r['short_alpha_24h']:>+10.4%} "
              f"{r['ew_24h']:>+8.4%}")

    print("\n  notes: leg alpha = 24h fwd return edge of the model's top/bottom-3")
    print("  vs the EW basket; vol terciles are full-sample (diagnostic only);")
    print("  IC_t uses N/24 effective observations for the 24h label overlap.")

    with open("checkpoints/research_regime_ic.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  saved: checkpoints/research_regime_ic.json")

    # register trial / 登记试验
    try:
        from tools.validation_stats import register_trial
        register_trial("regime_conditioned_ic_2026-07-13",
                       {"kind": "diagnostic", "buckets": list(out.keys())})
    except ImportError:
        pass


if __name__ == "__main__":
    main()
