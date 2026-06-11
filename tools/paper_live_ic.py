"""
Live rank IC from paper-trading logs.
从模拟盘日志计算 live 全截面 rank IC。

Requires daily_signals.all_closes (logged from 2026-06-10 onward by the
upgraded run_paper_daily.py). For each pair of consecutive logged days,
computes the Spearman rank correlation between day-t scores and the
day-t -> day-t+1 realized returns across all 20 assets, i.e. the live
counterpart of the backtest's OOS rank IC.

需要 daily_signals.all_closes 列（升级版 run_paper_daily.py 自 2026-06-10
起记录）。对每对相邻记录日，计算 t 日打分与 t→t+1 实现收益的横截面
Spearman 相关——即回测 OOS rank IC 的 live 对应物。

Usage: python tools/paper_live_ic.py
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "paper_daily.db"


def spearman(xs, ys) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: -v[i])
        r = [0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / max(vx * vy, 1e-12)


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT date, all_scores, all_closes FROM daily_signals "
        "WHERE all_closes IS NOT NULL ORDER BY date"
    ).fetchall()
    conn.close()

    if len(rows) < 2:
        print(f"Need >= 2 days with all_closes, have {len(rows)}. "
              f"Come back after a few more daily runs.")
        sys.exit(0)

    ics = []
    print(f"{'date_t':<12} {'date_t+1':<12} {'n_assets':>8} {'rank_IC':>9} {'gap_d':>6}")
    print("-" * 52)
    for (d0, s0, c0), (d1, _, c1) in zip(rows[:-1], rows[1:]):
        scores = json.loads(s0)
        closes0 = json.loads(c0)
        closes1 = json.loads(c1)
        common = [a for a in scores if a in closes0 and a in closes1
                  and closes0[a] and closes1[a] and closes0[a] > 0]
        if len(common) < 10:
            continue
        sc = [scores[a] for a in common]
        rt = [closes1[a] / closes0[a] - 1.0 for a in common]
        ic = spearman(sc, rt)
        ics.append(ic)
        from datetime import datetime
        gap = (datetime.strptime(d1, "%Y-%m-%d") - datetime.strptime(d0, "%Y-%m-%d")).days
        print(f"{d0:<12} {d1:<12} {len(common):>8} {ic:>+9.4f} {gap:>6}")

    if not ics:
        print("No computable day pairs yet.")
        return
    n = len(ics)
    mean = sum(ics) / n
    std = (sum((x - mean) ** 2 for x in ics) / max(n - 1, 1)) ** 0.5
    tstat = mean / max(std / math.sqrt(n), 1e-12) if n > 1 else float("nan")
    print("-" * 52)
    print(f"days={n}  mean IC={mean:+.4f}  std={std:.4f}  t-stat={tstat:+.2f}")
    print("(backtest OOS ensemble rank IC is the comparable number; "
          "expect noisy estimates below ~60 days)")


if __name__ == "__main__":
    main()
