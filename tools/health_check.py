"""
One-command health check for the unattended paper-trading system (read-only).
无人值守系统健康检查（只读）。Usage: python tools/health_check.py

WHAT: last-run recency, per-ledger continuity/gaps, cumulative returns,
carry funding income, O2 gross ramp, DB backups, run_meta error notes,
September gate countdowns. Exit code: 0 healthy, 1 warnings, 2 critical
(freshest ledger > 3 days stale).

WHY a deadman exists at all: both result-invalidating bugs (v11.1 random
weights, C-1 us timestamps) were SILENT — the pipeline kept "succeeding"
while the output was garbage, and a pipeline that silently STOPS is the
same failure shape wearing different clothes. --deadman (run nightly by the
watchdog scheduled task) therefore converts staleness into a VISIBLE
Desktop artifact (PAPER_RUN_FAILED.txt). It is the second leg of the F1
alert chain: leg 1 = the daily runner's nonzero exit makes loud failures
visible; leg 2 = this watchdog catches the case where the scheduler itself
died and nothing exits at all. The DB is opened read-only by design: a
health probe must never be able to perturb the evidence ledgers.
中文：两个历史致命 bug 都是静默失败——“安静地停摆”同样致命；--deadman
把停摆变成桌面可见哨兵（F1 告警链第二腿：第一腿靠非零退出报响亮故障，
本腿捕获调度器本身死掉、无人报错的情形）；只读打开数据库，体检永远
不可能扰动证据账本。
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "paper_daily.db"
# Pre-registered decision dates (ROADMAP_2026-07-13 Phase 3) — the criteria
# were frozen before the evidence; do not move them to fit the data.
# 预注册 gate 日期：判据先于证据写死，不得事后挪动。
GATES = {"carry 60d gate": "2026-09-11", "v13 90d gate": "2026-09-15"}
STALE_WARN_DAYS = 2   # 1-day lag is normal (timezones/late runs) / 1天滞后属正常
STALE_CRIT_DAYS = 3   # 3+ days = pipeline presumed dead -> Desktop sentinel / 视为停摆


def gaps_in(dates):
    ds = sorted(set(dates))
    out = []
    for a, b in zip(ds[:-1], ds[1:]):
        d0 = datetime.strptime(a, "%Y-%m-%d")
        d1 = datetime.strptime(b, "%Y-%m-%d")
        if (d1 - d0).days > 1:
            out.append(f"{a}..{b}")
    return out


def main():
    status = 0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("=" * 62)
    print(f"  PAPER SYSTEM HEALTH — {today} (UTC)")
    print("=" * 62)

    if not DB.exists():
        print("  CRITICAL: paper_daily.db missing")
        sys.exit(2)
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    ledgers = [
        ("v13 basket", "basket_pnl", "cumulative_ret"),
        ("carry", "carry_pnl", "cumulative_ret"),
        ("combo", "combo_pnl", "cumulative_ret"),
        ("o2", "o2_pnl", "cumulative_ret"),
    ]
    print(f"  {'ledger':<12} {'rows':>5} {'last':>12} {'age(d)':>7} "
          f"{'cum':>9} {'gaps(30d)':>10}")
    print("-" * 62)
    worst_age = 0
    for name, table, col in ledgers:
        rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if rows == 0:
            print(f"  {name:<12} {rows:>5} {'—':>12} {'—':>7} {'—':>9} "
                  f"{'(starts soon)':>12}")
            continue
        last, cum = conn.execute(
            f"SELECT date, {col} FROM {table} ORDER BY date DESC LIMIT 1").fetchone()
        age = (datetime.strptime(today, "%Y-%m-%d")
               - datetime.strptime(last, "%Y-%m-%d")).days
        worst_age = max(worst_age, age)
        recent = [r[0] for r in conn.execute(
            f"SELECT date FROM {table} WHERE date >= date('now', '-30 day') "
            f"ORDER BY date")]
        g = gaps_in(recent)
        print(f"  {name:<12} {rows:>5} {last:>12} {age:>7} {cum:>9.2%} "
              f"{(';'.join(g) if g else 'none'):>10}")

    # carry funding income / carry funding 收入
    f = conn.execute("SELECT SUM(funding_pnl), COUNT(*) FROM carry_pnl").fetchone()
    if f and f[1]:
        print(f"\n  carry funding income to date: {f[0]:+.4%} over {f[1]} days")
    # o2 ramp / o2 毛敞口爬坡
    r = conn.execute("SELECT date, weights FROM o2_state "
                     "ORDER BY date DESC LIMIT 1").fetchone()
    if r:
        w = json.loads(r[1])
        print(f"  o2 gross exposure: {sum(abs(x) for x in w.values()):.3f} "
              f"(target 1.0, GP ramp)")
    conn.close()

    # backups / 备份
    bdir = BASE / "backups"
    n_bak = len(list(bdir.glob("paper_daily_*.db"))) if bdir.exists() else 0
    print(f"  db backups present: {n_bak}")
    if n_bak == 0:
        print("  WARNING: no backups found")
        status = max(status, 1)

    # alert marker / 告警标记
    alert = Path.home() / "Desktop" / "PAPER_RUN_FAILED.txt"
    if alert.exists():
        print(f"  WARNING: alert marker exists: {alert}")
        status = max(status, 1)

    # recent run errors from run_meta / 近期运行错误
    try:
        conn2 = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        errs = conn2.execute(
            "SELECT date, notes FROM run_meta WHERE notes != '' "
            "AND date >= date('now', '-7 day') ORDER BY date DESC").fetchall()
        conn2.close()
        for d, n in errs[:5]:
            print(f"  WARNING: run_meta {d}: {n[:90]}")
            status = max(status, 1)
    except sqlite3.OperationalError:
        pass  # run_meta not created yet / 表尚未创建

    # staleness / 新鲜度
    if worst_age >= STALE_CRIT_DAYS:
        print(f"  CRITICAL: freshest ledger is {worst_age} days old")
        status = 2
    elif worst_age >= STALE_WARN_DAYS:
        print(f"  WARNING: freshest ledger is {worst_age} days old")
        status = max(status, 1)

    # deadman mode: raise/clear the Desktop sentinel on staleness so a
    # dead pipeline becomes VISIBLE without reading logs (review F1)
    # 看门狗模式：停摆超限直接在桌面立哨兵
    if "--deadman" in sys.argv:
        if status == 2:
            alert.write_text(
                f"Paper pipeline STALE: freshest ledger {worst_age} days old "
                f"as of {today} UTC.\nRun: python tools/health_check.py\n",
                encoding="utf-8")
            print(f"  deadman: sentinel written to {alert}")
        elif status == 0 and alert.exists():
            try:
                txt = alert.read_text(encoding="utf-8")
                if "STALE" in txt:  # only clear our own sentinel / 只清自己的哨兵
                    alert.unlink()
                    print("  deadman: stale sentinel cleared")
            except OSError:
                pass

    print()
    for gname, gdate in GATES.items():
        left = (datetime.strptime(gdate, "%Y-%m-%d")
                - datetime.strptime(today, "%Y-%m-%d")).days
        print(f"  {gname}: {gdate} ({left} days left)")

    print(f"\n  STATUS: {'OK' if status == 0 else ('WARN' if status == 1 else 'CRITICAL')}")
    sys.exit(status)


if __name__ == "__main__":
    main()
