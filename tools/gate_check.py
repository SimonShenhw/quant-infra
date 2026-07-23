"""
Pre-registered September gate evaluator (read-only). One command renders the
Phase-3 decision table of ROADMAP_2026-07-13 against the live ledgers.
九月 gate 裁决工具（只读）：一条命令把 ROADMAP Phase 3 的预注册判据
对着 live 账本算出当前读数。Usage: python tools/gate_check.py

WHY THIS EXISTS *NOW* (frozen 2026-07-23, ~7 weeks before the gates): the
gates were pre-registered so the criteria could not drift toward the data;
a judge IMPLEMENTED after seeing 90 days of evidence can drift the same way
through operationalization choices (which benchmark, which return series,
which annualization). So every ambiguity in the ROADMAP wording is pinned
here, in code, before the evidence completes:

  * "净值跑赢等权基准" (v13 clause B) — benchmark = equal-weight BUY & HOLD
    of the assets in basket_state's FIRST all_closes vector (2026-06-11,
    20 assets), marked at each day's all_closes, costless. This matches the
    BENCH row of RESEARCH_2026-07-02 ("等权买入持有"), the document the
    gate criteria came from. Costless is conservative (harder to beat).
    An asset that stops printing closes freezes at its last mark.
  * "live IC > 0" (v13 clause A) — judged on paper_live_ic.compute_live_ics(),
    the tool the ROADMAP already names as judge; imported, never re-derived
    (2026-06-10 crosscheck lesson: dual implementations drift).
  * "Sharpe > 0" (carry clause A) — mean/std × sqrt(365) of the DAILY
    all-in net return port_ret + funding_pnl − cost_est. All-in because the
    research that motivated the gate (RESEARCH_2026-07-02 variant D,
    Sharpe 0.53) counted funding inside equity, while the live ledger's
    cumulative_ret deliberately excludes funding (sleeves/book.py, T6).
    The all-in reading was USER-SIGNED as the registered reading on
    2026-07-23; the ex-funding reading stays printed for transparency.
  * carry clause B — SUM(carry_pnl.funding_pnl) > 0, same aggregation as
    health_check.
  * O2 gate (registered late, 2026-07-23, user-signed, at 9/60 marks —
    the track went live 07-14, after the ROADMAP froze): due at 60 daily
    marks (2026-09-12), net Sharpe > 0 AND gross mean daily w·r > 0. The
    gross clause is the live analogue of the w·y24 diagnostic (the number
    that killed every linear construction in the extended-window research):
    o2_pnl.port_ret is pre-cost by ledger convention, so mean(port_ret) IS
    mean daily w·r. If O2 dies, v14 falls back to a carry-only skeleton.
  * Sharpe treats each pnl row as one observation; frozen multi-day
    backfill rows (n_days > 1) count once — same convention as the daily
    research scripts. None exist to date (all n_days = 1 as of 2026-07-23).

Before the due dates every verdict line is labelled "if today were gate
day" — partial-window readings are NOISE, not verdicts (v13 needs ~88 days
just to detect IC 0.064 at t≈2).

中文要点：判据先于证据写死是本 repo 的纪律，但"判据的代码实现"若等到
九月才写，同样有被数据带偏的空间——所以裁决代码提前 ~7 周冻结，把
ROADMAP 文字里的每个歧义点（等权基准=首日 20 币等权买入持有、IC 判官
=paper_live_ic 单一实现、carry Sharpe=含 funding 的全口径日收益 √365
年化=2026-07-23 用户签核的注册口径、ex-funding 仅并示）逐条钉死。
O2 gate 为 07-23 追加注册（用户签核，9/60 天时冻结）：60 记账日到期
（2026-09-12），净 Sharpe>0 且 gross 日均 w·r>0（后者即扩窗研究里杀掉
全部线性构造的 w·y24 诊断的 live 版）；O2 死则 v14 退为 carry 单骨架。
到期前所有结论行都标注"若今天就是 gate 日"。

Exit code: always 0 (informational; health/alerting is health_check's job).
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "paper_daily.db"

# Single-source the registered constants: gate dates from health_check,
# the live-IC judge from paper_live_ic (crosscheck lesson: no second copy).
# 注册常量单一来源：日期取自 health_check，IC 判官取自 paper_live_ic。
try:  # run as `python tools/gate_check.py` (script dir on sys.path)
    from health_check import GATES
    from paper_live_ic import compute_live_ics
except ImportError:  # run as `python -m tools.gate_check` from repo root
    from tools.health_check import GATES
    from tools.paper_live_ic import compute_live_ics

CARRY_GATE = GATES["carry 60d gate"]   # 2026-09-11
V13_GATE = GATES["v13 90d gate"]       # 2026-09-15
O2_GATE = GATES["o2 60d gate"]         # 2026-09-12 (registered 2026-07-23)
V13_TARGET_DAYS = 90
CARRY_TARGET_DAYS = 60
O2_TARGET_DAYS = 60


def _days_between(a: str, b: str) -> int:
    return (datetime.strptime(b, "%Y-%m-%d")
            - datetime.strptime(a, "%Y-%m-%d")).days


def _sharpe_daily(rets) -> tuple[float, float]:
    """(annualized sharpe, t-stat of daily mean). Crypto trades every
    calendar day -> sqrt(365), same convention as the research scripts
    (research_carry_longonly.summarize uses a 365-day year).
    加密全年无休 -> √365 年化，与研究脚本口径一致。"""
    n = len(rets)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(rets) / n
    sd = (sum((x - m) ** 2 for x in rets) / (n - 1)) ** 0.5
    sharpe = m / max(sd, 1e-12) * math.sqrt(365)
    tstat = m / max(sd / math.sqrt(n), 1e-12)
    return sharpe, tstat


def ew_buy_hold_benchmark(conn) -> tuple[float, str, str, int]:
    """Equal-weight buy & hold over basket_state's first all_closes universe.
    Returns (cum_return, start_date, end_date, n_assets). Assets missing on a
    later day freeze at their last available mark (approximates realizing a
    delisting at last price; universe is stable to date so this is currently
    a no-op guard). 等权买入持有基准：首日宇宙，缺币按最后可得价冻结。"""
    rows = conn.execute(
        "SELECT date, all_closes FROM basket_state ORDER BY date").fetchall()
    if not rows:
        return float("nan"), "—", "—", 0
    entry = json.loads(rows[0][1])
    entry = {a: c for a, c in entry.items() if c and c > 0}
    last_mark = dict(entry)
    for _, cj in rows[1:]:
        closes = json.loads(cj)
        for a in entry:
            c = closes.get(a)
            if c and c > 0:
                last_mark[a] = c
    cum = sum(last_mark[a] / entry[a] for a in entry) / len(entry) - 1.0
    return cum, rows[0][0], rows[-1][0], len(entry)


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("=" * 70)
    print(f"  SEPTEMBER GATE CHECK — {today} (UTC)")
    print("  criteria: ROADMAP_2026-07-13 Phase 3 (pre-registered, frozen)")
    print("  judge code: tools/gate_check.py, frozen 2026-07-23 — BEFORE the")
    print("  evidence completes; operationalization pinned in its docstring")
    print("=" * 70)

    if not DB.exists():
        print("  paper_daily.db missing — nothing to judge")
        sys.exit(0)
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    # ---------------- evidence line 1: v13 live ----------------
    left = _days_between(today, V13_GATE)
    marks = conn.execute("SELECT COUNT(*) FROM basket_pnl").fetchone()[0]
    print(f"\n  [1] v13 live — due {V13_GATE} "
          f"({max(left, 0)} days left, {marks}/{V13_TARGET_DAYS} daily marks)")
    v13_alive = None
    if marks == 0:
        print("      no ledger rows yet")
    else:
        n_logged, pairs = compute_live_ics(DB)
        ics = [p[3] for p in pairs]
        v13_cum = conn.execute(
            "SELECT cumulative_ret FROM basket_pnl "
            "ORDER BY date DESC LIMIT 1").fetchone()[0]
        ew_cum, ew_d0, ew_d1, ew_n = ew_buy_hold_benchmark(conn)
        if ics:
            mean_ic = sum(ics) / len(ics)
            _, ic_t = _sharpe_daily(ics)  # t-stat of mean over pair ICs
            a_ok = mean_ic > 0
            print(f"      A  live rank IC > 0        : mean {mean_ic:+.4f} "
                  f"over {len(ics)} pairs (t={ic_t:+.2f})"
                  f"{'':<2}-> {_verdict(a_ok)}")
        else:
            a_ok = None
            print("      A  live rank IC > 0        : no computable pairs yet")
        b_ok = v13_cum > ew_cum
        print(f"      B  beats EW buy&hold bench : v13 {v13_cum:+.2%}  vs  "
              f"EW {ew_cum:+.2%}  (gap {(v13_cum - ew_cum) * 100:+.2f}pp, "
              f"{ew_n} assets since {ew_d0})  -> {_verdict(b_ok)}")
        if a_ok is not None:
            v13_alive = a_ok and b_ok
            tag = (f"if today were gate day — {left} days early" if left > 0
                   else "WINDOW COMPLETE — this is the registered reading")
            print(f"      line reading (A AND B)     : "
                  f"{'ALIVE' if v13_alive else 'DEAD'}  [{tag}]")

    # ---------------- evidence line 2: carry live ----------------
    left_c = _days_between(today, CARRY_GATE)
    crows = conn.execute(
        "SELECT port_ret, funding_pnl, cost_est FROM carry_pnl "
        "ORDER BY date").fetchall()
    print(f"\n  [2] carry live — due {CARRY_GATE} "
          f"({max(left_c, 0)} days left, {len(crows)}/{CARRY_TARGET_DAYS} "
          f"daily marks)")
    carry_alive = None
    if not crows:
        print("      no ledger rows yet")
    else:
        allin = [pr + fp - ce for pr, fp, ce in crows]
        exfund = [pr - ce for pr, _, ce in crows]  # = ledger cum chain (T6)
        s_all, t_all = _sharpe_daily(allin)
        s_ex, _ = _sharpe_daily(exfund)
        fund_sum = sum(fp for _, fp, _ in crows)
        a_all, a_ex = s_all > 0, s_ex > 0
        print(f"      A  Sharpe > 0 (all-in)     : {s_all:+.2f} ann. "
              f"(t={t_all:+.2f}; price+funding-cost)  -> {_verdict(a_all)}")
        print(f"         [ledger ex-funding      : {s_ex:+.2f} ann. — info only;"
              f" registered reading = all-in, user-signed 2026-07-23"
              f"{'' if a_all == a_ex else '; NOTE: signs currently DISAGREE'}]")
        b_ok = fund_sum > 0
        print(f"      B  funding PnL > 0         : {fund_sum:+.4%} "
              f"over {len(crows)} days  -> {_verdict(b_ok)}")
        carry_alive = a_all and b_ok
        tag = (f"if today were gate day — {left_c} days early" if left_c > 0
               else "WINDOW COMPLETE — this is the registered reading")
        print(f"      line reading (A AND B)     : "
              f"{'ALIVE' if carry_alive else 'DEAD'}  [{tag}]")

    # ---------------- evidence line 3: extended-window backtest ----------------
    print("\n  [3] extended-window backtest — RESOLVED 2026-07-13:")
    print("      short-leg tail monetization fragile across 2021-26; model")
    print("      sleeve survived only via magnitude-aware objective (O2).")
    print("      See RESEARCH_2026-07-13_extended_window.md + _objectives.md")

    # ---------------- decision table ----------------
    print("\n  DECISION TABLE (pre-registered; reading is provisional until")
    print("  both windows complete — do NOT act on partial-window noise):")
    if v13_alive is None or carry_alive is None:
        print("      insufficient data to place a reading")
    elif v13_alive and carry_alive:
        print("      -> v14 = multi-sleeve combo (PortfolioBook-level netting)")
    elif not v13_alive and carry_alive:
        print("      -> carry backbone, model demoted to overlay")
    elif not v13_alive and not carry_alive:
        print("      -> back to the research desk; retrain on extended window")
    else:  # v13 alive, carry dead — combination the ROADMAP table omits
        print("      -> v13 alive + carry dead: NOT in the pre-registered")
        print("         table — decide explicitly in September, do not improvise")

    # ---------------- evidence line 4: O2 live ----------------
    # Registered 2026-07-23 (user-signed) at 9/60 marks: the track went live
    # 2026-07-14, after the ROADMAP froze, so this line was added late but
    # still ~7 weeks before its window completes. 07-23 追加注册（用户签核）。
    left_o = _days_between(today, O2_GATE)
    orows = conn.execute(
        "SELECT port_ret, cost_est FROM o2_pnl ORDER BY date").fetchall()
    print(f"\n  [4] O2 live — due {O2_GATE} "
          f"({max(left_o, 0)} days left, {len(orows)}/{O2_TARGET_DAYS} "
          f"daily marks; gate registered 2026-07-23 at 9/60)")
    if orows:
        gross = 1.0
        net = 1.0
        for pr, ce in orows:
            gross *= 1 + pr
            net *= 1 + pr - ce
        s_o2, t_o2 = _sharpe_daily([pr - ce for pr, ce in orows])
        wy = sum(pr for pr, _ in orows) / len(orows)
        a_ok = s_o2 > 0
        b_ok = wy > 0
        print(f"      A  net Sharpe > 0          : {s_o2:+.2f} ann. "
              f"(t={t_o2:+.2f}; net cum {net - 1:+.2%})  -> {_verdict(a_ok)}")
        print(f"      B  gross mean w·r/day > 0  : {wy:+.4%} "
              f"(gross cum {gross - 1:+.2%})  -> {_verdict(b_ok)}")
        print(f"         (B = live analogue of the w·y24 diagnostic that")
        print(f"          killed the linear book; gross exposure still on GP")
        print(f"          ramp toward 1.0 — early Sharpe not comparable)")
        o2_alive = a_ok and b_ok
        tag = (f"if today were gate day — {left_o} days early" if left_o > 0
               else "WINDOW COMPLETE — this is the registered reading")
        print(f"      line reading (A AND B)     : "
              f"{'ALIVE' if o2_alive else 'DEAD'}  [{tag}]")
        print(f"      consequence if DEAD at gate: v14 = carry-only skeleton,")
        print(f"      model side returns to the research desk")
    else:
        print("      no ledger rows yet")

    conn.close()
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
