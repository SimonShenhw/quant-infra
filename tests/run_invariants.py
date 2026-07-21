"""
Invariant regression tests (ROADMAP 2026-07-13, cross-cutting).
不变量回归测试。Run BEFORE changing any shared module:

    python tests/run_invariants.py

WHY invariants rather than coverage: both historical result-invalidating
bugs (v11.1 random weights, C-1 us-timestamp corruption) were SILENT
failures that ordinary unit coverage would have blessed — the code ran fine,
the numbers were garbage. Each test therefore pins one FAILURE MODE to a
loud assertion; the ledger tests (T4-T6) assert against hand-derived
ANALYTIC values, never snapshots of current output, so the suite survives
refactors and only breaks when semantics break:

  T1  timestamp-unit golden test (us CSV -> exact hourly bar count)
      -> the C-1 class: ms/us mixing turned 97% of "1h bars" into 5m bars
  T2  no-lookahead (noise after t must not change factors at <= t)
      -> the lookahead class; also the invariant that legitimizes causal
         missed-day backfill (slice [:i+1] == truncated recompute)
  T3  checkpoint fingerprint (factor-list resolution + tamper detection)
      -> the v11.1 class: plausible inference on wrong weights / scrambled
         factor channels (M-2)
  T4  ledger conservation: basket/carry/combo bookkeeping vs analytic values
  T5  ledger conservation: ContinuousBook (O2) vs analytic GP values
  T6  upsert idempotence / gap handling / cumulative-chain telescoping
      -> integrity of the pre-registered September evidence ledgers

All DB tests run on throwaway temp files — the live paper_daily.db is never
touched. No pytest dependency; plain asserts; exit code 1 on any failure.
中文：每条不变量对应一类历史致命故障（共性=静默失败）；账本断言用解析值
而非输出快照，重构不误报、语义坏才报；全部用临时库，绝不碰线上账本。
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import polars as pl
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

import factors as _  # noqa: F401
from factors.base import FactorRegistry
from data.lake_loader import load_klines


def t1_timestamp_golden():
    """ms + us mixed archives -> normalized ms, exact 1h bucket count.

    Golden reconstruction of the C-1 incident: one ms day (2024/12) plus one
    us day (2025/01), exactly mirroring Binance Vision's 2025-01 unit switch.
    Before the fix, `// 3_600_000` bucketed us rows into 3.6-second bins and
    every 5m bar became its own "1h bar" — silently, with plausible output.
    Asserts the loader normalizes units AND the aggregation yields exactly
    48 hourly buckets of 12 bars each (counts, not distributions: a unit bug
    cannot hide in an exact-count golden).
    复刻 C-1 事故的最小黄金样本：ms/µs 各一天，断言归一化后恰好 48 个
    1h 桶、每桶 12 根——精确计数让单位 bug 无处遁形。"""
    with tempfile.TemporaryDirectory() as tmp:
        base_ms = 1_700_000_000_000 - (1_700_000_000_000 % 3_600_000)  # hour-aligned
        day = 24 * 12  # 288 five-minute bars / 一天288根5m
        for sub, unit, day_off in (("2024/12", 1, 0), ("2025/01", 1000, 1)):
            rows = []
            for i in range(day):
                ts_ms = base_ms + day_off * 86_400_000 + i * 300_000
                rows.append({"open_time": ts_ms * unit, "open": 1.0, "high": 2.0,
                             "low": 0.5, "close": 1.5, "volume": 10.0})
            d = Path(tmp) / "TESTUSDT" / sub
            d.mkdir(parents=True)
            pl.DataFrame(rows).write_parquet(d / "klines_5m.parquet")

        df = load_klines("TESTUSDT", data_lake=tmp)
        assert df.height == 2 * day, f"row count {df.height} != {2*day}"
        assert int(df["open_time"].max()) < 10 ** 14, "us timestamps not normalized"
        assert df["open_time"].is_sorted(), "not sorted after normalization"
        agg = df.with_columns((pl.col("open_time") // 3_600_000).alias("h")) \
                .group_by("h").len()
        assert agg.height == 48, f"1h buckets {agg.height} != 48 (golden)"
        assert (agg["len"] == 12).all(), "each hour must contain exactly 12 bars"


def t2_no_lookahead():
    """Replacing data AFTER t with noise must not change factors at <= t.

    Sweeps the ENTIRE registered factor stack (incl. the funding extra) in
    one shot instead of auditing factors one by one: any future-dependent
    primitive anywhere in the pipeline moves some value at <= t0. This is
    also the invariant that makes missed-day backfill honest — backfilled
    scores are computed by slicing rows [:i+1], which equals a true
    historical run ONLY if every factor is causal.
    一次性扫全因子栈的自动化前视检测：t 之后换噪声、t 之前必须逐位不变。
    补课打分的合法性（切片=截断重算）正建立在这条不变量上。"""
    g = torch.Generator().manual_seed(7)
    n, t0 = 400, 300
    c = (torch.randn(n, generator=g) * 0.01 + 1).cumprod(0) * 100
    o = c * (1 + torch.randn(n, generator=g) * 0.001)
    h = torch.maximum(o, c) * 1.001
    l = torch.minimum(o, c) * 0.999
    v = torch.rand(n, generator=g) * 1000 + 1
    fu = torch.randn(n, generator=g) * 1e-4
    names = FactorRegistry.list_factors()

    f1 = FactorRegistry.build_tensor(names, o, h, l, c, v,
                                     zscore_window=48, extras={"funding": fu})

    g2 = torch.Generator().manual_seed(999)
    def scramble(x):
        y = x.clone()
        y[t0 + 1:] = torch.rand(n - t0 - 1, generator=g2) * 500 + 1
        return y
    f2 = FactorRegistry.build_tensor(
        names, scramble(o), scramble(h), scramble(l), scramble(c), scramble(v),
        zscore_window=48, extras={"funding": scramble(fu)})

    diff = (f1[:t0 + 1] - f2[:t0 + 1]).abs().max().item()
    assert diff < 1e-5, (
        f"LOOKAHEAD DETECTED: factors at <= t changed by {diff:.2e} "
        f"when only future data was modified")


def t3_checkpoint_fingerprint():
    """Factor-list resolution must match the ckpt and fail loudly on tamper.

    Guards the v11.1 failure class: paper trading ran 12 days on RANDOM
    weights because initialization silently proceeded — and its cousin M-2,
    where a plausible-but-wrong factor list silently scrambles every input
    channel (v11 ckpts once stored 21 names for a 17-factor model). Asserts
    the production ckpt resolves cleanly against the registry AND that a
    tampered ckpt (count matching neither saved names nor fallback) raises
    instead of guessing.
    锁定 v11.1 类故障：因子清单必须与 ckpt 精确对账，对不上要炸而不是猜——
    静默错位会污染全部输入通道却照常出“像样”的信号。"""
    from run_paper_daily import _resolve_factor_names, load_checkpoint
    ckpt_path = BASE / "checkpoints" / "v13_production.pt"
    assert ckpt_path.exists(), "v13_production.pt missing"
    ckpt = load_checkpoint(ckpt_path, torch.device("cpu"))
    names = _resolve_factor_names(ckpt)
    assert len(names) == int(ckpt["n_factors"]), "factor count mismatch"
    assert names == ckpt["factor_names"], "resolved names differ from ckpt"
    registered = set(FactorRegistry.list_factors())
    assert set(names) <= registered, "ckpt references unregistered factors"

    tampered = dict(ckpt)
    tampered["n_factors"] = 18  # matches neither saved list nor fallback / 两边都对不上
    tampered["factor_names"] = None
    try:
        _resolve_factor_names(tampered)
        raise AssertionError("tampered ckpt did NOT raise")
    except RuntimeError:
        pass


def t4_ledger_conservation():
    """Bookkeeping must reproduce analytic values exactly (1e-9).

    The scenario is engineered so every number is derivable by hand (flat
    100.0 closes, +2%/-1% legs, constant funding rate -> accrual = 3*rate):
    asserting analytic values rather than snapshots is what lets this test
    survive refactors — it pinned the Phase 1 extraction of basket_step/
    carry_step into sleeves.PortfolioBook as bit-for-bit equivalent. Covers
    first-day init, a no-change mark, frozen-day compounding, and a band
    exit with its 2-slot cost.
    场景全部可手算（解析值而非快照），因此重构不误报——Phase 1 把记账抽入
    共享 PortfolioBook 时靠本测试锁定逐位一致；覆盖建仓/无变动/冻结复利/
    出带换槽四种记账日。"""
    from run_paper_daily import (init_db, basket_step, carry_step, combo_step,
                                 EST_COST_BPS)
    syms = [f"A{i:02d}" for i in range(1, 21)]
    ckpt = {"basket_k": 3, "enter_band": 3, "exit_band": 6}
    HOUR = 3_600_000
    t0 = 1_700_000_000_000 - (1_700_000_000_000 % HOUR)
    n_bars = 200
    ts = np.array([t0 + i * HOUR for i in range(n_bars)], dtype=np.int64)
    feats = {s: {"ts": ts} for s in syms}
    rate = 8e-4  # constant funding on every asset / 恒定费率便于解析
    funding = {s: torch.full((n_bars,), rate) for s in syms}

    scores = {s: float(20 - i) for i, s in enumerate(syms)}  # A01 best .. A20 worst
    close1 = {s: 100.0 for s in syms}

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "t4.db")
        conn = init_db(db)

        # day1: init / 首日建仓
        b1 = basket_step(conn, "2026-01-01", scores, close1, ckpt, "test", frozen=False)
        assert sorted(b1["long"]) == ["A01", "A02", "A03"]
        assert sorted(b1["short"]) == ["A18", "A19", "A20"]
        c1 = carry_step(conn, "2026-01-01", scores, close1, int(ts[100]),
                        funding, feats, frozen=False)
        assert c1.get("first")

        # day2: longs +2%, shorts -1%, no rank change / 多头+2%空头-1%排名不变
        close2 = dict(close1)
        for s in ("A01", "A02", "A03"):
            close2[s] = 102.0
        for s in ("A18", "A19", "A20"):
            close2[s] = 99.0
        b2 = basket_step(conn, "2026-01-02", scores, close2, ckpt, "test", frozen=False)
        assert abs(b2["long_ret"] - 0.02) < 1e-9
        assert abs(b2["short_ret"] - 0.01) < 1e-9
        assert abs(b2["port_ret"] - 0.015) < 1e-9
        assert b2["slot_changes"] == 0 and b2["cost_est"] == 0.0
        assert abs(b2["cumulative_ret"] - 0.015) < 1e-9

        # carry day2: 24 bars between marks -> accrual = 24*rate/8 = 3*rate
        # 两mark间24根bar，计提=3*rate；多头付、空头收
        acc = 24 * rate / 8.0
        c2 = carry_step(conn, "2026-01-02", scores, close2, int(ts[124]),
                        funding, feats, frozen=False)
        assert abs(c2["long_ret"] - (0.02 - acc)) < 1e-9
        assert abs(c2["short_ret"] - (0.01 + acc)) < 1e-9
        assert abs(c2["funding_pnl"] - (0.5 * (-acc) + 0.5 * acc)) < 1e-9

        # combo day2 = mean of the two nets / 组合=两净值均值
        co2 = combo_step(conn, "2026-01-02")
        expect = 0.5 * b2["port_ret"] + 0.5 * c2["port_ret"]
        assert abs(co2["combo_ret"] - expect) < 1e-9
        assert abs(co2["cumulative_ret"] - expect) < 1e-9

        # day3 frozen (backfill): longs +1%, cumulative compounds / 冻结日复利
        close3 = dict(close2)
        for s in ("A01", "A02", "A03"):
            close3[s] = close2[s] * 1.01
        b3 = basket_step(conn, "2026-01-03", scores, close3, ckpt, "test", frozen=True)
        assert b3["slot_changes"] == 0
        assert abs(b3["port_ret"] - 0.005) < 1e-9
        assert abs(b3["cumulative_ret"] - ((1.015) * (1.005) - 1)) < 1e-9

        # day4: A03 falls to rank 7 -> exits, A04 enters; cost = 2 slots
        # A03跌出exit band被A04替换，2个槽位成本
        scores4 = dict(scores)
        scores4["A03"] = 13.5  # between A07(14) and A08(13) -> rank 7 / 排名第7
        close4 = dict(close3)
        b4 = basket_step(conn, "2026-01-04", scores4, close4, ckpt, "test", frozen=False)
        assert sorted(b4["long"]) == ["A01", "A02", "A04"], b4["long"]
        assert b4["slot_changes"] == 2
        expect_cost = 2 * (0.5 / 3) * (EST_COST_BPS / 10000.0)
        assert abs(b4["cost_est"] - expect_cost) < 1e-12
        conn.close()


def t5_continuous_book_conservation():
    """ContinuousBook (O2 sleeve ledger) must match analytic GP values.

    Same analytic-values philosophy as T4, for the v14 continuous-weight
    ledger: the GP recursion w' = (1-tau)w + tau*target has closed-form
    turnover (tau on the ramp day, tau*(1-tau) the next flat day), so the
    ledger's turnover/cost/cum chain is asserted against those exact
    numbers. Guards the ramp-from-zero start that matches the gated
    backtest, frozen-day zero-turnover, and cost = |dw| x 8bps.
    GP 递推有闭式换手（首日 tau、次日 tau(1-tau)），账本必须精确复现；
    锁定从零爬坡、冻结日零换手与 |Δw|×8bps 成本口径。"""
    from run_paper_daily import init_db
    from sleeves import ContinuousBook
    syms = [f"A{i:02d}" for i in range(1, 17)]
    scores = {s: float(16 - i) for i, s in enumerate(syms)}
    close1 = {s: 100.0 for s in syms}
    tau, bps = 0.2, 8.0

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t5.db"))
        book = ContinuousBook("o2_state", "o2_pnl", cost_bps=bps, tau=tau)

        # day1: GP ramp from zero -> w = tau*target, turnover = tau (gross 1)
        # 首日从零起步：w=tau*target，换手=tau
        r1 = book.step(conn, "2026-01-01", scores, close1, 1000, frozen=False)
        assert r1.get("first")
        assert abs(r1["turnover"] - tau) < 1e-9
        assert abs(sum(abs(x) for x in r1["weights"].values()) - tau) < 1e-9

        # day2: flat prices, same scores -> port 0, turnover = tau*(1-tau)
        # 次日价格持平同打分：收益0，换手=tau(1-tau)
        r2 = book.step(conn, "2026-01-02", scores, close1, 2000, frozen=False)
        assert abs(r2["port_ret"]) < 1e-12
        assert abs(r2["turnover"] - tau * (1 - tau)) < 1e-9
        expect_cost = tau * (1 - tau) * bps / 10000.0
        assert abs(r2["cost_est"] - expect_cost) < 1e-12
        assert abs(r2["cumulative_ret"] - (-expect_cost)) < 1e-12

        # day3 frozen: +1% on every positive-weight asset -> port = sum(w+)*1%
        # 冻结日：正权重资产+1%
        w2 = r2["weights"]
        close3 = {s: (101.0 if w2[s] > 0 else 100.0) for s in syms}
        r3 = book.step(conn, "2026-01-03", scores, close3, 3000, frozen=True)
        pos_gross = sum(x for x in w2.values() if x > 0)
        assert abs(r3["port_ret"] - pos_gross * 0.01) < 1e-12
        assert r3["turnover"] == 0.0
        conn.close()


def t6_orchestration_upsert_and_chain():
    """Same-day rerun idempotence, multi-day gaps, and cumulative-chain
    telescoping — the evidence-integrity semantics the September gate
    depends on: a rerun must UPSERT one row recomputed from yesterday (not
    duplicate or compound on itself), a gap must record its true n_days and
    compound once, and every stored cum must equal the running product of
    stored nets — so the ledger can be audited end-to-end at gate time.
    同日重跑幂等（单行、从昨日重算）、跨日 gap 记真实天数且单次复利、
    存储的 cum 恒等于净收益连乘——九月裁决时账本可全程复核。"""
    from run_paper_daily import init_db, basket_step
    syms = [f"A{i:02d}" for i in range(1, 21)]
    ckpt = {"basket_k": 3, "enter_band": 3, "exit_band": 6}
    scores = {s: float(20 - i) for i, s in enumerate(syms)}

    def closes(mult_long):
        c = {s: 100.0 for s in syms}
        for s in ("A01", "A02", "A03"):
            c[s] = 100.0 * mult_long
        return c

    with tempfile.TemporaryDirectory() as tmp:
        conn = init_db(os.path.join(tmp, "t6.db"))
        basket_step(conn, "2026-01-01", scores, closes(1.00), ckpt, "t", False)

        # late run writes day2 with +1%, then same-day rerun revises to +3%:
        # must UPSERT (single row), recomputed from day1 closes / 同日重跑幂等
        basket_step(conn, "2026-01-02", scores, closes(1.01), ckpt, "t", False)
        r2 = basket_step(conn, "2026-01-02", scores, closes(1.03), ckpt, "t", False)
        n_rows = conn.execute(
            "SELECT COUNT(*) FROM basket_pnl WHERE date='2026-01-02'").fetchone()[0]
        assert n_rows == 1, "same-day rerun duplicated the pnl row"
        assert abs(r2["long_ret"] - 0.03) < 1e-9, "rerun did not recompute from day1"

        # 3-day gap: n_days recorded, chain compounds once / 跨日gap单次复利
        r5 = basket_step(conn, "2026-01-05", scores, closes(1.03 * 1.02),
                         ckpt, "t", False)
        assert r5["n_days"] == 3
        assert abs(r5["long_ret"] - 0.02) < 1e-9

        # telescoping: stored cum == product of stored nets / 复利链守恒
        rows = conn.execute(
            "SELECT port_ret, cost_est, cumulative_ret FROM basket_pnl "
            "ORDER BY date").fetchall()
        acc = 1.0
        for port, cost, cum in rows:
            acc *= (1 + port - cost)
            assert abs(cum - (acc - 1)) < 1e-9, "cumulative chain broken"
        conn.close()


def main():
    """Run every invariant even after a failure (one broken invariant must
    not mask another), print the x/6 tally, exit nonzero on any failure.
    全部跑完不早退（故障不互相遮蔽），任一失败以非零码退出。"""
    tests = [t1_timestamp_golden, t2_no_lookahead,
             t3_checkpoint_fingerprint, t4_ledger_conservation,
             t5_continuous_book_conservation, t6_orchestration_upsert_and_chain]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} invariants hold")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
