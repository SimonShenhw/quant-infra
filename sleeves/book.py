"""
PortfolioBook: one bookkeeping engine for every sleeve ledger.
组合账本：所有 sleeve 共用的记账引擎。(ROADMAP Phase 1)

WHAT — reproduces the exact row formats of the pre-refactor basket_step /
carry_step (paper_daily.db tables basket_state/basket_pnl and
carry_state/carry_pnl are LIVE series — schemas must not change). Two layouts:

  "basket": state(date, long, short, scores, closes, tag)
            pnl(date, prev_date, n_days, prev_long, prev_short,
                long_ret, short_ret, port_ret, slot_changes, cost_est, cum)
  "carry":  state(date, mark_ts, long, short, signals, closes)
            pnl(... + funding_pnl before slot_changes)

WHY schema fidelity is the whole point of this refactor: the ledgers are
the PRE-REGISTERED evidence for the September v14 gate (ROADMAP_2026-07-13).
A refactor that "improved" a column mid-series would fork that evidence, so
the Phase 1 extraction had to be bit-for-bit ledger-equivalent to the old
inline code — pinned by tests/run_invariants.py T4 against analytic values.

Semantics (identical to the originals):
  - PnL is always marked on the PREVIOUS state row: positions take effect
    at the NEXT daily mark (the H-1 no-decision-bar-lookahead rule).
  - frozen=True: positions unchanged, zero slots/cost. Missed days are
    days nobody traded, so freezing is the only honest accounting —
    recomputing positions retroactively would fabricate fills.
  - net = port_ret - cost_est compounds into cumulative_ret (T6 pins the
    telescoping of this chain).
  - cost = slots x (gross/k) x 8bps, charged single-sided per traded slot —
    the conservative flat quote cross-checked by the independent second
    engine (ENGINE_CROSSCHECK_2026-06-10 lower bound).
  - carry accrual: longs PAY the window's funding sum, shorts RECEIVE it.
中文：账本 schema 冻结（九月 gate 预注册证据），重构必须与旧实现逐位一致
（T4 用解析值锁定）；PnL 永远按前一状态行计（H-1 无前视）；冻结日=诚实
记账（没人交易就不假装成交）；成本口径与独立第二引擎交叉验证的保守
8bps/边一致；复利链守恒由 T6 锁定。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Dict, List, Optional

from sleeves.banding import banded_update_symbols


class PortfolioBook:
    """Date-keyed banded-basket ledger over a caller-owned sqlite connection.
    One instance per sleeve, pointed at that sleeve's FROZEN table pair;
    step() advances exactly one daily mark. The caller owns transactions
    (commit/rollback) so one sick ledger can be isolated without losing the
    others' day. / 每个 sleeve 一本账、表结构冻结；step() 推进一个记账日；
    事务由调用方掌控，便于账本间故障隔离。"""

    def __init__(
        self,
        layout: str,               # "basket" | "carry"
        state_table: str,
        pnl_table: str,
        cost_bps: float = 8.0,     # per traded slot notional / 每槽位成本
        gross_per_leg: float = 0.5,
    ) -> None:
        assert layout in ("basket", "carry"), layout
        self.layout = layout
        self.state_table = state_table
        self.pnl_table = pnl_table
        self.cost_bps = cost_bps
        self.gross = gross_per_leg

    # -- row IO / 行读写 ----------------------------------------------------

    def _read_prev(self, conn, date: str):
        """Most recent state row strictly BEFORE `date`. PnL marks against
        this row, so a same-day rerun recomputes from yesterday instead of
        compounding on its own earlier write (T6 idempotence).
        取严格早于当日的最近状态行：同日重跑因此从昨日重算而非自我叠加。"""
        if self.layout == "basket":
            row = conn.execute(
                f"SELECT date, long_assets, short_assets, all_closes "
                f"FROM {self.state_table} WHERE date < ? "
                f"ORDER BY date DESC LIMIT 1", (date,)).fetchone()
            if row is None:
                return None
            return {"date": row[0], "long": json.loads(row[1]),
                    "short": json.loads(row[2]), "closes": json.loads(row[3]),
                    "mark_ts": None}
        row = conn.execute(
            f"SELECT date, mark_ts, long_assets, short_assets, all_closes "
            f"FROM {self.state_table} WHERE date < ? "
            f"ORDER BY date DESC LIMIT 1", (date,)).fetchone()
        if row is None:
            return None
        return {"date": row[0], "mark_ts": int(row[1]),
                "long": json.loads(row[2]), "short": json.loads(row[3]),
                "closes": json.loads(row[4])}

    def _write_state(self, conn, date, new_l, new_s, sig, closes, tag, mark_ts):
        """Upsert today's state row (INSERT OR REPLACE on the date PK —
        rerun-idempotent, M-4). / 按 date 主键 upsert 状态行，重跑幂等。"""
        if self.layout == "basket":
            conn.execute(
                f"INSERT OR REPLACE INTO {self.state_table} VALUES (?,?,?,?,?,?)",
                (date, json.dumps(new_l), json.dumps(new_s),
                 json.dumps(sig), json.dumps(closes), tag))
        else:
            conn.execute(
                f"INSERT OR REPLACE INTO {self.state_table} VALUES (?,?,?,?,?,?)",
                (date, mark_ts, json.dumps(new_l), json.dumps(new_s),
                 json.dumps(sig), json.dumps(closes)))

    def _write_pnl(self, conn, date, prev, n_days, long_ret, short_ret,
                   port_ret, funding_pnl, slot_changes, cost_est, cum_ret):
        """Upsert today's pnl row; prev_long/prev_short record the basket
        the PnL was actually earned on (H-1). / upsert 损益行；prev_* 记录
        实际赚取该损益的持仓（H-1）。"""
        if self.layout == "basket":
            conn.execute(
                f"INSERT OR REPLACE INTO {self.pnl_table} VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (date, prev["date"], n_days,
                 json.dumps(prev["long"]), json.dumps(prev["short"]),
                 long_ret, short_ret, port_ret, slot_changes, cost_est, cum_ret))
        else:
            conn.execute(
                f"INSERT OR REPLACE INTO {self.pnl_table} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (date, prev["date"], n_days,
                 json.dumps(prev["long"]), json.dumps(prev["short"]),
                 long_ret, short_ret, port_ret, funding_pnl,
                 slot_changes, cost_est, cum_ret))

    # -- the step / 单日推进 -------------------------------------------------

    def step(
        self,
        conn, date: str,
        sig: Dict[str, float], closes: Dict[str, float],
        k: int, enter_band: int, exit_band: int,
        frozen: bool,
        tag: Optional[str] = None,
        mark_ts: Optional[int] = None,
        accrual_fn: Optional[Callable[[int, int], Dict[str, float]]] = None,
    ) -> Optional[Dict]:
        """Advance one ledger day: mark PnL on YESTERDAY's basket at today's
        closes, then write today's banded basket — in that order, so
        positions always take effect at the NEXT mark (H-1).

        frozen=True (missed-day backfill) keeps positions with zero
        slots/cost; the first-ever day only initializes (no prior basket to
        mark). accrual_fn (carry layout) maps the two mark timestamps to
        per-asset funding paid by longs over that window. Returns a summary
        dict, or None when frozen with no prior state.
        单日推进：先按昨日持仓、今日收盘记 PnL（H-1：仓位下个 mark 生效），
        再写今日篮子；冻结日零槽位零成本；首日仅建仓。"""
        prev = self._read_prev(conn, date)

        if prev is None:
            if frozen:
                return None  # nothing to freeze / 无仓可冻结
            new_l, new_s = banded_update_symbols(sig, [], [], k, enter_band, exit_band)
            self._write_state(conn, date, new_l, new_s, sig, closes, tag, mark_ts)
            return {"date": date, "long": new_l, "short": new_s,
                    "slot_changes": 2 * k, "first": True}

        acc: Dict[str, float] = {}
        if self.layout == "carry" and accrual_fn is not None:
            acc = accrual_fn(prev["mark_ts"], mark_ts)

        # Equal-weight mean within a leg; sign=-1 flips shorts so their gains
        # are positive. Funding uses the same convention: longs pay the
        # accrual, shorts receive it. A name missing a close drops out of the
        # mean (rather than counting as a fake 0% return).
        # 腿内等权；空头符号翻转；funding 多头付、空头收；缺价的币退出均值
        # 而不是被记成假的 0% 收益。
        def leg_ret(assets: List[str], sign: float) -> float:
            rets = []
            for a in assets:
                pc, tc = prev["closes"].get(a), closes.get(a)
                if pc and tc and pc > 0:
                    rets.append(sign * (tc / pc - 1.0) - sign * acc.get(a, 0.0))
            return sum(rets) / len(rets) if rets else 0.0

        long_ret = leg_ret(prev["long"], +1.0)
        short_ret = leg_ret(prev["short"], -1.0)
        port_ret = 0.5 * long_ret + 0.5 * short_ret
        funding_pnl = 0.0
        if self.layout == "carry":
            # Reporting decomposition ONLY: the accrual is already inside
            # long_ret/short_ret above — never added to port_ret again. It is
            # stored separately because the carry gate criterion is
            # "funding PnL > 0" (ROADMAP Phase 3).
            # 仅为分解展示：计提已含在腿收益内，不重复计入；单列存储是因为
            # carry gate 判据之一就是 funding PnL > 0。
            funding_pnl = (
                0.5 * (sum(-acc.get(a, 0.0) for a in prev["long"]) / max(len(prev["long"]), 1))
                + 0.5 * (sum(+acc.get(a, 0.0) for a in prev["short"]) / max(len(prev["short"]), 1)))
        # True calendar span of this mark (gaps once got booked as "one day",
        # review M-4); floor of 1 keeps same-date math safe.
        # 记录真实跨越天数（跨日 gap 曾被记成一天，M-4）。
        n_days = max((datetime.strptime(date, "%Y-%m-%d")
                      - datetime.strptime(prev["date"], "%Y-%m-%d")).days, 1)

        if frozen:
            new_l, new_s = list(prev["long"]), list(prev["short"])
            slot_changes = 0
        else:
            new_l, new_s = banded_update_symbols(sig, prev["long"], prev["short"],
                                                 k, enter_band, exit_band)
            # symmetric diff counts BOTH the leaver and the entrant of a swap
            # 对称差同时计入离场与进场，一次换人=2个槽位
            slot_changes = (len(set(prev["long"]) ^ set(new_l))
                            + len(set(prev["short"]) ^ set(new_s)))
        # Per-slot notional is gross/k; 8bps charged single-sided on each
        # changed slot — the conservative flat quote the second engine
        # cross-checked (vs ~5bps/side realized in the TWAP model).
        # 每槽位名义额 gross/k，按 8bps/边计费（交叉验证过的保守口径）。
        cost_est = slot_changes * (self.gross / k) * (self.cost_bps / 10000.0)

        cum_cur = conn.execute(
            f"SELECT cumulative_ret FROM {self.pnl_table} WHERE date < ? "
            f"ORDER BY date DESC LIMIT 1", (date,)).fetchone()
        prev_cum = cum_cur[0] if cum_cur else 0.0
        # NET return compounds (gross minus cost); the telescoping of this
        # chain — stored cum == product of stored nets — is invariant T6.
        # 复利链按净收益推进，链条守恒由 T6 锁定。
        cum_ret = (1 + prev_cum) * (1 + port_ret - cost_est) - 1.0

        self._write_pnl(conn, date, prev, n_days, long_ret, short_ret,
                        port_ret, funding_pnl, slot_changes, cost_est, cum_ret)
        self._write_state(conn, date, new_l, new_s, sig, closes, tag, mark_ts)

        out = {"date": date, "prev_date": prev["date"], "n_days": n_days,
               "long_ret": long_ret, "short_ret": short_ret,
               "port_ret": port_ret, "cost_est": cost_est,
               "cumulative_ret": cum_ret,
               "long": new_l, "short": new_s, "slot_changes": slot_changes}
        if self.layout == "carry":
            out["funding_pnl"] = funding_pnl
        return out


class ContinuousBook:
    """
    Ledger for continuous-weight sleeves (v14 O2 model: demeaned-score
    weights, Garleanu-Pedersen partial trading w_t = (1-tau)w + tau*target).
    连续权重账本（O2 sleeve：打分去均值权重 + GP 部分调仓）。

    WHY GP partial trading instead of full rebalance: the extended-window
    research killed every full-rebalance construction on the FULL 2021-2026
    window (six variants, all negative); "trade partially toward the aim"
    (Garleanu & Pedersen, JF 2013) with tau=1/5 is the exact configuration
    the pre-registered O2 gate backtest passed with
    (RESEARCH_2026-07-13_extended_window) — live replicates that setup, it
    does not "improve" on it.

    Tables (created by the caller's init_db):
      {state}: date PK, mark_ts, weights(json sym->w), all_scores, all_closes
      {pnl}:   date PK, prev_date, n_days, port_ret, turnover, cost_est, cum
    Semantics:
      - frozen=True keeps weights (missed-day backfill = honest accounting);
      - the first live day trades tau of the way from ZERO (GP ramp): the
        gated backtest starts from an empty book, so the live track must
        start the same way — gross approaches 1.0 only over ~1/tau days;
      - cost = turnover x 8bps single-sided with turnover = sum|dw| — the
        same conservative cross-checked flat quote as PortfolioBook;
      - conservation pinned by tests/run_invariants.py T5.
    语义：冻结日权重不动；首日从零按 GP 渐进建仓（与被 gate 的回测同起点，
    毛敞口需 ~1/tau 天爬到 1.0）；成本=|Δw|×8bps 单边（保守交叉验证口径）；
    记账守恒由 T5 锁定。
    """

    def __init__(self, state_table: str, pnl_table: str,
                 cost_bps: float = 8.0, tau: float = 0.2) -> None:
        self.state_table = state_table
        self.pnl_table = pnl_table
        self.cost_bps = cost_bps
        self.tau = tau

    @staticmethod
    def target_weights(scores: Dict[str, float]) -> Dict[str, float]:
        """Demeaned scores normalized to gross 1 (sum|w| = 1) — dollar-neutral
        by construction. Degenerate all-equal scores return an all-zero book
        instead of dividing by ~0.
        打分去均值再归一化到总敞口1（构造性美元中性）；打分全同退化为空仓。"""
        syms = sorted(scores)
        v = [scores[s] for s in syms]
        m = sum(v) / len(v)
        raw = {s: scores[s] - m for s in syms}
        g = sum(abs(x) for x in raw.values())
        if g < 1e-12:
            return {s: 0.0 for s in syms}
        return {s: x / g for s, x in raw.items()}

    def step(self, conn, date: str, scores: Dict[str, float],
             closes: Dict[str, float], mark_ts: Optional[int],
             frozen: bool) -> Optional[Dict]:
        """Advance one ledger day: mark PnL on YESTERDAY's weights at today's
        closes (H-1 — weights take effect at the next mark), then GP-step
        toward today's target unless frozen. Date-PK upserts keep same-day
        reruns idempotent. Returns a summary dict, or None when frozen with
        no prior state.
        单日推进：按昨日权重记 PnL（H-1），非冻结日再 GP 部分调仓；
        date 主键保证同日重跑幂等。"""
        prev = conn.execute(
            f"SELECT date, weights, all_closes FROM {self.state_table} "
            f"WHERE date < ? ORDER BY date DESC LIMIT 1", (date,)).fetchone()

        if prev is None:
            if frozen:
                return None
            target = self.target_weights(scores)
            w = {s: self.tau * x for s, x in target.items()}  # GP ramp from zero
            turnover = sum(abs(x) for x in w.values())
            cost_est = turnover * self.cost_bps / 10000.0
            conn.execute(
                f"INSERT OR REPLACE INTO {self.state_table} VALUES (?,?,?,?,?)",
                (date, mark_ts, json.dumps(w), json.dumps(scores),
                 json.dumps(closes)))
            return {"date": date, "weights": w, "turnover": turnover,
                    "cost_est": cost_est, "first": True}

        prev_date, wj, cj = prev
        w_prev = json.loads(wj)
        prev_closes = json.loads(cj)

        port_ret = 0.0
        for s, wi in w_prev.items():
            pc, tc = prev_closes.get(s), closes.get(s)
            if pc and tc and pc > 0:
                port_ret += wi * (tc / pc - 1.0)

        if frozen:
            w_new = dict(w_prev)
            turnover = 0.0
        else:
            target = self.target_weights(scores)
            # GP partial trading: move only tau of the way toward the aim —
            # the whole point is paying costs on tau*|target - w|, not |target - w|.
            # GP 部分调仓：每天只向目标走 tau，一次到位的换手成本正是被证伪的方案。
            w_new = {s: (1 - self.tau) * w_prev.get(s, 0.0) + self.tau * target[s]
                     for s in target}
            turnover = sum(abs(w_new[s] - w_prev.get(s, 0.0)) for s in w_new)
        cost_est = turnover * self.cost_bps / 10000.0
        n_days = max((datetime.strptime(date, "%Y-%m-%d")
                      - datetime.strptime(prev_date, "%Y-%m-%d")).days, 1)

        cum_cur = conn.execute(
            f"SELECT cumulative_ret FROM {self.pnl_table} WHERE date < ? "
            f"ORDER BY date DESC LIMIT 1", (date,)).fetchone()
        prev_cum = cum_cur[0] if cum_cur else 0.0
        cum_ret = (1 + prev_cum) * (1 + port_ret - cost_est) - 1.0

        conn.execute(
            f"INSERT OR REPLACE INTO {self.pnl_table} VALUES (?,?,?,?,?,?,?)",
            (date, prev_date, n_days, port_ret, turnover, cost_est, cum_ret))
        conn.execute(
            f"INSERT OR REPLACE INTO {self.state_table} VALUES (?,?,?,?,?)",
            (date, mark_ts, json.dumps(w_new), json.dumps(scores),
             json.dumps(closes)))
        return {"date": date, "prev_date": prev_date, "n_days": n_days,
                "port_ret": port_ret, "turnover": turnover,
                "cost_est": cost_est, "cumulative_ret": cum_ret,
                "weights": w_new}
