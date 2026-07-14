"""
PortfolioBook: one bookkeeping engine for every sleeve ledger.
组合账本：所有 sleeve 共用的记账引擎。(ROADMAP Phase 1)

Reproduces the exact row formats of the pre-refactor basket_step / carry_step
(paper_daily.db tables basket_state/basket_pnl and carry_state/carry_pnl are
LIVE series — schemas must not change). Two layouts:

  "basket": state(date, long, short, scores, closes, tag)
            pnl(date, prev_date, n_days, prev_long, prev_short,
                long_ret, short_ret, port_ret, slot_changes, cost_est, cum)
  "carry":  state(date, mark_ts, long, short, signals, closes)
            pnl(... + funding_pnl before slot_changes)

Semantics (identical to the originals, verified by tests/run_invariants.py T4):
  - frozen=True: positions unchanged (missed-day backfill), zero slots/cost
  - net = port_ret - cost_est compounds into cumulative_ret
  - carry accrual: longs PAY the window's funding sum, shorts RECEIVE it
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Dict, List, Optional

from sleeves.banding import banded_update_symbols


class PortfolioBook:
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
            funding_pnl = (
                0.5 * (sum(-acc.get(a, 0.0) for a in prev["long"]) / max(len(prev["long"]), 1))
                + 0.5 * (sum(+acc.get(a, 0.0) for a in prev["short"]) / max(len(prev["short"]), 1)))
        n_days = max((datetime.strptime(date, "%Y-%m-%d")
                      - datetime.strptime(prev["date"], "%Y-%m-%d")).days, 1)

        if frozen:
            new_l, new_s = list(prev["long"]), list(prev["short"])
            slot_changes = 0
        else:
            new_l, new_s = banded_update_symbols(sig, prev["long"], prev["short"],
                                                 k, enter_band, exit_band)
            slot_changes = (len(set(prev["long"]) ^ set(new_l))
                            + len(set(prev["short"]) ^ set(new_s)))
        cost_est = slot_changes * (self.gross / k) * (self.cost_bps / 10000.0)

        cum_cur = conn.execute(
            f"SELECT cumulative_ret FROM {self.pnl_table} WHERE date < ? "
            f"ORDER BY date DESC LIMIT 1", (date,)).fetchone()
        prev_cum = cum_cur[0] if cum_cur else 0.0
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

    Tables (created by the caller's init_db):
      {state}: date PK, mark_ts, weights(json sym->w), all_scores, all_closes
      {pnl}:   date PK, prev_date, n_days, port_ret, turnover, cost_est, cum
    Semantics: frozen=True keeps weights (missed-day backfill); the first
    live day moves tau from zero (GP ramp, matching the gated backtest).
    """

    def __init__(self, state_table: str, pnl_table: str,
                 cost_bps: float = 8.0, tau: float = 0.2) -> None:
        self.state_table = state_table
        self.pnl_table = pnl_table
        self.cost_bps = cost_bps
        self.tau = tau

    @staticmethod
    def target_weights(scores: Dict[str, float]) -> Dict[str, float]:
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
