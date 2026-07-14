"""
run_paper_daily.py — Daily batch paper trading with gap catch-up.
每日批处理模拟盘（带缺失日补课）。

Flow per run (~1 minute):
  1. Read DB to find the last recorded day; fetch enough CLOSED 1h bars to
     cover the gap (250 + 24/missed-day, multi-exchange okx -> bybit -> gate,
     HARD FAIL if any of the 20 assets is missing).
  2. Fetch real funding-rate history and align to bars (v13 checkpoints).
  3. BACKFILL each missed day at the 11:00 UTC (19:00 Beijing) mark:
       - signal scores recomputed from bars available at that mark (causal,
         deterministic — what the model WOULD have said; feeds live rank IC)
       - basket positions stay FROZEN (nobody traded on missed days); only
         daily PnL marks are recorded (slot_changes=0, cost=0)
  4. Reconcile + run TODAY's inference, banded basket update, log everything.
     Same-day reruns upsert instead of duplicating.

v13 upgrades vs the old script: see REVIEW_2026-06-10.md appendix
(H-2/H-3/H-4/H-5/M-4/L-2/L-3 fixes + banded top-K basket).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import factors as _  # trigger auto-discover / 触发自动发现
from factors.base import FactorRegistry
from model.cross_asset_attention import CrossAssetGRUAttention
from sleeves import CarrySleeve, ContinuousBook, ModelSleeve, PortfolioBook
from sleeves.banding import banded_update_symbols

DB_PATH: str = str(BASE_DIR / "paper_daily.db")

SYMBOLS: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "SUI/USDT", "INJ/USDT", "AAVE/USDT",
]

N_BARS_BASE: int = 250       # warmup: 48-bar z-score + EMAs converge well before seq tail / 暖机
N_BARS_MAX: int = 950        # gate/bybit honor up to 1000; okx caps at 300 (see fetch_bars)
                             # gate/bybit 支持约1000根；okx 静默截断300（见 fetch_bars）
MARK_HOUR_UTC: int = 23      # daily mark = close of the 22:00-23:00 UTC bar (~19:00 US Eastern,
                             # matches the 19:30 local scheduled run) / 每日记账时点，对齐本地19:30运行
MAX_BACKFILL_DAYS: int = 28  # beyond this, declare a break instead of backfilling / 超过则视为断档
EST_COST_BPS: float = 8.0    # per traded slot notional (fee+slippage est.) / 每槽位成本估计

# Carry sleeve (Phase 0, ROADMAP_2026-07-13): banded funding carry, model-free.
# Signal = NEGATIVE trailing 3d mean funding (long low/negative funding perps,
# short high-funding ones — shorts COLLECT positive funding). Same banding
# params as the model basket. See RESEARCH_2026-07-02.md variant D.
# carry sleeve：3日均funding取负做信号，与模型篮子同banding参数，不依赖模型。
CARRY_LOOKBACK: int = 72     # bars (3 days) / 信号回看窗口
CARRY_K: int = 3
CARRY_ENTER: int = 3
CARRY_EXIT: int = 6

# Match run_v12_final.py's training drops; v13 ckpts carry factor_names.
DROP_FACTORS_FALLBACK = {"volume_zscore", "volume_momentum", "macd", "klow"}


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ckpt_sha(path: Path) -> str:
    """Short sha256 of a checkpoint file — recorded per run so silent ckpt
    replacements show up as a dated regime change in run_meta (G5).
    ckpt 文件短哈希：入账后静默换模型会在 run_meta 留下带日期的痕迹。"""
    import hashlib
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def backup_db(db_path: str, keep: int = 10) -> None:
    """Rotate daily DB backups — the ledgers are the September evidence.
    每日备份轮转：账本即九月裁决证据。"""
    import shutil
    src = Path(db_path)
    if not src.exists():
        return
    bdir = src.parent / "backups"
    bdir.mkdir(exist_ok=True)
    dst = bdir / f"paper_daily_{utc_today()}.db"
    shutil.copy2(src, dst)
    baks = sorted(bdir.glob("paper_daily_*.db"))
    for old in baks[:-keep]:
        old.unlink()


def date_range_exclusive(d0: str, d1: str) -> List[str]:
    """Dates strictly between d0 and d1. / d0 与 d1 之间（不含端点）的日期。"""
    a = datetime.strptime(d0, "%Y-%m-%d")
    b = datetime.strptime(d1, "%Y-%m-%d")
    out = []
    cur = a + timedelta(days=1)
    while cur < b:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_signals (
            date TEXT, long_asset TEXT, short_asset TEXT,
            long_score REAL, short_score REAL,
            long_close REAL, short_close REAL,
            all_scores TEXT
        );
        CREATE TABLE IF NOT EXISTS daily_pnl (
            date TEXT, prev_long TEXT, prev_short TEXT,
            long_ret REAL, short_ret REAL, port_ret REAL,
            cumulative_ret REAL
        );
        CREATE TABLE IF NOT EXISTS basket_state (
            date TEXT PRIMARY KEY,
            long_assets TEXT, short_assets TEXT,
            all_scores TEXT, all_closes TEXT,
            ckpt TEXT
        );
        CREATE TABLE IF NOT EXISTS basket_pnl (
            date TEXT PRIMARY KEY,
            prev_date TEXT, n_days INTEGER,
            prev_long TEXT, prev_short TEXT,
            long_ret REAL, short_ret REAL, port_ret REAL,
            slot_changes INTEGER, cost_est REAL,
            cumulative_ret REAL
        );
        CREATE TABLE IF NOT EXISTS carry_state (
            date TEXT PRIMARY KEY, mark_ts INTEGER,
            long_assets TEXT, short_assets TEXT,
            all_signals TEXT, all_closes TEXT
        );
        CREATE TABLE IF NOT EXISTS carry_pnl (
            date TEXT PRIMARY KEY,
            prev_date TEXT, n_days INTEGER,
            prev_long TEXT, prev_short TEXT,
            long_ret REAL, short_ret REAL, port_ret REAL,
            funding_pnl REAL,
            slot_changes INTEGER, cost_est REAL,
            cumulative_ret REAL
        );
        CREATE TABLE IF NOT EXISTS combo_pnl (
            date TEXT PRIMARY KEY, prev_date TEXT,
            model_net REAL, carry_net REAL, combo_ret REAL,
            cumulative_ret REAL
        );
        CREATE TABLE IF NOT EXISTS o2_state (
            date TEXT PRIMARY KEY, mark_ts INTEGER,
            weights TEXT, all_scores TEXT, all_closes TEXT
        );
        CREATE TABLE IF NOT EXISTS o2_pnl (
            date TEXT PRIMARY KEY, prev_date TEXT, n_days INTEGER,
            port_ret REAL, turnover REAL, cost_est REAL,
            cumulative_ret REAL
        );
        CREATE TABLE IF NOT EXISTS run_meta (
            date TEXT PRIMARY KEY, run_utc TEXT,
            funding_degraded TEXT,
            v13_ckpt_sha TEXT, o2_ckpt_sha TEXT,
            notes TEXT
        );
    """)
    try:
        conn.execute("ALTER TABLE daily_signals ADD COLUMN all_closes TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists / 列已存在
    conn.commit()
    return conn


def last_recorded_date(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT MAX(date) FROM daily_signals").fetchone()
    return row[0] if row and row[0] else None


# ============================================================================
# Data fetch
# ============================================================================

def _drop_unclosed(ohlcv: List[List[float]]) -> List[List[float]]:
    """Keep only bars whose 1h window has fully elapsed (H-5).
    仅保留已完整收盘的 1h K线。"""
    now_ms = int(time.time() * 1000)
    return [k for k in ohlcv if k[0] + 3_600_000 <= now_ms]


def fetch_bars(symbols: List[str], n_bars: int) -> Dict[str, List[Tuple]]:
    """
    Latest N CLOSED 1h bars per symbol; okx -> bybit -> gate fill; raises if
    ANY symbol missing (positional asset embedding, H-3).
    Returns {clean_symbol: [(ts_ms, o, h, l, c, v), ...]}.
    """
    import ccxt
    result: Dict[str, List[Tuple]] = {}
    for name in ["okx", "bybit", "gate"]:
        missing = [s for s in symbols if s.replace("/", "") not in result]
        if not missing:
            break
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()
        except Exception as e:
            print(f"  [{name}] unavailable: {e}")
            continue
        got = 0
        for sym in missing:
            if sym not in ex.markets:
                continue
            try:
                ohlcv = ex.fetch_ohlcv(sym, "1h", limit=n_bars)
                ohlcv = _drop_unclosed(ohlcv or [])
                # okx silently caps limit at 300 (F4): require ~full depth so
                # deep-gap backfills fall through to bybit/gate (limit 1000).
                # okx 静默截断到300根：要求接近满额，深补课自动落到下一所。
                need = max(100, int(min(n_bars, 950) * 0.9))
                if len(ohlcv) >= need:
                    clean = sym.replace("/", "")
                    result[clean] = [(k[0], k[1], k[2], k[3], k[4], k[5]) for k in ohlcv]
                    got += 1
                time.sleep(0.25)
            except Exception:
                continue
        print(f"  [{name}] filled {got} symbols")

    missing = [s for s in symbols if s.replace("/", "") not in result]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} symbols after all exchanges: {missing}. "
            f"Refusing to run inference with a shifted asset universe (H-3)."
        )
    return result


def fetch_funding(
    symbols: List[str], bars: Dict[str, List[Tuple]], device: torch.device,
) -> Optional[Dict[str, Tensor]]:
    """Real 8h funding history forward-filled onto bar timestamps (H-4).
    真实资金费率前向填充到 bar 时间戳。okx -> bybit -> gate。"""
    import ccxt
    out: Dict[str, Tensor] = {}
    for name in ["okx", "bybit", "gate"]:
        missing = [s for s in symbols if s.replace("/", "") not in out]
        if not missing:
            break
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()
        except Exception as e:
            print(f"  [funding:{name}] unavailable: {e}")
            continue
        got = 0
        for sym in missing:
            clean = sym.replace("/", "")
            bar_ts = np.asarray([b[0] for b in bars[clean]], dtype=np.int64)
            swap = f"{sym}:USDT"
            try:
                hist = ex.fetch_funding_rate_history(swap, limit=100)  # ~33d at 8h
                if not hist:
                    continue
                ts = np.asarray([h["timestamp"] for h in hist], dtype=np.int64)
                rt = np.asarray([float(h["fundingRate"]) for h in hist], dtype=np.float32)
                order = np.argsort(ts)
                ts, rt = ts[order], rt[order]
                idx = np.searchsorted(ts, bar_ts, side="right") - 1
                vals = np.zeros(len(bar_ts), dtype=np.float32)
                ok = idx >= 0
                vals[ok] = rt[idx[ok]]
                out[clean] = torch.from_numpy(vals).to(device)
                got += 1
                time.sleep(0.15)
            except Exception:
                continue
        print(f"  [funding:{name}] filled {got} symbols")

    still = [s for s in symbols if s.replace("/", "") not in out]
    if still:
        print(f"  [funding] WARNING: {len(still)} symbols fell back to zeros: "
              f"{[s.replace('/', '') for s in still]}")
        for sym in still:
            clean = sym.replace("/", "")
            out[clean] = torch.zeros(len(bars[clean]), dtype=torch.float32, device=device)
    return out


# ============================================================================
# Model + features (built ONCE on the full window; slices are causal)
# ============================================================================

def load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run `python run_v13_final.py` (or v12) to train first."
        )
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(ckpt_path, map_location=device, weights_only=False)


def _resolve_factor_names(ckpt: dict) -> List[str]:
    n_expected = int(ckpt["n_factors"])
    saved = ckpt.get("factor_names")
    if isinstance(saved, list) and len(saved) == n_expected:
        return list(saved)
    fallback = [n for n in FactorRegistry.list_factors() if n not in DROP_FACTORS_FALLBACK]
    if len(fallback) == n_expected:
        return fallback
    raise RuntimeError(
        f"Cannot resolve factor list: ckpt expects {n_expected} factors, "
        f"saved factor_names has {len(saved) if saved else 'None'}, "
        f"fallback (after drops) has {len(fallback)}."
    )


def build_model(ckpt: dict, device: torch.device) -> CrossAssetGRUAttention:
    model = CrossAssetGRUAttention(
        n_factors=ckpt["n_factors"], d_model=ckpt["d_model"],
        gru_layers=ckpt["gru_layers"], n_cross_heads=ckpt["n_cross_heads"],
        n_cross_layers=ckpt["n_cross_layers"], d_ff=ckpt["d_ff"],
        dropout=0.0, seq_len=ckpt["seq_len"], max_assets=ckpt["max_assets"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_features(
    bars: Dict[str, List[Tuple]], ckpt: dict, device: torch.device,
    funding: Optional[Dict[str, Tensor]],
) -> Dict[str, dict]:
    """
    Per symbol: bar timestamps, full-window factor tensor, closes.
    All factor primitives are causal, so slicing rows [:i+1] is identical to
    recomputing on truncated input — backfilled scores are leak-free.
    每标的：时间戳、全窗口因子张量、收盘价。所有因子原语严格因果，
    按行切片等价于在截断输入上重算——补课打分无泄漏。
    """
    factor_names = _resolve_factor_names(ckpt)
    needs_funding = bool(ckpt.get("needs_real_funding", False))
    if needs_funding and funding is None:
        print("  WARNING: ckpt was trained on real funding but live funding "
              "unavailable — funding factor degrades to zeros this run.")
    feats: Dict[str, dict] = {}
    for sym in sorted(bars.keys()):
        b = bars[sym]
        o = torch.tensor([x[1] for x in b], dtype=torch.float32, device=device)
        h = torch.tensor([x[2] for x in b], dtype=torch.float32, device=device)
        l = torch.tensor([x[3] for x in b], dtype=torch.float32, device=device)
        c = torch.tensor([x[4] for x in b], dtype=torch.float32, device=device)
        v = torch.tensor([x[5] for x in b], dtype=torch.float32, device=device)
        extras = None
        if needs_funding:
            f_t = (funding or {}).get(sym)
            if f_t is None or f_t.numel() != c.numel():
                f_t = torch.zeros_like(c)
            extras = {"funding": f_t}
        f = FactorRegistry.build_tensor(factor_names, o, h, l, c, v,
                                        zscore_window=48, extras=extras)
        feats[sym] = {
            "ts": np.asarray([x[0] for x in b], dtype=np.int64),
            "factors": f,
            "closes": np.asarray([x[4] for x in b], dtype=np.float64),
        }
    return feats


def closes_at(feats: Dict[str, dict], idx_map: Dict[str, int]) -> Dict[str, float]:
    """Close price per symbol at its mark index. / 各标的在mark索引处的收盘价。"""
    return {sym: float(feats[sym]["closes"][idx_map[sym]]) for sym in feats}


def mark_indices_for_date(
    feats: Dict[str, dict], date: str,
) -> Optional[Dict[str, int]]:
    """
    Index of the daily-mark bar (open at MARK_HOUR-1 UTC, closes MARK_HOUR)
    per symbol. None if any symbol lacks a bar within 3h before the mark or
    has too little history before it.
    每标的找到记账时点对应的bar索引；任一标的缺数据则返回 None。
    """
    mark_open_ms = int(datetime.strptime(date, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc, hour=MARK_HOUR_UTC - 1)
                       .timestamp() * 1000)
    out: Dict[str, int] = {}
    for sym, d in feats.items():
        i = int(np.searchsorted(d["ts"], mark_open_ms, side="right")) - 1
        if i < 72:  # minimal factor warmup before a backfill mark / 最低暖机
            return None
        if mark_open_ms - d["ts"][i] > 3 * 3_600_000:
            return None
        out[sym] = i
    return out


# ============================================================================
# Banding + ledgers now live in the shared sleeves package (ROADMAP Phase 1);
# these names are kept as the stable public surface of this module.
# banding与账本移入共享sleeves包；此处保留同名入口保持兼容。
# ============================================================================

banded_update = banded_update_symbols

MODEL_BOOK = PortfolioBook("basket", "basket_state", "basket_pnl",
                           cost_bps=EST_COST_BPS)
CARRY_BOOK = PortfolioBook("carry", "carry_state", "carry_pnl",
                           cost_bps=EST_COST_BPS)


def load_o2_sleeve(device) -> Optional[Tuple]:
    """v14 O2 sleeve (continuous weights, own 16-symbol universe/factors);
    active only when its production checkpoint exists.
    O2 sleeve：有 o2_production.pt 才启用；自带 16 币宇宙与 18 因子。"""
    path = BASE_DIR / "checkpoints" / "o2_production.pt"
    if not path.exists():
        return None
    ckpt = load_checkpoint(path, device)
    book = ContinuousBook("o2_state", "o2_pnl", cost_bps=EST_COST_BPS,
                          tau=float(ckpt.get("construction", {}).get("tau", 0.2)))
    return ckpt, book


# ============================================================================
# Reconcile + logging (date-keyed upserts)
# ============================================================================

def reconcile_legacy(conn, closes: Dict[str, float], date: str) -> Optional[Dict]:
    row = conn.execute(
        "SELECT date, long_asset, short_asset, long_close, short_close "
        "FROM daily_signals WHERE date < ? ORDER BY date DESC LIMIT 1", (date,)
    ).fetchone()
    if row is None:
        return None
    prev_date, prev_long, prev_short, prev_lc, prev_sc = row
    today_lc = closes.get(prev_long, prev_lc)
    today_sc = closes.get(prev_short, prev_sc)
    long_ret = (today_lc / prev_lc - 1.0) if prev_lc and prev_lc > 0 else 0.0
    short_ret = -(today_sc / prev_sc - 1.0) if prev_sc and prev_sc > 0 else 0.0
    port_ret = 0.5 * long_ret + 0.5 * short_ret

    cum_cur = conn.execute(
        "SELECT cumulative_ret FROM daily_pnl WHERE date < ? "
        "ORDER BY date DESC LIMIT 1", (date,)
    ).fetchone()
    prev_cum = cum_cur[0] if cum_cur else 0.0
    cum_ret = (1 + prev_cum) * (1 + port_ret) - 1.0

    conn.execute("DELETE FROM daily_pnl WHERE date = ?", (date,))
    conn.execute(
        "INSERT INTO daily_pnl VALUES (?,?,?,?,?,?,?)",
        (date, prev_long, prev_short, long_ret, short_ret, port_ret, cum_ret),
    )
    return {"prev_date": prev_date, "long": prev_long, "short": prev_short,
            "long_ret": long_ret, "short_ret": short_ret,
            "port_ret": port_ret, "cumulative_ret": cum_ret}


def write_signal_row(conn, date: str, score_dict: Dict[str, float],
                     closes: Dict[str, float]) -> None:
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    long1, short1 = ranked[0][0], ranked[-1][0]
    conn.execute("DELETE FROM daily_signals WHERE date = ?", (date,))
    conn.execute(
        "INSERT INTO daily_signals "
        "(date, long_asset, short_asset, long_score, short_score, "
        " long_close, short_close, all_scores, all_closes) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (date, long1, short1, score_dict[long1], score_dict[short1],
         closes[long1], closes[short1],
         json.dumps(score_dict), json.dumps(closes)))


def basket_step(
    conn, date: str, score_dict: Dict[str, float], closes: Dict[str, float],
    ckpt: dict, ckpt_name: str, frozen: bool,
) -> Optional[Dict]:
    """Model-sleeve ledger day — delegates to the shared PortfolioBook.
    模型sleeve账本单日推进——委托共享PortfolioBook。"""
    return MODEL_BOOK.step(
        conn, date, score_dict, closes,
        k=int(ckpt.get("basket_k", 3)),
        enter_band=int(ckpt.get("enter_band", 3)),
        exit_band=int(ckpt.get("exit_band", 6)),
        frozen=frozen,
        tag=("backfill-frozen" if frozen else ckpt_name))


# ============================================================================
# Carry sleeve (Phase 0) — model-free funding carry, own ledger
# carry sleeve：不依赖模型的资金费率carry，独立账本
# ============================================================================

CARRY_SLEEVE = CarrySleeve(lookback=CARRY_LOOKBACK, k=CARRY_K,
                           enter_band=CARRY_ENTER, exit_band=CARRY_EXIT)


def carry_signal_at(
    funding: Dict[str, Tensor], feats: Dict[str, dict], idx_map: Dict[str, int],
) -> Optional[Dict[str, float]]:
    """Carry sleeve scores (delegates to sleeves.CarrySleeve). / 委托CarrySleeve。"""
    return CARRY_SLEEVE.compute_scores(feats, funding, idx_map)


def funding_accrual_between(
    funding: Dict[str, Tensor], feats: Dict[str, dict],
    prev_mark_ts: int, cur_mark_ts: int,
) -> Dict[str, float]:
    """Approximate funding paid by a LONG holder of each asset between marks:
    sum of forward-filled hourly rates / 8 (8h event cadence). Shorts receive.
    两个mark之间多头支付的funding近似值（小时ffill费率求和/8）；空头收取。"""
    acc: Dict[str, float] = {}
    for sym, d in feats.items():
        f = funding.get(sym)
        if f is None:
            acc[sym] = 0.0
            continue
        ts = d["ts"]
        mask = (ts > prev_mark_ts) & (ts <= cur_mark_ts)
        acc[sym] = float(f.cpu().numpy()[mask].sum() / 8.0)
    return acc


def carry_step(
    conn, date: str, sig: Dict[str, float], closes: Dict[str, float],
    mark_ts: int, funding: Dict[str, Tensor], feats: Dict[str, dict],
    frozen: bool,
) -> Optional[Dict]:
    """Carry-sleeve ledger day (price + funding accrual) — delegates to the
    shared PortfolioBook. carry账本单日推进——委托共享PortfolioBook。"""
    return CARRY_BOOK.step(
        conn, date, sig, closes,
        k=CARRY_K, enter_band=CARRY_ENTER, exit_band=CARRY_EXIT,
        frozen=frozen, mark_ts=mark_ts,
        accrual_fn=lambda prev_ts, cur_ts: funding_accrual_between(
            funding, feats, int(prev_ts), int(cur_ts)))


def combo_step(conn, date: str) -> Optional[Dict]:
    """Pure bookkeeping 50/50 ledger of the two sleeves' NET daily returns.
    Linear approximation (no capital netting) — see ROADMAP Phase 3.
    纯记账的50/50虚拟账本（线性近似，未做资金净额结算）。"""
    b = conn.execute("SELECT port_ret, cost_est FROM basket_pnl WHERE date=?",
                     (date,)).fetchone()
    c = conn.execute("SELECT port_ret, cost_est FROM carry_pnl WHERE date=?",
                     (date,)).fetchone()
    if b is None or c is None:
        return None
    model_net = b[0] - b[1]
    carry_net = c[0] - c[1]
    combo = 0.5 * model_net + 0.5 * carry_net
    prev = conn.execute(
        "SELECT date, cumulative_ret FROM combo_pnl WHERE date < ? "
        "ORDER BY date DESC LIMIT 1", (date,)).fetchone()
    prev_date, prev_cum = (prev[0], prev[1]) if prev else (None, 0.0)
    cum = (1 + prev_cum) * (1 + combo) - 1.0
    conn.execute("INSERT OR REPLACE INTO combo_pnl VALUES (?,?,?,?,?,?)",
                 (date, prev_date, model_net, carry_net, combo, cum))
    return {"model_net": model_net, "carry_net": carry_net,
            "combo_ret": combo, "cumulative_ret": cum}


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="auto",
                   help="'auto' (v13 if present, else v12), or v11/v12/v13")
    p.add_argument("--dry-run", action="store_true",
                   help="run everything but write nothing to the DB")
    p.add_argument("--db", default=DB_PATH, help="database path (testing)")
    args = p.parse_args()

    if args.ckpt == "auto":
        version = "v13" if (BASE_DIR / "checkpoints" / "v13_production.pt").exists() else "v12"
    else:
        version = args.ckpt
    ckpt_path = BASE_DIR / "checkpoints" / f"{version}_production.pt"

    now_utc = datetime.now(timezone.utc)
    print("=" * 60)
    print("  Daily Paper Trading — Batch Mode")
    print(f"  {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
          + ("  [DRY RUN]" if args.dry_run else ""))
    print(f"  Model: {ckpt_path.name}   DB: {Path(args.db).name}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conn = init_db(args.db)
    today = utc_today()

    run_errors: List[str] = []

    # gap detection BEFORE fetch so we pull enough history / 先测断档再定拉取量
    last_date = last_recorded_date(conn)
    if last_date and today < last_date:
        # clock skew would silently overwrite mid-series evidence rows (G4)
        # 时钟回拨会静默覆盖证据链中段——直接拒跑
        raise SystemExit(f"CLOCK SKEW: today {today} < last recorded {last_date}")
    missed: List[str] = date_range_exclusive(last_date, today) if last_date else []
    if len(missed) > MAX_BACKFILL_DAYS:
        print(f"  WARNING: {len(missed)} missed days > {MAX_BACKFILL_DAYS} cap — "
              f"backfilling only the most recent {MAX_BACKFILL_DAYS} (series break).")
        missed = missed[-MAX_BACKFILL_DAYS:]
    n_bars = min(N_BARS_BASE + 24 * (len(missed) + 1), N_BARS_MAX)

    print(f"\n[1/5] Fetching latest {n_bars} closed 1h bars "
          f"({len(missed)} missed day(s) to backfill) ...")
    bars = fetch_bars(SYMBOLS, n_bars)
    print(f"  Got all {len(bars)} symbols")

    ckpt = load_checkpoint(ckpt_path, device)
    is_v13 = bool(ckpt.get("label_horizon"))
    seq_len = int(ckpt["seq_len"])

    # funding is needed unconditionally now: model extras (v13 ckpt) AND the
    # carry sleeve both consume it / funding现在无条件抓取：模型extras与carry都用
    print("\n[2/5] Fetching real funding-rate history (model extras + carry) ...")
    funding = fetch_funding(SYMBOLS, bars, device) or {}

    print("\n[3/5] Building features + model ...")
    print(f"  Loaded {ckpt_path.name} (val_corr={ckpt['val_corr']:.4f}, "
          f"{ckpt['n_factors']} factors"
          + (f", label={ckpt.get('label_horizon')}h" if is_v13 else "") + ")")
    feats = build_features(bars, ckpt, device, funding)
    model = build_model(ckpt, device)
    model_sleeve = ModelSleeve(ckpt, model)

    # today-mark freshness: a stalled feed (delisting precursor) must not
    # silently enter the cross-section as a stale close (G3)
    # 当日mark新鲜度：停牌/停推的陈旧收盘价不得静默进入横截面
    last_ts = {sym: int(feats[sym]["ts"][-1]) for sym in feats}
    max_ts = max(last_ts.values())
    stale = {s: round((max_ts - t) / 3_600_000, 1)
             for s, t in last_ts.items() if max_ts - t > 2 * 3_600_000}
    if stale:
        raise RuntimeError(f"STALE last bars (hours behind): {stale} — "
                           f"refusing to build a misaligned cross-section (G3)")

    # funding degradation is recorded per run so September analysis can
    # identify degraded-signal days (G2)
    # funding退化落库：九月分析可甄别退化信号日
    funding_degraded = sorted(
        s.replace("/", "") for s in SYMBOLS
        if funding.get(s.replace("/", "")) is None
        or bool((funding[s.replace("/", "")] == 0).all()))
    if funding_degraded:
        print(f"  NOTE: funding degraded to zeros for {funding_degraded}")

    o2_sleeve = o2_book = o2_feats = None
    try:
        o2_loaded = load_o2_sleeve(device)
        if o2_loaded:
            o2_ckpt, o2_book = o2_loaded
            o2_bars = {s: bars[s] for s in o2_ckpt["symbols"]}
            o2_feats = build_features(o2_bars, o2_ckpt, device, funding)
            o2_sleeve = ModelSleeve(o2_ckpt, build_model(o2_ckpt, device))
            print(f"  Loaded o2_production.pt (v14 sleeve: {o2_ckpt['objective']}, "
                  f"{len(o2_bars)} symbols, {o2_ckpt['n_factors']} factors, "
                  f"tau={o2_book.tau})")
    except Exception as e:  # O2 failure must not kill v13/carry / O2失败不连坐
        o2_sleeve = None
        run_errors.append(f"o2 load: {e!r}")
        print(f"  ERROR: o2 sleeve disabled this run: {e!r}")

    # ---- Backfill missed days (signals recorded; basket FROZEN) ----
    # ---- 补课：缺失日重算信号入库；篮子持仓冻结，仅补每日PnL ----
    print(f"\n[4/5] Review & backfill ({len(missed)} missed day(s)) ...")
    n_backfilled = 0
    for d in missed:
        idx_map = mark_indices_for_date(feats, d)
        if idx_map is None:
            print(f"  [backfill {d}] SKIP — insufficient bars around the "
                  f"{MARK_HOUR_UTC:02d}:00 UTC mark")
            continue
        sc_d = model_sleeve.compute_scores(feats, funding, idx_map)
        cl_d = closes_at(feats, idx_map)
        if not args.dry_run:
            rec = reconcile_legacy(conn, cl_d, d)
            write_signal_row(conn, d, sc_d, cl_d)
            binfo = cinfo = oinfo = None
            mark_open_ms = int(datetime.strptime(d, "%Y-%m-%d")
                               .replace(tzinfo=timezone.utc, hour=MARK_HOUR_UTC - 1)
                               .timestamp() * 1000)
            # each ledger isolated: one failing must not lose the others' day
            # 账本间故障隔离：一本失败不连坐
            try:
                if is_v13:
                    binfo = basket_step(conn, d, sc_d, cl_d, ckpt,
                                        ckpt_path.name, frozen=True)
            except Exception as e:
                run_errors.append(f"backfill {d} basket: {e!r}")
            try:
                csig_d = carry_signal_at(funding, feats, idx_map)
                if csig_d:
                    cinfo = carry_step(conn, d, csig_d, cl_d, mark_open_ms,
                                       funding, feats, frozen=True)
                combo_step(conn, d)
            except Exception as e:
                run_errors.append(f"backfill {d} carry/combo: {e!r}")
            try:
                if o2_sleeve:
                    idx2 = {s: idx_map[s] for s in o2_feats}
                    sc2 = o2_sleeve.compute_scores(o2_feats, funding, idx2)
                    oinfo = o2_book.step(conn, d, sc2, cl_d, mark_open_ms,
                                         frozen=True)
            except Exception as e:
                run_errors.append(f"backfill {d} o2: {e!r}")
            conn.commit()
            msg = f"  [backfill {d}] signal logged"
            if rec:
                msg += f"; legacy port {rec['port_ret']:+.4%}"
            if binfo:
                msg += (f"; basket(frozen) {binfo['port_ret']:+.4%} "
                        f"cum {binfo['cumulative_ret']:+.4%}")
            if cinfo and "port_ret" in cinfo:
                msg += f"; carry(frozen) {cinfo['port_ret']:+.4%}"
            if oinfo and "port_ret" in oinfo:
                msg += f"; o2(frozen) {oinfo['port_ret']:+.4%}"
            print(msg)
        else:
            ranked = sorted(sc_d.items(), key=lambda x: x[1], reverse=True)
            print(f"  [backfill {d}] (dry) top={ranked[0][0]} bottom={ranked[-1][0]}")
        n_backfilled += 1
    if not missed:
        print("  No gap — last record is "
              + (f"{last_date}" if last_date else "absent (first run)"))

    # ---- Today's signal / 今日信号 ----
    today_idx = {sym: len(feats[sym]["ts"]) - 1 for sym in feats}
    score_dict = model_sleeve.compute_scores(feats, funding, today_idx)
    closes = closes_at(feats, today_idx)
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    long1, short1 = sorted_scores[0][0], sorted_scores[-1][0]

    today_mark_ts = int(min(feats[sym]["ts"][-1] for sym in feats))
    carry_sig = carry_signal_at(funding, feats, today_idx)

    o2_scores = None
    if o2_sleeve:
        idx2_today = {s: len(o2_feats[s]["ts"]) - 1 for s in o2_feats}
        o2_scores = o2_sleeve.compute_scores(o2_feats, funding, idx2_today)

    recon = None
    basket_info = None
    carry_info = None
    combo_info = None
    o2_info = None
    if not args.dry_run:
        recon = reconcile_legacy(conn, closes, today)
        write_signal_row(conn, today, score_dict, closes)
        try:
            if is_v13:
                basket_info = basket_step(conn, today, score_dict, closes,
                                          ckpt, ckpt_path.name, frozen=False)
        except Exception as e:
            run_errors.append(f"today basket: {e!r}")
        try:
            if carry_sig:
                carry_info = carry_step(conn, today, carry_sig, closes,
                                        today_mark_ts, funding, feats,
                                        frozen=False)
                combo_info = combo_step(conn, today)
            else:
                print("  WARNING: carry signal unavailable (funding degraded) "
                      "— carry ledger skipped today")
                run_errors.append("today carry: signal unavailable")
        except Exception as e:
            run_errors.append(f"today carry/combo: {e!r}")
        try:
            if o2_scores:
                o2_info = o2_book.step(conn, today, o2_scores, closes,
                                       today_mark_ts, frozen=False)
        except Exception as e:
            run_errors.append(f"today o2: {e!r}")
        conn.execute(
            "INSERT OR REPLACE INTO run_meta VALUES (?,?,?,?,?,?)",
            (today, now_utc.strftime("%Y-%m-%d %H:%M:%S"),
             json.dumps(funding_degraded),
             ckpt_sha(ckpt_path),
             ckpt_sha(BASE_DIR / "checkpoints" / "o2_production.pt"),
             "; ".join(run_errors)))
        conn.commit()
    elif is_v13:
        # dry-run preview of the banded update / 干跑时也预览篮子更新
        row = conn.execute(
            "SELECT long_assets, short_assets FROM basket_state "
            "WHERE date < ? ORDER BY date DESC LIMIT 1", (today,)).fetchone()
        pl_, ps_ = (json.loads(row[0]), json.loads(row[1])) if row else ([], [])
        nl, ns = banded_update(score_dict, pl_, ps_,
                               int(ckpt.get("basket_k", 3)),
                               int(ckpt.get("enter_band", 3)),
                               int(ckpt.get("exit_band", 6)))
        basket_info = {"long": nl, "short": ns,
                       "slot_changes": (len(set(pl_) ^ set(nl))
                                        + len(set(ps_) ^ set(ns)))}

    if recon:
        print(f"\n  [reconcile] {recon['prev_date']} -> {today}: "
              f"legacy port {recon['port_ret']:+.4%}, "
              f"cum {recon['cumulative_ret']:+.4%}")
    if basket_info and "port_ret" in basket_info:
        print(f"  [basket] {basket_info['prev_date']} -> {today}: "
              f"port {basket_info['port_ret']:+.4%} "
              f"(cost est {basket_info['cost_est']:.4%}), "
              f"cum {basket_info['cumulative_ret']:+.4%}")
    if carry_info and "port_ret" in carry_info:
        print(f"  [carry]  {carry_info['prev_date']} -> {today}: "
              f"port {carry_info['port_ret']:+.4%} "
              f"(funding {carry_info['funding_pnl']:+.4%}, "
              f"cost est {carry_info['cost_est']:.4%}), "
              f"cum {carry_info['cumulative_ret']:+.4%}")
    if combo_info:
        print(f"  [combo]  50/50 model+carry: {combo_info['combo_ret']:+.4%}, "
              f"cum {combo_info['cumulative_ret']:+.4%}")
    if o2_info and "port_ret" in o2_info:
        print(f"  [o2]     {o2_info['prev_date']} -> {today}: "
              f"port {o2_info['port_ret']:+.4%} "
              f"(turnover {o2_info['turnover']:.3f}, "
              f"cost est {o2_info['cost_est']:.4%}), "
              f"cum {o2_info['cumulative_ret']:+.4%}")
    elif o2_info and o2_info.get("first"):
        print(f"  [o2]     track initialized (GP ramp from zero, "
              f"turnover {o2_info['turnover']:.3f})")
    if o2_scores:
        tw = ContinuousBook.target_weights(o2_scores)
        top = sorted(tw.items(), key=lambda x: -x[1])
        print(f"    [o2] top weights: "
              + ", ".join(f"{s}{w:+.3f}" for s, w in top[:3])
              + " | " + ", ".join(f"{s}{w:+.3f}" for s, w in top[-3:]))

    print("\n  TODAY'S SIGNAL / 今日信号:")
    if basket_info:
        print(f"    [model] LONG  basket: {basket_info['long']}")
        print(f"    [model] SHORT basket: {basket_info['short']}")
        print(f"    [model] slot changes: {basket_info['slot_changes']}")
    else:
        print(f"    LONG:  {long1} (score={score_dict[long1]:+.4f})")
        print(f"    SHORT: {short1} (score={score_dict[short1]:+.4f})")
    if carry_sig:
        if carry_info:
            print(f"    [carry] LONG  basket: {carry_info['long']}")
            print(f"    [carry] SHORT basket: {carry_info['short']}")
        else:
            ranked_c = sorted(carry_sig.items(), key=lambda x: x[1], reverse=True)
            nl_c = sorted(s for s, _ in ranked_c[:CARRY_K])
            ns_c = sorted(s for s, _ in ranked_c[-CARRY_K:])
            print(f"    [carry] (preview) LONG {nl_c} SHORT {ns_c}")
    print("\n  Full ranking / 完整排名:")
    for i, (sym, sc) in enumerate(sorted_scores):
        tag = ""
        if basket_info:
            if sym in basket_info["long"]:
                tag = " <- LONG"
            elif sym in basket_info["short"]:
                tag = " <- SHORT"
        else:
            tag = " <- LONG" if i == 0 else (" <- SHORT" if i == len(sorted_scores) - 1 else "")
        print(f"    {i+1:2d}. {sym:12s} {sc:+.4f}{tag}")

    conn.close()
    if not args.dry_run:
        backup_db(args.db)
    print(f"\n[5/5] {'DRY RUN — nothing written' if args.dry_run else f'Logged to {args.db}'}"
          + (f' (backfilled {n_backfilled} day(s))' if n_backfilled else ''))
    if run_errors:
        # partial success: healthy ledgers are committed, but exit nonzero so
        # the scheduler wrapper raises the Desktop sentinel (F1 alert chain)
        # 部分成功：健康账本已入库，但以非零码退出触发桌面哨兵
        print("[DONE WITH ERRORS]")
        for e in run_errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    print("[DONE]")


if __name__ == "__main__":
    main()
