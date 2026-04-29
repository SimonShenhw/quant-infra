"""
run_paper_daily.py — Daily batch paper trading.
每日批处理模拟盘。

Run once per day (~30 seconds):
  1. Fetch latest 48h of 1h bars for 20 assets via CCXT
  2. Build factor tensor using checkpoint's saved factor_names (post-drop)
  3. Model inference + emit today's long/short pick
  4. Reconcile yesterday's signal vs realized returns
  5. Log everything to SQLite

Compatibility:
  - v12 (default): loads checkpoints/v12_production.pt; 17 factors
  - v11.1: pass --ckpt v11 to load checkpoints/v11_production.pt

Known caveats (TODO):
  - funding_rate factor falls back to OHLCV proxy here (no extras passed),
    while v11.1+/v12 trained on REAL Binance Vision funding. Slight domain
    shift. To fix: fetch live funding via CCXT and pass extras={"funding":...}
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

sys.path.insert(0, ".")

import factors as _  # trigger auto-discover / 触发自动发现
from factors.base import FactorRegistry
from model.cross_asset_attention import CrossAssetGRUAttention

DB_PATH: str = str(Path(__file__).resolve().parent / "paper_daily.db")

SYMBOLS: List[str] = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "UNI/USDT", "ATOM/USDT", "LTC/USDT", "NEAR/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "SUI/USDT", "INJ/USDT", "AAVE/USDT",
]


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
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
    """)
    conn.commit()
    return conn


def fetch_bars(symbols: List[str], n_bars: int = 50) -> Dict[str, List[Tuple]]:
    """Fetch latest N 1h bars per symbol via CCXT. / 通过CCXT获取每个标的最近N根1h K线。"""
    import ccxt
    exchanges = ["okx", "bybit", "gate"]
    ex = None
    for name in exchanges:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True, "timeout": 30000})
            ex.load_markets()
            break
        except Exception:
            continue
    if ex is None:
        raise RuntimeError("All exchanges failed")

    result: Dict[str, List[Tuple]] = {}
    for sym in symbols:
        if sym not in ex.markets:
            continue
        try:
            ohlcv = ex.fetch_ohlcv(sym, "1h", limit=n_bars)
            if ohlcv and len(ohlcv) >= 30:
                clean = sym.replace("/", "")
                result[clean] = [(k[1], k[2], k[3], k[4], k[5]) for k in ohlcv]
            time.sleep(0.3)
        except Exception:
            continue
    return result


# Match the noise factors dropped by v11.1+/v12 training (run_v11_final.py /
# run_v12_final.py). Used as fallback if the checkpoint doesn't carry a
# factor_names list. Keep in sync if training drops change.
# 与 v11.1+/v12 训练脚本剔除的因子保持一致；若 ckpt 自带 factor_names 优先用它。
DROP_FACTORS_FALLBACK = {"volume_zscore", "volume_momentum", "macd", "klow"}


def _resolve_factor_names(ckpt: dict) -> List[str]:
    """Pick the factor list that matches the checkpoint's n_factors.
    选用与 ckpt 的 n_factors 匹配的因子列表。"""
    n_expected = int(ckpt["n_factors"])
    saved = ckpt.get("factor_names")
    if isinstance(saved, list) and len(saved) == n_expected:
        return list(saved)
    # fallback: drop the noise factors and confirm the count matches
    fallback = [n for n in FactorRegistry.list_factors() if n not in DROP_FACTORS_FALLBACK]
    if len(fallback) == n_expected:
        return fallback
    raise RuntimeError(
        f"Cannot resolve factor list: ckpt expects {n_expected} factors, "
        f"saved factor_names has {len(saved) if saved else 'None'}, "
        f"fallback (after drops) has {len(fallback)}. "
        f"Update DROP_FACTORS_FALLBACK in run_paper_daily.py."
    )


def run_inference(
    bars: Dict[str, List[Tuple]], device: torch.device,
    seq_len: int = 24, ckpt_path: str = "checkpoints/v12_production.pt",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run model inference on latest bars. Returns (scores, closes). / 对最新K线运行推理。"""
    syms = sorted(bars.keys())
    if len(syms) < 5:
        raise ValueError(f"Only {len(syms)} symbols, need >= 5")

    # Load trained checkpoint FIRST so we know which factors to build
    # 先加载 ckpt，从中读出真正训练用的因子列表
    import os
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run `python run_v12_final.py` to train, or pass --ckpt v11 to use v11.1.\n"
            f"未找到 checkpoint。请先运行 run_v12_final.py 训练，或用 --ckpt v11 加载 v11.1。"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    factor_names = _resolve_factor_names(ckpt)
    print(f"  Loaded {ckpt_path} (val_corr={ckpt['val_corr']:.4f}, {len(factor_names)} factors)")

    all_factors = []
    closes = {}
    for sym in syms:
        b = bars[sym]
        o = torch.tensor([x[0] for x in b], dtype=torch.float32, device=device)
        h = torch.tensor([x[1] for x in b], dtype=torch.float32, device=device)
        l = torch.tensor([x[2] for x in b], dtype=torch.float32, device=device)
        c = torch.tensor([x[3] for x in b], dtype=torch.float32, device=device)
        v = torch.tensor([x[4] for x in b], dtype=torch.float32, device=device)
        f = FactorRegistry.build_tensor(factor_names, o, h, l, c, v, zscore_window=48)
        all_factors.append(f[-seq_len:])
        closes[sym] = b[-1][3]  # last close

    x = torch.stack(all_factors, dim=0).unsqueeze(0).to(device)  # (1, A, T, F)

    model = CrossAssetGRUAttention(
        n_factors=ckpt["n_factors"],
        d_model=ckpt["d_model"],
        gru_layers=ckpt["gru_layers"],
        n_cross_heads=ckpt["n_cross_heads"],
        n_cross_layers=ckpt["n_cross_layers"],
        d_ff=ckpt["d_ff"],
        dropout=0.0,  # no dropout at inference / 推理时关闭dropout
        seq_len=ckpt["seq_len"],
        max_assets=ckpt["max_assets"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        scores = model(x).squeeze(0)

    score_dict = {syms[i]: scores[i].item() for i in range(len(syms))}
    return score_dict, closes


def reconcile(conn: sqlite3.Connection, bars: Dict[str, List[Tuple]]) -> Optional[Dict]:
    """Compare yesterday's signal vs actual returns. / 对比昨日信号与实际收益。"""
    cur = conn.execute(
        "SELECT date, long_asset, short_asset, long_close, short_close "
        "FROM daily_signals ORDER BY date DESC LIMIT 1"
    )
    row = cur.fetchone()
    if row is None:
        return None

    prev_date, prev_long, prev_short, prev_lc, prev_sc = row

    # get today's close for those assets / 获取今日收盘价
    today_lc = bars.get(prev_long, [None])[-1][3] if prev_long in bars and bars[prev_long] else prev_lc
    today_sc = bars.get(prev_short, [None])[-1][3] if prev_short in bars and bars[prev_short] else prev_sc

    long_ret = (today_lc / prev_lc - 1.0) if prev_lc > 0 else 0.0
    short_ret = -(today_sc / prev_sc - 1.0) if prev_sc > 0 else 0.0
    port_ret = 0.5 * long_ret + 0.5 * short_ret

    # cumulative / 累计收益
    cum_cur = conn.execute(
        "SELECT cumulative_ret FROM daily_pnl ORDER BY date DESC LIMIT 1"
    ).fetchone()
    prev_cum = cum_cur[0] if cum_cur else 0.0
    cum_ret = (1 + prev_cum) * (1 + port_ret) - 1.0

    today = datetime.now().strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO daily_pnl VALUES (?,?,?,?,?,?,?)",
        (today, prev_long, prev_short, long_ret, short_ret, port_ret, cum_ret),
    )
    conn.commit()

    return {
        "prev_date": prev_date, "long": prev_long, "short": prev_short,
        "long_ret": long_ret, "short_ret": short_ret,
        "port_ret": port_ret, "cumulative_ret": cum_ret,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="v12",
                   help="model version: 'v12' (default) or 'v11' (loads checkpoints/<v>_production.pt)")
    args = p.parse_args()
    ckpt_path = f"checkpoints/{args.ckpt}_production.pt"

    print("=" * 60)
    print("  Daily Paper Trading — Batch Mode")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {ckpt_path}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conn = init_db()

    # Step 1: Fetch latest bars / 获取最新K线
    print("\n[1/4] Fetching latest 1h bars ...")
    bars = fetch_bars(SYMBOLS, n_bars=50)
    print(f"  Got {len(bars)} symbols")

    # Step 2: Reconcile yesterday / 对账昨日
    print("\n[2/4] Reconciling yesterday's signal ...")
    recon = reconcile(conn, bars)
    if recon:
        print(f"  Yesterday: long={recon['long']} ({recon['long_ret']:+.4%}) "
              f"short={recon['short']} ({recon['short_ret']:+.4%})")
        print(f"  Portfolio return: {recon['port_ret']:+.4%}")
        print(f"  Cumulative: {recon['cumulative_ret']:+.4%}")
    else:
        print("  No previous signal to reconcile (first run)")

    # Step 3: Run inference / 运行推理
    print("\n[3/4] Running model inference ...")
    score_dict, closes = run_inference(bars, device, ckpt_path=ckpt_path)
    sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    long_asset = sorted_scores[0][0]
    short_asset = sorted_scores[-1][0]

    print(f"\n  TODAY'S SIGNAL / 今日信号:")
    print(f"    LONG:  {long_asset} (score={score_dict[long_asset]:.4f}, "
          f"close=${closes[long_asset]:,.2f})")
    print(f"    SHORT: {short_asset} (score={score_dict[short_asset]:.4f}, "
          f"close=${closes[short_asset]:,.2f})")
    print(f"\n  Full ranking / 完整排名:")
    for i, (sym, sc) in enumerate(sorted_scores):
        marker = " ← LONG" if i == 0 else (" ← SHORT" if i == len(sorted_scores)-1 else "")
        print(f"    {i+1:2d}. {sym:12s} {sc:+.4f}{marker}")

    # Step 4: Log signal / 记录信号
    today = datetime.now().strftime("%Y-%m-%d")
    conn.execute(
        "INSERT INTO daily_signals VALUES (?,?,?,?,?,?,?,?)",
        (today, long_asset, short_asset,
         score_dict[long_asset], score_dict[short_asset],
         closes[long_asset], closes[short_asset],
         json.dumps(score_dict)),
    )
    conn.commit()
    conn.close()

    print(f"\n[4/4] Logged to {DB_PATH}")
    print("[DONE] Run again tomorrow!")


if __name__ == "__main__":
    main()
