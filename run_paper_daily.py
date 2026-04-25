"""
run_paper_daily.py — Daily batch paper trading.
每日批处理模拟盘。

Run once per day (~30 seconds):
  1. Fetch latest 48h of 1h bars for 20 assets via CCXT
  2. Compute 13 factors + model inference
  3. Output today's long/short recommendation
  4. Compare yesterday's recommendation vs actual returns (reconciliation)
  5. Log everything to SQLite

每天运行一次（约30秒）：
  1. 通过CCXT获取20个资产最近48h的1h K线
  2. 计算13因子 + 模型推理
  3. 输出今日多/空建议
  4. 对比昨日建议 vs 实际收益（对账）
  5. 全部记录到SQLite
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


def run_inference(
    bars: Dict[str, List[Tuple]], device: torch.device, seq_len: int = 24
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run model inference on latest bars. Returns (scores, closes). / 对最新K线运行推理。"""
    syms = sorted(bars.keys())
    if len(syms) < 5:
        raise ValueError(f"Only {len(syms)} symbols, need >= 5")

    factor_names = FactorRegistry.list_factors()
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

    # Load trained checkpoint / 加载训练好的checkpoint
    import os
    ckpt_path = "checkpoints/v11_production.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please run `python run_v11_final.py` first to train and save the model.\n"
            f"未找到checkpoint。请先运行 run_v11_final.py 训练并保存模型。"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    print(f"  Loaded checkpoint (val_corr={ckpt['val_corr']:.4f})")

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
    print("=" * 60)
    print("  Daily Paper Trading — Batch Mode")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    score_dict, closes = run_inference(bars, device)
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
