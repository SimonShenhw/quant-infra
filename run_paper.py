"""
run_paper.py — Paper Trading Entry Point.
模拟盘入口。

Loads a trained model and runs paper trading using live CCXT data.
加载训练好的模型，使用 CCXT 实时数据运行模拟盘。
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, ".")

from paper_trading.engine import PaperTradingEngine
from model.cross_asset_attention import CrossAssetGRUAttention


SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "UNIUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "INJUSDT", "AAVEUSDT",
]


def fetch_latest_bars(symbols: List[str]) -> Dict[str, Tuple[float, ...]]:
    """Fetch one bar per symbol from OKX via CCXT. / 通过CCXT从OKX获取每个标的一根K线。"""
    import ccxt
    ex = ccxt.okx({"enableRateLimit": True, "timeout": 30000})
    ex.load_markets()
    bars: Dict[str, Tuple[float, ...]] = {}
    for sym_clean in symbols:
        sym_ccxt = sym_clean.replace("USDT", "/USDT")
        if sym_ccxt not in ex.markets:
            continue
        try:
            ohlcv = ex.fetch_ohlcv(sym_ccxt, "1h", limit=1)
            if ohlcv:
                k = ohlcv[0]
                bars[sym_clean] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
        except Exception:
            continue
        time.sleep(0.2)
    return bars


def main() -> None:
    print("=" * 60)
    print("  Paper Trading — Live CCXT + Model Inference")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # init model (untrained — in production, load weights from checkpoint)
    # 初始化模型（未训练——生产环境中应从checkpoint加载权重）
    model = CrossAssetGRUAttention(
        n_factors=10, d_model=64, gru_layers=2,
        n_cross_heads=4, n_cross_layers=2, d_ff=128,
        dropout=0.0, seq_len=24, max_assets=len(SYMBOLS),
    ).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  NOTE: Using random weights. In production, load trained checkpoint.")

    engine = PaperTradingEngine(
        model=model, symbols=SYMBOLS, device=device,
        seq_len=24, initial_cash=100_000.0, min_hold_bars=3,
    )

    # demo: fetch a few bars to test the pipeline / 演示：获取几根K线测试管线
    print(f"\n[Paper] Fetching live bars from OKX ...")
    n_rounds = 5
    for i in range(n_rounds):
        bars = fetch_latest_bars(SYMBOLS)
        print(f"  Round {i+1}/{n_rounds}: got {len(bars)} bars", end="")

        action = engine.ingest_bars(bars)
        if action:
            print(f" → SIGNAL: long={action['long']} short={action['short']}")
        else:
            print(f" → (warming up, {engine._bar_count} bars buffered)")

        if i < n_rounds - 1:
            time.sleep(2)

    summary = engine.summary()
    print(f"\n[Paper] Session summary: {summary}")
    engine.close()
    print("[Paper] Done.")


if __name__ == "__main__":
    main()
