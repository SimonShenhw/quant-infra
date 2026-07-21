"""
On-chain data fetcher — Coinmetrics Community API (free, no auth).
链上数据获取器 — Coinmetrics Community API（免费，无需认证）。

Endpoint: https://community-api.coinmetrics.io/v4/timeseries/asset-metrics
Free metrics include:
  - AdrActCnt: Active addresses
  - TxCnt: Transaction count
  - TxTfrValAdjUSD: Adjusted transfer volume USD
  - HashRate (BTC, ETH): Network hash rate
  - SOPR (Spent Output Profit Ratio): Realized profit indicator
  - NVTAdj: NVT ratio (P/E equivalent for crypto)

These are SLOW-MOVING (daily granularity), good for regime detection
not 1h trading. Use as state features.

HYPOTHESIS: on-chain activity (addresses, transfers, valuation ratios) is
orthogonal to price/volume factors — it measures network usage rather than
exchange flow, so it could gate WHICH regime the price-based signal should
trade in, not generate 1h alpha itself.
经济假设：链上活动（地址数、转账量、估值比率）与价格/成交量因子正交——
度量的是网络使用而非交易所资金流，适合判定价格信号该在哪种 regime 下开仓，
而非自己产生 1h alpha。

STATUS — honest scope note: research/data-collection infrastructure from
the v11.2 alternative-data sweep. Not imported by any pipeline — no factor
consumes onchain.db, and it plays no part in the v13+ result path. Kept as
groundwork for future regime-feature research.
状态——如实说明：v11.2 另类数据尝试遗留的研究/采集基础设施。无任何管线
导入——没有因子消费 onchain.db，也不参与 v13+ 结果路径。保留作未来
regime 特征研究的地基。

这些指标更新慢（日级），适合做 regime 识别而非1h交易。作为状态特征使用。
"""
from __future__ import annotations

import json
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DB_PATH: str = str(Path(__file__).resolve().parent.parent / "onchain.db")
BASE: str = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


# Coinmetrics asset codes (subset of supported). / Coinmetrics 资产代码（部分）
CM_ASSETS: Dict[str, str] = {
    "BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol",
    "BNBUSDT": "bnb", "XRPUSDT": "xrp", "DOGEUSDT": "doge",
    "ADAUSDT": "ada", "AVAXUSDT": "avax", "LINKUSDT": "link",
    "DOTUSDT": "dot", "LTCUSDT": "ltc", "AAVEUSDT": "aave",
    "UNIUSDT": "uni", "ATOMUSDT": "atom",
}

# Free metrics (subset). / 免费指标（部分）。
DEFAULT_METRICS: List[str] = [
    "AdrActCnt",        # active addresses / 活跃地址
    "TxCnt",            # transaction count / 交易数
    "TxTfrValAdjUSD",   # adjusted transfer volume / 调整后转账量
]


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS onchain (
            asset TEXT, metric TEXT, ts INTEGER, value REAL,
            PRIMARY KEY (asset, metric, ts))
    """)
    conn.commit()
    return conn


def fetch_metric(asset: str, metric: str, days: int = 90) -> List[Tuple[int, float]]:
    """Fetch one (asset, metric) time series. / 获取单个(资产,指标)的时序。"""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    params = {
        "assets": asset,
        "metrics": metric,
        "start_time": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end_time": end.strftime("%Y-%m-%dT00:00:00Z"),
        "frequency": "1d",
    }
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())
        result = []
        for d in data.get("data", []):
            ts = int(datetime.fromisoformat(d["time"].replace("Z", "+00:00")).timestamp())
            val = d.get(metric)
            if val is not None:
                result.append((ts, float(val)))
        return result
    except Exception as e:
        print(f"  [FAIL] {asset}/{metric}: {e}")
        return []


def fetch_all(symbols: Optional[List[str]] = None,
              metrics: Optional[List[str]] = None,
              days: int = 90) -> Dict[str, int]:
    """Fetch all (symbol, metric) pairs and store in SQLite. / 批量入库。"""
    if symbols is None:
        symbols = list(CM_ASSETS.keys())
    if metrics is None:
        metrics = DEFAULT_METRICS

    conn = init_db()
    counts: Dict[str, int] = {}

    for sym in symbols:
        cm = CM_ASSETS.get(sym)
        if cm is None:
            continue
        total = 0
        for metric in metrics:
            recs = fetch_metric(cm, metric, days)
            if recs:
                conn.executemany(
                    "INSERT OR IGNORE INTO onchain VALUES (?, ?, ?, ?)",
                    [(sym, metric, ts, v) for ts, v in recs],
                )
                conn.commit()
                total += len(recs)
        counts[sym] = total
        if total:
            print(f"  [OK] {sym}: {total} records ({len(metrics)} metrics)")

    conn.close()
    return counts


if __name__ == "__main__":
    import time
    print(f"Fetching on-chain metrics for {len(CM_ASSETS)} assets ...")
    t0 = time.time()
    res = fetch_all(days=180)  # 6 months / 6个月
    print(f"Done in {time.time()-t0:.1f}s, {sum(res.values()):,} records")
