"""
Alphalens-style factor analyzer.
Alphalens 风格的因子分析器。

For each factor, computes:
  - IC (Information Coefficient): rank correlation with forward returns
  - IC IR (mean / std of IC): stability of the signal
  - Quintile returns: long top-quintile, short bottom-quintile spread
  - Decay: IC at 1h, 4h, 24h, 48h horizons

Usage:
    python tools/factor_analyzer.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch
from torch import Tensor

import factors as _  # auto-discover / 触发自动注册
from factors.base import FactorRegistry
from data.lake_loader import load_klines_multi, klines_to_tensors


def aggregate_5m_to_1h(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 5m klines to 1h (open_time already normalized to ms by lake_loader).
    将 5m K线聚合为 1h（open_time 已由 lake_loader 归一化为毫秒）。"""
    df = df.with_columns((pl.col("open_time") // 3_600_000 * 3_600_000).alias("hour_ts"))
    return df.group_by("hour_ts").agg([
        pl.col("open").first(), pl.col("high").max(),
        pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
    ]).sort("hour_ts").rename({"hour_ts": "open_time"})


def load_funding(symbol: str, bar_times_ms: np.ndarray, device: torch.device) -> Tensor:
    """Forward-fill real funding rates onto bar timestamps. / 真实资金费率前向填充到bar时间戳。"""
    import os
    import sqlite3
    db_path = str(Path(__file__).resolve().parent.parent / "funding_rates.db")
    if not os.path.exists(db_path):
        return torch.zeros(len(bar_times_ms), dtype=torch.float32, device=device)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT ts_ms, rate FROM funding WHERE symbol=? ORDER BY ts_ms", (symbol,)
    ).fetchall()
    conn.close()
    if not rows:
        return torch.zeros(len(bar_times_ms), dtype=torch.float32, device=device)
    ts_arr = np.asarray([r[0] for r in rows], dtype=np.int64)
    rate_arr = np.asarray([r[1] for r in rows], dtype=np.float32)
    idx = np.searchsorted(ts_arr, bar_times_ms, side="right") - 1
    out = np.zeros(len(bar_times_ms), dtype=np.float32)
    valid = idx >= 0
    out[valid] = rate_arr[idx[valid]]
    return torch.from_numpy(out).to(device)


def compute_rank_corr(a: Tensor, b: Tensor) -> Tensor:
    """Cross-sectional rank correlation per timestep. / 每个时间步的横截面排名相关性。"""
    ar = a.argsort(-1, descending=True).argsort(-1).float()
    br = b.argsort(-1, descending=True).argsort(-1).float()
    am = ar.mean(-1, keepdim=True)
    bm = br.mean(-1, keepdim=True)
    cov = ((ar - am) * (br - bm)).sum(-1)
    denom = ((ar - am).pow(2).sum(-1).sqrt() *
             (br - bm).pow(2).sum(-1).sqrt()).clamp(1e-8)
    return cov / denom


def analyze_factor(
    factor_values: Tensor,   # (T, A) factor values per (timestep, asset)
    closes_mat: Tensor,      # (T, A) close prices
    horizons: List[int] = [1, 6, 24, 48],
) -> Dict[str, float]:
    """
    Compute factor metrics across multiple horizons.
    IC at horizon h pairs factor[t] with the h-bar forward return
    close[t+h]/close[t] - 1 (true horizon return, no off-by-one).
    在多个时间跨度上计算因子指标。h 跨度的 IC 将 factor[t] 与
    close[t+h]/close[t]-1（真实 h-bar 前瞻收益）配对，无错位。
    """
    metrics: Dict[str, float] = {}

    for h in horizons:
        if h >= closes_mat.size(0):
            continue
        f = factor_values[:-h]
        r = closes_mat[h:] / closes_mat[:-h].clamp(min=1e-8) - 1.0
        ic_per_t = compute_rank_corr(f, r)
        ic_mean = ic_per_t.mean().item()
        ic_std = ic_per_t.std().item()
        ir = ic_mean / max(ic_std, 1e-9)
        metrics[f"ic_{h}h"] = ic_mean
        metrics[f"ir_{h}h"] = ir

    # quintile spread (top 20% long - bottom 20% short) at horizon=24
    # 五分位数差（top 20%做多 - bottom 20%做空，horizon=24，对齐持有期）
    h_q = 24 if closes_mat.size(0) > 24 else 1
    f = factor_values[:-h_q]
    r = closes_mat[h_q:] / closes_mat[:-h_q].clamp(min=1e-8) - 1.0
    n_assets = f.size(1)
    n_q = max(1, n_assets // 5)

    top_idx = f.topk(n_q, dim=-1).indices
    bot_idx = f.topk(n_q, dim=-1, largest=False).indices

    top_ret = r.gather(1, top_idx).mean(-1)
    bot_ret = r.gather(1, bot_idx).mean(-1)
    spread = (top_ret - bot_ret).mean().item()
    metrics["quintile_spread_24h"] = spread

    return metrics


def main():
    print("=" * 80)
    print("  Factor IC Analyzer (Alphalens-style)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data / 加载数据
    print("\n[Loading] Parquet data lake ...")
    raw = load_klines_multi(interval="5m", min_rows=40000)
    syms = sorted(raw.keys())[:20]

    # aggregate 5m -> 1h, then align all symbols on common timestamps
    # 5m 聚合为 1h，再按公共时间戳对齐所有标的（避免位置错位）
    agg = {s: aggregate_5m_to_1h(raw[s]) for s in syms}
    common_ts = None
    for s in syms:
        ts = agg[s]["open_time"].to_numpy()
        common_ts = ts if common_ts is None else np.intersect1d(common_ts, ts)
    aligned = {s: agg[s].filter(pl.col("open_time").is_in(common_ts)).sort("open_time")
               for s in syms}
    min_len = len(common_ts)
    print(f"  {len(syms)} assets, {min_len:,} aligned 1h bars "
          f"({raw[syms[0]].height:,} 5m rows for {syms[0]})")

    factor_names = FactorRegistry.list_factors()
    print(f"  {len(factor_names)} factors: {factor_names}")

    n_assets = len(syms)
    closes_mat = torch.zeros(min_len, n_assets, device=device)
    tensors = {}
    fundings = {}
    for j, sym in enumerate(syms):
        t = klines_to_tensors(aligned[sym], device)
        tensors[sym] = t
        closes_mat[:, j] = t["close"]
        bar_times = aligned[sym]["open_time"].to_numpy().astype(np.int64)
        fundings[sym] = load_funding(sym, bar_times, device)
    fund_cov = float(torch.stack([(f != 0).float().mean() for f in fundings.values()]).mean())
    print(f"  Real funding coverage: {fund_cov:.1%}")

    # for each factor, compute (T, A) values and analyze / 每个因子计算并分析
    print("\n[Analysis] Per-factor metrics (true 1h bars):")
    print(f"{'Factor':<22} {'IC_1h':>9} {'IR_1h':>9} {'IC_6h':>9} {'IC_24h':>9} {'IC_48h':>9} {'Sprd24h':>9}")
    print("-" * 80)

    results = {}
    for fname in factor_names:
        f_obj = FactorRegistry.get(fname)
        # (T, A) tensor / (时间, 资产) 张量
        f_vals = torch.zeros(min_len, n_assets, device=device)
        for j, sym in enumerate(syms):
            t = tensors[sym]
            try:
                import inspect as _inspect
                sig = _inspect.signature(f_obj.compute)
                if "extras" in sig.parameters:
                    vals = f_obj.compute(t["open"], t["high"], t["low"], t["close"], t["volume"],
                                         extras={"funding": fundings[sym]})
                else:
                    vals = f_obj.compute(t["open"], t["high"], t["low"], t["close"], t["volume"])
                f_vals[:, j] = torch.nan_to_num(vals, nan=0.0)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
                continue

        m = analyze_factor(f_vals, closes_mat)
        results[fname] = m
        print(f"{fname:<22} "
              f"{m.get('ic_1h', 0):>+9.4f} "
              f"{m.get('ir_1h', 0):>+9.4f} "
              f"{m.get('ic_6h', 0):>+9.4f} "
              f"{m.get('ic_24h', 0):>+9.4f} "
              f"{m.get('ic_48h', 0):>+9.4f} "
              f"{m.get('quintile_spread_24h', 0):>+9.4%}")

    # rank by absolute IC at the 24h holding horizon / 按持有期 |IC_24h| 排序
    print("\n[Ranking] Top factors by |IC_24h| (v13 label horizon):")
    sorted_factors = sorted(results.items(), key=lambda x: abs(x[1].get('ic_24h', 0)), reverse=True)
    for i, (n, m) in enumerate(sorted_factors):
        print(f"  {i+1:2d}. {n:<22} |IC_24h|={abs(m.get('ic_24h', 0)):.4f}  "
              f"IC_1h={m.get('ic_1h', 0):+.4f}  IR_24h={m.get('ir_24h', 0):+.3f}")


if __name__ == "__main__":
    main()
