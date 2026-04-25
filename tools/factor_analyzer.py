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

import torch
from torch import Tensor

import factors as _  # auto-discover / 触发自动注册
from factors.base import FactorRegistry
from data.lake_loader import load_klines_multi, klines_to_tensors


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
    fwd_returns: Tensor,     # (T, A) forward returns
    horizons: List[int] = [1, 6, 24, 48],
) -> Dict[str, float]:
    """
    Compute factor metrics across multiple horizons.
    在多个时间跨度上计算因子指标。
    """
    metrics: Dict[str, float] = {}

    for h in horizons:
        # shift returns by h bars / 收益率前移h个bar
        if h >= fwd_returns.size(0):
            continue
        f = factor_values[:-h]
        r = fwd_returns[h:]
        ic_per_t = compute_rank_corr(f, r)
        ic_mean = ic_per_t.mean().item()
        ic_std = ic_per_t.std().item()
        ir = ic_mean / max(ic_std, 1e-9)
        metrics[f"ic_{h}h"] = ic_mean
        metrics[f"ir_{h}h"] = ir

    # quintile spread (top 20% long - bottom 20% short) at horizon=1
    # 五分位数差（top 20%做多 - bottom 20%做空，horizon=1）
    f = factor_values[:-1]
    r = fwd_returns[1:]
    n_assets = f.size(1)
    n_q = max(1, n_assets // 5)

    top_idx = f.topk(n_q, dim=-1).indices
    bot_idx = f.topk(n_q, dim=-1, largest=False).indices

    top_ret = r.gather(1, top_idx).mean(-1)
    bot_ret = r.gather(1, bot_idx).mean(-1)
    spread = (top_ret - bot_ret).mean().item()
    metrics["quintile_spread"] = spread

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
    min_len = min(raw[s].height for s in syms)
    print(f"  {len(syms)} assets, {min_len:,} 5m bars")

    # build factor tensors per asset / 每个资产的因子张量
    factor_names = FactorRegistry.list_factors()
    print(f"  {len(factor_names)} factors: {factor_names}")

    n_assets = len(syms)
    factor_data: Dict[str, Tensor] = {}
    closes_mat = torch.zeros(min_len, n_assets, device=device)

    for j, sym in enumerate(syms):
        rows = raw[sym].head(min_len)
        t = klines_to_tensors(rows, device)
        closes_mat[:, j] = t["close"]

    # forward returns / 前瞻收益
    fwd = torch.zeros(min_len, n_assets, device=device)
    for j in range(n_assets):
        fwd[:-1, j] = closes_mat[1:, j] / closes_mat[:-1, j].clamp(min=1e-8) - 1.0

    # for each factor, compute (T, A) values and analyze / 每个因子计算并分析
    print("\n[Analysis] Per-factor metrics:")
    print(f"{'Factor':<22} {'IC_1h':>9} {'IR_1h':>9} {'IC_6h':>9} {'IC_24h':>9} {'IC_48h':>9} {'Spread':>9}")
    print("-" * 80)

    results = {}
    for fname in factor_names:
        f_obj = FactorRegistry.get(fname)
        # (T, A) tensor / (时间, 资产) 张量
        f_vals = torch.zeros(min_len, n_assets, device=device)
        for j, sym in enumerate(syms):
            rows = raw[sym].head(min_len)
            t = klines_to_tensors(rows, device)
            try:
                vals = f_obj.compute(t["open"], t["high"], t["low"], t["close"], t["volume"])
                f_vals[:, j] = torch.nan_to_num(vals, nan=0.0)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
                continue

        m = analyze_factor(f_vals, fwd)
        results[fname] = m
        print(f"{fname:<22} "
              f"{m.get('ic_1h', 0):>+9.4f} "
              f"{m.get('ir_1h', 0):>+9.4f} "
              f"{m.get('ic_6h', 0):>+9.4f} "
              f"{m.get('ic_24h', 0):>+9.4f} "
              f"{m.get('ic_48h', 0):>+9.4f} "
              f"{m.get('quintile_spread', 0):>+9.4%}")

    # rank by absolute IC / 按 |IC| 排序
    print("\n[Ranking] Top factors by |IC_1h|:")
    sorted_factors = sorted(results.items(), key=lambda x: abs(x[1].get('ic_1h', 0)), reverse=True)
    for i, (n, m) in enumerate(sorted_factors[:10]):
        print(f"  {i+1:2d}. {n:<22} |IC|={abs(m.get('ic_1h', 0)):.4f}  IR={m.get('ir_1h', 0):+.3f}")


if __name__ == "__main__":
    main()
