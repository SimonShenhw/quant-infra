"""
run_v11_final.py — 13 Factors + 6h Multi-Horizon Label + CPCV.

v11 changes vs v10:
  - 13 factors (added funding_rate, btc_dominance, volume_momentum)
  - 6-bar forward cumulative return as label (not 1-bar)
  - Uses FactorRegistry plugin system
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import polars as pl

sys.path.insert(0, ".")

from data.lake_loader import load_klines_multi, klines_to_tensors
from engine.cpcv import generate_cpcv_splits
from engine.twap_executor import TWAPExecutor
from factors.base import FactorRegistry
from model.cross_asset_attention import CrossAssetGRUAttention
from model.cross_sectional import listmle_loss

# trigger auto-discover of all 13 factors / 触发13个因子的自动发现
import factors  # noqa: F401


# ============================================================================
# Data: Parquet → aggregate → 13 factors + 6h label
# ============================================================================

def aggregate_5m_to_1h(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns((pl.col("open_time") // 3_600_000 * 3_600_000).alias("hour_ts"))
    return df.group_by("hour_ts").agg([
        pl.col("open").first(), pl.col("high").max(),
        pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
    ]).sort("hour_ts").rename({"hour_ts": "open_time"})


class DualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lv_r = nn.Parameter(torch.tensor(0.0))
        self.lv_d = nn.Parameter(torch.tensor(0.0))

    def forward(self, scores: Tensor, rets: Tensor) -> Tensor:
        lr = listmle_loss(scores, rets)
        med = rets.median(-1, keepdim=True).values
        tgt = (rets > med).float()
        logits = scores * 10.0
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        pt = p * tgt + (1 - p) * (1 - tgt)
        focal = ((1 - pt) ** 2 * ce).mean()
        pr, pd = torch.exp(-self.lv_r), torch.exp(-self.lv_d)
        return 0.5 * pr * lr + self.lv_r + 0.5 * pd * focal + self.lv_d


def build_from_parquet(
    seq_len: int, max_assets: int, device: torch.device,
    fwd_horizon: int = 6,
) -> Tuple[Tensor, Tensor, Tensor, List[str], int]:
    """Build 4D tensor with 13 plugin factors + 6h cumulative return label."""
    print("[Data] Loading from Parquet data lake ...")
    raw_5m = load_klines_multi(interval="5m", min_rows=40000)
    print(f"  Loaded {len(raw_5m)} symbols")

    syms = sorted(raw_5m.keys())[:max_assets]
    agg_dfs = {sym: aggregate_5m_to_1h(raw_5m[sym]) for sym in syms}
    min_len = min(df.height for df in agg_dfs.values())

    factor_names = FactorRegistry.list_factors()
    n_factors = len(factor_names)
    print(f"  {len(syms)} assets, {min_len} 1h bars, {n_factors} factors: {factor_names}")

    close_mat = torch.zeros(min_len, len(syms), device=device)
    all_factors: Dict[str, Tensor] = {}

    for j, sym in enumerate(syms):
        rows = agg_dfs[sym].head(min_len)
        t = klines_to_tensors(rows, device)
        close_mat[:, j] = t["close"]
        # use FactorRegistry with all 13 factors / 使用13因子注册表
        all_factors[sym] = FactorRegistry.build_tensor(
            factor_names, t["open"], t["high"], t["low"], t["close"], t["volume"],
            zscore_window=48,
        )

    # 6-bar forward cumulative return / 6bar前瞻累计收益
    fwd_ret = torch.zeros(min_len, len(syms), device=device)
    for j in range(len(syms)):
        c = close_mat[:, j]
        fwd_ret[:min_len - fwd_horizon, j] = (
            c[fwd_horizon:] / c[:min_len - fwd_horizon].clamp(min=1e-8) - 1.0
        )

    # sliding windows / 滑动窗口
    n_samples = min_len - seq_len - fwd_horizon
    X_list, y_list = [], []
    for i in range(n_samples):
        X_list.append(torch.stack([all_factors[s][i:i+seq_len] for s in syms], dim=0))
        y_list.append(fwd_ret[i + seq_len - 1])

    X = torch.stack(X_list)
    y = torch.stack(y_list)
    print(f"  X: {X.shape}, y: {y.shape}, fwd_horizon={fwd_horizon}h")
    return X, y, close_mat, syms, n_factors


# ============================================================================
# Train fold
# ============================================================================

def train_fold_indexed(
    model, loss_fn,
    X_all: Tensor, y_all: Tensor,
    train_idx: np.ndarray, val_idx: np.ndarray,
    epochs: int = 60, bs: int = 512, lr: float = 3e-4,
):
    """
    Zero-copy training: index into X_all per-batch instead of copying entire X_tr.
    零拷贝训练：按batch从X_all中索引，不复制整个X_tr。
    Eliminates the ~1.2GB VRAM spike at fold start.
    消除每个fold开始时的~1.2GB显存尖峰。
    """
    device = X_all.device
    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_idx_t = torch.from_numpy(val_idx).to(device)
    n_train = len(train_idx)

    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_corr, best_state = -1.0, None
    patience, no_imp = 12, 0

    for ep in range(1, epochs + 1):
        model.train(); loss_fn.train()
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, bs):
            e = min(s + bs, n_train)
            batch_idx = train_idx_t[perm[s:e]]
            # only copy bs samples (~26MB), not entire train set (~1.2GB)
            # 每次仅复制bs个样本，而非整个训练集
            scores = model(X_all[batch_idx])
            loss = loss_fn(scores, y_all[batch_idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()

        # validation: also chunk to avoid spike / 验证也分块避免尖峰
        model.eval()
        with torch.no_grad():
            val_scores_list = []
            for vs in range(0, len(val_idx), bs):
                ve = min(vs + bs, len(val_idx))
                val_scores_list.append(model(X_all[val_idx_t[vs:ve]]))
            vs_cat = torch.cat(val_scores_list, dim=0)
            vy = y_all[val_idx_t]
            pr = vs_cat.argsort(-1, descending=True).argsort(-1).float()
            tr = vy.argsort(-1, descending=True).argsort(-1).float()
            pm, tm = pr.mean(-1, True), tr.mean(-1, True)
            cov = ((pr-pm)*(tr-tm)).sum(-1)
            corr = (cov / ((pr-pm).pow(2).sum(-1).sqrt()*(tr-tm).pow(2).sum(-1).sqrt()).clamp(1e-8)).mean().item()

        if corr > best_corr:
            best_corr = corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state: model.load_state_dict(best_state)
    return model, best_corr


# ============================================================================
# CPCV + Backtest (reused from v10, with exec fix)
# ============================================================================

def run_cpcv(X, y, close_mat, syms, seq_len, n_factors, device):
    n_samples = X.size(0)
    n_assets = X.size(1)

    splits = generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                                   purge_bars=seq_len, embargo_bars=48)
    print(f"\n[CPCV] {len(splits)} splits, {n_factors} factors, 6h label")

    pred_matrix = torch.zeros(n_samples, n_assets, device=device)
    pred_count = torch.zeros(n_samples, device=device)
    fold_corrs = []
    t0_total = time.time()

    for fi, (train_idx, test_idx) in enumerate(splits):
        n_tr = len(train_idx)
        val_size = max(int(n_tr * 0.2), 20)
        val_idx = train_idx[-val_size:]
        actual_train_idx = train_idx[:-val_size]

        model = CrossAssetGRUAttention(
            n_factors=n_factors, d_model=128, gru_layers=2,
            n_cross_heads=4, n_cross_layers=3, d_ff=256,
            dropout=0.25, seq_len=seq_len, max_assets=n_assets,
        ).to(device)
        loss_fn = DualLoss().to(device)

        t0 = time.time()
        # zero-copy: pass indices, not sliced copies / 零拷贝：传索引而非切片拷贝
        model, corr = train_fold_indexed(
            model, loss_fn, X, y,
            actual_train_idx, val_idx,
        )
        fold_corrs.append(corr)

        # OOS prediction: also chunked to avoid VRAM spike / OOS预测也分块
        model.eval()
        test_idx_t = torch.from_numpy(test_idx).to(device)
        with torch.no_grad():
            for ts in range(0, len(test_idx), 512):
                te = min(ts + 512, len(test_idx))
                chunk_idx = test_idx_t[ts:te]
                chunk_scores = model(X[chunk_idx])
                pred_matrix[chunk_idx] += chunk_scores
                pred_count[chunk_idx] += 1

        if (fi + 1) % 5 == 0 or fi == 0:
            print(f"  Split {fi+1}/{len(splits)}: corr={corr:.4f} [{time.time()-t0:.0f}s]")
        del model, loss_fn  # free model before next fold / 释放模型以回收显存
        torch.cuda.empty_cache()

    valid_mask = pred_count > 0
    pred_matrix[valid_mask] /= pred_count[valid_mask].unsqueeze(-1)

    print(f"\n[CPCV] Done in {time.time()-t0_total:.0f}s, avg corr={sum(fold_corrs)/len(fold_corrs):.4f}")

    # backtest / 回测
    twap = TWAPExecutor(n_slices=4, favorable_reject_rate=0.60)
    equity = 1_000_000.0
    eq_curve = [equity]
    total_cost = 0.0
    cl, cs = -1, -1
    hc = 0
    min_hold = 48
    rebalances = 0
    hold_periods = []

    for t in range(n_samples):
        if not valid_mask[t]:
            eq_curve.append(equity); hc += 1; continue

        scores = pred_matrix[t]
        returns = y[t]
        need = cl < 0 or (hc >= min_hold and
               (scores.argmax().item() != cl or scores.argmin().item() != cs))

        if need and (hc >= min_hold or cl < 0):
            nl, ns = scores.argmax().item(), scores.argmin().item()
            cost_bar = 0.0
            legs = sum([cl!=nl and cl>=0, cs!=ns and cs>=0, cl!=nl, cs!=ns])
            if legs > 0:
                exec_bar = t + seq_len
                ci = min(exec_bar, close_mat.size(0)-1)
                fi_list = [min(exec_bar+k, close_mat.size(0)-1) for k in range(4)]
                ep = close_mat[ci, nl].item()
                fu = [close_mat[f, nl].item() for f in fi_list]
                if ep > 0 and fu:
                    _, cbps, _ = twap.execute_twap("BUY", equity*0.5/max(legs,1), ep, fu)
                    cost_bar = equity * 0.5 * cbps / 10000.0 * legs
            total_cost += cost_bar
            if cl >= 0: hold_periods.append(hc)
            cl, cs = nl, ns; hc = 0; rebalances += 1
        else:
            cost_bar = 0.0

        pr = 0.0
        if cl >= 0: pr += 0.5 * returns[cl].item()
        if cs >= 0: pr -= 0.5 * returns[cs].item()
        pr -= cost_bar / max(equity, 1.0)
        equity *= (1.0 + pr)
        eq_curve.append(equity); hc += 1

    rets = [(eq_curve[i]/eq_curve[i-1])-1 for i in range(1, len(eq_curve))]
    avg = sum(rets)/len(rets) if rets else 0
    std = (sum((r-avg)**2 for r in rets)/len(rets))**0.5 if rets else 1e-9
    sharpe = (avg / max(std, 1e-9)) * (24*365)**0.5
    peak = eq_curve[0]; max_dd = 0.0
    for e in eq_curve:
        peak = max(peak, e); max_dd = max(max_dd, (peak-e)/peak)

    return {
        "total_return": eq_curve[-1]/eq_curve[0]-1, "sharpe": sharpe,
        "max_drawdown": max_dd, "final_equity": eq_curve[-1],
        "total_cost": total_cost, "rebalances": rebalances,
        "avg_hold": sum(hold_periods)/max(len(hold_periods),1),
        "n_samples": n_samples, "n_splits": len(splits),
        "avg_corr": sum(fold_corrs)/max(len(fold_corrs),1),
        "fold_corrs": fold_corrs, **twap.stats(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 65)
    print("  v11.0 — 13 Factors + 6h Label + CPCV")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    SEQ_LEN = 24
    MAX_ASSETS = 20
    FWD_HORIZON = 6  # predict 6h cumulative return / 预测6h累计收益

    X, y, close_mat, syms, n_factors = build_from_parquet(
        SEQ_LEN, MAX_ASSETS, device, fwd_horizon=FWD_HORIZON,
    )

    summary = run_cpcv(X, y, close_mat, syms, SEQ_LEN, n_factors, device)

    print("\n" + "=" * 65)
    print("  v11.0 BACKTEST REPORT")
    print("=" * 65)
    print(f"  {'Factors':.<40s} {n_factors} (13 plugin)")
    print(f"  {'Label Horizon':.<40s} {FWD_HORIZON}h cumulative return")
    print(f"  {'CPCV Splits':.<40s} {summary['n_splits']}")
    print(f"  {'Samples':.<40s} {summary['n_samples']:,}")
    print(f"  {'Avg Rank Corr':.<40s} {summary['avg_corr']:.4f}")
    print(f"  {'--- PERFORMANCE ---':.<40s}")
    print(f"  {'Total Return':.<40s} {summary['total_return']:>10.4%}")
    print(f"  {'Sharpe Ratio':.<40s} {summary['sharpe']:>10.4f}")
    print(f"  {'Max Drawdown':.<40s} {summary['max_drawdown']:>10.4%}")
    print(f"  {'Final Equity':.<40s} {summary['final_equity']:>14,.2f}")
    print(f"  {'Total Cost':.<40s} {summary['total_cost']:>14,.2f}")
    print(f"  {'Rebalances':.<40s} {summary['rebalances']}")
    print(f"  {'Avg Hold (hours)':.<40s} {summary['avg_hold']:.1f}")
    ts = summary.get('total_slices', 0)
    if ts > 0:
        print(f"  {'Maker%':.<40s} {summary['maker_fill_pct']:.1%}")
        print(f"  {'Adverse%':.<40s} {summary['adverse_fill_pct']:.1%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
