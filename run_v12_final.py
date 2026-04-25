"""
run_v12_final.py — Cost-Aware Training (turnover penalty) + Vol-Filter + Longer Hold.

v12 changes vs v11.1 (all targeting fee reduction; v11.1 ran -$632K fees):
  - min_hold 48 -> 96 (halve baseline rebalance frequency)
  - Turnover penalty added to training loss: paired forward at (t, t+1) and
    L2 penalty on score difference, weight LAMBDA_TURNOVER. Trains the model
    to produce temporally smooth scores -> fewer flips in backtest.
  - Vol-filter in backtest: skip rebalance when basket realized vol is in
    top VOL_QUANTILE of distribution (high-noise periods). Holds current
    position through the spike rather than churning into it.

v11.1 changes vs v11.0:
  - Dropped 4 noise factors by factor_analyzer IC ranking:
    volume_zscore, volume_momentum, macd, klow (all |IC_1h| < 0.003)
  - Real funding rate from funding_rates.db (Binance Vision archive),
    8h cadence forward-filled to 1h bar timestamps

v11 changes vs v10:
  - 13 factors (added funding_rate, btc_dominance, volume_momentum)
  - Uses FactorRegistry plugin system
  - Label: 1h (1-bar) forward return for training and backtest
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path
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

# trigger auto-discover of all factors / 触发所有因子的自动发现
import factors  # noqa: F401

# Factors to exclude based on factor_analyzer IC ranking (|IC_1h| < 0.003).
# Rerun tools/factor_analyzer.py to refresh this list.
# 基于 factor_analyzer IC 排名剔除的因子（|IC_1h| < 0.003）。
DROP_FACTORS: set = {"volume_zscore", "volume_momentum", "macd", "klow"}

FUNDING_DB: str = str(Path(__file__).resolve().parent / "funding_rates.db")

# v12 hyperparameters / v12 超参
LAMBDA_TURNOVER: float = 0.1   # weight of turnover penalty in training loss / 训练 loss 中换手惩罚权重
VOL_QUANTILE: float = 0.15     # skip rebalance during top X% basket-vol periods / 跳过 top X% 高波动时段
MIN_HOLD: int = 96             # min bars to hold a position before rebalance (was 48) / 最小持仓 bar 数


# ============================================================================
# Data: Parquet → aggregate → N factors (post-drop) + 1h label
# ============================================================================

def aggregate_5m_to_1h(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns((pl.col("open_time") // 3_600_000 * 3_600_000).alias("hour_ts"))
    return df.group_by("hour_ts").agg([
        pl.col("open").first(), pl.col("high").max(),
        pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
    ]).sort("hour_ts").rename({"hour_ts": "open_time"})


def load_and_align_funding(
    symbol: str, bar_times_ms: np.ndarray, device: torch.device,
    db_path: str = FUNDING_DB,
) -> Tensor:
    """
    Load funding rates for `symbol` from SQLite, forward-fill onto 1h bar
    timestamps. Funding cadence is 8h so ~8 bars share the same rate.
    Bars earlier than the first funding record get 0.0.

    从 SQLite 加载资金费率并按 1h bar 时间戳前向填充。资金费率每 8 小时更新，
    约 8 个 bar 共享同一费率。首条记录之前的 bar 填 0。
    """
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
    # for each bar time T, find last funding ts <= T / 前向填充
    idx = np.searchsorted(ts_arr, bar_times_ms, side="right") - 1
    out = np.zeros(len(bar_times_ms), dtype=np.float32)
    valid = idx >= 0
    out[valid] = rate_arr[idx[valid]]
    return torch.from_numpy(out).to(device)


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
    """Build 4D tensor with FactorRegistry-discovered factors + 1h forward return label."""
    print("[Data] Loading from Parquet data lake ...")
    raw_5m = load_klines_multi(interval="5m", min_rows=40000)
    print(f"  Loaded {len(raw_5m)} symbols")

    syms = sorted(raw_5m.keys())[:max_assets]
    agg_dfs = {sym: aggregate_5m_to_1h(raw_5m[sym]) for sym in syms}
    min_len = min(df.height for df in agg_dfs.values())

    # Filter out low-IC noise factors (see DROP_FACTORS at top of file)
    # 按 IC 过滤掉低信号因子
    all_registered = FactorRegistry.list_factors()
    factor_names = [n for n in all_registered if n not in DROP_FACTORS]
    n_factors = len(factor_names)
    dropped = [n for n in all_registered if n in DROP_FACTORS]
    print(f"  {len(syms)} assets, {min_len} 1h bars, {n_factors} factors")
    print(f"    kept:    {factor_names}")
    print(f"    dropped: {dropped}")

    close_mat = torch.zeros(min_len, len(syms), device=device)
    all_factors: Dict[str, Tensor] = {}
    funding_coverage_nonzero: List[float] = []

    for j, sym in enumerate(syms):
        rows = agg_dfs[sym].head(min_len)
        t = klines_to_tensors(rows, device)
        close_mat[:, j] = t["close"]
        # Align real funding rate to 1h bar timestamps (8h cadence -> 1h, forward-fill)
        # 将真实资金费率按 1h bar 时间戳对齐（8h 间隔 -> 1h 前向填充）
        bar_times_ms = rows["open_time"].to_numpy().astype(np.int64)
        funding_1h = load_and_align_funding(sym, bar_times_ms, device)
        funding_coverage_nonzero.append((funding_1h != 0).float().mean().item())
        all_factors[sym] = FactorRegistry.build_tensor(
            factor_names, t["open"], t["high"], t["low"], t["close"], t["volume"],
            zscore_window=48,
            extras={"funding": funding_1h},
        )

    avg_cov = sum(funding_coverage_nonzero) / max(len(funding_coverage_nonzero), 1)
    print(f"  Funding coverage: {avg_cov:.1%} of bars have real funding data")

    # 1-bar forward return (for BOTH training and backtest) / 1bar前瞻收益（训练和回测共用）
    ret_1h = torch.zeros(min_len, len(syms), device=device)
    for j in range(len(syms)):
        c = close_mat[:, j]
        ret_1h[:-1, j] = c[1:] / c[:-1].clamp(min=1e-8) - 1.0

    # sliding windows / 滑动窗口
    n_samples = min_len - seq_len - 1
    X_list, y_list = [], []
    for i in range(n_samples):
        X_list.append(torch.stack([all_factors[s][i:i+seq_len] for s in syms], dim=0))
        y_list.append(ret_1h[i + seq_len - 1])

    X = torch.stack(X_list)
    y = torch.stack(y_list)  # 1h label for both training and backtest / 1h标签（训练+回测共用）
    print(f"  X: {X.shape}, y: {y.shape}, label=1h return")
    return X, y, y, close_mat, syms, n_factors  # r1h = y (same) / r1h=y（相同）


# ============================================================================
# Train fold
# ============================================================================

def train_fold_indexed(
    model, loss_fn,
    X_all: Tensor, y_all: Tensor,
    train_idx: np.ndarray, val_idx: np.ndarray,
    epochs: int = 60, bs: int = 512, lr: float = 3e-4,
    lambda_turnover: float = LAMBDA_TURNOVER,
):
    """
    Zero-copy training with v12 turnover penalty.
    零拷贝训练 + v12 换手惩罚。

    v12 change: build paired indices (t such that t+1 is also in train_idx),
    do paired forward at (t, t+1), and add `lambda_turnover * (s_{t+1} - s_t)^2`
    to the loss. This regularizes the model toward temporally smooth scores
    so backtest rebalances less.
    v12 改动：构建配对索引（保留 t 和 t+1 都在 train_idx 中的 t），单次
    forward 同时跑 (t, t+1)，加 L2 惩罚到 loss 让模型输出时间平滑的 score。
    """
    device = X_all.device

    # Build (t, t+1) pairs constrained to train_idx (avoid val/test leakage in pair)
    # 构建 (t, t+1) 对，且两端都在 train_idx 中
    train_set = set(int(x) for x in train_idx)
    pair_t_np = np.array(
        [int(t) for t in train_idx if (int(t) + 1) in train_set],
        dtype=np.int64,
    )
    use_pairs = len(pair_t_np) > 0
    if use_pairs:
        pair_t_t = torch.from_numpy(pair_t_np).to(device)
    else:
        # degrade to single-bar (no turnover) — only happens for tiny train splits
        # 降级到单 bar（无 turnover），仅极小 split 会触发
        pair_t_t = torch.from_numpy(train_idx).to(device)

    val_idx_t = torch.from_numpy(val_idx).to(device)
    n_iters_step = len(pair_t_t)

    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_corr, best_state = -1.0, None
    patience, no_imp = 12, 0

    for ep in range(1, epochs + 1):
        model.train(); loss_fn.train()
        perm = torch.randperm(n_iters_step, device=device)
        for s in range(0, n_iters_step, bs):
            e = min(s + bs, n_iters_step)
            t_idx = pair_t_t[perm[s:e]]
            if use_pairs:
                t1_idx = t_idx + 1
                # paired forward in one kernel launch / 单次 forward 跑 paired batch
                combined = torch.cat([t_idx, t1_idx], dim=0)
                combined_scores = model(X_all[combined])
                scores_t, scores_t1 = combined_scores.chunk(2, dim=0)
                cls_loss = loss_fn(scores_t, y_all[t_idx])
                turnover_loss = (scores_t1 - scores_t).pow(2).mean()
                loss = cls_loss + lambda_turnover * turnover_loss
            else:
                # fallback: no turnover term / 无 turnover 后备路径
                scores = model(X_all[t_idx])
                loss = loss_fn(scores, y_all[t_idx])
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

def run_cpcv(X, y, r1h, close_mat, syms, seq_len, n_factors, device):
    n_samples = X.size(0)
    n_assets = X.size(1)

    splits = generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                                   purge_bars=seq_len, embargo_bars=48)
    print(f"\n[CPCV] {len(splits)} splits, {n_factors} factors, 1h label, lambda_turnover={LAMBDA_TURNOVER}")

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
            dropout=0.30, seq_len=seq_len, max_assets=n_assets,
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

        # Save fold checkpoint for ensemble inference (v12 has its own folder)
        # 保存 fold 模型用于集成（v12 单独文件夹避免覆盖 v11.1）
        import os as _os
        _os.makedirs("checkpoints/folds_v12", exist_ok=True)
        torch.save({"state": model.state_dict(), "corr": corr},
                   f"checkpoints/folds_v12/fold_{fi:02d}.pt")

        del model, loss_fn  # free model before next fold / 释放模型以回收显存
        torch.cuda.empty_cache()

    valid_mask = pred_count > 0
    pred_matrix[valid_mask] /= pred_count[valid_mask].unsqueeze(-1)

    print(f"\n[CPCV] Done in {time.time()-t0_total:.0f}s, avg corr={sum(fold_corrs)/len(fold_corrs):.4f}")

    # ----- Train production model on ALL data + save checkpoint -----
    # 用全部数据训练生产模型 + 保存checkpoint
    print(f"\n[Production] Training final model on all {n_samples} samples ...")
    prod_model = CrossAssetGRUAttention(
        n_factors=n_factors, d_model=128, gru_layers=2,
        n_cross_heads=4, n_cross_layers=3, d_ff=256,
        dropout=0.30, seq_len=seq_len, max_assets=n_assets,
    ).to(device)
    prod_loss = DualLoss().to(device)
    # last 10% as holdout val for early stopping / 末尾10%做早停
    val_size = max(int(n_samples * 0.10), 100)
    all_idx = np.arange(n_samples)
    train_idx_prod = all_idx[:-val_size]
    val_idx_prod = all_idx[-val_size:]
    prod_t0 = time.time()
    prod_model, prod_corr = train_fold_indexed(
        prod_model, prod_loss, X, y,
        train_idx_prod, val_idx_prod,
    )
    print(f"  Production model trained: corr={prod_corr:.4f} [{time.time()-prod_t0:.0f}s]")

    # save checkpoint / 保存checkpoint
    import os
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "v12_production.pt")
    torch.save({
        "model_state": prod_model.state_dict(),
        "n_factors": n_factors,
        "d_model": 128,
        "gru_layers": 2,
        "n_cross_heads": 4,
        "n_cross_layers": 3,
        "d_ff": 256,
        "dropout": 0.30,
        "seq_len": seq_len,
        "max_assets": n_assets,
        "val_corr": prod_corr,
        "factor_names": [n for n in FactorRegistry.list_factors() if n not in DROP_FACTORS],
        "lambda_turnover": LAMBDA_TURNOVER,
        "min_hold": MIN_HOLD,
        "vol_quantile": VOL_QUANTILE,
    }, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    # backtest / 回测
    # ----- v12 vol-filter precompute: rolling 1d basket realized vol -----
    # v12 vol filter 预算：1d 滚动 basket 已实现波动
    vol_window = 24
    basket_ret_np = r1h.mean(dim=1).cpu().numpy()
    rolling_vol = np.zeros(n_samples, dtype=np.float64)
    for i in range(vol_window, n_samples):
        rolling_vol[i] = basket_ret_np[i - vol_window:i].std()
    valid_vol_mask = rolling_vol > 0
    if valid_vol_mask.any():
        vol_thresh = float(np.quantile(rolling_vol[valid_vol_mask], 1.0 - VOL_QUANTILE))
    else:
        vol_thresh = float("inf")
    print(f"  Vol filter: skip rebalance when 1d basket vol > {vol_thresh:.5f} (top {VOL_QUANTILE:.0%})")

    twap = TWAPExecutor(n_slices=4, favorable_reject_rate=0.60)
    equity = 1_000_000.0
    eq_curve = [equity]
    total_cost = 0.0
    cl, cs = -1, -1
    hc = 0
    min_hold = MIN_HOLD
    rebalances = 0
    hold_periods = []
    vol_skips = 0  # rebalances suppressed by vol filter / 被 vol filter 抑制的 rebalance

    for t in range(n_samples):
        if not valid_mask[t]:
            eq_curve.append(equity); hc += 1; continue

        scores = pred_matrix[t]
        returns = r1h[t]  # use 1h return for PnL / 用 1h 收益算 PnL
        need = cl < 0 or (hc >= min_hold and
               (scores.argmax().item() != cl or scores.argmin().item() != cs))

        # vol filter: only skip if already in a position (cl >= 0); never delay first entry
        # vol 过滤：仅当已有持仓时跳过；首次入场不跳
        high_vol_skip = (cl >= 0 and rolling_vol[t] > vol_thresh)

        if need and (hc >= min_hold or cl < 0) and not high_vol_skip:
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
            if need and high_vol_skip:
                vol_skips += 1

        pr = 0.0
        if cl >= 0: pr += 0.5 * returns[cl].item()
        if cs >= 0: pr -= 0.5 * returns[cs].item()
        pr -= cost_bar / max(equity, 1.0)
        pr = max(min(pr, 0.10), -0.10)  # clamp ±10% per bar to prevent compounding explosion / 限制单bar±10%防止复利爆炸
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
        "vol_skips": vol_skips, "vol_thresh": vol_thresh,
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
    print(f"  v12 — Cost-Aware Training (lambda={LAMBDA_TURNOVER})")
    print(f"        + Vol-Filter (top {VOL_QUANTILE:.0%}) + Min Hold {MIN_HOLD}h")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    SEQ_LEN = 24
    MAX_ASSETS = 20
    X, y, r1h, close_mat, syms, n_factors = build_from_parquet(
        SEQ_LEN, MAX_ASSETS, device,
    )

    summary = run_cpcv(X, y, r1h, close_mat, syms, SEQ_LEN, n_factors, device)

    print("\n" + "=" * 65)
    print("  v12 BACKTEST REPORT")
    print("=" * 65)
    print(f"  {'Factors':.<40s} {n_factors}")
    print(f"  {'Label':.<40s} 1h forward return")
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
    print(f"  {'Vol-Filter Skips':.<40s} {summary['vol_skips']}")
    print(f"  {'Vol Threshold':.<40s} {summary['vol_thresh']:.5f}")
    print(f"  {'Avg Hold (hours)':.<40s} {summary['avg_hold']:.1f}")
    ts = summary.get('total_slices', 0)
    if ts > 0:
        print(f"  {'Maker%':.<40s} {summary['maker_fill_pct']:.1%}")
        print(f"  {'Adverse%':.<40s} {summary['adverse_fill_pct']:.1%}")
    print("=" * 65)


if __name__ == "__main__":
    main()
