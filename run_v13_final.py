"""
run_v13_final.py — Fixed data + horizon-aligned label + banded portfolio.

v13 changes vs v12 (each tagged with the review finding / literature it answers):

  DATA
  - C-1 fix: lake_loader now normalizes Binance Vision us->ms timestamps, so
    aggregate_5m_to_1h produces TRUE 1h bars (~13.1K, was 97% raw 5m bars).
    All v11/v12 headline numbers were computed on corrupted aggregation.
  - Symbols aligned on common timestamps (intersection), not positional head().
  - DROP_FACTORS refreshed from factor_analyzer re-run on true 1h bars:
    only {macd, volume_zscore} are noise at both 1h and 24h horizons.
    (old v12 list wrongly dropped klow — #5 by |IC_24h| on clean data.)

  LABEL (H-6 fix)
  - 24-bar (24h) forward return label instead of 1-bar: matches the daily
    decision cadence of paper trading and the multi-day effective hold.

  BACKTEST
  - H-1 fix: positions take effect on the bar AFTER the decision bar
    (TWAP executes during the decision->next-bar window). v12 credited the
    label bar itself to the new position — pure look-ahead PnL.
  - Banded top-K portfolio (Novy-Marx & Velikov RFS 2016 buy/hold spread;
    qlib TopkDropout): enter a long slot only when rank <= ENTER_BAND,
    exit only when rank falls below EXIT_BAND. Cuts turnover while keeping
    signal exposure. Decisions evaluated every DECISION_EVERY bars (daily),
    same cadence as paper trading -> backtest and paper now test the SAME
    strategy.
  - M-6 fix: per-slot TWAP cost on each asset's own price path.
  - M-1 fix: vol filter threshold is an EXPANDING (causal) quantile.
  - M-5 fix: seeded RNG -> reproducible cost simulation.

  VALIDATION
  - PSR + DSR (Bailey & Lopez de Prado) printed for every config;
    DSR benchmark = expected max Sharpe over N_TRIALS tried configurations.

  TRAINING
  - M-13 fix: purge gap (seq_len + LABEL_H) between train tail and val slice.
  - Turnover penalty kept from v12 (lambda=0.1, arXiv:2509.04541 motivates
    band-style turnover regularization).
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
from tools.validation_stats import probabilistic_sharpe, deflated_sharpe

import factors  # noqa: F401  trigger auto-discover / 触发因子自动发现

SEED: int = 42

# Refreshed via tools/factor_analyzer.py on TRUE 1h bars (2026-06-10 rerun):
# macd |IC_1h|=0.0013 |IC_24h|=0.0055, volume_zscore 0.0070/0.0033 — noise at
# both horizons. Everything else kept (old v12 drop of klow was an artifact
# of the corrupted 5m aggregation).
# 基于真 1h 数据重跑 factor_analyzer 后的剔除名单：仅 macd 与 volume_zscore
# 在两个 horizon 都接近零。v12 错杀的 klow 恢复。
DROP_FACTORS: set = {"macd", "volume_zscore"}

FUNDING_DB: str = str(Path(__file__).resolve().parent / "funding_rates.db")

# v13 hyperparameters / v13 超参
LABEL_H: int = 24          # label horizon in bars (24h), matches decision cadence / 标签前瞻期
DECISION_EVERY: int = 24   # evaluate rebalance once per 24 bars (daily) / 每24根bar(每日)决策一次
BASKET_K: int = 3          # assets per leg / 每腿持仓数
ENTER_BAND: int = 3        # enter long slot only if rank < ENTER_BAND / 进入阈值
EXIT_BAND: int = 6         # exit long slot only if rank >= EXIT_BAND / 退出阈值
LAMBDA_TURNOVER: float = 0.1
VOL_QUANTILE: float = 0.15
N_TRIALS_DSR: int = 50     # honest-ish count of configs tried across v1..v13 / 历史尝试配置数


# ============================================================================
# Data: Parquet -> true 1h bars -> 19 factors + 24h label
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
    """Forward-fill real 8h funding onto 1h bar timestamps (both in ms).
    将真实 8h 资金费率前向填充到 1h bar 时间戳（均为毫秒）。"""
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


def build_from_parquet(
    seq_len: int, max_assets: int, device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[str], int]:
    """4D factor tensor + 24h label + 1h bar returns (for PnL accounting).
    Returns (X, y24, r1h, close_mat, syms, n_factors)."""
    print("[Data] Loading from Parquet data lake (timestamps normalized) ...")
    raw_5m = load_klines_multi(interval="5m", min_rows=40000)
    print(f"  Loaded {len(raw_5m)} symbols")

    syms = sorted(raw_5m.keys())[:max_assets]
    agg_dfs = {sym: aggregate_5m_to_1h(raw_5m[sym]) for sym in syms}

    # align on common timestamps — positional head() can misalign cross-sections
    # 按公共时间戳对齐——位置对齐在起始时间不同步时会错位
    common_ts = None
    for sym in syms:
        ts = agg_dfs[sym]["open_time"].to_numpy()
        common_ts = ts if common_ts is None else np.intersect1d(common_ts, ts)
    aligned = {
        sym: agg_dfs[sym].filter(pl.col("open_time").is_in(common_ts)).sort("open_time")
        for sym in syms
    }
    min_len = len(common_ts)

    all_registered = FactorRegistry.list_factors()
    factor_names = [n for n in all_registered if n not in DROP_FACTORS]
    n_factors = len(factor_names)
    print(f"  {len(syms)} assets, {min_len} aligned TRUE 1h bars, {n_factors} factors")
    print(f"    dropped: {sorted(DROP_FACTORS)}")

    close_mat = torch.zeros(min_len, len(syms), device=device)
    all_factors: Dict[str, Tensor] = {}
    funding_cov: List[float] = []

    for j, sym in enumerate(syms):
        rows = aligned[sym]
        t = klines_to_tensors(rows, device)
        close_mat[:, j] = t["close"]
        bar_times_ms = rows["open_time"].to_numpy().astype(np.int64)
        funding_1h = load_and_align_funding(sym, bar_times_ms, device)
        funding_cov.append((funding_1h != 0).float().mean().item())
        all_factors[sym] = FactorRegistry.build_tensor(
            factor_names, t["open"], t["high"], t["low"], t["close"], t["volume"],
            zscore_window=48,
            extras={"funding": funding_1h},
        )

    print(f"  Funding coverage: {sum(funding_cov)/len(funding_cov):.1%} of bars (real data)")

    # bar returns r1h[t] = close[t+seq_len]/close[t+seq_len-1]-1: the bar
    # IMMEDIATELY AFTER decision point t (used only for PnL accounting).
    # 24h label y24[t] = close[t+seq_len-1+24]/close[t+seq_len-1]-1.
    n_samples = min_len - seq_len - LABEL_H
    close_np = close_mat
    dec = torch.arange(n_samples, device=device) + seq_len - 1  # decision close index
    y24 = close_np[dec + LABEL_H] / close_np[dec].clamp(min=1e-8) - 1.0
    r1h = close_np[dec + 1] / close_np[dec].clamp(min=1e-8) - 1.0

    X_list = []
    for i in range(n_samples):
        X_list.append(torch.stack([all_factors[s][i:i + seq_len] for s in syms], dim=0))
    X = torch.stack(X_list)
    print(f"  X: {tuple(X.shape)}, y24: {tuple(y24.shape)}, label={LABEL_H}h forward return")
    return X, y24, r1h, close_mat, syms, n_factors


# ============================================================================
# Loss + fold training (v12 turnover penalty kept, M-13 val purge added)
# ============================================================================

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


def train_fold_indexed(
    model, loss_fn,
    X_all: Tensor, y_all: Tensor,
    train_idx: np.ndarray, val_idx: np.ndarray,
    epochs: int = 60, bs: int = 512, lr: float = 3e-4,
    lambda_turnover: float = LAMBDA_TURNOVER,
):
    device = X_all.device
    train_set = set(int(x) for x in train_idx)
    pair_t_np = np.array(
        [int(t) for t in train_idx if (int(t) + 1) in train_set], dtype=np.int64,
    )
    use_pairs = len(pair_t_np) > 0
    pair_t_t = torch.from_numpy(pair_t_np if use_pairs else train_idx).to(device)
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
                combined = torch.cat([t_idx, t1_idx], dim=0)
                combined_scores = model(X_all[combined])
                scores_t, scores_t1 = combined_scores.chunk(2, dim=0)
                cls_loss = loss_fn(scores_t, y_all[t_idx])
                turnover_loss = (scores_t1 - scores_t).pow(2).mean()
                loss = cls_loss + lambda_turnover * turnover_loss
            else:
                scores = model(X_all[t_idx])
                loss = loss_fn(scores, y_all[t_idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        sched.step()

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
            cov = ((pr - pm) * (tr - tm)).sum(-1)
            corr = (cov / ((pr - pm).pow(2).sum(-1).sqrt()
                           * (tr - tm).pow(2).sum(-1).sqrt()).clamp(1e-8)).mean().item()

        if corr > best_corr:
            best_corr = corr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_corr


def split_train_val(train_idx: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Tail 20% as val, with a purge gap (M-13) so train labels don't
    overlap val features. / 末尾20%做val，train与val间留purge空隙。"""
    n_tr = len(train_idx)
    val_size = max(int(n_tr * 0.2), 20)
    val_idx = train_idx[-val_size:]
    gap = seq_len + LABEL_H
    head = train_idx[:-val_size]
    actual = head[head < (int(val_idx[0]) - gap)]
    if len(actual) == 0:
        actual = head
    return actual, val_idx


# ============================================================================
# Backtest engine: banded top-K long-short with next-bar effect (H-1 fix)
# ============================================================================

# Banding now lives in the shared sleeves package (used by backtest, paper
# trading, and research alike) — single implementation, tested by
# tests/run_invariants.py. Name kept for research-script imports.
# banding 移入共享 sleeves 包（回测/模拟盘/研究共用一份实现）。
from sleeves.banding import banded_targets as _banded_targets  # noqa: E402


def run_backtest(
    name: str,
    pred_matrix: Tensor, valid_mask: Tensor,
    r1h: Tensor, close_mat: Tensor,
    seq_len: int,
    k: int, enter_band: int, exit_band: int,
    decision_every: int, min_hold: int,
    use_vol_filter: bool,
) -> Dict:
    device = pred_matrix.device
    n_samples, A = pred_matrix.shape
    scores_np = pred_matrix.cpu().numpy()
    valid_np = valid_mask.cpu().numpy()
    r1h_np = r1h.cpu().numpy()
    close_np = close_mat.cpu().numpy()
    T_full = close_np.shape[0]

    # expanding-quantile vol filter (M-1: no full-sample lookahead)
    # expanding 分位数 vol filter（M-1：不使用全样本未来信息）
    vol_thresh = np.full(n_samples, np.inf)
    if use_vol_filter:
        vol_window = 24
        basket_ret = r1h_np.mean(axis=1)
        rolling_vol = np.zeros(n_samples)
        for i in range(vol_window, n_samples):
            rolling_vol[i] = basket_ret[i - vol_window:i].std()
        warmup = 24 * 30  # need a month of history before filtering / 一个月预热
        for i in range(warmup, n_samples, 24):
            hist = rolling_vol[vol_window:i]
            q = np.quantile(hist[hist > 0], 1.0 - VOL_QUANTILE) if (hist > 0).any() else np.inf
            vol_thresh[i:i + 24] = q

    twap = TWAPExecutor(n_slices=4, favorable_reject_rate=0.60)
    equity = 1_000_000.0
    eq_curve = [equity]
    w = np.zeros(A)
    long_set: Set[int] = set()
    short_set: Set[int] = set()
    total_cost = 0.0
    rebalances = 0
    slot_trades = 0
    hc = 10 ** 9  # bars since last change / 距上次换仓bar数

    def slot_cost(asset: int, t: int, notional: float, side: str) -> float:
        di = t + seq_len - 1                      # decision close index
        ep = close_np[di, asset]
        fu = [close_np[min(di + 1 + j, T_full - 1), asset] for j in range(4)]
        if ep <= 0:
            return 0.0
        _, cbps, _ = twap.execute_twap(side, notional, float(ep), fu)
        return notional * cbps / 10000.0

    for t in range(n_samples):
        # H-1 fix: PnL first, with weights set at most at t-1; a position
        # changed at t starts earning from bar t+1 (TWAP executes in between).
        # H-1 修复：先用旧权重计算本bar收益；t 时刻的换仓从 t+1 起才生效。
        pnl = float((w * r1h_np[t]).sum())
        cost_bar = 0.0

        if valid_np[t] and (t % decision_every == 0):
            can_trade = (hc >= min_hold) or (len(long_set) == 0)
            vol_ok = not (use_vol_filter and len(long_set) > 0
                          and vol_thresh[t] < np.inf
                          and t >= 24 and _rolling_vol_at(r1h_np, t) > vol_thresh[t])
            if can_trade and vol_ok:
                new_l, new_s = _banded_targets(
                    scores_np[t], long_set, short_set, k, enter_band, exit_band)
                if new_l != long_set or new_s != short_set:
                    slot_notional = equity * 0.5 / k
                    for a in (long_set - new_l):
                        cost_bar += slot_cost(a, t, slot_notional, "SELL")
                    for a in (new_l - long_set):
                        cost_bar += slot_cost(a, t, slot_notional, "BUY")
                    for a in (short_set - new_s):
                        cost_bar += slot_cost(a, t, slot_notional, "BUY")
                    for a in (new_s - short_set):
                        cost_bar += slot_cost(a, t, slot_notional, "SELL")
                    slot_trades += (len(long_set - new_l) + len(new_l - long_set)
                                    + len(short_set - new_s) + len(new_s - short_set))
                    long_set, short_set = new_l, new_s
                    w = np.zeros(A)
                    for a in long_set:
                        w[a] = 0.5 / k
                    for a in short_set:
                        w[a] = -0.5 / k
                    rebalances += 1
                    hc = 0
        total_cost += cost_bar

        ret = pnl - cost_bar / max(equity, 1.0)
        ret = max(min(ret, 0.10), -0.10)
        equity *= (1.0 + ret)
        eq_curve.append(equity)
        hc += 1

    rets = [(eq_curve[i] / eq_curve[i - 1]) - 1 for i in range(1, len(eq_curve))]
    avg = sum(rets) / len(rets)
    std = (sum((r - avg) ** 2 for r in rets) / len(rets)) ** 0.5
    sharpe = (avg / max(std, 1e-9)) * (24 * 365) ** 0.5
    peak, max_dd = eq_curve[0], 0.0
    for e in eq_curve:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak)
    years = len(rets) / (24 * 365)
    psr = probabilistic_sharpe(rets, 0.0)

    return {
        "name": name,
        "total_return": eq_curve[-1] / eq_curve[0] - 1,
        "sharpe": sharpe, "max_drawdown": max_dd,
        "total_cost": total_cost, "rebalances": rebalances,
        "slot_trades": slot_trades,
        "slot_trades_per_year": slot_trades / max(years, 1e-9),
        "psr": psr, "rets": rets,
        "final_equity": eq_curve[-1],
        **twap.stats(),
    }


def _rolling_vol_at(r1h_np: np.ndarray, t: int, window: int = 24) -> float:
    s = max(0, t - window)
    return float(r1h_np[s:t].mean(axis=1).std()) if t > s else 0.0


# ============================================================================
# CPCV + production model
# ============================================================================

def run_cpcv(X, y24, seq_len, n_factors, device):
    n_samples = X.size(0)
    n_assets = X.size(1)
    purge = seq_len + LABEL_H  # train labels span LABEL_H bars past features / 标签跨度需纳入purge

    splits = generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                                  purge_bars=purge, embargo_bars=48)
    print(f"\n[CPCV] {len(splits)} splits, {n_factors} factors, {LABEL_H}h label, "
          f"purge={purge}, lambda_turnover={LAMBDA_TURNOVER}")

    pred_matrix = torch.zeros(n_samples, n_assets, device=device)
    pred_count = torch.zeros(n_samples, device=device)
    fold_corrs = []
    t0_total = time.time()

    for fi, (train_idx, test_idx) in enumerate(splits):
        actual_train_idx, val_idx = split_train_val(train_idx, seq_len)

        model = CrossAssetGRUAttention(
            n_factors=n_factors, d_model=128, gru_layers=2,
            n_cross_heads=4, n_cross_layers=3, d_ff=256,
            dropout=0.30, seq_len=seq_len, max_assets=n_assets,
        ).to(device)
        loss_fn = DualLoss().to(device)

        t0 = time.time()
        model, corr = train_fold_indexed(model, loss_fn, X, y24,
                                         actual_train_idx, val_idx)
        fold_corrs.append(corr)

        model.eval()
        test_idx_t = torch.from_numpy(test_idx).to(device)
        with torch.no_grad():
            for ts in range(0, len(test_idx), 512):
                te = min(ts + 512, len(test_idx))
                chunk_idx = test_idx_t[ts:te]
                pred_matrix[chunk_idx] += model(X[chunk_idx])
                pred_count[chunk_idx] += 1

        print(f"  Split {fi+1}/{len(splits)}: val_corr={corr:.4f} [{time.time()-t0:.0f}s]")

        os.makedirs("checkpoints/folds_v13", exist_ok=True)
        torch.save({"state": model.state_dict(), "corr": corr},
                   f"checkpoints/folds_v13/fold_{fi:02d}.pt")
        del model, loss_fn
        torch.cuda.empty_cache()

    valid_mask = pred_count > 0
    pred_matrix[valid_mask] /= pred_count[valid_mask].unsqueeze(-1)
    print(f"\n[CPCV] Done in {time.time()-t0_total:.0f}s, "
          f"avg val corr={sum(fold_corrs)/len(fold_corrs):.4f}")

    return pred_matrix, valid_mask, fold_corrs


def oos_rank_ic(pred_matrix: Tensor, valid_mask: Tensor, target: Tensor) -> float:
    """Mean cross-sectional rank IC of the OOS ensemble vs a target return.
    OOS 集成预测对目标收益的平均横截面 rank IC。"""
    with torch.no_grad():
        p = pred_matrix[valid_mask]
        t = target[valid_mask]
        pr = p.argsort(-1, descending=True).argsort(-1).float()
        tr = t.argsort(-1, descending=True).argsort(-1).float()
        pm, tm = pr.mean(-1, True), tr.mean(-1, True)
        cov = ((pr - pm) * (tr - tm)).sum(-1)
        corr = cov / ((pr - pm).pow(2).sum(-1).sqrt()
                      * (tr - tm).pow(2).sum(-1).sqrt()).clamp(1e-8)
        return corr.mean().item()


def train_production(X, y24, seq_len, n_factors, device, factor_names):
    n_samples = X.size(0)
    n_assets = X.size(1)
    print(f"\n[Production] Training final model on all {n_samples} samples ...")
    model = CrossAssetGRUAttention(
        n_factors=n_factors, d_model=128, gru_layers=2,
        n_cross_heads=4, n_cross_layers=3, d_ff=256,
        dropout=0.30, seq_len=seq_len, max_assets=n_assets,
    ).to(device)
    loss_fn = DualLoss().to(device)
    all_idx = np.arange(n_samples)
    actual_train, val_idx = split_train_val(all_idx, seq_len)
    t0 = time.time()
    model, corr = train_fold_indexed(model, loss_fn, X, y24, actual_train, val_idx)
    print(f"  Production model: val_corr={corr:.4f} [{time.time()-t0:.0f}s]")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/v13_production.pt"
    torch.save({
        "model_state": model.state_dict(),
        "n_factors": n_factors, "d_model": 128, "gru_layers": 2,
        "n_cross_heads": 4, "n_cross_layers": 3, "d_ff": 256,
        "dropout": 0.30, "seq_len": seq_len, "max_assets": n_assets,
        "val_corr": corr,
        "factor_names": factor_names,
        "label_horizon": LABEL_H,
        "decision_every": DECISION_EVERY,
        "basket_k": BASKET_K, "enter_band": ENTER_BAND, "exit_band": EXIT_BAND,
        "lambda_turnover": LAMBDA_TURNOVER,
        "needs_real_funding": True,
    }, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")
    return corr


# ============================================================================
# Main
# ============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  v13 — Fixed 1h data + 24h label + banded top-K portfolio")
    print(f"        K={BASKET_K}, enter<{ENTER_BAND}, exit>={EXIT_BAND}, "
          f"decision every {DECISION_EVERY}h")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    SEQ_LEN = 24
    MAX_ASSETS = 20
    X, y24, r1h, close_mat, syms, n_factors = build_from_parquet(
        SEQ_LEN, MAX_ASSETS, device)
    factor_names = [n for n in FactorRegistry.list_factors() if n not in DROP_FACTORS]

    pred_matrix, valid_mask, fold_corrs = run_cpcv(X, y24, SEQ_LEN, n_factors, device)

    prod_corr = train_production(X, y24, SEQ_LEN, n_factors, device, factor_names)

    # ---- Backtest configs / 回测对照组 ----
    configs = [
        # A: v12-style strategy on FIXED data (top1/bottom1, hourly, min_hold 96h, vol filter)
        dict(name="A_v12_style_top1_hourly", k=1, enter_band=1, exit_band=1,
             decision_every=1, min_hold=96, use_vol_filter=True),
        # B: current paper-trading strategy (top1/bottom1, daily, no band)
        dict(name="B_paper_top1_daily", k=1, enter_band=1, exit_band=1,
             decision_every=24, min_hold=0, use_vol_filter=False),
        # C: v13 proposal (top3/bottom3, daily, enter<3 / exit>=6 banding)
        dict(name="C_v13_band_top3_daily", k=BASKET_K, enter_band=ENTER_BAND,
             exit_band=EXIT_BAND, decision_every=DECISION_EVERY, min_hold=0,
             use_vol_filter=False),
    ]

    results = []
    for cfg in configs:
        random.seed(SEED)  # same execution randomness for every config / 各配置同一执行随机性
        res = run_backtest(
            cfg["name"], pred_matrix, valid_mask, r1h, close_mat, SEQ_LEN,
            cfg["k"], cfg["enter_band"], cfg["exit_band"],
            cfg["decision_every"], cfg["min_hold"], cfg["use_vol_filter"])
        results.append(res)

    # DSR across configs / 跨配置 DSR
    sr_trials = []
    for r in results:
        m = sum(r["rets"]) / len(r["rets"])
        s = (sum((x - m) ** 2 for x in r["rets"]) / len(r["rets"])) ** 0.5
        sr_trials.append(m / max(s, 1e-18))
    dsr_info = deflated_sharpe(results[-1]["rets"], sr_trials, N_TRIALS_DSR)

    print("\n" + "=" * 70)
    print("  v13 BACKTEST REPORT (all on FIXED true-1h data)")
    print("=" * 70)
    print(f"  {'Samples':.<42s} {X.size(0):,}")
    print(f"  {'Label':.<42s} {LABEL_H}h forward return")
    print(f"  {'CPCV avg val rank-corr (24h)':.<42s} {sum(fold_corrs)/len(fold_corrs):.4f}")
    print(f"  {'Fold corr range':.<42s} "
          f"[{min(fold_corrs):.4f}, {max(fold_corrs):.4f}]")
    ic24 = oos_rank_ic(pred_matrix, valid_mask, y24)
    ic1 = oos_rank_ic(pred_matrix, valid_mask, r1h)
    print(f"  {'OOS ensemble rank IC vs 24h label':.<42s} {ic24:.4f}")
    print(f"  {'OOS ensemble rank IC vs 1h return':.<42s} {ic1:.4f}")
    print(f"  {'Production val corr':.<42s} {prod_corr:.4f}")
    for r in results:
        print(f"\n  --- {r['name']} ---")
        print(f"  {'Total Return':.<42s} {r['total_return']:>10.2%}")
        print(f"  {'Sharpe (annualized)':.<42s} {r['sharpe']:>10.3f}")
        print(f"  {'PSR (vs SR*=0)':.<42s} {r['psr']:>10.3f}")
        print(f"  {'Max Drawdown':.<42s} {r['max_drawdown']:>10.2%}")
        print(f"  {'Total Cost':.<42s} {r['total_cost']:>12,.0f}")
        print(f"  {'Rebalance events':.<42s} {r['rebalances']}")
        print(f"  {'Slot trades (total / per year)':.<42s} "
              f"{r['slot_trades']} / {r['slot_trades_per_year']:.0f}")
        print(f"  {'Adverse fill %':.<42s} {r['adverse_fill_pct']:.1%}")
    print(f"\n  --- DSR (config C, N_trials={N_TRIALS_DSR}) ---")
    print(f"  {'SR* (expected max under null)':.<42s} {dsr_info['sr_star_expected_max']:.5f}")
    print(f"  {'PSR vs 0':.<42s} {dsr_info['psr_vs_zero']:.3f}")
    print(f"  {'DSR':.<42s} {dsr_info['dsr']:.3f}")
    print("  (DSR < 0.95 => not distinguishable from best-of-N luck)")
    print("=" * 70)

    # persist summary for later analysis / 保存汇总供后续分析
    out = {
        "fold_corrs": fold_corrs,
        "prod_corr": prod_corr,
        "configs": [{k2: v for k2, v in r.items() if k2 != "rets"} for r in results],
        "dsr": dsr_info,
        "drop_factors": sorted(DROP_FACTORS),
        "label_h": LABEL_H,
    }
    with open("checkpoints/v13_backtest_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("  Summary saved to checkpoints/v13_backtest_summary.json")


if __name__ == "__main__":
    main()
