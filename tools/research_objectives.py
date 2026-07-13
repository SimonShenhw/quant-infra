"""
Magnitude-aware training objectives — the model sleeve's LAST SHOT.
magnitude-aware 训练目标——模型 sleeve 的最后一搏。(v14 option b)

Diagnosis being tested (RESEARCH_2026-07-13_extended_window.md): ListMLE
delivers rank order; the model wins in rank space (IC + every year) but
loses in dollar space (w.y24 < 0) because large-magnitude names end up on
the wrong side. If magnitude-aware objectives cannot flip dollar-space
alpha positive, the model sleeve retires and v14 goes carry-backbone.

Objectives (same FULL 2021-2026 window, 16 symbols, 18 factors, same CPCV,
same turnover-penalty mechanics as v13):
  O1 volnorm_mse   MSE on z = y24 / vol24 (vol-normalized 24h return)
  O2 pair_magwt    pairwise logistic ranking weighted by |z_i - z_j|
                   (big spreads dominate the loss)
  O3 softport      differentiable portfolio: maximize w(s).z with
                   w = demeaned scores / gross

PRE-REGISTERED PASS GATE (per objective, decided before running):
  1. OOS w.y24 (score-proportional demeaned weights, label-aligned gross,
     no costs) > 0 overall, AND
  2. positive in >= 4 of 6 calendar years, AND
  3. flat-8bps GP-smoothed (tau=1/5) backtest Sharpe > 0.
No objective passes => model sleeve retires. Trials are registered.

Run (HPC): python tools/research_objectives.py --obj O1|O2|O3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from engine.cpcv import generate_cpcv_splits
from model.cross_asset_attention import CrossAssetGRUAttention
from run_v13_final import train_fold_indexed, split_train_val, LABEL_H, SEED
from tools.research_extended_window import build_window, WINDOWS, SEQ_LEN
from tools.validation_stats import probabilistic_sharpe, register_trial

import factors  # noqa: F401

COST_BPS = 8.0
DECISION_EVERY = 24
TAU = 1 / 5
VOL_WINDOW = 720  # 30d trailing hourly vol / 30天滚动小时波动


# ---------------------------------------------------------------------------
# Objectives (loss_fn(scores, targets) interface = train_fold_indexed compat)
# ---------------------------------------------------------------------------

class VolnormMSE(nn.Module):
    """O1: regression on vol-normalized 24h returns. / vol归一化收益回归。"""
    def forward(self, scores: Tensor, z: Tensor) -> Tensor:
        return nn.functional.mse_loss(scores, z)


class PairMagWeighted(nn.Module):
    """O2: pairwise logistic ranking, pair weight = |z_i - z_j|.
    配对排序损失，权重=收益差幅度（大行情对主导损失）。"""
    def forward(self, scores: Tensor, z: Tensor) -> Tensor:
        ds = scores.unsqueeze(-1) - scores.unsqueeze(-2)   # (B, A, A)
        dz = z.unsqueeze(-1) - z.unsqueeze(-2)
        w = dz.abs()
        loss = w * nn.functional.softplus(-torch.sign(dz) * ds)
        return loss.sum() / w.sum().clamp(min=1e-8)


class SoftPortfolio(nn.Module):
    """O3: maximize w(s)·z directly, w = demeaned scores / gross.
    可微组合：直接最大化组合的vol归一化收益。"""
    def forward(self, scores: Tensor, z: Tensor) -> Tensor:
        w_raw = scores - scores.mean(-1, keepdim=True)
        w = w_raw / w_raw.abs().sum(-1, keepdim=True).clamp(min=1e-8)
        return -(w * z).sum(-1).mean()


OBJECTIVES = {"O1": VolnormMSE, "O2": PairMagWeighted, "O3": SoftPortfolio}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trailing_vol24(r1h: np.ndarray) -> np.ndarray:
    """(n, A) causal 24h vol from trailing hourly vol. / 因果24h波动率。"""
    n, A = r1h.shape
    out = np.full((n, A), np.nan)
    csum = np.cumsum(r1h, axis=0)
    csum2 = np.cumsum(r1h ** 2, axis=0)
    for t in range(VOL_WINDOW, n):
        m = (csum[t - 1] - csum[t - VOL_WINDOW - 1 if t > VOL_WINDOW else 0]) / VOL_WINDOW
        v = (csum2[t - 1] - csum2[t - VOL_WINDOW - 1 if t > VOL_WINDOW else 0]) / VOL_WINDOW - m ** 2
        out[t] = np.sqrt(np.maximum(v, 1e-12)) * np.sqrt(24)
    med = np.nanmedian(out[VOL_WINDOW:])
    out[:VOL_WINDOW] = med  # warmup rows get a neutral scale / 预热期用中位数
    return np.maximum(out, med * 0.25)


def score_weights(s_row: np.ndarray) -> np.ndarray:
    w = s_row - s_row.mean()
    g = np.abs(w).sum()
    return w / g if g > 1e-12 else np.zeros_like(w)


def spearman_rows(a: np.ndarray, b: np.ndarray) -> float:
    ra = (-a).argsort(1).argsort(1).astype(float)
    rb = (-b).argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return float((num / np.maximum(den, 1e-12)).mean())


def flat_gp_backtest(scores: np.ndarray, r1h: np.ndarray) -> Dict:
    n, A = scores.shape
    w = np.zeros(A)
    eq = 1.0
    rets = []
    for t in range(n):
        pnl = float(w @ r1h[t])
        cost = 0.0
        if t % DECISION_EVERY == 0:
            target = score_weights(scores[t])
            new_w = (1 - TAU) * w + TAU * target
            cost = float(np.abs(new_w - w).sum()) * COST_BPS / 10000.0
            w = new_w
        r = max(min(pnl - cost, 0.10), -0.10)
        eq *= 1 + r
        rets.append(r)
    rets = np.asarray(rets)
    return {"total_return": float(eq - 1),
            "sharpe": float(rets.mean() / max(rets.std(), 1e-12) * np.sqrt(24 * 365)),
            "psr": probabilistic_sharpe(list(rets), 0.0)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True, choices=sorted(OBJECTIVES))
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  MAGNITUDE-AWARE OBJECTIVE {args.obj} "
          f"({OBJECTIVES[args.obj].__name__}) — model sleeve last shot")
    print("=" * 70, flush=True)

    X, y24_t, r1h_t, close_t, bar_ms, n_factors = build_window(WINDOWS["FULL"], device)
    y24 = y24_t.cpu().numpy()
    r1h = r1h_t.cpu().numpy()
    vol24 = trailing_vol24(r1h)
    z = np.clip(y24 / vol24, -5.0, 5.0)
    z_t = torch.from_numpy(z.astype(np.float32)).to(device)
    n, A = z.shape

    splits = generate_cpcv_splits(n, n_groups=6, n_test_groups=2,
                                  purge_bars=SEQ_LEN + LABEL_H, embargo_bars=48)
    pred = torch.zeros(n, A, device=device)
    cnt = torch.zeros(n, device=device)
    fold_dir = f"checkpoints/folds_obj_{args.obj}"
    os.makedirs(fold_dir, exist_ok=True)

    for fi, (train_idx, test_idx) in enumerate(splits):
        tr, va = split_train_val(train_idx, SEQ_LEN)
        model = CrossAssetGRUAttention(
            n_factors=n_factors, d_model=128, gru_layers=2, n_cross_heads=4,
            n_cross_layers=3, d_ff=256, dropout=0.30, seq_len=SEQ_LEN,
            max_assets=A).to(device)
        loss_fn = OBJECTIVES[args.obj]().to(device)
        t0 = time.time()
        # val early-stopping still uses rank corr vs z (scale-free) / 早停用z的rank corr
        model, corr = train_fold_indexed(model, loss_fn, X, z_t, tr, va)
        model.eval()
        t_idx = torch.from_numpy(test_idx).to(device)
        with torch.no_grad():
            for s in range(0, len(test_idx), 512):
                e = min(s + 512, len(test_idx))
                pred[t_idx[s:e]] += model(X[t_idx[s:e]])
                cnt[t_idx[s:e]] += 1
        torch.save({"state": model.state_dict(), "corr": corr, "n_samples": n},
                   f"{fold_dir}/fold_{fi:02d}.pt")
        print(f"  fold {fi+1}/{len(splits)}: val_corr={corr:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del model, loss_fn
        torch.cuda.empty_cache()

    valid = (cnt > 0)
    pred[valid] /= cnt[valid].unsqueeze(-1)
    scores = pred.cpu().numpy()

    # ---- pre-registered evaluation / 预注册评估 ----
    ic = spearman_rows(scores, y24)
    daily_idx = np.arange(0, n, DECISION_EVERY)
    wy = np.array([score_weights(scores[t]) @ y24[t] for t in daily_idx])
    yrs = ((bar_ms[daily_idx] // 1000).astype("datetime64[s]")
           .astype("datetime64[Y]").astype(int) + 1970)
    per_year = {str(y): float(wy[yrs == y].mean()) for y in sorted(set(yrs))}
    n_pos_years = sum(v > 0 for v in per_year.values())
    bt = flat_gp_backtest(scores, r1h)

    gate = (wy.mean() > 0) and (n_pos_years >= 4) and (bt["sharpe"] > 0)
    print(f"\n  OOS Spearman IC vs y24 .......... {ic:+.4f}")
    print(f"  w·y24 gross (PRIMARY) ........... {wy.mean():+.4%}/day")
    for y, v in per_year.items():
        print(f"    {y}: {v:+.4%}")
    print(f"  positive years .................. {n_pos_years}/6")
    print(f"  flat-8bps GP(1/5) backtest ...... {bt['total_return']:+.2%} "
          f"(Sharpe {bt['sharpe']:.3f}, PSR {bt['psr']:.2f})")
    print(f"\n  PRE-REGISTERED GATE: {'PASS' if gate else 'FAIL'}")

    out = {"objective": args.obj, "ic": ic,
           "w_y24_all": float(wy.mean()), "w_y24_per_year": per_year,
           "n_pos_years": n_pos_years, "backtest": bt, "gate_pass": bool(gate)}
    os.makedirs("checkpoints", exist_ok=True)
    with open(f"checkpoints/research_objective_{args.obj}.json", "w") as f:
        json.dump(out, f, indent=2)
    register_trial(f"objectives_2026-07-13/{args.obj}",
                   {"kind": "training_objective", "gate_pass": bool(gate)})
    print(f"  saved: checkpoints/research_objective_{args.obj}.json")


if __name__ == "__main__":
    main()
