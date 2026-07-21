"""
O2 production model — v14 model sleeve, selected by the pre-registered
ablation rule (RESEARCH_2026-07-13_extended_window.md).
O2 生产模型训练——v14 模型 sleeve（预注册规则选定）。

WHY O2 AND NOT M010: the objective ablation (research_objectives.py) left
two families of gate-passers. M010 (O2-dominant mix) posted the highest
backtest Sharpe (0.75, PSR 0.96) but its alpha is FRONT-LOADED in 2021-22
(+56/+19bp/d) and NEGATIVE in 2026 (-13bp/d). The pre-registered selection
rule — most positive years first, Sharpe only as tie-break — picks the O2
endpoint: 5/6 positive years INCLUDING 2026 (+10.8bp/d), the only variant
positive in the deployment year. For a 2026 deployment, "small wins
everywhere" beats "big wins five years ago"; chasing M010's Sharpe after
seeing the results would be exactly the selection bias the rule was frozen
to prevent. M010 is archived as backup.

WHAT: trains PairMagWeighted (pairwise |dz|-weighted ranking) on the FULL
2021-2026 window (16 full-history symbols, 18 factors, 24h label,
z = y24/vol24), all samples with a purged tail-val early stop — the
production analogue of the CPCV folds that passed the gate. Went live
2026-07-14 as the THIRD paper track (o2_state/o2_pnl, ContinuousBook:
continuous weights + GP partial trading), running to the September gate.

Checkpoint carries EVERYTHING the paper side needs: architecture, factor
list, the 16-symbol universe (the asset embedding is POSITIONAL — serving
a different universe would silently misassign identities, the H-3 failure
mode; universe change = retrain), and the continuous-GP construction
params (tau, daily decisions, gross 1.0).

为什么选 O2 而非 M010：消融中 M010 回测 Sharpe 最高（0.75），但收益高度
前置于 2021-22、在 2026 年为负（-13bp/天）；预注册选择规则（正年份数优先、
Sharpe 仅破平）选中 O2 端点——5/6 年为正且唯一在部署年 2026 为正
（+10.8bp/天）。对 2026 年部署而言"到处小赚"胜过"五年前大赚"；见结果后
改追 M010 正是规则要防的选择偏差。ckpt 自带 16 币宇宙与构建参数：资产
嵌入按位置编码，换宇宙必须重训。2026-07-14 起作为第三条 paper track
（o2_state/o2_pnl，连续权重 + GP 渐进建仓）上线，跑到 9 月 gate。

Run (HPC): python tools/train_o2_production.py
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from factors.base import FactorRegistry
from model.cross_asset_attention import CrossAssetGRUAttention
from run_v13_final import train_fold_indexed, split_train_val, LABEL_H, SEED
from tools.research_extended_window import (
    build_window, WINDOWS, EXT_SYMBOLS, SEQ_LEN, DROP_FACTORS_EXT,
)
from tools.research_objectives import PairMagWeighted, trailing_vol24

import factors  # noqa: F401

# Construction params ship INSIDE the checkpoint so the paper side replays
# exactly the construction the gate was passed with (tau=0.2 = move 1/5 of
# the way to the aim portfolio per daily decision, Garleanu-Pedersen style)
# — backtest and live remain the SAME strategy (the v13 lesson).
# 构建参数随 ckpt 落盘：模拟盘按通过 gate 的同一构建执行（tau=0.2 即每日向
# 目标组合移动 1/5，GP 式渐进建仓），确保回测与 live 是同一个策略。
CONSTRUCTION = {"type": "continuous_gp", "tau": 0.2,
                "decision_every": 24, "gross": 1.0}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  O2 PRODUCTION TRAIN (v14 model sleeve)")
    print("=" * 70, flush=True)

    X, y24_t, r1h_t, close_t, bar_ms, n_factors = build_window(WINDOWS["FULL"], device)
    r1h = r1h_t.cpu().numpy()
    z = np.clip(y24_t.cpu().numpy() / trailing_vol24(r1h), -5.0, 5.0)
    z_t = torch.from_numpy(z.astype(np.float32)).to(device)
    n, A = z.shape

    model = CrossAssetGRUAttention(
        n_factors=n_factors, d_model=128, gru_layers=2, n_cross_heads=4,
        n_cross_layers=3, d_ff=256, dropout=0.30, seq_len=SEQ_LEN,
        max_assets=A).to(device)
    loss_fn = PairMagWeighted().to(device)
    tr, va = split_train_val(np.arange(n), SEQ_LEN)
    t0 = time.time()
    model, corr = train_fold_indexed(model, loss_fn, X, z_t, tr, va)
    print(f"  production val_corr(z)={corr:.4f} [{time.time()-t0:.0f}s]")

    factor_names = [f for f in FactorRegistry.list_factors()
                    if f not in DROP_FACTORS_EXT]
    assert len(factor_names) == n_factors
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_factors": n_factors, "d_model": 128, "gru_layers": 2,
        "n_cross_heads": 4, "n_cross_layers": 3, "d_ff": 256,
        "dropout": 0.30, "seq_len": SEQ_LEN, "max_assets": A,
        "val_corr": corr,
        "factor_names": factor_names,
        "symbols": EXT_SYMBOLS,          # positional embedding universe / 位置嵌入宇宙
        "label_horizon": LABEL_H,
        "objective": "O2_pair_magwt",
        "construction": CONSTRUCTION,
        "needs_real_funding": False,     # 18-factor set excludes funding / 无funding因子
        "train_window": "2021-01..2026-03",
        "n_train_samples": n,
    }, "checkpoints/o2_production.pt")
    print("  saved: checkpoints/o2_production.pt")


if __name__ == "__main__":
    main()
