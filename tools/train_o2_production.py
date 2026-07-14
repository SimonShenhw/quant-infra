"""
O2 production model — v14 model sleeve, selected by the pre-registered
ablation rule (RESEARCH_2026-07-13_extended_window.md).
O2 生产模型训练——v14 模型 sleeve（预注册规则选定）。

Trains PairMagWeighted (pairwise |dz|-weighted ranking) on the FULL
2021-2026 window (16 full-history symbols, 18 factors, 24h label,
z = y24/vol24), all samples with a purged tail-val early stop — the
production analogue of the CPCV folds that passed the gate.

Checkpoint carries EVERYTHING the paper side needs: architecture, factor
list, the 16-symbol universe (positional asset embedding!), and the
continuous-GP construction params (tau=1/5, daily decisions, gross 1.0).

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
