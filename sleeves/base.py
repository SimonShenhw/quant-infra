"""
Sleeve interface: a signal source with banding parameters.
sleeve 接口：带 banding 参数的信号源。(ROADMAP Phase 1)

A Sleeve turns market context (per-symbol features / funding / mark indices)
into a cross-sectional score dict; the PortfolioBook turns scores into a
banded basket and keeps the ledger. ModelSleeve (v13 GRU checkpoint) and
CarrySleeve (trailing-mean funding) are the two live instances.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import Tensor


class Sleeve(ABC):
    name: str = ""
    k: int = 3
    enter_band: int = 3
    exit_band: int = 6

    @abstractmethod
    def compute_scores(
        self,
        feats: Dict[str, dict],
        funding: Dict[str, Tensor],
        idx_map: Dict[str, int],
    ) -> Optional[Dict[str, float]]:
        """Cross-sectional scores at each symbol's mark index, or None if the
        sleeve has no usable signal. / 各标的在mark索引处的截面打分；无信号返回None。"""
        ...


class ModelSleeve(Sleeve):
    """Scores from a trained cross-asset model checkpoint. / 模型信号。"""

    name = "model"

    def __init__(self, ckpt: dict, model: Any) -> None:
        self.ckpt = ckpt
        self.model = model
        self.seq_len = int(ckpt["seq_len"])
        self.k = int(ckpt.get("basket_k", 3))
        self.enter_band = int(ckpt.get("enter_band", 3))
        self.exit_band = int(ckpt.get("exit_band", 6))

    def compute_scores(self, feats, funding, idx_map):
        syms = sorted(feats.keys())
        xs = []
        for sym in syms:
            i = idx_map[sym]
            f = feats[sym]["factors"][: i + 1]
            if f.size(0) < self.seq_len:
                raise ValueError(
                    f"{sym}: only {f.size(0)} bars before mark, need {self.seq_len}")
            xs.append(f[-self.seq_len:])
        x = torch.stack(xs, dim=0).unsqueeze(0)
        with torch.no_grad():
            scores = self.model(x).squeeze(0)
        return {syms[i]: scores[i].item() for i in range(len(syms))}


class CarrySleeve(Sleeve):
    """Negative trailing-mean funding: long low/negative-funding perps, short
    high-funding ones (shorts collect). Model-free. / 反向均值funding，无模型。"""

    name = "carry"

    def __init__(self, lookback: int = 72,
                 k: int = 3, enter_band: int = 3, exit_band: int = 6) -> None:
        self.lookback = lookback
        self.k = k
        self.enter_band = enter_band
        self.exit_band = exit_band

    def compute_scores(self, feats, funding, idx_map):
        sig: Dict[str, float] = {}
        for sym, i in idx_map.items():
            f = funding.get(sym)
            if f is None or f.numel() != len(feats[sym]["ts"]):
                return None
            lo = max(i - self.lookback + 1, 0)
            sig[sym] = -float(f[lo:i + 1].mean().item())
        if all(v == 0.0 for v in sig.values()):
            return None  # funding degraded to zeros / funding全零无信号
        return sig
