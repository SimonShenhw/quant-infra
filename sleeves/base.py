"""
Sleeve interface: a signal source with banding parameters.
sleeve 接口：带 banding 参数的信号源。(ROADMAP Phase 1)

WHAT — a Sleeve turns market context (per-symbol features / funding / mark
indices) into a cross-sectional score dict; the PortfolioBook (book.py)
turns scores into a banded basket and keeps the ledger. ModelSleeve (v13
GRU checkpoint) and CarrySleeve (trailing-mean funding) are the two live
instances; the v14 O2 track reuses ModelSleeve with a ContinuousBook.

WHY an interface at all: the September v14 gate composes MULTIPLE sleeves
(model + carry per the ROADMAP_2026-07-13 decision table), so composition
must be "iterate sleeves over one scoring/ledger contract", not a fork of
run_paper_daily.py per strategy. Deliberately minimal (Phase 1): scoring and
banding params only — bookkeeping stays in the books.

CONTRACT / 契约:
  - compute_scores is a PURE, causal function of its inputs at the given
    mark indices — deterministic scoring is what makes missed-day backfill
    and same-day rerun idempotence legitimate.
  - Returning None means "no usable signal today": the caller SKIPS the day
    (visible, recoverable) instead of booking zeros as if they were signal.
中文：Sleeve 只负责打分（自带 banding 参数），账本只负责记账；v14 多 sleeve
组合复用同一契约。打分必须是纯因果函数（补课/重跑幂等的前提）；无信号
返回 None、整日跳过，不把零当信号入账。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import Tensor


class Sleeve(ABC):
    """A signal source carrying its own banding parameters (k/enter/exit
    travel WITH the sleeve, so a basket is always built with the exact
    parameters the sleeve was validated with — never hand-edited at the
    call site). / 信号源自带 banding 参数：篮子构建永远用该 sleeve 被验证
    时的同一组参数，调用方不得临场改配。"""

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
    """Scores from a trained cross-asset model checkpoint (v13 GRU basket or
    v14 O2 continuous). / 模型信号（v13 篮子或 v14 O2 连续权重共用）。

    Banding params come from the ckpt itself: live must replicate the exact
    configuration that was backtested and gated. CONSTRAINT (H-3): the model
    assigns asset identity embeddings BY POSITION in the sorted symbol list —
    a shifted, padded, or partial universe still emits plausible-looking
    scores, hence the sorted iteration and hard failure on short history
    below. Universe change = model retrain.
    banding 参数取自 ckpt（只复现被验证过的配置）；资产身份嵌入按排序位置
    分配（H-3）——宇宙错位照样输出“看似合理”的信号，必须硬失败；改宇宙
    即需重训模型。"""

    name = "model"

    def __init__(self, ckpt: dict, model: Any) -> None:
        self.ckpt = ckpt
        self.model = model
        self.seq_len = int(ckpt["seq_len"])
        self.k = int(ckpt.get("basket_k", 3))
        self.enter_band = int(ckpt.get("enter_band", 3))
        self.exit_band = int(ckpt.get("exit_band", 6))

    def compute_scores(self, feats, funding, idx_map):
        """One deterministic forward pass on the seq_len window ending at
        each symbol's mark bar (model is dropout=0/eval; funding already
        baked into feats). / 各标的 mark 截止的 seq_len 窗口一次确定性前向。"""
        syms = sorted(feats.keys())  # sorted order IS the embedding index (H-3) / 排序即嵌入位置
        xs = []
        for sym in syms:
            i = idx_map[sym]
            f = feats[sym]["factors"][: i + 1]  # causal slice: nothing after the mark / 因果切片
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
    high-funding ones (shorts COLLECT positive funding).
    反向 3 日均 funding：多低费率、空高费率（空头收取正费率）。

    Model-free BY DESIGN: no trained parameters = no overfitting surface,
    and funding is one of crypto's few SLOW signals (BIS WP 1087 "Crypto
    Carry"), so a trailing 3d mean loses little to the daily cadence.
    Research variant D (RESEARCH_2026-07-02): standalone Sharpe 0.53 with
    the bulk of PnL from funding collection, half the model sleeve's
    turnover, and corr(model, carry) = -0.01 — the diversifying leg of the
    50/50 combo (Sharpe 0.92). Live 60d gate ~2026-09-11.
    刻意无模型：零过拟合面；funding 是 crypto 少数慢信号（BIS Crypto
    Carry），慢采样几乎无损；与模型信号零相关，是组合的分散腿。"""

    name = "carry"

    def __init__(self, lookback: int = 72,
                 k: int = 3, enter_band: int = 3, exit_band: int = 6) -> None:
        self.lookback = lookback
        self.k = k
        self.enter_band = enter_band
        self.exit_band = exit_band

    def compute_scores(self, feats, funding, idx_map):
        """Score = NEGATIVE mean funding over the trailing lookback window
        ending at each symbol's mark bar. / 各标的 mark 前回看窗内费率均值取负。"""
        sig: Dict[str, float] = {}
        for sym, i in idx_map.items():
            f = funding.get(sym)
            # One missing/misaligned funding series invalidates the whole
            # cross-section (scores are relative ranks) — skip the day
            # rather than rank a partial universe.
            # 单币缺失即整日跳过：排名是相对的，残缺截面不可比。
            if f is None or f.numel() != len(feats[sym]["ts"]):
                return None
            lo = max(i - self.lookback + 1, 0)
            sig[sym] = -float(f[lo:i + 1].mean().item())
        if all(v == 0.0 for v in sig.values()):
            return None  # funding degraded to zeros / funding全零无信号
        return sig
