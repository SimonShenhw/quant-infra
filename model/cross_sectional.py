"""
Cross-Sectional Multi-Asset Transformer with ListMLE Ranking Loss.

4D tensor architecture: [Batch, Assets, Seq_len, Features]
The model learns to RANK assets by relative performance, not predict
absolute returns. This avoids the MSE-on-returns trap entirely.

WHY ranking instead of return regression (the v4 decision): absolute
crypto returns are dominated by market beta and noise — v1-v3 regression
either overfit or predicted nothing.  Cross-sectional RANKING removes
the common market component by construction and directly targets what a
long-short book trades on (Poh, Lim, Zohren, Roberts, arXiv 2012.07149).
Ranking beat regression decisively in v4 and carried through v13.

HONEST UPDATE (2026-07-13, RESEARCH_2026-07-13_extended_window.md):
pure rank objectives later proved NOT to monetize in dollar space over
the full 2021-2026 window — Spearman IC was positive every single year,
yet every linear monetization of the rank signal (banded top-K,
rank-weighted book, vol-scaled, GP-smoothed) lost money: the model ranks
the many small moves correctly (which is what props up IC) while putting
the few LARGE movers on the wrong side.  ListMLE delivered exactly what
it was asked to optimize; the ask was wrong.  Production consequently
moved to magnitude-aware objectives (tools/research_objectives.py; O2 =
pairwise magnitude-weighted ranking is the v14 model sleeve).  ListMLE
remains the v13 CONTROL-TRACK objective — its paper track runs unchanged
to the September gate so the pre-registered comparison stays intact.

Key components:
  - Per-asset temporal encoder (shared weights)
  - Cross-asset attention (learns inter-asset dependencies)
  - ListMLE ranking loss (from Learning to Rank literature)

Status: `listmle_loss` is imported by the v13 result path
(run_v13_final.py) and the research tools; the CrossSectionalTransformer
CLASS below is the v4/v5-era model, superseded by CrossAssetGRUAttention
since v7 and now used only by legacy run scripts (run_v5_final.py,
run_v6_lowfreq.py, run_cross_sectional.py, hyperparam_search.py).

截面多资产Transformer，带ListMLE排序损失。

4D张量架构: [批次, 资产, 序列长度, 特征]
模型学习按相对收益对资产进行排序，而非预测绝对收益。
完全避免了"对收益率做MSE回归"的陷阱。

为什么用排序而非回归（v4 决策）：加密货币绝对收益被市场 beta 与噪声
主导，v1-v3 的回归要么过拟合要么学不到东西；横截面排序在构造上消去了
共同市场分量，直指多空组合真正交易的对象（Poh 等，arXiv 2012.07149）。

诚实更新（2026-07-13）：纯 rank 目标后来被证明在美元空间不变现——
2021-2026 每一年 Spearman IC 均为正，但 rank 信号的所有线性变现方式
全部亏损：模型排对了大量小幅度名字（撑起 IC），却系统性把少数大幅度
行情放错边。ListMLE 交付了它被要求优化的东西，是要求本身错了。生产
已转向幅度感知目标（tools/research_objectives.py，O2 = v14 模型
sleeve）；ListMLE 保留为 v13 对照 track 的目标函数，照跑到 9 月 gate、
维持预注册对比不中断。

核心组件:
  - 每资产时序编码器（共享权重）
  - 跨资产注意力（学习资产间依赖关系）
  - ListMLE排序损失（来自Learning to Rank文献）

状态：`listmle_loss` 仍在 v13 出结果路径上；下方的
CrossSectionalTransformer 类是 v4/v5 时期模型，v7 起被
CrossAssetGRUAttention 取代，现仅用于遗留 run 脚本。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ListMLE Ranking Loss (Xia et al., 2008)
# ListMLE排序损失（Xia等，2008）
# ---------------------------------------------------------------------------

def listmle_loss(scores: Tensor, relevance: Tensor) -> Tensor:
    """
    ListMLE: Listwise ranking loss based on Plackett-Luce model.

    Given predicted scores and true relevance labels for a set of items,
    computes the negative log-likelihood of the permutation defined by
    sorting items by true relevance.

    NOTE — magnitude-blind by construction: a +40% mover and a +0.4% mover
    are just adjacent ranks to this loss.  That blindness is precisely what
    the 2026-07-13 research identified as the monetization failure mode
    (see module HONEST UPDATE); kept as the v13 control-track objective.

    ListMLE: 基于Plackett-Luce模型的列表级排序损失。
    给定预测分数和真实相关性标签，计算按真实相关性排序所定义排列的负对数似然。
    注意——构造上对幅度盲视：+40% 与 +0.4% 在该损失眼中只是相邻名次。
    这正是 2026-07-13 研究定位的变现失败根源（见模块诚实更新）；
    作为 v13 对照 track 目标保留。

    Parameters / 参数
    ----------
    scores : Tensor
        Predicted scores, shape (B, N) where N = number of assets.
        预测分数，形状 (B, N)，N为资产数。
    relevance : Tensor
        True relevance/returns, shape (B, N). Higher = better.
        真实相关性/收益率，形状 (B, N)。值越大越好。

    Returns / 返回
    -------
    Tensor
        Scalar loss.
        标量损失值。
    """
    # sort indices by true relevance (descending) / 按真实相关性降序排列索引
    _, sorted_idx = relevance.sort(dim=-1, descending=True)
    # gather predicted scores in true-relevance order / 按真实相关性顺序收集预测分数
    sorted_scores: Tensor = scores.gather(dim=-1, index=sorted_idx)

    # ListMLE: for each position i, compute log-softmax over remaining items
    # ListMLE: 对每个位置i，计算剩余项的log-softmax
    n: int = sorted_scores.size(1)
    # Plackett-Luce NLL needs, at each position i, logsumexp over the SUFFIX
    # s_i..s_n.  Trick: flip → logcumsumexp (prefix) → flip back = suffix
    # logsumexp in one vectorized, numerically-stable pass (no O(n^2) loop,
    # no overflow from raw exp).
    # log P(pi) = sum_{i=1}^{n} [s_{pi(i)} - logsumexp(s_{pi(i)}, ..., s_{pi(n)})]
    # Plackett-Luce 负对数似然需要每个位置 i 对后缀 s_i..s_n 做 logsumexp。
    # 技巧：翻转 → logcumsumexp（前缀）→ 再翻转 = 后缀 logsumexp，
    # 一次向量化完成且数值稳定（无 O(n^2) 循环、无裸 exp 溢出）。
    cumsums: Tensor = torch.logcumsumexp(sorted_scores.flip(dims=[1]), dim=1).flip(dims=[1])
    loss: Tensor = -(sorted_scores - cumsums).mean()
    return loss


# ---------------------------------------------------------------------------
# Cross-Sectional Transformer
# 截面Transformer
# ---------------------------------------------------------------------------

class CrossSectionalTransformer(nn.Module):
    """
    Multi-asset ranking model (v4/v5 era — LEGACY: superseded by
    CrossAssetGRUAttention since v7; only legacy run scripts and
    hyperparam_search.py still instantiate this class).

    多资产排序模型（v4/v5 时期——遗留：v7 起被 CrossAssetGRUAttention
    取代，现仅遗留脚本使用）。

    Architecture:
      1. FactorProjection: (B, A, T, F) -> (B, A, T, D)
      2. TemporalEncoder: shared Transformer encoder over time axis
         (B*A, T, D) -> (B*A, D) via mean pooling
      3. CrossAssetAttention: Transformer over asset axis
         (B, A, D) -> (B, A, D)
      4. RankingHead: (B, A, D) -> (B, A) scores

    The output scores are used with ListMLE loss for ranking.

    多资产排序模型。

    架构:
      1. 因子投影: (B, A, T, F) -> (B, A, T, D)
      2. 时序编码器: 共享Transformer编码器处理时间轴
         (B*A, T, D) -> (B*A, D) 通过均值池化
      3. 跨资产注意力: Transformer处理资产轴
         (B, A, D) -> (B, A, D)
      4. 排序头: (B, A, D) -> (B, A) 分数

    输出分数与ListMLE损失配合用于排序。
    """

    def __init__(
        self,
        n_factors: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_temporal_layers: int = 2,
        n_cross_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 20,
        max_assets: int = 20,
    ) -> None:
        super().__init__()
        self.n_factors: int = n_factors
        self.d_model: int = d_model
        self.seq_len: int = seq_len

        # factor projection / 因子投影
        self.factor_proj: nn.Linear = nn.Linear(n_factors, d_model)
        self.factor_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # learnable positional encoding for time axis / 时间轴可学习位置编码
        self.time_pe: nn.Parameter = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # temporal encoder (shared across assets) / 时序编码器（跨资产共享）
        temp_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            temp_layer, num_layers=n_temporal_layers
        )

        # learnable asset positional encoding / 可学习资产位置编码
        self.asset_pe: nn.Parameter = nn.Parameter(
            torch.randn(1, max_assets, d_model) * 0.02
        )

        # cross-asset attention / 跨资产注意力
        cross_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            cross_layer, num_layers=n_cross_layers
        )

        # ranking head: project to scalar score per asset / 排序头: 每个资产投影为标量分数
        self.rank_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters / 参数
        ----------
        x : Tensor
            Shape (B, A, T, F) — batch, assets, time, features.
            形状 (B, A, T, F) — 批次、资产、时间、特征。

        Returns / 返回
        -------
        Tensor
            Shape (B, A) — ranking scores per asset.
            形状 (B, A) — 每个资产的排序分数。
        """
        B, A, T, _F = x.shape
        D: int = self.d_model

        # 1. Factor projection: (B, A, T, F) -> (B, A, T, D) / 因子投影
        h: Tensor = self.factor_norm(F.gelu(self.factor_proj(x)))

        # 2. Temporal encoding: reshape to (B*A, T, D) for shared encoder / 时序编码: 重塑为共享编码器输入
        h = h.reshape(B * A, T, D)
        h = h + self.time_pe[:, :T, :]
        h = self.temporal_encoder(h)  # (B*A, T, D)

        # pool over time -> (B*A, D) / 时间维度池化
        h = h.mean(dim=1)

        # 3. Reshape to (B, A, D) for cross-asset attention / 重塑为跨资产注意力输入
        h = h.reshape(B, A, D)
        h = h + self.asset_pe[:, :A, :]
        h = self.cross_encoder(h)  # (B, A, D)

        # 4. Ranking head → (B, A, 1) → (B, A) / 排序头
        scores: Tensor = self.rank_head(h).squeeze(-1)
        return scores
