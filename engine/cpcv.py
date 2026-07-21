"""
Combinatorial Purged Cross-Validation (CPCV).
组合净化交叉验证。

WHAT — implements López de Prado's CPCV (Advances in Financial Machine
Learning) for financial time-series:
  - Splits data into N contiguous groups
  - Tests all C(N, k) combinations of k test groups
  - Purges training samples near test boundaries (remove lookback overlap)
  - Adds embargo gap after test boundaries (remove feature leakage)

WHY CPCV instead of walk-forward (the v10 decision — see README version
history): WFO leaks information at every train/test boundary and yields
only ONE backtest path, while each fold trains on just the data before it.
CPCV explicitly purges + embargoes every boundary and produces C(N, k)
paths: every sample gets a leakage-free out-of-sample prediction (later
ensemble-averaged), and each fold trains on ~(N-k)/N of ALL data — more
training data per fold than any expanding-window scheme.

Key invariants / 关键不变量:
  - Groups are CONTIGUOUS in time; shuffling would destroy the temporal
    structure that purge/embargo protect.
  - purge_bars must cover label horizon + feature lookback overlap
    (v13: purge=48 for the 24h label with seq_len=24 on true 1h bars).
  - CPCV controls leakage but NOT researcher degrees of freedom accumulated
    across 13 versions — hence v13 reports the Deflated Sharpe Ratio (0.11)
    next to every headline number.

实现 de Prado 的 CPCV 方法:
  - 将数据分为 N 个连续组
  - 测试所有 C(N,k) 种 k 组测试组合
  - 在测试边界附近净化训练样本（移除回看重叠）
  - 在测试边界后添加 embargo 间隔（移除特征泄露）

为什么用 CPCV 而非 walk-forward（v10 决策，见 README 版本史）：WFO 在每个
训练/测试边界都有泄露且只产生一条回测路径；CPCV 显式净化/隔离每个边界，
产生 C(N,k) 条路径——每根 bar 都获得无泄露的 OOS 预测（下游做集成平均），
且每折仍能用约 (N-k)/N 的全量数据训练。注意：CPCV 只控泄露、不控多版本
迭代累积的研究者自由度——因此 v13 在 Sharpe 旁并列披露 DSR。
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Tuple

import numpy as np


def generate_cpcv_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_bars: int = 24,
    embargo_bars: int = 48,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate all CPCV train/test splits with purging and embargo.
    生成所有 CPCV 训练/测试划分（含净化和隔离）。

    Parameters / 参数
    ----------
    n_samples : int
        Total number of samples. / 总样本数。
    n_groups : int
        Number of contiguous groups to divide data into. / 连续组数。
    n_test_groups : int
        Number of groups held out for testing per split. / 每个划分中的测试组数。
    purge_bars : int
        Remove training samples within this many bars BEFORE each test group start.
        Must cover label horizon + feature lookback so no train label overlaps
        a test feature window (v13: 48 = 24h label + seq_len 24).
        在每个测试组起点之前，移除此范围内的训练样本。
        必须覆盖标签持有期+特征回看窗口（v13 取 48 = 24h 标签 + 24 步序列）。
    embargo_bars : int
        Remove training samples within this many bars AFTER each test group end.
        Guards against serial correlation carrying test-period information into
        training samples that immediately follow the test block.
        在每个测试组终点之后，移除此范围内的训练样本。
        防止序列相关性把测试期信息带入紧随其后的训练样本。

    Returns / 返回
    -------
    List of (train_indices, test_indices) tuples.
    (训练索引, 测试索引) 元组列表。
    """
    # divide into N contiguous groups / 分为N个连续组
    group_size: int = n_samples // n_groups
    group_bounds: List[Tuple[int, int]] = []
    for g in range(n_groups):
        start: int = g * group_size
        end: int = (g + 1) * group_size if g < n_groups - 1 else n_samples
        group_bounds.append((start, end))

    all_indices: np.ndarray = np.arange(n_samples)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    # enumerate all C(N, k) combinations / 枚举所有 C(N,k) 组合
    for test_group_ids in combinations(range(n_groups), n_test_groups):
        # collect test indices / 收集测试索引
        test_mask: np.ndarray = np.zeros(n_samples, dtype=bool)
        for gid in test_group_ids:
            gs, ge = group_bounds[gid]
            test_mask[gs:ge] = True

        # start with all non-test as train / 初始训练集 = 所有非测试样本
        train_mask: np.ndarray = ~test_mask.copy()

        # purge + embargo around each test group boundary / 在每个测试组边界做净化+隔离
        for gid in test_group_ids:
            gs, ge = group_bounds[gid]

            # purge: remove train samples within purge_bars BEFORE test start
            # 净化：移除测试起点前 purge_bars 范围内的训练样本
            purge_start: int = max(0, gs - purge_bars)
            train_mask[purge_start:gs] = False

            # embargo: remove train samples within embargo_bars AFTER test end
            # 隔离：移除测试终点后 embargo_bars 范围内的训练样本
            embargo_end: int = min(n_samples, ge + embargo_bars)
            train_mask[ge:embargo_end] = False

        train_idx: np.ndarray = all_indices[train_mask]
        test_idx: np.ndarray = all_indices[test_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits
