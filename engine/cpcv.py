"""
Combinatorial Purged Cross-Validation (CPCV).
组合净化交叉验证。

Implements de Prado's CPCV for financial time-series:
  - Splits data into N contiguous groups
  - Tests all C(N, k) combinations of k test groups
  - Purges training samples near test boundaries (remove lookback overlap)
  - Adds embargo gap after test boundaries (remove feature leakage)

实现 de Prado 的 CPCV 方法:
  - 将数据分为 N 个连续组
  - 测试所有 C(N,k) 种 k 组测试组合
  - 在测试边界附近净化训练样本（移除回看重叠）
  - 在测试边界后添加 embargo 间隔（移除特征泄露）
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
        在每个测试组起点之前，移除此范围内的训练样本。
    embargo_bars : int
        Remove training samples within this many bars AFTER each test group end.
        在每个测试组终点之后，移除此范围内的训练样本。

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
