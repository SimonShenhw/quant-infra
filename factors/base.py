"""
Base class and registry for pluggable factors.
可插拔因子的基类和注册表。

Each factor is an independent .py file with a class extending BaseFactor.
每个因子是一个独立的 .py 文件，包含一个继承 BaseFactor 的类。

WHY a decorator registry + auto-discover: factor experiments are hot-swappable —
adding/removing a factor is adding/deleting one file, with zero edits to the
training pipeline. The pipeline consumes factors by NAME (checkpoints store
`factor_names`), so the registry is the single source of truth for what exists.
为什么用装饰器注册表 + 自动发现：因子实验可热插拔——增删因子 = 增删一个文件，
训练管线零改动。管线按名称消费因子（checkpoint 存 `factor_names`），
注册表是"存在哪些因子"的唯一事实来源。

Invariants / 不变量:
  - All factor computations must be strictly causal (no look-ahead); enforced by
    tests/run_invariants.py::t2_no_lookahead — scrambling data after bar t must
    not change any factor value at <= t. Run it before touching this module.
    所有因子计算必须严格因果（无前视）；由 tests/run_invariants.py 的 t2 保证——
    打乱 t 之后的数据不得改变 <= t 处的任何因子值。改本模块前必跑该测试。
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch
from torch import Tensor

# global registry / 全局注册表
_FACTOR_REGISTRY: Dict[str, Type["BaseFactor"]] = {}


def register_factor(cls: Type["BaseFactor"]) -> Type["BaseFactor"]:
    """Decorator to register a factor class. / 注册因子类的装饰器。"""
    _FACTOR_REGISTRY[cls.name] = cls
    return cls


class BaseFactor(ABC):
    """
    Abstract base for all factors.
    所有因子的抽象基类。

    Subclasses must define:
      - name: str (unique identifier) / 名称：唯一标识
      - compute(open_, high, low, close, volume) -> Tensor / 计算方法
    """
    name: str = ""

    @abstractmethod
    def compute(
        self,
        open_: Tensor,
        high: Tensor,
        low: Tensor,
        close: Tensor,
        volume: Tensor,
    ) -> Tensor:
        """Compute factor values. Returns 1D tensor of same length as input. / 计算因子值。"""
        ...


class FactorRegistry:
    """
    Discovers and manages factor plugins.
    发现并管理因子插件。
    """

    @staticmethod
    def auto_discover() -> None:
        """Import all factor modules to trigger @register_factor. / 导入所有因子模块以触发注册。"""
        import importlib
        import pkgutil
        from pathlib import Path
        factors_dir = Path(__file__).parent
        for _, module_name, _ in pkgutil.iter_modules([str(factors_dir)]):
            if module_name not in ("base", "__init__"):
                importlib.import_module(f"factors.{module_name}")

    @staticmethod
    def list_factors() -> List[str]:
        """Return names of all registered factors. / 返回所有已注册因子的名称。"""
        return sorted(_FACTOR_REGISTRY.keys())

    @staticmethod
    def get(name: str) -> BaseFactor:
        """Get a factor instance by name. / 按名称获取因子实例。"""
        return _FACTOR_REGISTRY[name]()

    @staticmethod
    def build_tensor(
        factor_names: List[str],
        open_: Tensor, high: Tensor, low: Tensor,
        close: Tensor, volume: Tensor,
        zscore_window: int = 48,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Build factor tensor using named factors from registry.
        使用注册表中的命名因子构建因子张量。

        `extras` is an optional per-symbol payload (e.g. pre-aligned funding rate
        tensor) — forwarded only to factors whose compute() signature accepts it.
        extras 是可选的逐 symbol 载荷（如预对齐的资金费率张量），只转发给
        compute() 签名中声明接收它的因子。

        WHY normalize here (not per-factor): raw factor scales differ by orders
        of magnitude (RSI in [-1,1] vs volume ratios); a shared rolling z-score
        makes columns comparable so no factor dominates the model input purely
        by scale. The z-score window is trailing-only — strictly causal
        (invariant test t2). Clamp to ±5 sigma + nan_to_num afterwards: crypto
        returns are fat-tailed, and a single 50-sigma outlier (flash crash,
        exchange glitch) would otherwise dominate gradients / blow up training.
        为什么在这里统一归一化（而非各因子自理）：原始因子量纲相差数个数量级
        （RSI 在 [-1,1]，量比可达数十），共享滚动 z-score 让各列可比，避免某因子
        仅凭量纲主导模型输入。z-score 窗口只看过去——严格因果（不变量测试 t2）。
        之后 clamp ±5 σ + nan_to_num：crypto 收益厚尾，单个 50σ 异常值
        （闪崩、交易所故障）否则会主导梯度/炸掉训练。

        Returns (T, len(factor_names)) tensor, z-score normalized.
        返回 (T, len(factor_names)) 张量，经 z-score 归一化。
        """
        from model.features import _rolling_zscore
        cols: List[Tensor] = []
        for name in factor_names:
            if name not in _FACTOR_REGISTRY:
                raise KeyError(f"Factor '{name}' not registered. Available: {list(_FACTOR_REGISTRY.keys())}")
            factor = _FACTOR_REGISTRY[name]()
            # forward extras only if the factor's compute() declares it / 仅当因子声明接收时转发
            sig = inspect.signature(factor.compute)
            if extras is not None and "extras" in sig.parameters:
                raw = factor.compute(open_, high, low, close, volume, extras=extras)
            else:
                raw = factor.compute(open_, high, low, close, volume)
            normalized = _rolling_zscore(raw, zscore_window)
            cols.append(normalized)
        result = torch.stack(cols, dim=-1)
        # fat-tail guard: cap at ±5 sigma, zero out NaNs from warm-up windows
        # 厚尾防护：截断到 ±5σ，暖机窗口产生的 NaN 置零
        result = result.clamp(-5.0, 5.0)
        result = torch.nan_to_num(result, nan=0.0)
        return result
