"""
Base class and registry for pluggable factors.
可插拔因子的基类和注册表。

Each factor is an independent .py file with a class extending BaseFactor.
每个因子是一个独立的 .py 文件，包含一个继承 BaseFactor 的类。
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
        result = result.clamp(-5.0, 5.0)
        result = torch.nan_to_num(result, nan=0.0)
        return result
