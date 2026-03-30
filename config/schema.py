"""
Config schema — all backtest hyperparameters in typed dataclasses.
配置模式 — 所有回测超参数以类型化 dataclass 定义。

Replaces hardcoded values scattered across 9 run_*.py files.
替代散落在 9 个 run_*.py 文件中的硬编码值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data source configuration. / 数据源配置。"""
    timeframe: str = "5m"
    aggregate_to: str = "1h"
    min_bars: int = 40000
    exchanges: List[str] = field(default_factory=lambda: ["binance", "okx"])
    symbols: Optional[List[str]] = None  # None = auto-discover from lake / None=从数据湖自动发现


@dataclass
class FeatureConfig:
    """Feature engineering configuration. / 因子工程配置。"""
    zscore_window: int = 48
    n_factors: int = 10
    factor_list: List[str] = field(default_factory=lambda: [
        "log_return", "sma5_ratio", "sma20_ratio", "ema10_ratio",
        "rsi", "macd", "bollinger", "volume_zscore",
        "trade_imbalance", "price_impact",
    ])


@dataclass
class ModelConfig:
    """Model architecture configuration. / 模型架构配置。"""
    d_model: int = 64
    gru_layers: int = 2
    n_cross_heads: int = 4
    n_cross_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.25
    seq_len: int = 24
    max_assets: int = 20


@dataclass
class CVConfig:
    """Cross-validation configuration. / 交叉验证配置。"""
    method: str = "cpcv"       # "cpcv" or "wfo" / "cpcv" 或 "wfo"
    n_groups: int = 6          # CPCV: number of groups / CPCV: 组数
    n_test_groups: int = 2     # CPCV: test groups per split / CPCV: 每个划分的测试组数
    purge_bars: int = 24       # CPCV: purge window = seq_len / CPCV: 净化窗口
    embargo_bars: int = 48     # CPCV: embargo window = zscore_window / CPCV: 隔离窗口
    # WFO params (used when method="wfo") / WFO参数（method="wfo"时使用）
    train_bars: int = 1440
    step_bars: int = 720


@dataclass
class TrainConfig:
    """Training configuration. / 训练配置。"""
    epochs: int = 60
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    grad_clip: float = 1.0


@dataclass
class ExecutionConfig:
    """Execution simulation configuration. / 执行模拟配置。"""
    twap_slices: int = 4
    favorable_reject_rate: float = 0.60
    taker_fee_bps: float = 4.0
    maker_fee_bps: float = 1.0
    min_hold_bars: int = 48


@dataclass
class PortfolioConfig:
    """Portfolio configuration. / 组合配置。"""
    initial_cash: float = 1_000_000.0
    max_drawdown: float = 0.25
    top_k_pct: float = 0.05    # top/bottom 5% for long/short / 头尾5%做多/做空


@dataclass
class BacktestConfig:
    """
    Top-level config composing all sub-configs.
    顶层配置，组合所有子配置。
    """
    name: str = "v10_cpcv"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
