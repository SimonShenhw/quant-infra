"""
Config system — load BacktestConfig from YAML or use defaults.
配置系统 — 从 YAML 加载 BacktestConfig 或使用默认值。
"""
from config.schema import (
    BacktestConfig, DataConfig, FeatureConfig, ModelConfig,
    CVConfig, TrainConfig, ExecutionConfig, PortfolioConfig,
)

__all__ = [
    "BacktestConfig", "DataConfig", "FeatureConfig", "ModelConfig",
    "CVConfig", "TrainConfig", "ExecutionConfig", "PortfolioConfig",
    "load_config", "default_config",
]


def load_config(path: str) -> BacktestConfig:
    """Load config from YAML file. / 从YAML文件加载配置。"""
    import yaml
    from dacite import from_dict
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return from_dict(data_class=BacktestConfig, data=raw)


def default_config() -> BacktestConfig:
    """Return default config matching v10 hardcoded values. / 返回匹配v10硬编码值的默认配置。"""
    return BacktestConfig()
