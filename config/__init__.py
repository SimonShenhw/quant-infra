"""
Config system — load BacktestConfig from YAML or use defaults.
配置系统 — 从 YAML 加载 BacktestConfig 或使用默认值。

WHY dacite for the YAML→dataclass hop: it recursively validates nested
sections and field types at load time, so a typo'd key or wrong-typed value
dies at startup instead of hours into a run (see config/schema.py for the
full rationale and the honest note that v13+ scripts don't consume this).
为什么用 dacite 做 YAML→dataclass 转换：它在加载时递归校验嵌套段与字段
类型，键名手滑或类型错误在启动即失败，而不是跑了几小时才炸（完整理由及
"v13+ 脚本并不消费本包"的如实说明见 config/schema.py）。
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
