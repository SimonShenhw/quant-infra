# Auto-discover all factor plugins on import / 导入时自动发现所有因子插件
from factors.base import FactorRegistry, BaseFactor, register_factor
FactorRegistry.auto_discover()
