# Auto-discover all factor plugins on import / 导入时自动发现所有因子插件
#
# WHY at import time: consumers (run scripts, paper trading, invariant tests)
# resolve factors by NAME from checkpoints' `factor_names`; `import factors`
# must therefore be sufficient to populate the registry — no explicit
# per-factor imports anywhere else, so adding a factor stays a one-file change.
# 为什么在导入时发现：下游（run 脚本、模拟盘、不变量测试）按 checkpoint 里的
# `factor_names` 名称解析因子；`import factors` 必须足以填满注册表——
# 其他地方无需逐因子显式导入，新增因子保持"只改一个文件"。
from factors.base import FactorRegistry, BaseFactor, register_factor
FactorRegistry.auto_discover()
