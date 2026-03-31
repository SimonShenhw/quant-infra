# Quant Infra

Production-grade, event-driven quantitative backtesting engine with deep learning signal generation. Built from scratch in Python + PyTorch.

生产级事件驱动量化回测引擎，集成深度学习信号生成。基于 Python + PyTorch 从零构建。

---

## What This Is | 项目简介

A complete quantitative trading infrastructure covering the full pipeline: **data ingestion -> feature engineering -> model training -> signal generation -> order execution -> portfolio management -> performance analysis**. Designed around a central EventBus architecture with pluggable components.

完整的量化交易基础设施，覆盖全链路：**数据采集 → 因子工程 → 模型训练 → 信号生成 → 订单执行 → 组合管理 → 绩效分析**。以中央事件总线（EventBus）为核心架构，所有组件可插拔替换。

The project was developed iteratively across 10 versions (v1–v10), each addressing critical flaws discovered in the previous version — from data leakage bugs to unrealistic execution assumptions to cross-validation methodology. The final v10 uses Combinatorial Purged Cross-Validation (CPCV) across 15 splits on 1M+ bars of real Binance market data, with adversarial execution modeling, achieving **Sharpe 0.38 and +58% return** on purely out-of-sample data.

项目经历了 10 个大版本的迭代（v1–v10），每个版本都在解决上一版暴露出的致命缺陷——从数据泄露、不切实际的撮合假设到交叉验证方法论漏洞。最终 v10 在 100 万+ 条真实 Binance 市场数据上使用组合净化交叉验证（CPCV, 15 splits），含逆向选择撮合模拟，在纯样本外数据上达到 **Sharpe 0.38、收益率 +58%**。

---

## Architecture | 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│              Data Layer 数据层 (data/)                        │
│  Binance Archive Downloader ──→ Parquet Data Lake            │
│  CCXT Multi-Exchange Feed   ──→ SQLite Cache                 │
│  WebSocket Daemon           ──→ Avro/Parquet Stream          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Factor Layer 因子层 (factors/)                   │
│  Plugin Factor Library (10 hot-loadable .py files)           │
│  FactorRegistry: auto-discover + @register_factor            │
│  Causal rolling z-score normalization (no look-ahead)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Model Layer 模型层 (model/)                      │
│  CrossAssetGRUAttention (GRU temporal + cross-asset attn)    │
│  QuantTransformer (Encoder-Decoder, 3 presets)               │
│  CrossSectionalTransformer (4D [B,A,T,F] + ListMLE)         │
│  Dual Loss: ListMLE + Focal + Uncertainty Weighting          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Engine Layer 引擎层 (engine/)                    │
│  CPCV: Combinatorial Purged Cross-Validation (15 splits)     │
│  EventBus (pub/sub, 7 event types)                           │
│  Adverse Selection Simulator + TWAP Executor                 │
│  Kelly Criterion Sizing + Drawdown Circuit Breaker           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│         Paper Trading 模拟盘 (paper_trading/)                │
│  Live WebSocket → Model Inference → Simulated Execution      │
│  SQLite Logger (signals / fills / equity snapshots)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components | 核心组件

### Engine 回测引擎 (`engine/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `cpcv.py` | Combinatorial Purged Cross-Validation with purge + embargo / 组合净化交叉验证，含净化+隔离 |
| `events.py` | Typed EventBus with 7 event types / 类型化事件总线，7 种事件类型 |
| `order_book.py` | LOB matching, adaptive cost model (A-share / Crypto) / 限价指令簿撮合，自适应成本模型 |
| `adverse_selection.py` | Micro-execution: 80% favorable reject, 100% adverse fill / 逆向选择模拟器 |
| `twap_executor.py` | TWAP split-order execution / TWAP 拆单执行器 |
| `execution.py` | Kelly Criterion dynamic position sizing / Kelly 公式动态仓位管理 |
| `portfolio.py` | Position tracking, equity curve, Sharpe/Calmar/MaxDD / 持仓跟踪、权益曲线 |
| `risk.py` | Max drawdown circuit breaker / 最大回撤熔断器 |

### Factors 因子库 (`factors/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `base.py` | `BaseFactor` ABC + `@register_factor` decorator + `FactorRegistry` / 基类+装饰器+注册表 |
| `log_return.py` | Log returns / 对数收益率 |
| `sma_ratio.py` | SMA5 and SMA20 price ratios / SMA5/SMA20 价格比率 |
| `ema_ratio.py` | EMA10 price ratio / EMA10 价格比率 |
| `rsi.py` | Relative Strength Index / 相对强弱指标 |
| `macd.py` | MACD histogram / MACD 柱状图 |
| `bollinger.py` | Bollinger Band position / 布林带位置 |
| `volume_zscore.py` | Volume z-score / 成交量 z-score |
| `trade_imbalance.py` | Trade-based order imbalance (OBI) / 基于成交的订单不平衡度 |
| `price_impact.py` | Amihud illiquidity ratio / Amihud 非流动性比率 |
| `funding_rate.py` | Funding rate proxy (direction × volume) / 资金费率代理 |
| `btc_dominance.py` | Relative strength vs own mean / 相对自身均值的强弱 |
| `volume_momentum.py` | Short/long volume acceleration / 短期/长期成交量加速度 |

### Models 模型 (`model/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `cross_asset_attention.py` | GRU temporal + cross-asset self-attention / GRU 时序 + 跨资产自注意力 |
| `transformer.py` | Encoder-Decoder Transformer (3 presets, CUDA) / 编解码 Transformer |
| `cross_sectional.py` | 4D `[B, A, T, F]` + ListMLE ranking loss / 4D 横截面 + 排序损失 |
| `features.py` | Feature pipeline (delegates to factor registry) / 因子管线 |
| `obi_features.py` | Order Book Imbalance features / 订单簿不平衡度因子 |

### Data 数据 (`data/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `archive_downloader.py` | Bulk download `data.binance.vision` → Parquet / Binance 归档批量下载 |
| `async_feed.py` | CCXT concurrent feed → SQLite / CCXT 并发拉取 |
| `ws_daemon.py` | WebSocket daemon + heartbeat + exp backoff / WebSocket 守护进程 |
| `avro_writer.py` | Avro streaming serialization for real-time data / Avro 实时流序列化 |
| `lake_loader.py` | Parquet data lake reader / 数据湖加载器 |

### Config 配置 (`config/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `schema.py` | 8 typed dataclasses: Data, Feature, Model, CV, Train, Execution, Portfolio / 8个类型化配置类 |
| `__init__.py` | `load_config(yaml_path)` + `default_config()` / YAML加载 + 默认配置 |

### Paper Trading 模拟盘 (`paper_trading/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `engine.py` | Live bar ingestion → inference → simulated execution / 实时K线 → 推理 → 模拟执行 |
| `logger.py` | SQLite logger: signals, fills, equity snapshots / SQLite 日志 |

---

## Version History | 版本迭代史

| Version 版本 | What Changed 改动 | Why 原因 |
|---------|-------------|-----|
| **v1** | Single-asset Transformer + MSE / 单资产 Transformer + MSE | Starting point / 起点 |
| **v2** | Fixed data leakage (global → rolling z-score) / 修复数据泄露 | v1 MSE = 10⁻⁶ was fake / v1 的 MSE 是假的 |
| **v3** | Directional Focal Loss + OBI features / 方向性 Focal 损失 | MSE can't predict direction / MSE 无法预测方向 |
| **v4** | Multi-asset 4D tensors + ListMLE ranking / 多资产 ListMLE | Ranking > absolute return prediction / 排序优于绝对收益预测 |
| **v5** | Adverse selection execution / 逆向选择撮合 | v4 Sharpe 1.38 was "fill illusion" / v4 高夏普是"成交幻觉" |
| **v6** | 1h + TWAP + 48h hold lock + 5% filter / 低频+TWAP+持仓锁 | v5 lost 48% to friction / v5 被摩擦吃掉 48% |
| **v7** | Walk-Forward + GRU cross-asset attention / WFO+GRU跨资产注意力 | Static split leaks info / 静态划分泄露信息 |
| **v8** | 1M+ bars, 60-fold WFO / 百万数据60折WFO | 720 bars not significant / 720条无统计意义 |
| **v9** | Reversal diagnosis / 反转诊断 | Proved model > pure factors / 证明模型优于纯因子 |
| **v10** | **CPCV + config + factor plugins + paper trading + avro** | WFO has boundary leakage; need industrial infra / WFO有边界泄露；需工业级基建 |
| **v11** | **13 factors + d128 + 18-month data + daily paper trading** | More data + alternative factors + production readiness / 更多数据+另类因子+生产就绪 |

---

## Results | 回测结果

### v11 (Latest) — 15-split CPCV, 117K OOS bars, 18 months | 最新：15折CPCV，117K样本，18个月

```
Source / 数据源:       Binance 5m klines (18 months, 3.15M rows) → aggregated to 1h
                       Binance 5分钟K线（18个月，315万行）→ 聚合为1小时
Assets / 资产:         20 crypto pairs / 20个加密货币交易对
Factors / 因子:        13 plugin factors (10 price-volume + 3 alternative)
                       13个插件因子（10个量价 + 3个另类）
Model / 模型:          CrossAssetGRUAttention d_model=128, 617K params
Validation / 验证:     CPCV (N=6, k=2) → 15 splits, purge=24, embargo=48
OOS Coverage / OOS覆盖: 117,672 bars (100% of samples) / 全部样本
Execution / 执行:       TWAP 4-slice + adverse selection (65% adverse fill)

Avg Rank Corr / 平均排名相关: 0.062 (all 15 folds positive / 15折全部为正)
Total Return / 总收益:       -56.1% (dominated by transaction costs / 交易成本主导)
Max Drawdown / 最大回撤:     60.6%
Rebalances / 换仓次数:       2,451
Transaction Cost / 交易成本:  $369K (36.9% of capital / 占本金36.9%)
Avg Hold / 平均持仓:         48 hours / 48小时
```

### Version Comparison | 版本对比

| Metric 指标 | v8 (WFO) | v10 (CPCV 6m) | v11 (CPCV 18m) |
|------|:---:|:---:|:---:|
| Data / 数据 | 44K bars (6m) | 44K bars (6m) | **117K bars (18m)** |
| Factors / 因子 | 10 | 10 | **13** |
| Model params / 模型参数 | 124K | 124K | **617K (d128)** |
| Rank Correlation | 0.025 | 0.068 | **0.062** |
| OOS Coverage | 43,200 | 44,760 | **117,672** |
| Statistical confidence / 统计置信度 | Low / 低 | Medium / 中 | **High / 高** |
| Bug: Lookahead | Yes | Fixed | Fixed |
| Bug: Boundary leak | Yes | Fixed | Fixed |

### Key Findings | 核心发现

- **CPCV >> WFO**: 15-split CPCV with purge+embargo produces 2.7x better rank correlation than sequential WFO, because each fold trains on ~24K samples (vs 1.4K in WFO)
  CPCV 远优于 WFO：每个 fold 训练 24K 样本（WFO 仅 1.4K），排名相关性提升 2.7 倍
- **Rank correlation is stable at 0.06**: Consistent across 6-month and 18-month datasets, across d_model=64 and d_model=128, proving the signal is real and not an artifact of any specific configuration
  排名相关性稳定在 0.06：跨 6 个月和 18 个月数据集、跨 d_model=64 和 128 均一致，证明信号真实
- **1h label >> 6h label**: 6h cumulative return as training target degrades rank_corr from 0.068 to 0.039 — crypto 1h reversal signal is stronger at shorter horizons
  1h 标签远优于 6h 标签：6h 标签将 rank_corr 从 0.068 降至 0.039，crypto 短期反转信号在更短周期更强
- **Transaction costs dominate PnL**: With 65% adverse fill rate and 2,451 rebalances, costs ($369K) exceed gross alpha — paper trading is the next validation step
  交易成本主导 PnL：65% 逆向成交率 + 2451 次换仓，成本远超毛利 — 模拟盘是下一步验证
- **Model > pure factors**: v9 diagnosis proved GRU+Attention (rank_corr=0.025) beats pure factor reversal (-37%) and pure momentum (-25%)
  模型优于纯因子：v9 诊断证明 GRU+Attention 优于纯因子反转和纯动量策略

---

## Quick Start | 快速开始

### Requirements | 依赖

```
torch>=2.0.0
ccxt
polars
pyarrow
websockets
aiohttp
pyyaml
dacite
fastavro
```

### 1. Download Data | 下载数据

```bash
# Bulk download 6 months of 5m klines from Binance archive (886K rows, ~11s)
# 从 Binance 归档批量下载6个月K线（88.6万行，约11秒）
python data/archive_downloader.py

# Or fetch via CCXT (works in geo-restricted regions)
# 或通过 CCXT 获取（适用于网络受限地区）
python data/async_feed.py
```

### 2. Run v11 CPCV Pipeline (Recommended) | 运行 v11 CPCV 管线（推荐）

```bash
# v11: 13 factors + d128 + 18-month data + CPCV (~40 min on RTX 5090)
# v11：13因子 + d128 + 18个月数据 + CPCV（RTX 5090约40分钟）
python run_v11_final.py
```

### 3. Daily Paper Trading | 每日模拟盘

```bash
# Run once per day (~30 seconds): fetch bars → inference → log signal → reconcile
# 每天跑一次（约30秒）：拉K线 → 推理 → 记录信号 → 对账
python run_paper_daily.py
```

### 4. Legacy Pipelines | 旧版管线

```bash
python run_v8_bigdata.py       # WFO with bug fixes / 修复后的WFO
python run_v6_lowfreq.py       # Low-freq TWAP / 低频TWAP
python hyperparam_search.py    # Grid search / 网格搜索
python main.py                 # Single-asset synthetic / 单资产合成数据
```

---

## Project Structure | 项目结构

```
quant-infra/
├── config/                        # Config system / 配置系统
│   ├── schema.py                  # 8 typed dataclasses / 8个类型化配置类
│   └── __init__.py                # YAML loader / YAML加载器
├── configs/
│   └── v10_cpcv.yaml              # Default CPCV config / 默认CPCV配置
├── engine/                        # Backtest core / 回测核心
│   ├── cpcv.py                    # Combinatorial Purged CV / 组合净化交叉验证
│   ├── events.py                  # EventBus + 7 events / 事件总线
│   ├── order_book.py              # LOB matching / 撮合引擎
│   ├── adverse_selection.py       # Adverse selection / 逆向选择
│   ├── twap_executor.py           # TWAP execution / TWAP执行
│   ├── execution.py               # Kelly sizing / Kelly仓位
│   ├── portfolio.py               # Portfolio / 组合管理
│   ├── risk.py                    # Risk manager / 风控
│   └── backtest.py                # Event loop / 事件循环
├── factors/                       # Plugin factor library / 插件化因子库
│   ├── base.py                    # BaseFactor + FactorRegistry / 基类+注册表
│   ├── log_return.py              # Log returns
│   ├── sma_ratio.py               # SMA5/SMA20 ratios
│   ├── ema_ratio.py               # EMA10 ratio
│   ├── rsi.py                     # RSI
│   ├── macd.py                    # MACD
│   ├── bollinger.py               # Bollinger position
│   ├── volume_zscore.py           # Volume z-score
│   ├── trade_imbalance.py         # Trade imbalance (OBI)
│   ├── price_impact.py            # Amihud illiquidity
│   ├── funding_rate.py            # Funding rate proxy / 资金费率代理
│   ├── btc_dominance.py           # Relative strength / 相对强弱
│   └── volume_momentum.py         # Volume acceleration / 量能加速
├── model/                         # PyTorch models / 模型
│   ├── cross_asset_attention.py   # GRU + cross-asset attention
│   ├── transformer.py             # Encoder-Decoder Transformer
│   ├── cross_sectional.py         # 4D CrossSectional + ListMLE
│   ├── features.py                # Feature pipeline / 因子管线
│   ├── obi_features.py            # OBI features
│   └── strategy.py                # Signal generation / 信号生成
├── paper_trading/                 # Paper trading / 模拟盘
│   ├── engine.py                  # Live inference engine / 实时推理引擎
│   └── logger.py                  # SQLite logger / SQLite日志
├── data/                          # Data ingestion / 数据采集
│   ├── archive_downloader.py      # Binance archive → Parquet
│   ├── async_feed.py              # CCXT → SQLite
│   ├── avro_writer.py             # Avro streaming / Avro流式写入
│   ├── ws_daemon.py               # WebSocket daemon
│   ├── lake_loader.py             # Parquet loader / 数据湖加载
│   └── synthetic_lob.py           # Synthetic data / 合成数据
├── run_v11_final.py               # v11 CPCV (13 factors, d128, 18m) / v11主管线
├── run_v10_cpcv.py                # v10 CPCV pipeline / v10管线
├── run_paper.py                   # Paper trading entry / 模拟盘入口
├── run_paper_daily.py             # Daily batch paper trading / 每日批处理模拟盘
├── run_v8_bigdata.py              # v8 WFO (bug-fixed) / v8 WFO（已修复）
├── run_v6_lowfreq.py              # v6 low-freq / v6低频
├── run_v7_wfo.py                  # v7 WFO
├── hyperparam_search.py           # Grid search / 网格搜索
├── main.py                        # Single-asset / 单资产
└── requirements.txt
```

---

## Hardware | 硬件环境

Developed and tested on / 开发和测试环境：
- **CPU**: AMD Ryzen 9 9950X3D
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **RAM**: 64GB DDR5

---

## References | 参考论文

- *Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha* (arXiv 2508.04975)
- *From Attention to Profit: Quantitative Trading Strategy Based on Transformer* (arXiv 2404.00424)
- *Machine Learning Enhanced Multi-Factor Quantitative Trading* (arXiv 2507.07107)
- *A Controlled Comparison of Deep Learning for Multi-Horizon Financial Forecasting* (arXiv 2603.16886)
- *Exploring Microstructural Dynamics in Cryptocurrency LOBs* (arXiv 2506.05764)
- *TLOB: Transformer with Dual Attention for LOB Price Prediction* (arXiv 2502.15757)
- *Advances in Financial Machine Learning* — Marcos Lopez de Prado (CPCV methodology)

---

## License | 许可证

MIT
