# Quant Infra

Production-grade, event-driven quantitative backtesting engine with deep learning signal generation. Built from scratch in Python + PyTorch.

生产级事件驱动量化回测引擎，集成深度学习信号生成。基于 Python + PyTorch 从零构建。

---

## What This Is | 项目简介

A complete quantitative trading infrastructure covering the full pipeline: **data ingestion -> feature engineering -> model training -> signal generation -> order execution -> portfolio management -> performance analysis**. Designed around a central EventBus architecture with pluggable components.

完整的量化交易基础设施，覆盖全链路：**数据采集 → 因子工程 → 模型训练 → 信号生成 → 订单执行 → 组合管理 → 绩效分析**。以中央事件总线（EventBus）为核心架构，所有组件可插拔替换。

The project was developed iteratively across 9 versions (v1–v9), each addressing critical flaws discovered in the previous version — from data leakage bugs to unrealistic execution assumptions. The final version runs Walk-Forward Optimization across 60 folds on 1M+ bars of real Binance market data with adversarial execution modeling.

项目经历了 9 个大版本的迭代（v1–v9），每个版本都在解决上一版暴露出的致命缺陷——从数据泄露 bug 到不切实际的撮合假设。最终版本在 100 万+ 条真实 Binance 市场数据上运行了 60 折 Walk-Forward 滚动优化，并包含逆向选择撮合模拟。

---

## Architecture | 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│              Data Layer 数据层 (data/)                        │
│  Binance Archive Downloader ──→ Parquet Data Lake            │
│  CCXT Multi-Exchange Feed   ──→ SQLite Cache                 │
│  WebSocket Daemon           ──→ Real-time Parquet Stream     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Model Layer 模型层 (model/)                      │
│  Feature Engineering (10 factors, causal rolling z-score)     │
│  QuantTransformer (Encoder-Decoder, 3 presets)                │
│  CrossAssetGRUAttention (GRU + cross-asset self-attention)    │
│  CrossSectionalTransformer (4D [B,A,T,F] + ListMLE)          │
│  Dual Loss: ListMLE + Focal + Uncertainty Weighting           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Engine Layer 引擎层 (engine/)                    │
│  EventBus (pub/sub, 7 event types)                           │
│  LOB Matching Engine (price-time priority)                    │
│  Adverse Selection Simulator (80% reject / 100% adverse)     │
│  TWAP Executor (4-slice split orders)                         │
│  Kelly Criterion Position Sizing                              │
│  Portfolio + Risk Manager (drawdown circuit breaker)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components | 核心组件

### Engine 回测引擎 (`engine/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `events.py` | Typed EventBus with 7 event types (Tick, Market, Signal, Order, Fill, Risk) / 类型化事件总线，7 种事件类型 |
| `order_book.py` | LOB with price-time priority matching, adaptive cost model (A-share / Crypto) / 限价指令簿撮合引擎，自适应成本模型（A 股/加密货币） |
| `adverse_selection.py` | Micro-execution simulator: favorable moves 80% rejected, adverse 100% filled / 逆向选择模拟器：有利行情 80% 拒单，不利行情 100% 成交 |
| `twap_executor.py` | Time-Weighted Average Price split-order execution / TWAP 拆单执行器，分 N 个 Bar 缓慢吃单 |
| `execution.py` | Kelly Criterion dynamic position sizing / Kelly 公式动态仓位管理 |
| `portfolio.py` | Position tracking, equity curve, Sharpe/Calmar/MaxDD / 持仓跟踪、权益曲线、夏普/卡尔玛/最大回撤 |
| `risk.py` | Max drawdown circuit breaker / 最大回撤熔断器 |
| `backtest.py` | Event loop orchestrator / 事件循环主控 |

### Models 模型 (`model/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `transformer.py` | Encoder-Decoder Transformer (small/medium/large presets, CUDA) / 编码器-解码器 Transformer，3 档预设，CUDA 加速 |
| `cross_asset_attention.py` | GRU temporal + cross-asset self-attention for lead-lag / GRU 时序编码 + 跨资产自注意力，捕捉领先-滞后关系 |
| `cross_sectional.py` | 4D `[Batch, Assets, Seq, Features]` + ListMLE ranking loss / 4D 张量横截面架构 + ListMLE 排序损失 |
| `features.py` | 10 causal factors, rolling z-score (no look-ahead) / 10 个因果因子，滚动 z-score 归一化（无未来函数） |
| `obi_features.py` | Order Book Imbalance: trade imbalance, Amihud illiquidity / 订单簿不平衡度：交易不平衡、Amihud 非流动性 |
| `strategy.py` | Dynamic vol-adjusted thresholds, stop-loss/take-profit / 动态波动率阈值、止损止盈、冷却期 |

### Data 数据 (`data/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `archive_downloader.py` | Bulk download from `data.binance.vision` → Parquet / 从 Binance 公开归档批量下载 → Parquet |
| `async_feed.py` | CCXT multi-exchange concurrent feed → SQLite / CCXT 多交易所并发拉取 → SQLite |
| `ws_daemon.py` | WebSocket daemon + heartbeat + exponential backoff / WebSocket 守护进程 + 心跳 + 指数退避重连 |
| `lake_loader.py` | Parquet data lake reader, partitioned `{asset}/{year}/{month}/` / Parquet 数据湖加载器，按资产/年/月分区 |
| `synthetic_lob.py` | Synthetic LOB with regime-switching microstructure / 合成 LOB 数据生成器，含 regime 切换微结构 |

---

## Version History | 版本迭代史

Each version addressed specific failures discovered in the previous iteration:

每个版本都针对上一版暴露出的具体缺陷进行修复：

| Version 版本 | What Changed 改动 | Why 原因 |
|---------|-------------|-----|
| **v1** | Basic single-asset Transformer + MSE / 单资产 Transformer + MSE 损失 | Starting point / 起点 |
| **v2** | Fixed data leakage (global → rolling z-score) / 修复数据泄露（全局归一化 → 滚动 z-score） | v1 MSE = 10⁻⁶ was fake — future info in features / v1 的 MSE 是假的，特征中混入了未来信息 |
| **v3** | Directional Focal Loss + OBI features / 方向性 Focal 损失 + OBI 因子 | MSE can't predict direction (arXiv 2603.16886) / MSE 训练的模型无法预测方向 |
| **v4** | Multi-asset 4D tensors + ListMLE ranking / 多资产 4D 张量 + ListMLE 排序 | Absolute return prediction is hopeless; ranking is tractable / 绝对收益预测无望，排序可行 |
| **v5** | Adverse selection execution model / 逆向选择撮合模型 | v4 Sharpe 1.38 was a "fill illusion" / v4 的高夏普是"成交幻觉" |
| **v6** | 1h + TWAP + 48h holding lock + top/bottom 5% / 1h 级别 + TWAP + 48h 持仓锁 + 头尾 5% 过滤 | v5 lost 48% to friction / v5 被摩擦成本吃掉 48% |
| **v7** | Walk-Forward Optimization + GRU cross-asset attention / WFO 滚动优化 + GRU 跨资产注意力 | Static split leaks info across time / 静态划分在时间维度上泄露信息 |
| **v8** | 1M+ bars from Binance archive, 60-fold WFO / Binance 归档 100 万+ 条数据，60 折 WFO | 720 bars was not statistically significant / 720 条数据不具备统计显著性 |
| **v9** | Reversal signal diagnosis / 反转信号诊断 | Proved model beats pure factors / 证明模型优于纯因子策略 |

---

## Results | 回测结果

### v8 — 60-fold Walk-Forward, 43K OOS bars | 60 折滚动验证，43K 条样本外数据

```
Source / 数据源:       Binance 5m klines (6 months) aggregated to 1h
                       Binance 5分钟K线（6个月）聚合为1小时
Assets / 资产:         20 crypto pairs / 20个加密货币交易对
WFO Folds / 滚动折数:  60 (each: 2-month train, 1-month OOS)
                       60折（每折：2个月训练，1个月样本外）
OOS Periods / OOS样本:  43,200 bars / 43,200条
Execution / 执行:       TWAP 4-slice + adverse selection (51% adverse fill)
                       TWAP 4片拆单 + 逆向选择（51%逆向成交率）

Total Return / 总收益:       +1.22%
Sharpe Ratio / 夏普比率:     0.08
Max Drawdown / 最大回撤:     16.4%
Rebalances / 换仓次数:       900
Avg Hold / 平均持仓:         48 hours / 48小时
Avg Rank Corr / 平均排名相关: 0.025 (positive in 59/60 folds / 59/60折为正)
Transaction Cost / 交易成本:  $307K (30.7% of capital / 占本金30.7%)
```

### Key Findings | 核心发现

- **Crypto 1h cross-section is mean-reverting** (factor IC = -0.05), not momentum
  加密货币 1h 横截面是均值回归市场（因子 IC = -0.05），而非动量市场
- **Model rank_corr = 0.025 is the only profitable signal** — pure factor reversal loses -37%, pure momentum loses -25%, only the GRU+Attention model is net positive
  模型的 rank_corr = 0.025 是唯一盈利信号——纯因子反转亏 -37%，纯动量亏 -25%，只有 GRU+Attention 模型净正
- **Transaction costs dominate**: 48h holding lock reduced cost from $970K to $307K
  交易成本是最大敌人：48h 持仓锁将成本从 $970K 降到 $307K
- **Adverse selection is brutal**: 51% of limit order fills are adverse
  逆向选择极其残酷：51% 的限价单成交都是逆向的（价格朝不利方向移动后成交）

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
```

### 1. Download Data | 下载数据

```bash
# Bulk download 6 months of 5m klines from Binance archive
# 从 Binance 公开归档批量下载6个月5分钟K线
python data/archive_downloader.py

# Or fetch via CCXT (works in geo-restricted regions)
# 或通过 CCXT 获取（适用于网络受限地区）
python data/async_feed.py
```

### 2. Run the Full Pipeline | 运行完整管线

```bash
# v8: Walk-Forward with 1M+ bars (recommended)
# v8：100万+条数据的滚动回测（推荐）
python run_v8_bigdata.py

# v6: Low-frequency TWAP backtest on 1h bars
# v6：1小时低频 TWAP 回测
python run_v6_lowfreq.py

# Hyperparameter grid search
# 超参数网格搜索
python hyperparam_search.py
```

### 3. Single-Asset Quick Test | 单资产快速测试

```bash
# Synthetic data + single Transformer (fast, good for understanding the codebase)
# 合成数据 + 单 Transformer（速度快，适合理解代码结构）
python main.py

# BTC/USDT OOS test with real data
# BTC/USDT 真实数据样本外测试
python run_btc_oos.py
```

---

## Project Structure | 项目结构

```
quant-infra/
├── engine/                        # Event-driven backtest core / 事件驱动回测核心
│   ├── events.py                  # EventBus + 7 typed events / 事件总线 + 7种事件
│   ├── order_book.py              # LOB matching engine / 限价指令簿撮合引擎
│   ├── adverse_selection.py       # Micro-execution simulator / 逆向选择模拟器
│   ├── twap_executor.py           # TWAP split-order execution / TWAP拆单执行
│   ├── execution.py               # Kelly position sizing / Kelly仓位管理
│   ├── portfolio.py               # Portfolio + metrics / 组合管理 + 绩效指标
│   ├── risk.py                    # Drawdown circuit breaker / 回撤熔断器
│   └── backtest.py                # Main event loop / 主事件循环
├── model/                         # PyTorch models / PyTorch 模型
│   ├── transformer.py             # Encoder-Decoder Transformer
│   ├── cross_asset_attention.py   # GRU + cross-asset attention / GRU + 跨资产注意力
│   ├── cross_sectional.py         # 4D CrossSectional + ListMLE / 4D横截面 + 排序损失
│   ├── features.py                # 10-factor engineering / 10因子工程
│   ├── obi_features.py            # Order book imbalance / 订单簿不平衡度
│   └── strategy.py                # Signal generation / 信号生成策略
├── data/                          # Data ingestion / 数据采集
│   ├── archive_downloader.py      # Binance archive → Parquet
│   ├── async_feed.py              # CCXT concurrent → SQLite
│   ├── ws_daemon.py               # WebSocket real-time daemon / 实时WebSocket守护进程
│   ├── lake_loader.py             # Parquet data lake loader / 数据湖加载器
│   └── synthetic_lob.py           # Synthetic LOB generator / 合成LOB生成器
├── main.py                        # Single-asset pipeline (v1-v3) / 单资产管线
├── run_cross_sectional.py         # Multi-asset ranking (v4) / 多资产排序
├── run_v5_final.py                # Adverse selection backtest / 逆向选择回测
├── run_v6_lowfreq.py              # Low-freq TWAP + holding lock / 低频TWAP
├── run_v7_wfo.py                  # Walk-Forward Optimization / 滚动优化
├── run_v8_bigdata.py              # 1M+ bars WFO (main) / 百万级数据主管线
├── run_v9_reversal.py             # Reversal signal analysis / 反转信号分析
├── run_btc_oos.py                 # BTC OOS test / BTC样本外测试
├── hyperparam_search.py           # Grid search / 网格搜索
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

Architecture and methodology informed by / 架构和方法参考：
- *Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha* (arXiv 2508.04975)
- *From Attention to Profit: Quantitative Trading Strategy Based on Transformer* (arXiv 2404.00424)
- *Machine Learning Enhanced Multi-Factor Quantitative Trading* (arXiv 2507.07107)
- *A Controlled Comparison of Deep Learning for Multi-Horizon Financial Forecasting* (arXiv 2603.16886)
- *Exploring Microstructural Dynamics in Cryptocurrency LOBs* (arXiv 2506.05764)
- *TLOB: Transformer with Dual Attention for LOB Price Prediction* (arXiv 2502.15757)

---

## License | 许可证

MIT
