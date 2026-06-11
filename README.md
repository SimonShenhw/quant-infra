# Quant Infra

Production-grade, event-driven quantitative backtesting engine with deep learning signal generation. Built from scratch in Python + PyTorch.

生产级事件驱动量化回测引擎，集成深度学习信号生成。基于 Python + PyTorch 从零构建。

---

## What This Is | 项目简介

A complete quantitative trading infrastructure covering the full pipeline: **data ingestion -> feature engineering -> model training -> signal generation -> order execution -> portfolio management -> performance analysis**. Designed around a central EventBus architecture with pluggable components.

完整的量化交易基础设施，覆盖全链路：**数据采集 → 因子工程 → 模型训练 → 信号生成 → 订单执行 → 组合管理 → 绩效分析**。以中央事件总线（EventBus）为核心架构，所有组件可插拔替换。

The project was developed iteratively across 13 versions (v1–v13), each addressing critical flaws discovered in the previous version — from data leakage bugs to unrealistic execution assumptions to cross-validation methodology, including two self-discovered result-invalidating bugs documented below (paper trading on random weights; a ms→µs timestamp switch that silently corrupted bar aggregation). The current v13 trains on true 1h bars with a 24h horizon-aligned label, validates via 15-split CPCV (all 15 folds positive, OOS rank IC 0.064), and monetizes the weak signal with a banded top-K portfolio (Novy-Marx–Velikov buy/hold bands): **+32.6% / Sharpe 0.81** under the TWAP adverse-selection cost model, **+18.2% / Sharpe 0.53 conservative lower bound** under a flat 8bps/side cost cross-checked by an independent second engine. Deflated Sharpe Ratio (0.11) is disclosed: the edge is not yet statistically separable from multiple-testing luck — live paper trading is the arbiter.

项目经历了 13 个大版本的迭代（v1–v13），每个版本都在解决上一版暴露出的致命缺陷——从数据泄露、不切实际的撮合假设到交叉验证方法论漏洞，包括两个自查发现、推翻全部已发布结果的 bug（模拟盘跑随机权重；Binance 时间戳 ms→µs 切换导致聚合静默损坏）。当前 v13 在真实 1h bar 上以 24h 持有期对齐标签训练，15 折 CPCV 全部为正（OOS rank IC 0.064），用双阈值 banded top-K 组合（Novy-Marx–Velikov buy/hold bands）将弱信号变现：TWAP 逆向选择成本模型下 **+32.6% / Sharpe 0.81**，独立第二引擎交叉验证的固定 8bps/边保守下界 **+18.2% / Sharpe 0.53**。同时如实披露 DSR=0.11：该收益尚不能与多重测试运气区分——模拟盘是最终裁判。

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
| `qlib_pack.py` | **8 Qlib-inspired factors**: kmid, klen, kup, klow, roc10, corr_pv, std20, max20_ratio |
| `multi_timeframe.py` | Multi-TF wrapper: factors at 1h+4h+24h scales / 多时间尺度因子封装 |

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
| `realtime_feed.py` | **Binance WebSocket** kline feed (replaces 6s REST) / WebSocket 实时K线 |

### Tools 分析工具 (`tools/`)

| Module 模块 | Description 描述 |
|--------|-------------|
| `factor_analyzer.py` | **Alphalens-style** IC analysis across 1h/6h/24h/48h horizons / 因子IC分析 |

### v11.2 New Modules | v11.2 新增模块

| Module 模块 | Description 描述 |
|--------|-------------|
| `data/funding_fetcher.py` | Binance Futures **real funding rate** historical API / 真实资金费率 |
| `data/onchain_fetcher.py` | Coinmetrics community on-chain metrics (free) / 链上指标（免费） |
| `engine/numba_backtest.py` | **Numba JIT** backtest loop (~50x faster) / Numba加速回测 |
| `engine/adaptive_sizing.py` | RL-inspired Kelly with drawdown awareness / 自适应Kelly仓位 |
| `model/patch_tst.py` | **PatchTST** alternative (ICLR 2023) cross-asset variant / PatchTST模型 |

### v13 New Modules | v13 新增模块

| Module 模块 | Description 描述 |
|--------|-------------|
| `run_v13_final.py` | 24h label + banded top-K backtest comparison + production ckpt / 24h标签+banding对照回测 |
| `tools/validation_stats.py` | **PSR + Deflated Sharpe Ratio** (Bailey & López de Prado) / 概率夏普+紧缩夏普 |
| `tools/paper_live_ic.py` | Live cross-sectional rank IC from paper-trading logs / 模拟盘实时rank IC |
| `tools/crosscheck_v13_engines.py` | Independent second backtest engine for cost cross-validation / 独立第二引擎交叉验证 |
| `REVIEW_2026-06-10.md` | Full code review + literature survey + v13 implementation log / 全项目审查+文献调研+实施记录 |
| `ENGINE_CROSSCHECK_2026-06-10.md` | Engine divergence + cost-model sensitivity report / 引擎分歧与成本敏感性报告 |

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
| **v11.1** | **Checkpoint save/load + paper trading bug fix** | Discovered paper trading was running on RANDOM weights for 12 days (41.7% win rate ≈ random) / 发现 paper trading 跑了12天随机权重模型 |
| **v11.2** | **10-feature optimization sweep** | Major upgrade: 21 factors, fold ensemble, Numba backtest, PatchTST, multi-TF, real funding rate, on-chain data, WS feed, adaptive Kelly, factor IC analyzer / 21因子+折集成+Numba回测+PatchTST等 |
| **v12** | **Cost-aware training: turnover penalty in loss + vol filter + min_hold 96** | v11.1 paid $632K in fees — train the model itself to output temporally smooth scores / v11.1 手续费 63 万：让模型原生输出低换手信号 |
| **v13** | **Timestamp bug fix (ALL v11/v12 results invalidated and rerun) + 24h label + banded top-K portfolio + PSR/DSR + gap-healing paper trading** | Binance Vision switched kline timestamps to microseconds in 2025-01 — 97% of "1h bars" were raw 5m bars; horizon mismatch (1h label vs multi-day hold) + per-bar full rebalance were killing net PnL / Binance 时间戳切微秒导致 97% 假聚合；标签与持有期错配 + 全量换仓吃掉净收益 |

---

## Results | 回测结果

### v13 (Current) — fixed true-1h data, 24h label, banded top-K portfolio | 当前版本

```
Source / 数据源:       Binance 5m klines (19 months) → TRUE 1h bars after the
                       ms/µs timestamp fix / 时间戳修复后的真实1h bar
Assets / 资产:         20 crypto pairs, timestamp-intersection aligned / 20个交易对，时间戳对齐
Bars / 样本:           13,153 hourly bars → 13,105 samples
Factors / 因子:        19 (re-ranked by IC on TRUE 1h data; dropped macd + volume_zscore)
Model / 模型:          CrossAssetGRUAttention d_model=128 + turnover-penalty loss
Label / 标签:          24h forward return — matched to the daily decision cadence
Validation / 验证:     CPCV (N=6, k=2) → 15 splits, purge=48, embargo=48,
                       purged train/val gap inside folds
Execution / 执行:       TWAP 4-slice + adverse selection, per-leg cost on each
                       asset's own path, positions effective NEXT bar (no
                       decision-bar lookahead PnL)

CPCV val rank corr / 折内验证:  0.0874 avg (range 0.044–0.138, 15/15 positive)
OOS ensemble rank IC / 样本外:  0.064 (24h label)

Portfolio construction comparison (same OOS predictions, same cost model):
组合构建对照（同一OOS预测、同一成本模型）:
  top1/bottom1, hourly, min_hold 96   ->  -44.5%  (Sharpe -0.76)
  top1/bottom1, daily full rebalance  ->  -10.7%  (Sharpe  0.02)
  banded top3/bottom3 (enter<3/exit>=6) -> +32.6%  (Sharpe  0.81, maxDD 23.4%)

Cost-model sensitivity (independent second-engine cross-check):
成本口径敏感性（独立第二引擎交叉验证，见 ENGINE_CROSSCHECK_2026-06-10.md）:
  TWAP adverse-selection model (~5bps/side realized) -> +32.6% / Sharpe 0.81
  Flat 8bps/side, conservative lower bound           -> +18.2% / Sharpe 0.53
  Engine-mechanics divergence at identical costs:    -0.8 ~ +1.3pp

Honest statistics / 诚实统计:
  PSR (vs SR*=0)                  0.84
  Deflated Sharpe Ratio (N=50)    0.11  <- NOT yet separable from
                                          multiple-testing luck; paper
                                          trading is the arbiter
```

### Key Findings | 核心发现

- **The signal is real but weak, and the money is in not trading**: identical predictions span -44.5% to +32.6% depending purely on portfolio construction. Buy/hold banding (Novy-Marx & Velikov, RFS 2016) cuts effective turnover while keeping signal exposure
  信号真实但弱，钱省在"不交易"上：同一预测矩阵仅因组合构建不同就横跨 -44.5% 到 +32.6%。双阈值 banding 在保留信号暴露的同时大幅降低有效换手
- **Label horizon must match the holding period**: v11/v12 trained on 1-bar labels but held for days — the predicted bar was never tradable. v13's 24h label aligns training, backtest, and paper trading into the SAME strategy
  标签必须对齐持有期：v11/v12 用 1-bar 标签却持仓数日，被预测的那根 bar 根本不可交易；v13 的 24h 标签让训练、回测、模拟盘成为同一个策略
- **Always re-derive factor rankings after a data fix**: on true 1h bars the IC ranking inverted — vol factors (std20, klen) lead at the 24h horizon, and a previously-dropped factor (klow) was #5. The old drop list was an artifact of corrupted aggregation
  数据修复后必须重排因子：真 1h 数据上 IC 排名彻底洗牌，波动率类因子领跑 24h 周期，旧剔除名单是污染数据的伪影
- **Report DSR next to Sharpe**: a +32.6% backtest with DSR 0.11 is a hypothesis, not an edge. CPCV alone cannot control researcher degrees of freedom across 13 versions of iteration
  Sharpe 旁边必须放 DSR：DSR 0.11 的 +32.6% 是一个待验证假设而非已证实的 edge；CPCV 无法控制 13 个版本迭代积累的研究者自由度

### Two Self-Discovered Result-Invalidating Bugs | 两个自查发现的"推翻结果"级 bug

**Case 2 — v13: 97% of "1h bars" were raw 5m bars | 时间戳静默损坏**

Binance Vision switched kline CSV timestamps from milliseconds to **microseconds** in 2025-01 archives. The 1h aggregation (`open_time // 3_600_000`) silently bucketed µs rows into 3.6-second bins — every 5m bar became its own "1h bar". 97% of v11/v12's "117K hourly bars" were raw 5m bars: labels were 5m returns, "48h holds" were ~4h, and the funding factor was both look-ahead and constant-zero after normalization. Fixed by unit normalization in the lake loader; every v11/v12 headline number was discarded and rerun as v13.

Binance Vision 自 2025-01 起将 K 线时间戳改为**微秒**，按毫秒分桶的 1h 聚合把每根 5m bar 静默地变成独立"1h bar"——v11/v12 的"117K 小时样本"中 97% 是原始 5m bar：标签实为 5m 收益、"48h 持仓"实为 4 小时、funding 因子既前视又恒为零。在数据湖加载层做单位归一化修复后，v11/v12 全部结果作废并以 v13 重跑。

**Case 1 — v11.1: Paper Trading Was Running Random Weights | 模拟盘跑的是随机权重**

After 12 days of paper trading (Mar 30 – Apr 24), reviewing accumulated data revealed:
- Win rate: **41.7%** (5W / 7L) — close to random baseline
- Cumulative return: -0.74%
- Sharpe: -0.15
- Day-to-day volatility: 2.84% (way too high for market-neutral)

Root cause: `run_paper_daily.py` initialized the model with **random weights** at every run instead of loading the trained checkpoint. The model was effectively a random number generator — explaining why results matched random chance.

12天模拟盘数据的 review 揭露：胜率 41.7% 接近随机基准，累计 -0.74%，夏普 -0.15。
根因：`run_paper_daily.py` 每次初始化随机权重而非加载训练好的checkpoint，模型实质上是随机数生成器，所以结果等同于随机选择。

**Fix in v11.1 / 修复方案**:
- `run_v11_final.py` now trains a final production model on all data and saves checkpoint to `checkpoints/v11_production.pt`
- `run_paper_daily.py` now loads that checkpoint at inference time
- Old `paper_daily.db` archived as `paper_daily_random_weights_backup.db`
- Need to retrain (run_v11_final.py once) before next paper trading session

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

### 2. Run v13 Pipeline (Recommended) | 运行 v13 管线（推荐）

```bash
# v13: 19 factors + 24h label + CPCV + banded-portfolio backtest comparison
#      + PSR/DSR (~8 min on RTX 5090)
# v13：19因子 + 24h标签 + CPCV + banding组合对照回测 + PSR/DSR（5090约8分钟）
python run_v13_final.py
```

### 3. Daily Paper Trading | 每日模拟盘

```bash
# Run once per day (~1 min): fetch CLOSED bars (multi-exchange, all-20-or-die)
# -> real funding -> inference -> banded basket update -> reconcile -> SQLite.
# Self-heals gaps: missed days are backfilled at the daily mark (signals
# recomputed causally; basket positions stay frozen). Same-day reruns upsert.
# 每日一次（约1分钟）：已收盘K线（跨所补抓，20币强制全齐）→ 真实funding →
# 推理 → banding篮子更新 → 对账入库。断档自愈：缺失日自动补课（信号因果
# 回算；篮子持仓冻结）。同日重跑 upsert 不重复记账。
python run_paper_daily.py            # auto-selects v13 checkpoint / 自动选v13
python run_paper_daily.py --dry-run  # test without writing / 测试不写库

# Live signal quality once data accumulates / 数据积累后查看 live rank IC:
python tools/paper_live_ic.py
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
├── tools/                         # Analysis & validation / 分析与验证
│   ├── factor_analyzer.py         # Alphalens-style IC (true 1h) / 因子IC分析
│   ├── validation_stats.py        # PSR + Deflated Sharpe / 统计验证
│   ├── paper_live_ic.py           # Live rank IC from paper logs / 实时IC
│   ├── crosscheck_v13_engines.py  # Second-engine cost cross-check / 引擎交叉验证
│   └── recompute_backtest.py      # Recompute from fold ckpts / 复算工具
├── run_v13_final.py               # v13: 24h label + banded portfolio / v13主管线
├── run_v12_final.py               # v12 cost-aware training / v12管线
├── run_v11_final.py               # v11 CPCV (13 factors, d128, 18m) / v11管线
├── run_v10_cpcv.py                # v10 CPCV pipeline / v10管线
├── run_paper.py                   # Paper trading entry / 模拟盘入口
├── run_paper_daily.py             # Daily paper trading + gap backfill / 每日模拟盘（断档自愈）
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

Portfolio construction & transaction costs / 组合构建与交易成本:
- *Dynamic Trading with Predictable Returns and Transaction Costs* — Gârleanu & Pedersen, JF 2013 (aim in front of the target, partial rebalancing)
- *A Taxonomy of Anomalies and Their Trading Costs* — Novy-Marx & Velikov, RFS 2016 (buy/hold banding — basis of v13 portfolio construction)
- *Multi-Period Trading via Convex Optimization* — Boyd et al. (arXiv 1705.00109)
- *Finance-Grounded Optimization For Algorithmic Trading* (arXiv 2509.04541, band turnover regularization)

Validation / 统计验证:
- *Advances in Financial Machine Learning* — Marcos López de Prado (CPCV methodology)
- *The Deflated Sharpe Ratio* — Bailey & López de Prado, JPM 2014 (v13 reports DSR)
- *Implementation Risk in Portfolio Backtesting* (arXiv 2603.20319 — motivated the v13 second-engine cross-check)

Models & signals / 模型与信号:
- *Building Cross-Sectional Systematic Strategies By Learning to Rank* — Poh, Lim, Zohren, Roberts (arXiv 2012.07149, ListMLE for cross-sections)
- *Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha* (arXiv 2508.04975)
- *From Attention to Profit: Quantitative Trading Strategy Based on Transformer* (arXiv 2404.00424)
- *Machine Learning Enhanced Multi-Factor Quantitative Trading* (arXiv 2507.07107)
- *A Controlled Comparison of Deep Learning for Multi-Horizon Financial Forecasting* (arXiv 2603.16886)
- *Exploring Microstructural Dynamics in Cryptocurrency LOBs* (arXiv 2506.05764)
- *TLOB: Transformer with Dual Attention for LOB Price Prediction* (arXiv 2502.15757)
- *Crypto Carry* — Schmeling, Schrimpf & Todorov, BIS WP 1087

---

## License | 许可证

MIT
