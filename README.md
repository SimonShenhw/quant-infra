# Quant Infra

Production-grade, event-driven quantitative backtesting engine with deep learning signal generation. Built from scratch in Python + PyTorch.

## What This Is

A complete quantitative trading infrastructure that covers the full pipeline: **data ingestion -> feature engineering -> model training -> signal generation -> order execution -> portfolio management -> performance analysis**. Designed around a central EventBus architecture with pluggable components.

The project was developed iteratively across 9 versions (v1-v9), each addressing critical flaws discovered in the previous version — from data leakage bugs to unrealistic execution assumptions. The final version runs Walk-Forward Optimization across 60 folds on 1M+ bars of real Binance market data with adversarial execution modeling.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer (data/)                       │
│  Binance Archive Downloader ──→ Parquet Data Lake            │
│  CCXT Multi-Exchange Feed   ──→ SQLite Cache                 │
│  WebSocket Daemon           ──→ Real-time Parquet Stream     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Model Layer (model/)                        │
│  Feature Engineering (10 factors, causal rolling z-score)     │
│  QuantTransformer (Encoder-Decoder, 3 presets)                │
│  CrossAssetGRUAttention (GRU + cross-asset self-attention)    │
│  CrossSectionalTransformer (4D [B,A,T,F] + ListMLE)          │
│  Dual Loss: ListMLE + Focal + Uncertainty Weighting           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Engine Layer (engine/)                       │
│  EventBus (pub/sub, 7 event types)                           │
│  LOB Matching Engine (price-time priority)                    │
│  Adverse Selection Simulator (80% reject / 100% adverse)     │
│  TWAP Executor (4-slice split orders)                         │
│  Kelly Criterion Position Sizing                              │
│  Portfolio + Risk Manager (drawdown circuit breaker)          │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### Engine (`engine/`)

| Module | Description |
|--------|-------------|
| `events.py` | Typed EventBus with 7 event types (Tick, Market, Signal, Order, Fill, Risk) |
| `order_book.py` | Limit Order Book with price-time priority matching, adaptive cost model (A-share / Crypto) |
| `adverse_selection.py` | Micro-execution simulator: favorable moves 80% rejected, adverse moves 100% filled |
| `twap_executor.py` | Time-Weighted Average Price split-order execution over N bars |
| `execution.py` | Kelly Criterion dynamic position sizing with running win-rate calibration |
| `portfolio.py` | Position tracking, mark-to-market, equity curve, Sharpe/Calmar/MaxDD |
| `risk.py` | Max drawdown circuit breaker, concentration limits |
| `backtest.py` | Event loop orchestrator, bar-based liquidity seeding |

### Models (`model/`)

| Module | Description |
|--------|-------------|
| `transformer.py` | Encoder-Decoder Transformer with 3 presets (small/medium/large), CUDA-optimized |
| `cross_asset_attention.py` | GRU temporal encoder + Transformer cross-asset self-attention for lead-lag modeling |
| `cross_sectional.py` | 4D tensor architecture `[Batch, Assets, Seq, Features]` + ListMLE ranking loss |
| `features.py` | 10 causal factors with rolling z-score normalization (no look-ahead bias) |
| `obi_features.py` | Order Book Imbalance features (trade imbalance, Amihud illiquidity) |
| `strategy.py` | Dynamic volatility-adjusted thresholds, stop-loss/take-profit, cooldown periods |

### Data (`data/`)

| Module | Description |
|--------|-------------|
| `archive_downloader.py` | Bulk download from `data.binance.vision` (ThreadPool, ZIP→Parquet) |
| `async_feed.py` | CCXT multi-exchange concurrent feed with pagination → SQLite |
| `ws_daemon.py` | WebSocket daemon with heartbeat ping/pong + exponential backoff reconnection |
| `lake_loader.py` | Parquet data lake reader: partitioned by `{asset}/{year}/{month}/` |
| `synthetic_lob.py` | Synthetic LOB generator with regime-switching microstructure patterns |

## Version History

Each version addressed specific failures discovered in the previous iteration:

| Version | What Changed | Why |
|---------|-------------|-----|
| **v1** | Basic single-asset Transformer + MSE loss | Starting point |
| **v2** | Fixed data leakage (global normalization → rolling z-score) | v1 MSE was 10^-6 — fake, caused by future information in features |
| **v3** | Directional Focal Loss + OBI features | MSE-trained models can't predict direction (arXiv 2603.16886) |
| **v4** | Multi-asset 4D tensors + ListMLE ranking loss | Absolute return prediction is hopeless; ranking is tractable |
| **v5** | Adverse selection execution model | v4 Sharpe 1.38 was a "fill illusion" — limit orders face adverse selection |
| **v6** | 1h bars + TWAP + 48h holding lock + top/bottom 5% filter | v5 lost 48% to friction — needed lower frequency and split execution |
| **v7** | Walk-Forward Optimization + GRU cross-asset attention | Static train/test split leaks information across time |
| **v8** | 1M+ bars from Binance archive, 60-fold WFO | 720 bars was not statistically significant |
| **v9** | Reversal signal diagnosis | Proved the model (rank_corr=0.025) beats pure factor strategies |

## Results (v8 — 60-fold Walk-Forward, 43K OOS bars)

```
Source:          Binance 5m klines (6 months) aggregated to 1h
Assets:          20 crypto pairs
WFO Folds:       60 (each: 2-month train, 1-month OOS)
OOS Periods:     43,200 bars
Execution:       TWAP 4-slice + adverse selection (51% adverse fill rate)

Total Return:    +1.22%
Sharpe Ratio:    0.08
Max Drawdown:    16.4%
Rebalances:      900
Avg Hold:        48 hours
Avg Rank Corr:   0.025 (positive in 59/60 folds)
Transaction Cost: $307K (30.7% of capital)
```

### Key Findings

- **Crypto 1h cross-section is mean-reverting** (factor IC = -0.05), not momentum
- **Model rank_corr = 0.025 is the only profitable signal** — pure factor reversal loses -37%, pure momentum loses -25%, only the GRU+Attention model is net positive
- **Transaction costs dominate**: 48h holding lock reduced cost from $970K to $307K
- **Adverse selection is brutal**: 51% of limit order fills are adverse (price moved against you)

## Quick Start

### Requirements

```
torch>=2.0.0
ccxt
polars
pyarrow
websockets
aiohttp
```

### 1. Download Data

```python
# Bulk download 6 months of 5m klines from Binance archive
python data/archive_downloader.py

# Or fetch via CCXT (works in geo-restricted regions)
python data/async_feed.py
```

### 2. Run the Full Pipeline

```python
# v8: Walk-Forward with 1M+ bars (recommended)
python run_v8_bigdata.py

# v6: Low-frequency TWAP backtest on 1h bars
python run_v6_lowfreq.py

# Hyperparameter grid search
python hyperparam_search.py
```

### 3. Single-Asset Quick Test

```python
# Synthetic data + single Transformer (fast, good for understanding the codebase)
python main.py

# BTC/USDT OOS test with real data
python run_btc_oos.py
```

## Project Structure

```
quant-infra/
├── engine/                    # Event-driven backtest core
│   ├── events.py              # EventBus + 7 typed events
│   ├── order_book.py          # LOB matching engine
│   ├── adverse_selection.py   # Micro-execution simulator
│   ├── twap_executor.py       # TWAP split-order execution
│   ├── execution.py           # Kelly position sizing
│   ├── portfolio.py           # Portfolio + performance metrics
│   ├── risk.py                # Drawdown circuit breaker
│   └── backtest.py            # Main event loop
├── model/                     # PyTorch models
│   ├── transformer.py         # Encoder-Decoder Transformer
│   ├── cross_asset_attention.py # GRU + cross-asset self-attention
│   ├── cross_sectional.py     # 4D CrossSectionalTransformer + ListMLE
│   ├── features.py            # 10-factor causal feature engineering
│   ├── obi_features.py        # Order book imbalance features
│   └── strategy.py            # Signal generation with dynamic thresholds
├── data/                      # Data ingestion
│   ├── archive_downloader.py  # Binance archive → Parquet
│   ├── async_feed.py          # CCXT concurrent feed → SQLite
│   ├── ws_daemon.py           # WebSocket real-time daemon
│   ├── lake_loader.py         # Parquet data lake loader
│   └── synthetic_lob.py       # Synthetic LOB generator
├── main.py                    # Single-asset pipeline (v1-v3)
├── run_cross_sectional.py     # Multi-asset ranking (v4)
├── run_v5_final.py            # Adverse selection backtest
├── run_v6_lowfreq.py          # Low-freq TWAP + holding lock
├── run_v7_wfo.py              # Walk-Forward Optimization
├── run_v8_bigdata.py          # 1M+ bars WFO (main pipeline)
├── run_v9_reversal.py         # Reversal signal analysis
├── run_btc_oos.py             # BTC real-data OOS test
├── hyperparam_search.py       # Automated grid search
└── requirements.txt
```

## Hardware

Developed and tested on:
- **CPU**: AMD Ryzen 9 9950X3D
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **RAM**: 64GB DDR5

## References

Architecture and methodology informed by:
- *Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha* (arXiv 2508.04975)
- *From Attention to Profit: Quantitative Trading Strategy Based on Transformer* (arXiv 2404.00424)
- *Machine Learning Enhanced Multi-Factor Quantitative Trading* (arXiv 2507.07107)
- *A Controlled Comparison of Deep Learning for Multi-Horizon Financial Forecasting* (arXiv 2603.16886)
- *Exploring Microstructural Dynamics in Cryptocurrency LOBs* (arXiv 2506.05764)
- *TLOB: Transformer with Dual Attention for LOB Price Prediction* (arXiv 2502.15757)

## License

MIT
