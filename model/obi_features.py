"""
Order Book Imbalance (OBI) Features for LOB-based prediction.

Implements microstructure features from:
  - arXiv 2506.05764: I^1, I^5, weighted mid-price change
  - arXiv 2505.22678: OFI (Order Flow Imbalance)
  - arXiv 1512.03492: Queue imbalance as directional predictor

All features are computed as PyTorch tensors for CUDA acceleration.

订单簿失衡(OBI)特征，用于基于LOB的预测。

实现的微观结构特征来源:
  - arXiv 2506.05764: I^1, I^5, 加权中间价变化
  - arXiv 2505.22678: OFI（订单流失衡）
  - arXiv 1512.03492: 队列失衡作为方向性预测因子

所有特征以PyTorch张量计算，支持CUDA加速。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


def compute_obi_level1(bid_vol: Tensor, ask_vol: Tensor) -> Tensor:
    """
    First-level Order Book Imbalance (from arXiv 2506.05764):

      I^1_t = (Q^b_{t,1} - Q^a_{t,1}) / (Q^b_{t,1} + Q^a_{t,1})

    Range: [-1, 1].  Positive = buy pressure, negative = sell pressure.

    一档订单簿失衡（来自arXiv 2506.05764）。
    范围: [-1, 1]。正值=买压，负值=卖压。
    """
    denom: Tensor = (bid_vol + ask_vol).clamp(min=1e-8)
    return (bid_vol - ask_vol) / denom


def compute_obi_multi_level(
    bid_vols: List[Tensor],
    ask_vols: List[Tensor],
) -> Tensor:
    """
    Multi-level aggregate OBI:

      I^L_t = (sum(Q^b) - sum(Q^a)) / (sum(Q^b) + sum(Q^a))

    多档聚合订单簿失衡。
    """
    total_bid: Tensor = torch.stack(bid_vols, dim=0).sum(dim=0)
    total_ask: Tensor = torch.stack(ask_vols, dim=0).sum(dim=0)
    denom: Tensor = (total_bid + total_ask).clamp(min=1e-8)
    return (total_bid - total_ask) / denom


def compute_vpin(
    buy_volume: Tensor,
    sell_volume: Tensor,
    window: int = 20,
) -> Tensor:
    """
    Volume-synchronised Probability of Informed Trading (VPIN):

      VPIN = |V_buy - V_sell| / (V_buy + V_sell)

    Measures toxicity of order flow. High VPIN = adverse selection risk.

    量同步知情交易概率(VPIN)。
    衡量订单流毒性。高VPIN = 逆向选择风险。
    """
    total: Tensor = (buy_volume + sell_volume).clamp(min=1e-8)
    vpin: Tensor = (buy_volume - sell_volume).abs() / total
    return vpin


def compute_trade_imbalance(
    price: Tensor,
    volume: Tensor,
    window: int = 10,
) -> Tensor:
    """
    Trade-based order imbalance using tick rule:
    Classify trades as buys (uptick) or sells (downtick),
    then compute imbalance over rolling window.

    基于tick规则的交易失衡:
    将成交分类为买入（上涨tick）或卖出（下跌tick），
    然后在滚动窗口上计算失衡度。
    """
    tick_dir: Tensor = torch.zeros_like(price)
    tick_dir[1:] = torch.sign(price[1:] - price[:-1])
    # forward-fill zeros (no price change → keep previous direction) / 前向填充零值（无价格变化 → 保持前一方向）
    for i in range(1, tick_dir.size(0)):
        if tick_dir[i] == 0:
            tick_dir[i] = tick_dir[i - 1]

    buy_vol: Tensor = volume * (tick_dir > 0).float()
    sell_vol: Tensor = volume * (tick_dir < 0).float()

    # rolling sum / 滚动求和
    kernel: Tensor = torch.ones(
        1, 1, window, device=price.device, dtype=price.dtype
    )
    pad: int = window - 1

    def _rolling_sum(x: Tensor) -> Tensor:
        first: Tensor = x[0].expand(pad)
        padded: Tensor = torch.cat([first, x])
        return torch.nn.functional.conv1d(
            padded.unsqueeze(0).unsqueeze(0), kernel
        ).squeeze(0).squeeze(0)

    rolling_buy: Tensor = _rolling_sum(buy_vol)
    rolling_sell: Tensor = _rolling_sum(sell_vol)
    denom: Tensor = (rolling_buy + rolling_sell).clamp(min=1e-8)
    return (rolling_buy - rolling_sell) / denom


def compute_price_impact(
    close: Tensor,
    volume: Tensor,
    window: int = 20,
) -> Tensor:
    """
    Amihud illiquidity ratio (rolling):
      ILLIQ = |return| / volume
    Measures price impact per unit volume.

    Amihud非流动性比率（滚动）:
      ILLIQ = |收益率| / 成交量
    衡量单位成交量的价格冲击。
    """
    ret: Tensor = torch.zeros_like(close)
    ret[1:] = (close[1:] / close[:-1].clamp(min=1e-8) - 1.0).abs()
    illiq: Tensor = ret / volume.clamp(min=1e-8)

    # rolling mean / 滚动均值
    kernel: Tensor = torch.ones(
        1, 1, window, device=close.device, dtype=close.dtype
    ) / window
    pad: int = window - 1
    first: Tensor = illiq[0].expand(pad)
    padded: Tensor = torch.cat([first, illiq])
    result: Tensor = torch.nn.functional.conv1d(
        padded.unsqueeze(0).unsqueeze(0), kernel
    ).squeeze(0).squeeze(0)
    return result
