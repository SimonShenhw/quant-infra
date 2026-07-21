"""
Funding Rate Factor — 资金费率因子。

Perpetual futures funding rate: a crowding/carry signal. Extreme positive
funding = longs crowded and paying to hold (reversal-short pressure); extreme
negative = shorts crowded (reversal-long). Funding is also one of the few
SLOW signals in crypto (8h cadence) — the same data feeds the v13 carry sleeve.
永续合约资金费率：拥挤度/carry 信号。极端正值=多头拥挤、持仓付费（反转做空压力）；
极端负值=空头拥挤（反转做多）。funding 也是 crypto 少数"慢信号"之一（8 小时更新），
同一数据还驱动 v13 的 carry sleeve。

Real data is PREFERRED: loaded from funding_rates.db (populated by
data/funding_archive_downloader.py from data.binance.vision). The caller
passes a pre-aligned 1h funding tensor via `extras["funding"]`.
优先使用真实数据：来自 funding_rates.db（由 data/funding_archive_downloader.py
从 data.binance.vision 落库），调用方经 `extras["funding"]` 传入预对齐的 1h 张量。

If real data is unavailable, falls back to an OHLCV-derived proxy:
  funding_proxy = (close - open) / ATR * volume_ratio
真实数据不可用时退化为 OHLCV proxy。

TRAIN/SERVE CAVEAT (REVIEW_2026-06-10 H-4): train and serve MUST use the same
source. v11-era paper trading trained on (broken) real funding but served the
proxy — the same input channel carried two different physical quantities.
If extras is supplied in training, it must be supplied at inference too.
Also note funding_rates.db only covers 2024-09+; extended-window research
(2021+) drops this factor entirely rather than split the channel's semantics.
训练/推理一致性警告（REVIEW H-4）：两端必须同源。v11 时期训练用（已损坏的）真实
funding、模拟盘却用 proxy——同一输入通道喂了两种物理量。训练传了 extras，推理也
必须传。另外 funding_rates.db 仅覆盖 2024-09 之后；扩窗研究（2021+）直接剔除本
因子，而不是让通道语义分裂。
"""
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from factors.base import BaseFactor, register_factor
from model.features import compute_sma, compute_ema


@register_factor
class FundingRate(BaseFactor):
    name = "funding_rate"

    def compute(
        self,
        open_: Tensor, high: Tensor, low: Tensor,
        close: Tensor, volume: Tensor,
        extras: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        # Real funding rate path (preferred) / 优先使用真实资金费率
        if extras is not None and "funding" in extras:
            f: Tensor = extras["funding"]
            # length must match bar count — a misaligned tensor silently falls
            # through to the proxy, which is exactly the H-4 failure mode
            # 长度必须与 bar 数一致——错位张量会静默落入 proxy，正是 H-4 的失效模式
            if f.numel() == close.numel():
                return f.to(close.device, dtype=close.dtype)

        # Proxy fallback (OHLCV-only): signed intrabar move in ATR units,
        # amplified by abnormal volume — a rough "directional pressure" stand-in
        # 后备 OHLCV proxy：以 ATR 为单位的带符号日内动量，乘以放量倍数——
        # 粗略代理"方向性压力"
        tr = high - low
        atr = compute_ema(tr, 14).clamp(min=1e-8)
        direction = (close - open_) / atr
        vol_sma = compute_sma(volume, 20).clamp(min=1e-8)
        vol_ratio = volume / vol_sma
        return direction * vol_ratio
