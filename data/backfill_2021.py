"""
Extended-window backfill: 2021-01 .. 2024-08 monthly 5m klines.
数据扩窗：2021-01 至 2024-08 的月度 5m K线（ROADMAP Phase 2）。

WHY extend to 2021: v13 trains on ~18 months (2024-09+) — one regime.
The extended-window research (tools/research_extended_window.py) needs the
2021 bull / 2022 crash / 2023 chop to test whether the signal survives
regime diversity, so this script grows the SAME lake backwards in time.
为什么扩到 2021：v13 只在约 18 个月（2024-09 起）上训练——单一市场状态。
扩窗研究需要 2021 牛市/2022 崩盘/2023 震荡来检验信号跨状态存活性，
本脚本把同一个数据湖向过去生长。

Reuses archive_downloader.download_and_convert, inheriting its two key
properties: skip-if-exists idempotence (a crashed backfill just reruns —
monthly parquet partitions already on disk are skipped), and silent
404-tolerance (APT/ARB/OP/SUI etc. listed mid-window, so their early months
legitimately don't exist; reported as "months unavailable", not failures).
Note: archives in this range use MILLISECOND timestamps (the µs switch is
2025-01+); lake_loader normalizes units at load time anyway, so mixed-unit
partitions in one lake are safe.
复用 download_and_convert，继承其两个关键性质：已存在即跳过的幂等性
（回填中断直接重跑——磁盘上已有的月度 parquet 分区自动跳过）、静默容忍 404
（APT/ARB/OP/SUI 等窗口中途才上市，早期月份本就不存在；按"缺失月数"汇报而
非报错）。注意：该区间归档为毫秒时间戳（微秒切换在 2025-01+）；lake_loader
在加载时统一归一化，同一湖内混单位分区是安全的。

Usage: python data/backfill_2021.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Tuple

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.archive_downloader import (
    BASE_URL, SEMAPHORE_LIMIT, TOP_20_SYMBOLS, download_and_convert,
)

START = (2021, 1)
END = (2024, 8)  # inclusive; 2024-09 onward already in the lake / 起止均含


def month_range(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    out = []
    y, m = start
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


async def main() -> None:
    months = month_range(START, END)
    print(f"[Backfill] {len(TOP_20_SYMBOLS)} symbols x {len(months)} months "
          f"({START[0]}-{START[1]:02d} .. {END[0]}-{END[1]:02d})")
    t0 = time.time()
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    per_symbol = {s: 0 for s in TOP_20_SYMBOLS}
    missing = {s: 0 for s in TOP_20_SYMBOLS}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sym in TOP_20_SYMBOLS:
            for (y, m) in months:
                url = f"{BASE_URL}/klines/{sym}/5m/{sym}-5m-{y}-{m:02d}.zip"
                tasks.append((sym, y, m, url))
        batch = 20
        for i in range(0, len(tasks), batch):
            chunk = tasks[i:i + batch]
            res = await asyncio.gather(*[
                download_and_convert(session, sem, url, sym, y, m, "klines", "5m")
                for sym, y, m, url in chunk
            ], return_exceptions=True)
            for (sym, y, m, _), r in zip(chunk, res):
                if isinstance(r, Exception):
                    print(f"  [FAIL] {sym} {y}-{m:02d}: {r}")
                    continue
                _, n = r
                if n > 0:
                    per_symbol[sym] += n
                else:
                    missing[sym] += 1
            done = min(i + batch, len(tasks))
            print(f"  [{done}/{len(tasks)}] {sum(per_symbol.values()):,} rows", flush=True)

    print(f"\n[Backfill] done in {time.time()-t0:.0f}s")
    for sym in sorted(per_symbol):
        note = f" ({missing[sym]} months unavailable)" if missing[sym] else ""
        print(f"  {sym:12s} {per_symbol[sym]:>10,} rows{note}")


if __name__ == "__main__":
    asyncio.run(main())
