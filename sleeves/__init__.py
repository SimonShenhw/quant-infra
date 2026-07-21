"""Sleeve framework: signal sources + shared banding + portfolio ledgers.
sleeve 框架：信号源 + 共享 banding + 组合账本。(ROADMAP Phase 1)

WHAT — the package run_paper_daily.py (live) and the backtest/research
scripts both import for everything between "scores" and "ledger rows":
  banding.py  banded top-K construction — THE single shared implementation
  base.py     Sleeve interface + ModelSleeve / CarrySleeve (live signals)
  book.py     PortfolioBook / ContinuousBook (frozen-schema ledger engines)

WHY a package: ROADMAP Phase 1's deliberately minimal refactor — exactly two
abstractions (Sleeve, PortfolioBook), no more. Extracted so v14 multi-sleeve
composition reuses one scoring/ledger contract, and so backtest and paper
trading execute the SAME banding code (the engine crosscheck attributed part
of its ±1pp divergence to duplicated implementations). The engine/ EventBus
layer is intentionally untouched.

INVARIANT: everything here is on the live path writing the pre-registered
September-gate evidence — run tests/run_invariants.py before any change.
中文：本包是 live 与回测共用的“打分→账本”层；只抽象两件事（克制式重构）；
共享 banding 消除双实现漂移（±1pp 引擎分歧的来源之一）；改动前必跑不变量
测试（账本即九月 gate 预注册证据）。"""
from sleeves.banding import banded_targets, banded_update_symbols
from sleeves.base import CarrySleeve, ModelSleeve, Sleeve
from sleeves.book import ContinuousBook, PortfolioBook

__all__ = ["banded_targets", "banded_update_symbols",
           "Sleeve", "ModelSleeve", "CarrySleeve",
           "PortfolioBook", "ContinuousBook"]
