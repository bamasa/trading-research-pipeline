"""Pairs trading on bars: a cointegration screen and a walk-forward backtest of two legs.

* :mod:`.stats` — the hedge ratio and half-life the ``spread`` strategy plugin
  fits (:mod:`trading_research.strategies.spread`), an Engle-Granger test with
  a simulated null, the correlation of returns, Holm's adjusted p-values;
* :mod:`.data` — bars of several symbols on one gap-free clock, or a refusal;
* :mod:`.costs` — the taker fee and slippage of one fill, from the fee tables;
* :mod:`.screen` — every pair on a formation window, corrected and ranked;
* :mod:`.backtest` — formation then trading, stepped; both legs charged;
* :mod:`.report` — the pages the ``trp pairs`` commands write.

See ``docs/pairs.md``.
"""

from __future__ import annotations

from trading_research.pairs.backtest import PairRule, PairsResult, backtest
from trading_research.pairs.costs import LegCosts, leg_costs
from trading_research.pairs.data import AlignmentError, align, load
from trading_research.pairs.screen import screen
from trading_research.pairs.stats import PairsError, engle_granger, half_life

__all__ = [
    "AlignmentError",
    "LegCosts",
    "PairRule",
    "PairsError",
    "PairsResult",
    "align",
    "backtest",
    "engle_granger",
    "half_life",
    "leg_costs",
    "load",
    "screen",
]
