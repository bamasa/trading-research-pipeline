"""What one day of one row computes, shared by D's scoring and every block read.

A row's day (:mod:`.runs_prints` for the print rows, :mod:`.runs_book` for the
book rows) starts from the frozen (or the fold's) fits' probabilities on every
row of the day's grid and the hold-first thresholds of each cell, and returns
one record of plain values: the true run's daily ``net_bp_trade`` at the
verdict tier and its parts, the placebos on the true run's entry decision
times, the neighbourhood cells, the 110 ms re-run, the brackets, the long-only
variant, every tier of the fee table, the inventory's own P&L, the account's
void conditions, the capacity ratio, the IC, the day's signal tape for the
one-day shift, and, on D, Null A's hour-block sign flips. **A day without a
trade has no value**: every run's daily value is NaN then (the walk tables'
``day_values`` give 0, which :func:`table_bp` turns into NaN), and a run
without a trade on the block scores 0 when the statuses read it.

The random directions and Null A's flips are drawn from
``numpy.random.default_rng([seed, day ordinal])``, so a day's draws do not
depend on which days were read before it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES
from trading_research.spot.costs import FeeNotSourced, FeeTable, fill_fee
from trading_research.spot.plan import NS_PER_S, Cell
from trading_research.strategies.hold_first import HoldFirstRule

BUY_COLUMN, SELL_COLUMN = CLASSES.index(1), CLASSES.index(-1)
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
HOUR_NS = 3_600 * NS_PER_S


def rank_ic(score: np.ndarray, outcome: np.ndarray) -> float:
    """Spearman correlation on rows where both are finite; NaN when undefined."""
    ok = np.isfinite(score) & np.isfinite(outcome)
    if ok.sum() < 3:
        return math.nan
    a = pd.Series(np.asarray(score)[ok]).rank().to_numpy()
    b = pd.Series(np.asarray(outcome)[ok]).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def signal(proba: np.ndarray) -> np.ndarray:
    """``p_long - p_short`` per row (NaN where unscored): the IC's score."""
    return np.asarray(proba[:, BUY_COLUMN] - proba[:, SELL_COLUMN], dtype=np.float64)


def decisions(proba: np.ndarray, rule: HoldFirstRule) -> tuple[np.ndarray, np.ndarray]:
    """Row positions and sides the hold-first rule decides (an unscored row holds)."""
    side = rule.decide_proba(np.nan_to_num(proba, nan=0.0)).astype(np.int64)
    side[~np.isfinite(proba).all(axis=1)] = 0
    rows = np.flatnonzero(side != 0).astype(np.int64)
    return rows, side[rows]


def table_bp(trades: Any) -> float:
    """A run's daily ``net_bp_trade`` from walk tables; NaN on a day without a trade."""
    return float(trades.day_values()["net_bp"]) if trades.trades else math.nan


def random_sides(n: int, seed: int, day: date) -> np.ndarray:
    return np.random.default_rng([seed, day.toordinal()]).choice(np.array([-1, 1]), size=n)


def hour_flips(decision_ns: np.ndarray, start_ns: int, seed: int, day: date) -> np.ndarray:
    """Null A's sign per decision: one random sign per hour block of the day."""
    signs = np.random.default_rng([seed, day.toordinal()]).choice(np.array([-1, 1]), size=24)
    hours = np.clip((np.asarray(decision_ns) - start_ns) // HOUR_NS, 0, 23).astype(np.int64)
    return signs[hours]


def quantiles(values: np.ndarray) -> list[float | None]:
    found = np.asarray(values, dtype=np.float64)
    found = found[np.isfinite(found)]
    if not len(found):
        return [None for _ in QUANTILES]
    return [float(np.quantile(found, q)) for q in QUANTILES]


def tier_values(
    trades: pd.DataFrame, table: FeeTable, instrument: str, fee_bp_at: Any
) -> dict[str, float]:
    """The day's ``net_bp_trade`` at every tier of the table; NaN on a day the tier
    does not cover (a measurement is refused there) or without a trade.
    ``fee_bp_at(tier)`` gives each trade's (entry, exit) taker fee in bp."""
    out: dict[str, float] = {}
    notional = float(trades["notional"].sum()) if len(trades) else 0.0
    for tier in table.tiers(instrument):
        if notional <= 0:
            out[tier] = math.nan
            continue
        try:
            entry_bp, exit_bp = fee_bp_at(tier)
        except FeeNotSourced:
            out[tier] = math.nan
            continue
        size = trades["size"].to_numpy(dtype=np.float64)
        fees = fill_fee(trades["entry_price"].to_numpy() * size, entry_bp) + fill_fee(
            trades["exit_price"].to_numpy() * size, exit_bp
        )
        out[tier] = float((trades["gross"].sum() - fees.sum()) / notional * 1e4)
    out["zero_fees"] = float(trades["gross"].sum() / notional * 1e4) if notional > 0 else math.nan
    return out


@dataclass
class RowModels:
    """What a row's day reads from its fits: per fit (hold, fee) a model, per cell a
    rule, and the scale its inputs are cast with."""

    row: str
    models: Mapping[tuple[float, float], Any]
    rules: Mapping[Cell, HoldFirstRule]
    layout: Any

    def cells(self, cells: Sequence[Cell]) -> list[Cell]:
        return [c for c in cells if c in self.rules and c.fit in self.models]


def shift_rows(grid_ns: np.ndarray, start_ns: int, tod_ns: np.ndarray) -> np.ndarray:
    """Another day's decisions at the same times of day: the first row stamped at or
    after each (a time after the day's last row has none, ``len(grid)``)."""
    return np.searchsorted(grid_ns, start_ns + np.asarray(tod_ns, dtype=np.int64), side="left")
