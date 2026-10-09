"""The EV gate of stratum R, and one entry per event under one position at a time.

With ``mu`` and ``sigma`` the predicted forward move over the hold in basis
points, ``c`` the round trip's fees (2 x 5.5 bp at the base tier), ``w_t`` the
cost of walking the current book for the clip and back
(:func:`trading_research.backtest.walk.walk_cost_bp`) and ``beta`` a buffer:

    long  when  mu - k * sigma > c + w_t + beta
    short when -mu - k * sigma > c + w_t + beta

An **event** is a run of rows on which the gate is open, on either side. One
entry is taken per event: a new entry needs the gate to close and reopen. The
market gate and the regime guard act afterwards and only remove decision rows,
so events are found on the EV gate alone and the removed rows simply cannot be
entries.

Units (K-units)
---------------
Every input is in basis points computed from prices or predicted in basis
points. A column that :class:`~trading_research.features.normalise.RollingNormaliser`
rewrote keeps its name, so it is refused by its tag: a z-scored spread fed in
as ``w_t`` would otherwise pass for a cost of a fraction of a basis point. A
negative ``sigma`` or ``w_t`` cannot be a standard deviation or a cost, and is
refused too. Profit and loss is never the sum of ``mu`` over entries; it comes
from the fills of the execution path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.backtest.costs import BYBIT_BASE, FeeTier
from trading_research.features.normalise import normalised_columns


class UnitsError(ValueError):
    """An input to a gate or a cost is not in basis points from prices."""


def check_bp_input(name: str, values: np.ndarray | pd.Series) -> np.ndarray:
    """Refuse a normalised column; return the values as float64."""
    if isinstance(values, pd.Series) and normalised_columns(values):
        raise UnitsError(f"{name} is a normalised column ({values.name!r}); bp from prices only")
    return np.asarray(values, dtype="float64")


@dataclass(frozen=True)
class EVGate:
    """``mu - k * sigma`` against fees, the walk and a buffer, all in bp."""

    k: float
    beta_bp: float
    cost_bp: float = 2.0 * BYBIT_BASE.taker_bp

    def __post_init__(self) -> None:
        if self.k < 0 or self.beta_bp < 0 or self.cost_bp < 0:
            raise ValueError("k, beta_bp and cost_bp must be non-negative")

    @classmethod
    def at_tier(cls, tier: FeeTier, *, k: float, beta_bp: float) -> EVGate:
        """The gate with ``c`` at a tier's taker fee, charged on both sides."""
        return cls(k=k, beta_bp=beta_bp, cost_bp=2.0 * tier.taker_bp)

    def sides(
        self,
        mu_bp: np.ndarray | pd.Series,
        sigma_bp: np.ndarray | pd.Series,
        walk_cost_bp: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """+1, -1 or 0 per row; a NaN input keeps the gate closed."""
        mu = check_bp_input("mu", mu_bp)
        sigma = check_bp_input("sigma", sigma_bp)
        walk = check_bp_input("walk cost", walk_cost_bp)
        if np.any(sigma < 0):
            raise UnitsError("sigma is negative; it is not a standard deviation in bp")
        if np.any(walk < 0):
            raise UnitsError("the walk cost is negative; it is not a cost in bp")
        hurdle = self.cost_bp + walk + self.beta_bp
        with np.errstate(invalid="ignore"):
            long = mu - self.k * sigma > hurdle
            short = -mu - self.k * sigma > hurdle
        out = np.zeros(len(mu), dtype=np.int8)
        out[long] = 1
        out[short] = -1
        return out


def gate_events(sides: np.ndarray) -> np.ndarray:
    """An id per run of open rows (either side), -1 where the gate is closed."""
    open_ = np.asarray(sides) != 0
    starts = open_ & ~np.r_[False, open_[:-1]]
    ids = np.cumsum(starts) - 1
    return np.where(open_, ids, -1).astype(np.int64)


def thin_entries(
    sides: np.ndarray,
    busy_until: np.ndarray,
    events: np.ndarray | None = None,
) -> np.ndarray:
    """Entry rows under one position at a time and, with ``events``, one per event.

    ``busy_until[r]`` is the row of the snapshot that closes a position decided
    at row ``r`` (``len(sides)`` or more where none follows); the next decision
    may be taken from the row after it. A candidate whose position would not
    close inside the data is not taken, and nothing after it is. Candidates of
    an event that already entered are skipped, whether the position is still
    open or not; an event that opens while a position is held can still enter
    at its first free row. The loop jumps from one entry's free row to the next
    candidate, so it runs once per entry rather than once per row.
    """
    sides = np.asarray(sides)
    busy_until = np.asarray(busy_until, dtype=np.int64)
    n = len(sides)
    candidates = np.flatnonzero(sides != 0)
    if events is not None:
        events = np.asarray(events, dtype=np.int64)
        if np.any(events[candidates] < 0):
            raise ValueError("a candidate row lies outside every event")
        last_row = np.full(int(events.max(initial=-1)) + 1, -1, dtype=np.int64)
        np.maximum.at(last_row, events[events >= 0], np.flatnonzero(events >= 0))

    accepted: list[int] = []
    last_event = -1
    i = 0
    while i < len(candidates):
        row = int(candidates[i])
        if events is not None and events[row] == last_event:
            i = int(np.searchsorted(candidates, last_row[last_event] + 1))
            continue
        close = int(busy_until[row])
        if close >= n:
            break
        accepted.append(row)
        if events is not None:
            last_event = int(events[row])
        i = int(np.searchsorted(candidates, close + 1))
    return np.asarray(accepted, dtype=np.int64)
