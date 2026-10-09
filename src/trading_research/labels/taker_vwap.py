"""Labels in time on the replayed book: walked-book taker P&L, forward move, control.

All three read the book forward from a decision row stamped ``t`` over a hold
``h`` in seconds, measured on row stamps, never in rows: "``t + x``" is the
first row stamped at or after ``t + x``.

**The walked-book taker label.** The entry snapshot is the first row at or
after ``t`` + 10 ms (the order's latency), the exit snapshot the first at or
after the entry snapshot's stamp + ``h``. Both are walked for the traded clip
by :func:`trading_research.backtest.walk.walk`, the function the taker executor
prices with. In basis points of the entry snapshot's mid::

    long  = (bid VWAP at exit  - ask VWAP at entry) / mid * 1e4 - 2 * fee
    short = (bid VWAP at entry - ask VWAP at exit)  / mid * 1e4 - 2 * fee

The class is long if the long P&L is positive, short if the short P&L is, hold
otherwise; the two cannot both be positive, since every ask is above every bid.
The fee is a parameter of the label, charged on both sides. A row without an
exit snapshot is unlabelled, and so is one whose entry clip exceeds the visible
book, which the executor would skip.

**The forward move** is the mid at the exit snapshot over the mid at the entry
snapshot, minus one, in bp: a simple return between the same two snapshots.

**The control label** is the log of the mean mid over the 20 rows ending at the
first row at or after ``t + h``, over the mean mid over the 20 rows ending at
``t``, times 1e4. It averages the present as well as the future, so it is partly
known at the decision; it exists only to train the control models.

The purge
---------
Each label reports its exit row, so a split can purge exactly. A bound in rows
also follows from the hold: at most one row falls in each 100 ms interval, so
the rows after ``t`` up to the exit snapshot are at most ``ceil(h / grid) + 3``
(one before the entry snapshot, the interval the hold spans, and the exit row).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.backtest.walk import walk_rows
from trading_research.features.timed import at_or_after, seconds_ns, stamps_ns
from trading_research.labels.directional import BUY, HOLD, SELL

BP: float = 1e4

#: The registered order latency: an order is live 10 ms after its decision.
LATENCY_MS: float = 10.0


def entry_exit_snapshots(
    stamps: np.ndarray, hold_s: float, latency_ms: float = LATENCY_MS
) -> tuple[np.ndarray, np.ndarray]:
    """Entry and exit snapshot indices for every row; ``len(stamps)`` where none."""
    n = len(stamps)
    entry = at_or_after(stamps, stamps + round(latency_ms * 1_000_000))
    exit_ = np.full(n, n, dtype=np.int64)
    known = entry < n
    exit_[known] = at_or_after(stamps, stamps[entry[known]] + seconds_ns(hold_s))
    return entry, exit_


def purge_rows(hold_s: float, grid_ms: int = 100) -> int:
    """Rows after a decision that its label can read, on a grid of ``grid_ms``."""
    return math.ceil(hold_s * 1000 / grid_ms) + 3


def _mid(rows: pd.DataFrame) -> np.ndarray:
    return (rows["bid_price_0"].to_numpy("float64") + rows["ask_price_0"].to_numpy("float64")) / 2


@dataclass(frozen=True)
class WalkedTakerLabel:
    """The three-class walked-book taker label for one hold, clip and fee."""

    hold_s: float
    clip: float = 0.010
    label_fee_bp: float = 0.0
    latency_ms: float = LATENCY_MS
    depth: int = 10

    def __post_init__(self) -> None:
        if not self.hold_s > 0:
            raise ValueError(f"hold_s must be positive, got {self.hold_s}")
        if not self.clip > 0:
            raise ValueError(f"clip must be positive, got {self.clip}")
        if self.label_fee_bp < 0 or self.latency_ms < 0:
            raise ValueError("label_fee_bp and latency_ms must be non-negative")

    @property
    def horizon_s(self) -> float:
        """The least time the label reads past its decision: latency and hold."""
        return self.hold_s + self.latency_ms / 1000.0

    def purge_rows(self, grid_ms: int = 100) -> int:
        return purge_rows(self.hold_s, grid_ms)

    def outcomes(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Snapshots, walks and both sides' P&L in bp for every row."""
        stamps = stamps_ns(rows)
        n = len(stamps)
        entry, exit_ = entry_exit_snapshots(stamps, self.hold_s, self.latency_ms)
        buy = walk_rows(rows, "ask", self.clip, self.depth)
        sell = walk_rows(rows, "bid", self.clip, self.depth)
        mid = _mid(rows)

        known = exit_ < n
        e, x = entry[known], exit_[known]
        fees = 2.0 * self.label_fee_bp
        long_bp = np.full(n, np.nan)
        short_bp = np.full(n, np.nan)
        long_bp[known] = (sell.vwap[x] - buy.vwap[e]) / mid[e] * BP - fees
        short_bp[known] = (sell.vwap[e] - buy.vwap[x]) / mid[e] * BP - fees
        entry_exceeds = np.zeros(n, dtype=bool)
        exit_exceeds = np.zeros(n, dtype=bool)
        entry_exceeds[known] = buy.exceeds[e] | sell.exceeds[e]
        exit_exceeds[known] = buy.exceeds[x] | sell.exceeds[x]
        return pd.DataFrame(
            {
                "entry_index": entry,
                "exit_index": exit_,
                "long_bp": long_bp,
                "short_bp": short_bp,
                "entry_exceeds": entry_exceeds,
                "exit_exceeds": exit_exceeds,
            },
            index=rows.index,
        )

    def __call__(self, rows: pd.DataFrame) -> pd.Series:
        """-1 / 0 / +1, NaN where the outcome is unknown or the entry is skipped."""
        out = self.outcomes(rows)
        label = np.full(len(out), float(HOLD))
        label[out["long_bp"].to_numpy() > 0] = float(BUY)
        label[out["short_bp"].to_numpy() > 0] = float(SELL)
        unknown = out["long_bp"].isna().to_numpy() | out["entry_exceeds"].to_numpy()
        label[unknown] = np.nan
        return pd.Series(label, index=rows.index, name="label")

    def describe(self) -> dict[str, object]:
        return {
            "kind": "walked_taker",
            "hold_s": self.hold_s,
            "clip": self.clip,
            "label_fee_bp": self.label_fee_bp,
            "latency_ms": self.latency_ms,
            "depth": self.depth,
            "horizon_s": self.horizon_s,
        }


def forward_move_bp(rows: pd.DataFrame, hold_s: float, latency_ms: float = LATENCY_MS) -> pd.Series:
    """Mid at the exit snapshot over mid at the entry snapshot, minus one, in bp."""
    stamps = stamps_ns(rows)
    n = len(stamps)
    entry, exit_ = entry_exit_snapshots(stamps, hold_s, latency_ms)
    mid = _mid(rows)
    move = np.full(n, np.nan)
    known = exit_ < n
    move[known] = (mid[exit_[known]] / mid[entry[known]] - 1.0) * BP
    return pd.Series(move, index=rows.index, name=f"forward_move_bp_{hold_s:g}s")


def control_move_bp(rows: pd.DataFrame, hold_s: float, smoothing: int = 20) -> pd.Series:
    """The time-based smoothed control label, in bp; NaN without 20 rows behind."""
    stamps = stamps_ns(rows)
    n = len(stamps)
    smoothed = pd.Series(_mid(rows)).rolling(smoothing, min_periods=smoothing).mean().to_numpy()
    later = at_or_after(stamps, stamps + seconds_ns(hold_s))
    move = np.full(n, np.nan)
    known = later < n
    with np.errstate(invalid="ignore"):
        move[known] = np.log(smoothed[later[known]] / smoothed[known]) * BP
    return pd.Series(move, index=rows.index, name=f"control_move_bp_{hold_s:g}s")
