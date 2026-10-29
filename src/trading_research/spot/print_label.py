"""The print label (S25-PF, S25-PB, S25-PFT) and the forward move on prints.

**The label** (the registration, "Labels, features and fits"). At a print-grid
row stamped ``t``, with hold ``h``, for the clip: the long buys by the print
rule at ``t`` + 10 ms; its exit, a sell, is decided at the first row stamped at
or after the entry fill's print + ``h`` and fills by the same rule, never at a
print before the entry's. The short is symmetric. Both are
:func:`trading_research.spot.fills.round_trips` with ``flatten=False``, the
function the executor and the walk tables call. In basis points::

    long  = (exit sell price - entry buy price) / mid proxy at the long's entry fill * 1e4 - 2 * fee
    short = (entry sell price - exit buy price) / mid proxy at the short's entry fill * 1e4 - 2 * fee

The class is long if the long P&L is positive, short if the short P&L is, the
larger of the two when both are (prints can make both positive), hold on a
tie or otherwise. A row whose long or short has no entry or exit fill before
its day ends has no label and is not a training row; so has a row whose mid
proxy at an entry fill is not yet defined (a side that has not printed that
day). The mid proxy at a fill is the mean of the last buyer- and
seller-initiated prints stamped at or before the fill's print, every print of
the day counting (:func:`trading_research.spot.grid.proxies_at`).

**The forward move** (the IC on prints): from the mid proxy at the entry fill
to the mid proxy at the first row stamped at or after the fill + ``h``, in bp.
Its entry fill is side-free, the first reachable print of either side stamped
at or after ``t`` + 10 ms, so the move is a property of the row and not of a
signal's sign.

**The purge.** ``horizon_s`` is the least the label reads past its decision;
a thin day's entry can wait far longer, so each row reports
``reads_until_ns``, the later of its two exit fills' stamps, for an exact purge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.features.timed import seconds_ns
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.spot.fills import PrintFillSettings, PrintTape, round_trips
from trading_research.spot.grid import proxies_at

BP: float = 1e4
LATENCY_MS: float = 10.0


def mid_proxy_at(tape: PrintTape, stamps_ns: np.ndarray) -> np.ndarray:
    """The mid proxy right after every print of the day stamped at or before each stamp."""
    return proxies_at(tape.ts, tape.price, tape.side, np.asarray(stamps_ns, dtype=np.int64))[2]


def classify(long_bp: np.ndarray, short_bp: np.ndarray) -> np.ndarray:
    """-1 / 0 / +1 by the registered rule; NaN where either side is unknown."""
    long_bp, short_bp = np.asarray(long_bp, float), np.asarray(short_bp, float)
    label = np.full(len(long_bp), float(HOLD))
    up, down = long_bp > 0, short_bp > 0
    label[up & ~down] = float(BUY)
    label[down & ~up] = float(SELL)
    both = up & down
    label[both & (long_bp > short_bp)] = float(BUY)
    label[both & (short_bp > long_bp)] = float(SELL)
    label[np.isnan(long_bp) | np.isnan(short_bp)] = np.nan
    return label


@dataclass(frozen=True)
class PrintLabel:
    """The three-class print label for one hold, clip and label fee."""

    hold_s: float
    clip: float = 0.010
    label_fee_bp: float = 0.0
    latency_ms: float = LATENCY_MS

    def __post_init__(self) -> None:
        if not self.hold_s > 0:
            raise ValueError(f"hold_s must be positive, got {self.hold_s}")
        if self.label_fee_bp < 0 or self.latency_ms < 0:
            raise ValueError("label_fee_bp and latency_ms must be non-negative")

    @property
    def settings(self) -> PrintFillSettings:
        return PrintFillSettings(clip=self.clip, latency_ns=round(self.latency_ms * 1_000_000))

    @property
    def horizon_s(self) -> float:
        """The least time the label reads past its decision: latency and hold."""
        return self.hold_s + self.latency_ms / 1000.0

    def outcomes(self, tape: PrintTape, grid_ns: np.ndarray) -> pd.DataFrame:
        """Both sides' fills and P&L in bp, after the label fee, for every grid row."""
        grid = np.asarray(grid_ns, dtype=np.int64)
        hold, settings = seconds_ns(self.hold_s), self.settings
        long = round_trips(tape, grid, grid, 1, hold, flatten=False, settings=settings)
        short = round_trips(tape, grid, grid, -1, hold, flatten=False, settings=settings)
        fees = 2.0 * self.label_fee_bp
        columns: dict[str, np.ndarray] = {}
        for name, trips, sign in (("long", long, 1.0), ("short", short, -1.0)):
            done = trips.complete
            mid = np.full(len(grid), np.nan)
            mid[done] = mid_proxy_at(tape, trips.entry.print_ns[done])
            pnl = sign * (trips.exit.rule - trips.entry.rule) / mid * BP - fees
            columns[f"{name}_bp"] = np.where(done, pnl, np.nan)
            columns[f"{name}_entry_ns"] = np.where(done, trips.entry.print_ns, -1)
            columns[f"{name}_exit_ns"] = np.where(done, trips.exit.print_ns, -1)
        known = ~(np.isnan(columns["long_bp"]) | np.isnan(columns["short_bp"]))
        reads = np.maximum(columns["long_exit_ns"], columns["short_exit_ns"])
        columns["reads_until_ns"] = np.where(known, reads, -1)
        return pd.DataFrame(columns)

    def __call__(self, tape: PrintTape, grid_ns: np.ndarray) -> pd.Series:
        """-1 / 0 / +1 per grid row, NaN where the row has no label."""
        out = self.outcomes(tape, grid_ns)
        return pd.Series(
            classify(out["long_bp"].to_numpy(), out["short_bp"].to_numpy()), name="label"
        )

    def describe(self) -> dict[str, object]:
        return {
            "kind": "print",
            "hold_s": self.hold_s,
            "clip": self.clip,
            "label_fee_bp": self.label_fee_bp,
            "latency_ms": self.latency_ms,
            "horizon_s": self.horizon_s,
        }


def forward_move_bp(
    tape: PrintTape, grid_ns: np.ndarray, hold_s: float, latency_ms: float = LATENCY_MS
) -> np.ndarray:
    """The IC's forward move per grid row (module docstring); NaN where it is unknown."""
    grid = np.asarray(grid_ns, dtype=np.int64)
    reachable = np.flatnonzero(~tape.rpi)
    times = tape.ts[reachable]
    j = np.searchsorted(times, grid + round(latency_ms * 1_000_000), side="left")
    move = np.full(len(grid), np.nan)
    has = j < len(times)
    fill = times[j[has]]
    later = np.searchsorted(grid, fill + seconds_ns(hold_s), side="left")
    ok = later < len(grid)
    rows = np.flatnonzero(has)[ok]
    start = mid_proxy_at(tape, fill[ok])
    end = mid_proxy_at(tape, grid[later[ok]])
    move[rows] = (end / start - 1.0) * BP
    return move


def label_column(hold_s: float, fee_bp: float) -> str:
    return f"label_{hold_s:g}_{fee_bp:g}"


def forward_column(hold_s: float) -> str:
    return f"forward_{hold_s:g}"


def reads_column(hold_s: float) -> str:
    """The stamp of the last print (or book row) the hold's labels read, for its purge."""
    return f"reads_until_ns_{hold_s:g}"


def print_targets(
    tape: PrintTape,
    grid_ns: np.ndarray,
    holds_s: Sequence[float],
    fees_bp: Sequence[float],
    *,
    clip: float = 0.010,
    at: np.ndarray | None = None,
) -> pd.DataFrame:
    """Every target at the grid positions ``at`` (all rows by default).

    For each hold, one walk of both sides serves every label fee (a fee ``f``
    takes ``2 f`` off both sides' P&L), as the 2025 targets do; the forward
    move; and the stamp each hold's labels read up to (``reads_until_ns_<hold>``),
    which purges that hold's fits, with the latest over the holds beside them.
    """
    grid = np.asarray(grid_ns, dtype=np.int64)
    rows = np.arange(len(grid)) if at is None else np.asarray(at, dtype=np.int64)
    columns: dict[str, np.ndarray] = {"row": rows, "timestamp": grid[rows]}
    reads = np.full(len(rows), -1, dtype=np.int64)
    for hold in holds_s:
        out = PrintLabel(hold, clip=clip).outcomes(tape, grid).iloc[rows]
        long_bp, short_bp = out["long_bp"].to_numpy(), out["short_bp"].to_numpy()
        for fee in fees_bp:
            columns[label_column(hold, fee)] = classify(long_bp - 2 * fee, short_bp - 2 * fee)
        columns[forward_column(hold)] = forward_move_bp(tape, grid, hold)[rows]
        columns[reads_column(hold)] = out["reads_until_ns"].to_numpy(dtype=np.int64)
        reads = np.maximum(reads, columns[reads_column(hold)])
    columns["reads_until_ns"] = reads
    return pd.DataFrame(columns)
