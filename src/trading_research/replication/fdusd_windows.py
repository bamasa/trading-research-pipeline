"""The replication's two windows and its days as the spot loader reads them.

S25-PF's cell (hold 50 s, 240 trades a day, label fee 0) and its nine print
members on the print grid (``spot.features_prints.day_frame``), reused by
import. Each window is the spot study's :class:`~trading_research.spot.plan.Fold`
over its own dates, so its purges are the spot study's: the hold plus 61
minutes before the inner validation and before the test days, and a training
label that reads past the inner validation's first stamp dropped.

Warm and cold as in the spot study: a day whose previous calendar day is a
usable replication day reads that day's prints as history; any other day
starts cold and its first 61 minutes are not scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from trading_research.replication.fdusd_fetch import ROOT, ReplicationAccess, span_days
from trading_research.spot import days as dy
from trading_research.spot import loader as ld
from trading_research.spot.archives import BINANCE_PRINTS
from trading_research.spot.features_prints import day_frame
from trading_research.spot.fills import PrintTape
from trading_research.spot.plan import (
    LOOKBACK_MINUTES,
    NS_PER_S,
    OWN_CELL,
    PRINT_INPUTS,
    Fold,
    midnight_ns,
)
from trading_research.spot.print_taker import print_tape

HOLD_S, RATE, LABEL_FEE_BP = OWN_CELL.hold_s, OWN_CELL.rate, OWN_CELL.fee_bp
INPUTS: tuple[str, ...] = PRINT_INPUTS
LAYOUT = dy.layout("binance", None)


@dataclass(frozen=True)
class Window:
    """Training days, the inner validation (thresholds) and the test days, inclusive."""

    name: str
    train: tuple[date, date]
    inner: tuple[date, date]
    test: tuple[date, date]

    def fold(self) -> Fold:
        """The spot study's fold: purges before ``inner[0]`` and before ``test[0]``."""
        return Fold(self.name, tuple(span_days(*self.train)), tuple(span_days(*self.inner)),
                    tuple(span_days(*self.test)), self.inner[0], self.test[0])  # fmt: skip


WINDOWS: dict[str, Window] = {
    "A": Window("A", (date(2024, 11, 5), date(2025, 1, 14)), (date(2025, 1, 15), date(2025, 1, 21)),
                (date(2025, 2, 4), date(2025, 2, 26))),
    "B": Window("B", (date(2025, 5, 1), date(2025, 5, 31)), (date(2025, 6, 1), date(2025, 6, 7)),
                (date(2025, 6, 8), date(2025, 6, 30))),
}  # fmt: skip


@dataclass(frozen=True)
class Day:
    """One day's prints as the spot loader cut them, warm or cold."""

    prints: ld.PrintsDay
    warm: bool
    history: pd.DataFrame | None

    @property
    def day(self) -> date:
        return self.prints.day

    @property
    def usable(self) -> bool:
        return self.prints.usable and self.prints.frame is not None

    @property
    def scored_from(self) -> int:
        start = midnight_ns(self.day)
        return start if self.warm else start + LOOKBACK_MINUTES * 60 * NS_PER_S

    def features(self) -> pd.DataFrame:
        assert self.prints.frame is not None
        return day_frame(self.prints.frame, names=INPUTS, history=self.history,
                         registry=dy.registry())  # fmt: skip

    def tape(self) -> PrintTape:
        return print_tape(self.prints)


def load(day: date, root: Path | str = ROOT) -> ld.PrintsDay:
    return ld.load_prints_day(BINANCE_PRINTS, day, root, access=ReplicationAccess())


def open_day(day: date, root: Path | str = ROOT) -> Day:
    before = day - timedelta(days=1)
    previous = load(before, root) if before in ReplicationAccess().days else None
    warm = bool(previous is not None and previous.usable and previous.frame is not None)
    return Day(load(day, root), warm, previous.frame if warm and previous else None)
