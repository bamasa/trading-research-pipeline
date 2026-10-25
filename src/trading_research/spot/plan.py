"""What a run of the spot study covers: its scale, the rows and their cells, and D's folds.

The registered study runs at :data:`REGISTERED`: every day of every block, the
registered draws and the repository's model defaults. The dry run
(:data:`DRY_RUN`) runs the same code on a synthetic venue that serves a few days
of each block, with fewer null draws and smaller trees; nothing else differs.

**The rows.** The family's three slots and the two measured rows
(``docs/preregistration/spot_2026.md``, "The slots"), each with its venue, its
grid (prints or book), its inputs and the cells it is fitted or scored at: the
print slots' and S25-BB's neighbourhood is hold {20, 50, 120} s x rate {100, 240}
x label fee {0, 10} bp; S25-T's is the 2025 study's, at label fees {0, 5.5}; the
measured rows run their own cell only (50 s, 240 a day, 0 bp).

**The folds** of D ("Labels, features and fits"): June (trained 05-01 to 05-24,
theta on 05-25 to 05-31), July (05-01 to 06-23, theta on 06-24 to 06-30) and the
frozen fits (05-01 to 07-24, theta on 07-25 to 07-31, scored on H and P). The
purge of the hold plus 61 minutes is taken before each inner validation and each
test month; on prints the fitter also drops a row whose label reads past the
cutoff (:meth:`Fold.train_keep`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np

from trading_research.spot import access as sa
from trading_research.spot.book_inputs import S25_SELECTED
from trading_research.spot.features_prints import PRINT_MEMBERS, TOUCH_PROXY_MEMBERS

NS_PER_S = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_S
#: The longest lookback in minutes: a 60 s window and 3,600 s of winsorising, rounded up.
LOOKBACK_MINUTES = 61
#: The every-tenth-row store's stride, and the rows a fit takes at most.
TARGET_STRIDE = 10
MAX_TRAINING_ROWS = 4_000_000
HOLDS_S: tuple[float, ...] = (20.0, 50.0, 120.0)
RATES: tuple[float, ...] = (100.0, 240.0)
LABEL_FEES_BP: tuple[float, ...] = (0.0, 10.0)
LABEL_FEES_2025_BP: tuple[float, ...] = (0.0, 5.5)
OWN = (50.0, 240.0, 0.0)


def midnight_ns(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * NS_PER_S


def purge_ns(hold_s: float) -> int:
    """The registered purge before an inner validation or a test month: hold plus 61 min."""
    return round((float(hold_s) + 60.0 * LOOKBACK_MINUTES) * NS_PER_S)


@dataclass(frozen=True, order=True)
class Cell:
    """One (hold, rate, label fee) cell of a row."""

    hold_s: float
    rate: float
    fee_bp: float

    @property
    def key(self) -> str:
        return f"hold_s={self.hold_s:g}|trades_per_day={self.rate:g}|label_fee_bp={self.fee_bp:g}"

    @property
    def hold_ns(self) -> int:
        return round(self.hold_s * NS_PER_S)

    @property
    def fit(self) -> tuple[float, float]:
        """The fit a cell's decisions come from: its hold and label fee."""
        return (self.hold_s, self.fee_bp)


def neighbourhood(fees: Sequence[float] = LABEL_FEES_BP) -> tuple[Cell, ...]:
    """The 12 cells, in (hold, rate, fee) order."""
    return tuple(Cell(h, r, f) for h in HOLDS_S for r in RATES for f in fees)


OWN_CELL = Cell(*OWN)


@dataclass(frozen=True)
class Row:
    """A slot of the family or a measured row: where it runs and what it is fitted on."""

    name: str
    instrument: str
    grid: str  # "prints" or "book"
    inputs: tuple[str, ...]
    #: The input set (store) its features come from: a venue's plane and its columns.
    store: str
    cells: tuple[Cell, ...]
    fitted: bool = True

    @property
    def fits(self) -> tuple[tuple[float, float], ...]:
        """(hold, label fee) of every fit, in cell order without repeats."""
        return tuple(dict.fromkeys(c.fit for c in self.cells))

    @property
    def own(self) -> Cell:
        return OWN_CELL


PRINT_INPUTS = tuple(PRINT_MEMBERS)
PROXY_INPUTS = (*PRINT_MEMBERS, *TOUCH_PROXY_MEMBERS)
ROWS: dict[str, Row] = {
    "S25-PF": Row("S25-PF", sa.BINANCE, "prints", PRINT_INPUTS, "binance", neighbourhood()),
    "S25-BB": Row("S25-BB", sa.BYBIT, "book", S25_SELECTED, "book", neighbourhood()),
    "S25-T": Row("S25-T", sa.BYBIT, "book", S25_SELECTED, "book",
                 neighbourhood(LABEL_FEES_2025_BP), fitted=False),
    "S25-PB": Row("S25-PB", sa.BYBIT, "prints", PRINT_INPUTS, "bybit_prints", (OWN_CELL,)),
    "S25-PFT": Row("S25-PFT", sa.BINANCE, "prints", PROXY_INPUTS, "binance", (OWN_CELL,)),
}  # fmt: skip
#: The input sets, the columns each store keeps, and the venue each is computed on.
STORES: dict[str, tuple[str, tuple[str, ...]]] = {
    "binance": (sa.BINANCE, PROXY_INPUTS),
    "bybit_prints": (sa.BYBIT, PRINT_INPUTS),
    "book": (sa.BYBIT, tuple(S25_SELECTED)),
}


def fit_key(row: str, hold_s: float, fee_bp: float) -> str:
    """A fit's file name: ``models/spot_2026/weights/<fit key>`` for the frozen ones."""
    return f"{row}__hold_s={hold_s:g}__label_fee_bp={fee_bp:g}.pkl"


@dataclass(frozen=True)
class Fold:
    """One fit's split of D's usable days; the purge is taken before ``inner_start``
    and before ``test_start`` (UTC midnights), the instants taken as the inner
    validation's first stamp and the fold's cutoff."""

    name: str
    train: tuple[date, ...]
    inner: tuple[date, ...]
    test: tuple[date, ...]
    inner_start: date
    test_start: date

    @property
    def inner_start_ns(self) -> int:
        return midnight_ns(self.inner_start)

    @property
    def cutoff_ns(self) -> int:
        return midnight_ns(self.test_start)

    def train_keep(self, stamps: np.ndarray, reads_until: np.ndarray, hold_s: float) -> np.ndarray:
        """Training rows: stamped before the inner validation less the purge, with a
        label that reads nothing after the inner validation's first stamp."""
        stamps, reads = np.asarray(stamps, np.int64), np.asarray(reads_until, np.int64)
        return (
            (stamps < self.inner_start_ns - purge_ns(hold_s))
            & (reads >= 0)
            & (reads <= self.inner_start_ns)
        )

    def inner_keep(self, stamps: np.ndarray, hold_s: float) -> np.ndarray:
        """Inner-validation rows: stamped from the inner validation's first stamp to the
        cutoff less the purge, with a label that reads nothing after the cutoff. A
        label reads nothing after its own day's end, so that end bounds each row's
        ``reads_until_ns`` exactly."""
        stamps = np.asarray(stamps, np.int64)
        bound = (stamps // NS_PER_DAY + 1) * NS_PER_DAY
        inside = (stamps >= self.inner_start_ns) & (stamps < self.cutoff_ns - purge_ns(hold_s))
        return inside & (bound <= self.cutoff_ns)


#: The registered splits (``models.fits``): (train first, train last), (theta first,
#: theta last), the test month's first and last day (None for the frozen fits).
FOLDS: dict[str, tuple[tuple[date, date], tuple[date, date], tuple[date, date] | None]] = {
    "june": ((date(2026, 5, 1), date(2026, 5, 24)), (date(2026, 5, 25), date(2026, 5, 31)),
             (date(2026, 6, 1), date(2026, 6, 30))),
    "july": ((date(2026, 5, 1), date(2026, 6, 23)), (date(2026, 6, 24), date(2026, 6, 30)),
             (date(2026, 7, 1), date(2026, 7, 31))),
    "frozen": ((date(2026, 5, 1), date(2026, 7, 24)), (date(2026, 7, 25), date(2026, 7, 31)),
               None),
}  # fmt: skip
WALK_FORWARD: tuple[str, ...] = ("june", "july")


def folds(usable: Sequence[date]) -> dict[str, Fold]:
    """The three folds over D's usable days (the frozen fits' cutoff is 2026-08-01)."""
    out = {}
    for name, (train, inner, test) in FOLDS.items():

        def within(span: tuple[date, date]) -> tuple[date, ...]:
            return tuple(d for d in usable if span[0] <= d <= span[1])

        test_start = test[0] if test else inner[1] + timedelta(days=1)
        out[name] = Fold(name, within(train), within(inner), within(test) if test else (),
                         inner[0], test_start)  # fmt: skip
    return out


def _dry_calendar() -> dict[str, tuple[date, ...]]:
    """A few days of each block: every fold has training, theta and test days; one of
    H's five days has sequence gaps (excluded and put back, the venue still readable);
    P's first and last day, for the publication check."""
    d = date
    return {
        sa.READER_TEST: sa.READER_TEST_DAYS,
        "W": (d(2026, 4, 29), d(2026, 4, 30)),
        "D": (d(2026, 5, 1), d(2026, 5, 2), d(2026, 5, 12), d(2026, 5, 20), d(2026, 5, 26),
              d(2026, 5, 30), d(2026, 6, 3), d(2026, 6, 11), d(2026, 6, 19), d(2026, 6, 25),
              d(2026, 6, 29), d(2026, 7, 3), d(2026, 7, 13), d(2026, 7, 21), d(2026, 7, 26),
              d(2026, 7, 30)),
        "H": (d(2026, 8, 8), d(2026, 8, 21), d(2026, 9, 9), d(2026, 9, 28), d(2026, 10, 21)),
        "P": (d(2026, 11, 2), d(2026, 12, 31)),
    }  # fmt: skip


@dataclass(frozen=True)
class Scale:
    """How much of the registered study a run covers (``calendar`` None: every day)."""

    name: str
    calendar: Mapping[str, tuple[date, ...]] | None = None
    null_a_draws: int = 999
    max_training_rows: int = MAX_TRAINING_ROWS
    model_params: Mapping[str, Any] = field(default_factory=dict)
    #: Rows of a W day the feature leakage checks read, and the streaming positions.
    leakage_rows: int = 6_000
    leakage_positions: int = 120
    #: Decisions per day and hold in stage 0's parity test.
    parity_decisions: int = 2_000

    @property
    def dry(self) -> bool:
        return self.calendar is not None

    def days(self, block: str) -> tuple[date, ...]:
        if self.calendar is not None:
            return tuple(self.calendar.get(block, ()))
        return tuple(sa.block_days(block))

    def null_b_shifts(self, n: int) -> range:
        """The registered 7 <= s <= 54 on 61 test days; on fewer days the same share of
        the circle is left out at each end."""
        low = max(1, math.ceil(7 * n / 61))
        return range(low, n - low + 1)


REGISTERED = Scale("registered")
DRY_RUN = Scale(
    "dry run",
    calendar=_dry_calendar(),
    null_a_draws=19,
    max_training_rows=200_000,
    model_params={"n_estimators": 25, "max_depth": 3},
    leakage_rows=1_200,
    leakage_positions=12,
    parity_decisions=300,
)
