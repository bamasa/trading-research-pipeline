"""What a run covers: its scale, its calendar and the walk-forward folds.

The registered study runs at :data:`REGISTERED`: every day of every block,
every hold of the catalogue, 999 Null A draws, 200 random-direction draws and
the network at its registered size. The dry run (:data:`DRY_RUN`) runs the same
code on a synthetic venue that serves a few days of each block, scores the
catalogue's cells at two holds, and draws fewer replications; nothing else
differs, so a dry run exercises every stage's code path.

The folds follow the registration's walk-forward over D: for each test month
from February to June, the inner validation is D's days in the seven calendar
days before the month, training is D's days before those, and a purge of the
hold plus the longest feature lookback is taken before the inner validation and
before the test month (on row stamps, :func:`purge_ns`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from trading_research.data import study_access as sa

#: N, the longest feature lookback in minutes (``features.longest_lookback_minutes``).
LONGEST_LOOKBACK_MINUTES = 61
#: D's walk-forward test months.
TEST_MONTHS: tuple[str, ...] = ("2025-02", "2025-03", "2025-04", "2025-05", "2025-06")
INNER_VALIDATION_DAYS = 7
#: The frozen fits: trained before this day, thresholds and checkpoints on it to D's end.
FROZEN_INNER_START = date(2025, 6, 24)
#: A0: trained on 01-01 to 01-27, thresholds on 01-28 to 02-03, run on O.
A0_INNER_START = date(2025, 1, 28)
A0_INNER_END = date(2025, 2, 3)
#: The every-tenth-row store's stride, and the rows a fit takes at most.
TARGET_STRIDE = 10
MAX_TRAINING_ROWS = 4_000_000
NS_PER_S = 1_000_000_000


def purge_ns(hold_s: float, lookback_minutes: int = LONGEST_LOOKBACK_MINUTES) -> int:
    """The purge before an inner validation or a test span: hold plus N."""
    return round((float(hold_s) + 60.0 * lookback_minutes) * NS_PER_S)


def midnight_ns(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * NS_PER_S


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: its training, inner-validation and test days.

    ``inner_start`` and ``test_start`` are the instants the purge is taken
    before (UTC midnights, in ns)."""

    name: str
    train: tuple[date, ...]
    inner: tuple[date, ...]
    test: tuple[date, ...]
    inner_start: date
    test_start: date

    def train_end_ns(self, hold_s: float) -> int:
        """Training rows must be stamped before this."""
        return midnight_ns(self.inner_start) - purge_ns(hold_s)

    def inner_end_ns(self, hold_s: float) -> int:
        """Inner-validation rows must be stamped before this."""
        return midnight_ns(self.test_start) - purge_ns(hold_s)


def _month_start(month: str) -> date:
    year, number = (int(part) for part in month.split("-"))
    return date(year, number, 1)


def _month_end(month: str) -> date:
    start = _month_start(month)
    following = date(start.year + start.month // 12, start.month % 12 + 1, 1)
    return following - timedelta(days=1)


def walk_forward(days: Sequence[date], months: Sequence[str] = TEST_MONTHS) -> list[Fold]:
    """The folds over ``days`` (D's usable days), one per test month."""
    d_start, _ = sa.BLOCKS["D"]
    out = []
    for month in months:
        start, end = _month_start(month), _month_end(month)
        inner_start = start - timedelta(days=INNER_VALIDATION_DAYS)
        train = tuple(d for d in days if d_start <= d < inner_start)
        inner = tuple(d for d in days if inner_start <= d < start)
        test = tuple(d for d in days if start <= d <= end)
        out.append(Fold(month, train, inner, test, inner_start, start))
    return out


def frozen_fold(days: Sequence[date]) -> Fold:
    """The frozen fits' split: train before 06-24, thresholds on 06-24 to 06-30."""
    d_start, d_end = sa.BLOCKS["D"]
    train = tuple(d for d in days if d_start <= d < FROZEN_INNER_START)
    inner = tuple(d for d in days if FROZEN_INNER_START <= d <= d_end)
    return Fold("frozen", train, inner, (), FROZEN_INNER_START, d_end + timedelta(days=1))


def a0_fold(days: Sequence[date]) -> Fold:
    """A0's protocol: trained on 01-01 to 01-27, thresholds on 01-28 to 02-03, run on O."""
    d_start, _ = sa.BLOCKS["D"]
    o_start, o_end = sa.O_WINDOW
    train = tuple(d for d in days if d_start <= d < A0_INNER_START)
    inner = tuple(d for d in days if A0_INNER_START <= d <= A0_INNER_END)
    test = tuple(d for d in days if o_start <= d <= o_end)
    return Fold("A0", train, inner, test, A0_INNER_START, o_start)


def _dry_calendar() -> dict[str, tuple[date, ...]]:
    """The dry run's days: a few of each block, so that every fold has training,
    inner-validation and test days, H has days on both sides of 08-01, and F and P
    have days to read; the reader-test days as registered."""
    d_days = tuple(date(2025, m, d) for m in range(1, 7) for d in (12, 27, 28))
    return {
        "W": (date(2024, 12, 30), date(2024, 12, 31)),
        "D": d_days,
        "H": (date(2025, 7, 8), date(2025, 7, 21), date(2025, 8, 11)),
        "F": (date(2025, 8, 21), date(2025, 9, 15), date(2025, 10, 20)),
        "P": (date(2025, 11, 17), date(2025, 12, 15)),
        sa.READER_TEST: sa.READER_TEST_DAYS,
    }


@dataclass(frozen=True)
class Scale:
    """How much of the registered study a run covers.

    ``calendar`` maps a block to the days the run plans (None: every day of the
    block); ``holds_s`` the holds whose cells are scored (None: every hold).
    The replication counts are the registered ones at :data:`REGISTERED`.
    """

    name: str
    calendar: Mapping[str, tuple[date, ...]] | None = None
    holds_s: tuple[float, ...] | None = None
    null_a_draws: int = 999
    random_direction_draws: int = 200
    taker_flag_sets: int = 200
    simulator_flag_sets: int = 20
    #: Hours of each test month the flag-causality test mutates after (None: every hour).
    causality_hours: int | None = None
    max_training_rows: int = MAX_TRAINING_ROWS
    #: Constructor arguments per model name, over the repository's defaults.
    model_params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Rows of one W day the feature leakage checks run on, and the positions
    #: (evenly spaced) streaming parity recomputes each feature at.
    leakage_rows: int = 12_000
    leakage_positions: int = 240
    #: The two 2024 days stage 0 is timed on.
    timing_days: tuple[date, ...] = sa.TIMING_DAYS
    #: The refit replication's circular shift of the training labels, in days.
    refit_shift_days: int = 28

    @property
    def dry(self) -> bool:
        return self.calendar is not None

    def days(self, block: str) -> tuple[date, ...]:
        """The days of ``block`` (or the reader-test days) this run plans."""
        if self.calendar is not None:
            return tuple(self.calendar.get(block, ()))
        return tuple(sa.block_days(block))

    def null_b_shifts(self, n: int) -> range:
        """The registered 7 to 143 on 150 days; on fewer days the same share of
        the circle is left out at each end."""
        low = max(1, math.ceil(7 * n / 150))
        return range(low, n - low + 1) if n >= 150 else range(low, n - low)

    def params(self, model: str) -> dict[str, Any]:
        return dict(self.model_params.get(model, {}))


REGISTERED = Scale("registered")
DRY_RUN = Scale(
    "dry run",
    calendar=_dry_calendar(),
    holds_s=(20.0, 50.0),
    null_a_draws=49,
    random_direction_draws=20,
    taker_flag_sets=10,
    simulator_flag_sets=5,
    causality_hours=1,
    max_training_rows=200_000,
    model_params={
        "gbm": {"n_estimators": 25, "max_depth": 3},
        "logistic": {"max_iter": 100},
        "tcn": {
            "window": 16,
            "channels": 8,
            "levels": 2,
            "max_epochs": 2,
            "patience": 1,
            "batch_size": 256,
            "device": "cpu",
        },
    },
    leakage_rows=2_000,
    leakage_positions=8,
    refit_shift_days=15,
)
