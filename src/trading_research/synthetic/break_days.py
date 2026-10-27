"""Book days with one regime break at a known moment, and the same days without it.

What is planted
---------------
The stream of :mod:`.book_days` run continuously over whole days (the log mid
carries from one day to the next), and from ``break_at_s`` seconds after the
first midnight onward a different regime: every part of the return (the
unpredictable noise and the planted drifts alike) ``vol_ratio`` times larger,
the spread at ``spread_ticks_after`` ticks, and every planted drift's sign times
``edge_after`` (``-1`` flips the edge, ``0`` switches it off). ``truth.json`` records the break's UTC time to the snapshot.

The null
--------
``null=True`` draws the same days from the same seed with no break: the regime
before the break holds throughout.

What it can prove, and what it cannot
-------------------------------------
It can show how long the repository's detectors take to flag a change of known
size and kind, and how often they flag one where none was planted, on data
whose only change is the planted one. It cannot show how they behave on a
market, whose changes are gradual, several at once and of no known time, nor
that a flag would have helped a strategy there.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from trading_research.synthetic.book_days import (
    BookDataset,
    BookDaysSpec,
    Regime,
    generate,
    planted_truth,
)
from trading_research.synthetic.common import NS_PER_S, day_list, day_start_ns

GENERATOR = "trading_research.synthetic.break_days.v1"

PROVES = (
    "How long the repository's detectors take to flag a change of known size and time, "
    "and how often they flag one where none was planted."
)
DOES_NOT_PROVE = (
    "How the detectors behave on a market, whose changes are gradual and of no known "
    "time, or that a flag would have helped a strategy."
)


def _whole_days() -> BookDaysSpec:
    return BookDaysSpec(days=2, session_s=86_400)


@dataclass(frozen=True)
class BreakDaysSpec:
    """A book-days spec run over whole days, and the break planted in it."""

    book: BookDaysSpec = field(default_factory=_whole_days)
    #: Seconds after the first midnight at which the new regime starts.
    break_at_s: float = 30 * 3600.0
    vol_ratio: float = 1.6
    spread_ticks_after: int = 4
    edge_after: float = -1.0
    null: bool = False

    def __post_init__(self) -> None:
        if self.book.session_s != 86_400:
            raise ValueError("a break stream runs over whole days: session_s must be 86400")
        if not 0 < self.break_at_s < self.book.days * 86_400:
            raise ValueError("the break must fall inside the generated days")
        if self.vol_ratio <= 0 or self.spread_ticks_after < 1:
            raise ValueError("the volatility ratio and the spread after must be positive")

    @property
    def break_ns(self) -> int:
        first = day_list(self.book.start, 1)[0]
        step_ns = self.book.step_ms * 1_000_000
        raw = day_start_ns(first) + int(self.break_at_s * NS_PER_S)
        return raw - raw % step_ns

    def with_book(self, **changes: object) -> BreakDaysSpec:
        return replace(self, book=replace(self.book, **changes))  # type: ignore[arg-type]


def generate_breaks(spec: BreakDaysSpec | None = None) -> BookDataset:
    """Generate the stream; with ``null`` the same draws and no break."""
    spec = spec or BreakDaysSpec()
    at = datetime.fromtimestamp(spec.break_ns / NS_PER_S, tz=UTC)
    planted = {
        "break_at_utc": None if spec.null else at.isoformat(),
        "break_ns": None if spec.null else spec.break_ns,
        "vol_ratio": 1.0 if spec.null else spec.vol_ratio,
        "spread_ticks_before": spec.book.spread_ticks,
        "spread_ticks_after": spec.book.spread_ticks if spec.null else spec.spread_ticks_after,
        "edge_after": 1.0 if spec.null else spec.edge_after,
        "size_bp_after": spec.book.size_bp
        * (1.0 if spec.null else spec.vol_ratio * spec.edge_after),
    }
    base = planted_truth(spec.book, GENERATOR)
    truth = replace(
        base,
        null=spec.null,
        planted={**base.planted, **planted},
        spec={"break": dict(planted.items()), "book": base.spec},
        proves=PROVES,
        does_not_prove=DOES_NOT_PROVE,
    )
    regime = Regime(spec.vol_ratio, spec.spread_ticks_after, spec.edge_after)
    if spec.null:
        return generate(spec.book, truth=truth)
    return generate(spec.book, break_ns=spec.break_ns, regime=regime, truth=truth)
