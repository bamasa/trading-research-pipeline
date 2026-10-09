"""One instrument's book days for the 2025 study: read, checked, and excluded by rule.

A day is excluded for what its book shows, counted by reason, and never
repaired:

- **non-positive price**: a ``positive_prices`` error in
  :func:`trading_research.data.validate.validate_book`'s report. The function
  returns its findings and does not raise, so the loader reads the report;
- **off-grid price**: a price that is not a multiple of the day's own tick,
  derived from its book by :func:`trading_research.market_making.events.derive_tick`
  as the simulator's day loader derives it. A day whose tick differs from the
  venue's specification, or is not a multiple of it, is noted and kept;
- **sequence gaps**: more than ``max_sequence_gaps``, the 95th percentile of
  the development block's per-day counts rounded up (:func:`max_sequence_gaps`),
  one value for every block;
- **no book file**: the day is not on disk (the fetch log says why).

Every other finding of the report is kept by check and severity, so a day's
record can be set beside another archive's.

The clip is fixed per instrument (:data:`CLIP`) and must sit on the venue's lot
grid (:class:`StudyInstrument`).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Base-currency size of one trade, per instrument.
CLIP: dict[str, float] = {"BTCUSDT": 0.010}

#: The venue's public instrument specification, no credentials.
INSTRUMENT_INFO_URL = (
    "https://api.bybit.com/v5/market/instruments-info?category=linear&symbol={symbol}"
)

NO_BOOK = "no_book_file"
NON_POSITIVE_PRICE = "non_positive_price"
OFF_GRID_PRICE = "off_grid_price"
SEQUENCE_GAPS = "sequence_gaps"

#: Grid tolerance, as a share of the step: the arrays are floats.
_GRID_TOLERANCE = 1e-6


def clip_for(symbol: str) -> float:
    if symbol not in CLIP:
        raise KeyError(f"no clip registered for {symbol}; registered: {sorted(CLIP)}")
    return CLIP[symbol]


def on_grid(value: float, step: float) -> bool:
    """True when ``value`` is a whole multiple of ``step``, to float tolerance."""
    units = value / step
    return abs(units - round(units)) <= _GRID_TOLERANCE * max(1.0, abs(units))


@dataclass(frozen=True)
class StudyInstrument:
    """An instrument as the study trades it: its tick and lot, and its clip."""

    symbol: str
    tick: float
    lot: float
    clip: float

    def __post_init__(self) -> None:
        if not (self.tick > 0 and self.lot > 0 and self.clip > 0):
            raise ValueError(f"tick, lot and clip must be positive: {self}")
        if not on_grid(self.clip, self.lot):
            raise ValueError(f"the clip {self.clip} is not a multiple of the lot {self.lot}")

    @classmethod
    def from_spec(cls, symbol: str, tick: float, lot: float) -> StudyInstrument:
        return cls(symbol=symbol, tick=tick, lot=lot, clip=clip_for(symbol))


def parse_instrument_info(payload: dict[str, Any], symbol: str) -> tuple[float, float]:
    """The tick and lot steps from the venue's instrument-info response."""
    if payload.get("retCode") != 0:
        raise ValueError(f"venue returned retCode={payload.get('retCode')!r}")
    for row in payload.get("result", {}).get("list", []):
        if row.get("symbol") == symbol:
            tick = float(row["priceFilter"]["tickSize"])
            lot = float(row["lotSizeFilter"]["qtyStep"])
            return tick, lot
    raise KeyError(f"{symbol} is not in the instrument list")


def fetch_instrument(symbol: str, *, timeout: float = 30.0, opener: Any = None) -> StudyInstrument:
    """Today's specification of ``symbol`` from the venue, with its registered clip."""
    import json
    import urllib.request

    open_url = opener or urllib.request.urlopen
    with open_url(INSTRUMENT_INFO_URL.format(symbol=symbol), timeout=timeout) as response:
        payload = json.loads(response.read())
    tick, lot = parse_instrument_info(payload, symbol)
    return StudyInstrument.from_spec(symbol, tick, lot)


def max_sequence_gaps(counts: Sequence[int]) -> int:
    """The 95th percentile of per-day gap counts, rounded up.

    Numpy's default (linear) interpolation between order statistics; the value
    is rounded to nine decimals before rounding up, so float noise in the
    interpolation (3.0000000000000004) does not add a gap.
    """
    if not len(counts):
        raise ValueError("no per-day gap counts to take a percentile of")
    value = float(np.percentile(np.asarray(counts, dtype=np.float64), 95))
    return math.ceil(round(value, 9))


@dataclass(frozen=True)
class BookDay:
    """One book day as the study's loader found it."""

    symbol: str
    day: date
    frame: pd.DataFrame | None
    sequence_gaps: int | None
    tick: float | None
    #: Every finding of the validation report: check -> severity -> rows
    #: affected (at least one per finding, so a descriptive one shows).
    findings: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Reasons the day is excluded from every verdict, if any.
    exclusions: tuple[str, ...] = ()
    #: Kept facts that exclude nothing (a tick unlike the specification's).
    notes: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return not self.exclusions


def _findings(report: Iterable[Any]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for finding in report:
        by_severity = out.setdefault(finding.check, {})
        severity = str(finding.severity.value)
        by_severity[severity] = by_severity.get(severity, 0) + max(int(finding.count), 1)
    return out


def _day_tick(frame: pd.DataFrame, depth: int) -> tuple[float | None, bool]:
    """The day's own tick, and whether every price sits on it."""
    from trading_research.market_making.events import OffGridPrice, derive_tick, to_ticks

    columns = [f"{side}_price_{level}" for side in ("bid", "ask") for level in range(depth)]
    distinct = np.concatenate([pd.unique(frame[c].to_numpy(dtype=np.float64)) for c in columns])
    try:
        tick = derive_tick(distinct)
        for column in columns:
            to_ticks(frame[column].to_numpy(dtype=np.float64), tick)
    except OffGridPrice:
        return None, False
    return tick, True


def load_book_day(
    symbol: str,
    day: date,
    root: Path | str,
    *,
    max_sequence_gaps: int | None,
    spec_tick: float | None = None,
    depth: int = 10,
) -> BookDay:
    """Read one replayed book day and apply the exclusion rules.

    ``max_sequence_gaps`` is None only while it is being computed, on the
    development block itself; the gap count is then recorded but excludes
    nothing. ``spec_tick`` is the venue's tick; a day whose own tick differs
    from it, or is not a multiple of it, gets a note and is kept.
    """
    from trading_research.data.study_fetch import book_path
    from trading_research.data.validate import Severity, validate_book

    path = book_path(symbol, day, root)
    if not path.exists() or path.stat().st_size == 0:
        return BookDay(symbol, day, None, None, None, exclusions=(NO_BOOK,))
    frame = pd.read_parquet(path)
    gaps = int(frame.attrs.get("sequence_gaps", 0))
    report = validate_book(frame, depth=depth, sampled=True)
    exclusions: list[str] = []
    notes: list[str] = []
    if any(f.check == "positive_prices" and f.severity is Severity.ERROR for f in report):
        exclusions.append(NON_POSITIVE_PRICE)
    tick, gridded = _day_tick(frame, depth)
    if not gridded:
        exclusions.append(OFF_GRID_PRICE)
    elif spec_tick is not None and tick is not None and not math.isclose(tick, spec_tick):
        relation = "a multiple of" if on_grid(tick, spec_tick) else "not a multiple of"
        notes.append(
            f"own tick {tick} differs from the specification's {spec_tick} ({relation} it)"
        )
    if max_sequence_gaps is not None and gaps > max_sequence_gaps:
        exclusions.append(SEQUENCE_GAPS)
    return BookDay(
        symbol, day, frame, gaps, tick, _findings(report), tuple(exclusions), tuple(notes)
    )


def exclusion_counts(days: Iterable[BookDay]) -> Counter[str]:
    """Excluded days by reason; a day with two reasons counts under each."""
    return Counter(reason for day in days for reason in day.exclusions)


def block_unreadable(excluded_days: int, block_days: int) -> bool:
    """More than a quarter of a held-out block's days excluded."""
    if block_days <= 0:
        raise ValueError("a block has at least one day")
    return excluded_days * 4 > block_days
