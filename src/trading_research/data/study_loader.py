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

Every reader here takes a :class:`~trading_research.data.study_access.StudyAccess`
and asks it before any file of the day is opened; without one it reads under the
development access, which refuses every 2025 day after 2025-06-30 in every plane.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trading_research.data import study_access as sa

if TYPE_CHECKING:
    from trading_research.market_making.events import DayEvents

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
    access: sa.StudyAccess | None = None,
) -> BookDay:
    """Read one replayed book day and apply the exclusion rules.

    ``max_sequence_gaps`` is None only while it is being computed, on the
    development block itself; the gap count is then recorded but excludes
    nothing. ``spec_tick`` is the venue's tick; a day whose own tick differs
    from it, or is not a multiple of it, gets a note and is kept. ``access``
    must permit the day's book (the development access when None).
    """
    sa.resolve(access).require(symbol, [day], sa.BOOK, what="load")
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


# ---------------------------------------------------------------------------
# The other planes, each read only inside the access
# ---------------------------------------------------------------------------


def load_prints_day(
    symbol: str, day: date, root: Path | str, *, access: sa.StudyAccess | None = None
) -> pd.DataFrame:
    """One day of prints with aggressor side, if ``access`` permits it."""
    from trading_research.data import bybit_trades

    sa.resolve(access).require(symbol, [day], sa.PRINTS, what="load")
    return bybit_trades.load_day(symbol, day, root)


def load_funding_day(
    symbol: str, day: date, root: Path | str, *, access: sa.StudyAccess | None = None
) -> pd.DataFrame:
    """One day of funding settlements, if ``access`` permits it."""
    from trading_research.data import bybit_funding

    sa.resolve(access).require(symbol, [day], sa.FUNDING, what="load")
    return bybit_funding.load_day(symbol, day, root)


def bars_path(symbol: str, day: date, root: Path | str, interval: str = "1h") -> Path:
    """Where one day of Binance futures bars lives (``data/binance.py``'s layout)."""
    return Path(root) / symbol / f"klines-{interval}" / f"{day.isoformat()}.parquet"


def load_bars(
    symbol: str,
    days: Sequence[date],
    root: Path | str,
    *,
    interval: str = "1h",
    access: sa.StudyAccess | None = None,
) -> pd.DataFrame:
    """Binance futures bars for ``days``, one daily file each, in time order.

    Refused unless ``access`` permits every day's bars, and refused while a
    cached monthly archive of the interval runs past the last day of the
    highest opened block. A day without a file raises ``FileNotFoundError``:
    the fetch log says why it is missing.
    """
    permitted = sa.resolve(access)
    permitted.require(symbol, days, sa.BARS, what="load")
    permitted.refuse_monthly_bars(symbol, root, interval)
    frames = []
    for day in sorted(set(days)):
        path = bars_path(symbol, day, root, interval)
        if not path.exists():
            raise FileNotFoundError(f"no {interval} bars of {symbol} on {day} under {root}")
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_events_day(
    symbol: str,
    day: date,
    *,
    book_roots: Sequence[Path],
    trades_root: Path,
    funding_root: Path | None,
    depth: int = 10,
    access: sa.StudyAccess | None = None,
) -> DayEvents:
    """The simulator's merged day (:func:`~trading_research.market_making.events.load_day`),
    if ``access`` permits the day's book, prints and, when read, its funding.

    The passive strategies call the simulator directly on days loaded here.
    """
    from trading_research.market_making.events import load_day

    permitted = sa.resolve(access)
    planes = (sa.BOOK, sa.PRINTS) + ((sa.FUNDING,) if funding_root is not None else ())
    for plane in planes:
        permitted.require(symbol, [day], plane, what="load")
    return load_day(
        symbol,
        day,
        book_roots=book_roots,
        trades_root=trades_root,
        funding_root=funding_root,
        depth=depth,
    )
