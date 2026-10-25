"""The spot study's day loaders: read inside the access, cut to the UTC day, excluded by rule.

On the pattern of :mod:`trading_research.data.study_loader`, whose
``within_day``, ``max_sequence_gaps``, ``block_unreadable`` and ``BookDay`` are
reused by import. Every loader asks its access (:class:`~trading_research.spot.fetch.AccessGuard`)
before any file of the day is opened, and keeps only the rows stamped in
``[midnight(day), midnight(day + 1))``, counting the rows it drops, so that no
label, walk or flatten of a day reads the next day.

Exclusions are per day and per venue, counted by reason, never repaired:

- **Bybit's book** (:func:`load_book_day`): a ``positive_prices`` error in
  :func:`trading_research.data.validate.validate_book`'s report; a price off the
  day's own tick (:func:`trading_research.market_making.events.derive_tick`);
  more sequence gaps than ``max_sequence_gaps`` (one value, from D, by
  :func:`gap_threshold`); no file, or no row inside the day;
- **either venue's prints** (:func:`load_prints_day`): no file; no print inside
  the day; a price off the day's own tick, derived from its prints the same way;
- **the fetch** (:func:`fetch_exclusions`): a missing plane (HTTP 404 only), an
  unreadable archive or an unfetchable day, as the fetch log ended them.

A tick unlike the venue's specification is noted and excludes nothing. A day
excluded on any plane of a venue is excluded on that venue (:func:`excluded_days`);
a venue's block is unreadable when more than a quarter of its days are
(:func:`unreadable`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data import fetch_log as fl
from trading_research.data.study_loader import (
    NO_BOOK,
    NON_POSITIVE_PRICE,
    OFF_GRID_PRICE,
    SEQUENCE_GAPS,
    BookDay,
    block_unreadable,
    max_sequence_gaps,
    on_grid,
    within_day,
)
from trading_research.data.validate import Finding, Severity, validate_book
from trading_research.market_making.events import OffGridPrice, derive_tick, to_ticks
from trading_research.spot.archives import BOOK, BYBIT_BOOK, BYBIT_SPOT, PRINTS, Plane, day_path
from trading_research.spot.fetch import AccessGuard

NO_PRINTS_FILE = "no_prints_file"
NO_PRINT = "no_print"
#: The fetch log's excluding statuses, as exclusion reasons.
FETCH_REASONS: dict[str, str] = {
    fl.MISSING_PLANE: "missing_plane",
    fl.UNREADABLE: "unreadable",
    fl.UNFETCHABLE: "unfetchable",
}
REASONS: tuple[str, ...] = (
    NO_BOOK,
    NON_POSITIVE_PRICE,
    OFF_GRID_PRICE,
    SEQUENCE_GAPS,
    NO_PRINTS_FILE,
    NO_PRINT,
    *FETCH_REASONS.values(),
)


def own_tick(prices: np.ndarray) -> tuple[float | None, bool]:
    """A day's own tick from its distinct prices, and whether every price sits on it.

    Fewer than two distinct prices give no tick, and nothing is off it.
    """
    distinct = pd.unique(np.asarray(prices, dtype=np.float64))
    if len(distinct) < 2:
        return None, True
    try:
        tick = derive_tick(distinct)
        to_ticks(distinct, tick)
    except OffGridPrice:
        return None, False
    return tick, True


def _tick_note(tick: float | None, spec_tick: float | None) -> tuple[str, ...]:
    if tick is None or spec_tick is None or math.isclose(tick, spec_tick):
        return ()
    relation = "a multiple of" if on_grid(tick, spec_tick) else "not a multiple of"
    return (f"own tick {tick} differs from the specification's {spec_tick} ({relation} it)",)


def _findings(report: Iterable[Finding]) -> dict[str, dict[str, int]]:
    """Every finding of a report: check -> severity -> rows affected (at least one)."""
    out: dict[str, dict[str, int]] = {}
    for finding in report:
        by_severity = out.setdefault(finding.check, {})
        severity = str(finding.severity.value)
        by_severity[severity] = by_severity.get(severity, 0) + max(int(finding.count), 1)
    return out


def load_book_day(
    day: date,
    root: Path | str,
    *,
    access: AccessGuard,
    max_sequence_gaps: int | None,
    spec_tick: float | None = None,
    depth: int = 10,
) -> BookDay:
    """One replayed book day of Bybit spot, cut to its UTC day, with its exclusions.

    ``max_sequence_gaps`` is None only while it is being computed on D: the gap
    count is then recorded and excludes nothing.
    """
    access.require(BYBIT_SPOT, [day], BOOK, what="load")
    path = day_path(BYBIT_BOOK, day, root)
    if not path.exists() or path.stat().st_size == 0:
        return BookDay(BYBIT_SPOT, day, None, None, None, exclusions=(NO_BOOK,))
    frame, outside = within_day(pd.read_parquet(path), day)
    gaps = int(frame.attrs.get("sequence_gaps", 0))
    if frame.empty:
        return BookDay(BYBIT_SPOT, day, None, gaps, None, exclusions=(NO_BOOK,),
                       rows_outside_day=outside)  # fmt: skip
    report = validate_book(frame, depth=depth, sampled=True)
    exclusions: list[str] = []
    if any(f.check == "positive_prices" and f.severity is Severity.ERROR for f in report):
        exclusions.append(NON_POSITIVE_PRICE)
    columns = [f"{side}_price_{level}" for side in ("bid", "ask") for level in range(depth)]
    tick, gridded = own_tick(frame[columns].to_numpy(dtype=np.float64).ravel())
    if not gridded:
        exclusions.append(OFF_GRID_PRICE)
    if max_sequence_gaps is not None and gaps > max_sequence_gaps:
        exclusions.append(SEQUENCE_GAPS)
    return BookDay(BYBIT_SPOT, day, frame, gaps, tick, _findings(report), tuple(exclusions),
                   _tick_note(tick, spec_tick), outside)  # fmt: skip


@dataclass(frozen=True)
class PrintsDay:
    """One day of one venue's prints as the loader found it."""

    plane: str
    day: date
    frame: pd.DataFrame | None
    tick: float | None
    exclusions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    rows_outside_day: int = 0
    #: Prints no order can reach (kept in the frame for the features).
    rpi_rows: int = 0

    @property
    def usable(self) -> bool:
        return not self.exclusions


def load_prints_day(
    plane: Plane,
    day: date,
    root: Path | str,
    *,
    access: AccessGuard,
    spec_tick: float | None = None,
) -> PrintsDay:
    """One day of ``plane``'s prints, cut to its UTC day, with its exclusions."""
    if plane.kind != PRINTS:
        raise ValueError(f"{plane.name} is not a print plane")
    access.require(plane.instrument, [day], PRINTS, what="load")
    path = day_path(plane, day, root)
    if not path.exists() or path.stat().st_size == 0:
        return PrintsDay(plane.name, day, None, None, exclusions=(NO_PRINTS_FILE,))
    frame, outside = within_day(pd.read_parquet(path), day)
    if frame.empty:
        return PrintsDay(plane.name, day, None, None, (NO_PRINT,), rows_outside_day=outside)
    tick, gridded = own_tick(frame["price"].to_numpy(dtype=np.float64))
    notes = _tick_note(tick, spec_tick)
    if tick is None and gridded:
        notes += ("one distinct price: no own tick",)
    return PrintsDay(plane.name, day, frame, tick, () if gridded else (OFF_GRID_PRICE,), notes,
                     outside, int(frame["rpi"].sum()))  # fmt: skip


def fetch_exclusions(log: fl.FetchLog, plane: Plane) -> dict[date, str]:
    """The days of ``plane`` the fetch ended without a file, by reason."""
    return {
        date.fromisoformat(r.day): FETCH_REASONS[r.status]
        for r in log.of(plane.name)
        if r.status in FETCH_REASONS
    }


def excluded_days(
    days: Iterable[BookDay | PrintsDay], fetched: Iterable[Mapping[date, str]] = ()
) -> dict[date, tuple[str, ...]]:
    """Every excluded day of one venue, with the reasons of all its planes."""
    reasons: dict[date, set[str]] = {}
    for loaded in days:
        if loaded.exclusions:
            reasons.setdefault(loaded.day, set()).update(loaded.exclusions)
    for ended in fetched:
        for day, reason in ended.items():
            reasons.setdefault(day, set()).add(reason)
    return {day: tuple(sorted(found)) for day, found in sorted(reasons.items())}


def gap_threshold(book_days: Iterable[BookDay]) -> int:
    """``max_sequence_gaps``: the 95th percentile, rounded up, over every day whose book replayed."""
    return max_sequence_gaps([d.sequence_gaps for d in book_days if d.sequence_gaps is not None])


def unreadable(excluded: Mapping[date, tuple[str, ...]], block_days: int) -> bool:
    """More than a quarter of a venue's block days excluded."""
    return block_unreadable(sum(1 for reasons in excluded.values() if reasons), block_days)
