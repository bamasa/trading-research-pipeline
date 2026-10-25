"""A block's days as the study reads them: loaded inside an access, excluded per venue, warm or cold.

For each day (:func:`load_day`) the three planes are loaded through
:mod:`trading_research.spot.loader` inside the access the stage was given:
Binance's prints, Bybit's prints and Bybit's book (replayed at the walk depth).
A day is excluded on a venue for any reason on any of its planes, for its fetch
log's terminal status (a missing plane, an unreadable archive, an unfetchable
day) and for the fee table (``fee not sourced``), counted by reason.

**Warm and cold** (the registration, "Blocks"): a day whose previous calendar
day is usable on the venue and lies in the same block, or in W before D, reads
its rolling state from that day; any other day starts cold, and its first 61
minutes from midnight are not scored (:attr:`DayInputs.scored_from`).

**Inputs** per store (:data:`trading_research.spot.plan.STORES`): the print
members and touch-proxy members on Binance's print grid, the print members on
Bybit's, and S25's 40 inputs on Bybit's book (unscaled; the book rows the
features read are the first ten levels, the walk reads the walk depth).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import fetch_log as fl
from trading_research.data.feature_store import StoreLayout, to_stored
from trading_research.data.study_loader import BookDay
from trading_research.features.study_set import RAW_SIZE_COLUMNS
from trading_research.features.timed import stamps_ns
from trading_research.spot import access as sa
from trading_research.spot import archives as ar
from trading_research.spot import loader as ld
from trading_research.spot.book_inputs import book_day_frame
from trading_research.spot.costs import FeeTable
from trading_research.spot.features_prints import day_frame, print_registry
from trading_research.spot.fetch import AccessGuard
from trading_research.spot.plan import LOOKBACK_MINUTES, NS_PER_S, STORES, midnight_ns
from trading_research.spot.print_taker import print_tape

#: The planes of each venue.
VENUE_PLANES: dict[str, tuple[ar.Plane, ...]] = {
    sa.BINANCE: (ar.BINANCE_PRINTS,),
    sa.BYBIT: (ar.BYBIT_BOOK, ar.BYBIT_PRINTS),
}
FEATURE_LEVELS = 10


def block_name(day: date) -> str | None:
    return sa.block_of(day)


def follows(day: date, before: date) -> bool:
    """``before`` is the calendar day before ``day``, in the same block or W before D."""
    if (day - before).days != 1:
        return False
    a, b = block_name(before), block_name(day)
    return a == b or (a, b) == ("W", "D")


@dataclass
class DayInputs:
    """One day of both venues as loaded, with each venue's exclusions."""

    day: date
    binance: ld.PrintsDay
    bybit: ld.PrintsDay
    book: BookDay
    reasons: dict[str, tuple[str, ...]]
    #: Each venue's warm state: True when the previous day's rolling state is read.
    warm: dict[str, bool] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)

    def usable(self, name: str) -> bool:
        return not self.reasons[name]

    def gap_only(self, name: str) -> bool:
        """Excluded on Bybit for its sequence gaps alone (put back as a measurement)."""
        return self.reasons[name] == (ld.SEQUENCE_GAPS,)

    def scored_from(self, name: str) -> int:
        """The first stamp a decision may be scored at on the venue's day."""
        start = midnight_ns(self.day)
        return start if self.warm.get(name) else start + LOOKBACK_MINUTES * 60 * NS_PER_S

    def prints(self, name: str) -> ld.PrintsDay:
        return self.binance if name == sa.BINANCE else self.bybit


def fetch_reasons(logs: Sequence[fl.FetchLog], plane: ar.Plane, day: date) -> list[str]:
    found: list[str] = []
    for log in logs:
        reason = ld.fetch_exclusions(log, plane).get(day)
        if reason is not None:
            found.append(reason)
    return found


def load_day(
    day: date,
    root: Any,
    *,
    access: AccessGuard,
    logs: Sequence[fl.FetchLog],
    table: FeeTable,
    max_sequence_gaps: int | None,
    depth: int,
    spec: Mapping[str, tuple[float, float]] | None = None,
) -> DayInputs:
    """Every plane of ``day`` inside ``access``, with each venue's reasons."""
    tick = {name: None if spec is None else spec[name][0] for name in VENUE_PLANES}
    binance = ld.load_prints_day(ar.BINANCE_PRINTS, day, root, access=access,
                                 spec_tick=tick[sa.BINANCE])  # fmt: skip
    bybit = ld.load_prints_day(ar.BYBIT_PRINTS, day, root, access=access,
                               spec_tick=tick[sa.BYBIT])  # fmt: skip
    book = ld.load_book_day(day, root, access=access, max_sequence_gaps=max_sequence_gaps,
                            spec_tick=tick[sa.BYBIT], depth=depth)  # fmt: skip
    reasons = {}
    for name, loaded in ((sa.BINANCE, (binance,)), (sa.BYBIT, (book, bybit))):
        found = {r for one in loaded for r in one.exclusions}
        for plane in VENUE_PLANES[name]:
            found.update(fetch_reasons(logs, plane, day))
        fee = table.exclusion(name, day)
        if fee is not None:
            found.add(fee)
        reasons[name] = tuple(sorted(found))
    return DayInputs(day, binance, bybit, book, reasons)


def book_tail(
    frame: pd.DataFrame | None, minutes: int = LOOKBACK_MINUTES + 1
) -> pd.DataFrame | None:
    """The last ``minutes`` of a day's book rows: what the next day's features read."""
    if frame is None or frame.empty:
        return frame
    stamps = stamps_ns(frame)
    return frame[stamps >= stamps[-1] - minutes * 60 * NS_PER_S]


def walk_days(days: Sequence[date], load: Any) -> Iterator[DayInputs]:
    """Each of ``days`` loaded in order with its warm state and the history it reads;
    a day whose previous calendar day is not in ``days`` loads that day for it."""
    previous: DayInputs | None = None
    for day in days:
        current = load(day)
        before = day - timedelta(days=1)
        if follows(day, before) and (previous is None or previous.day != before):
            previous = load(before)
        for name in VENUE_PLANES:
            current.warm[name] = bool(
                previous is not None
                and previous.day == before
                and previous.usable(name)
                and follows(day, before)
            )
        if previous is not None and current.warm[sa.BINANCE]:
            current.history["binance"] = previous.binance.frame
        if previous is not None and current.warm[sa.BYBIT]:
            current.history["bybit_prints"] = previous.bybit.frame
            current.history["book"] = (book_tail(previous.book.frame), previous.bybit.frame)
        yield current
        previous = current


def ten_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """The book rows the features read: every column but levels from the eleventh on."""
    deep = [c for c in frame.columns
            if c.rsplit("_", 1)[-1].isdigit() and int(c.rsplit("_", 1)[-1]) >= FEATURE_LEVELS
            and c.split("_")[0] in ("bid", "ask")]  # fmt: skip
    return frame.drop(columns=deep) if deep else frame


_REGISTRY: list[Any] = []


def registry() -> Any:
    if not _REGISTRY:
        _REGISTRY.append(print_registry())
    return _REGISTRY[0]


def store_frame(store: str, inputs: DayInputs) -> pd.DataFrame:
    """``timestamp`` and the store's inputs at every row of the day's grid (unscaled)."""
    name, columns = STORES[store]
    if store == "book":
        if inputs.book.frame is None or inputs.bybit.frame is None:
            raise ValueError(f"{inputs.day}: no book or prints on Bybit")
        history = inputs.history.get("book")
        held = None if history is None else (ten_levels(history[0]), history[1])
        return book_day_frame(ten_levels(inputs.book.frame), inputs.bybit.frame,
                              names=columns, history=held)  # fmt: skip
    prints = inputs.prints(name).frame
    if prints is None:
        raise ValueError(f"{inputs.day}: no prints on {name}")
    return day_frame(prints, names=columns, history=inputs.history.get(store),
                     registry=registry())  # fmt: skip


def layout(store: str, touch_size: float | None) -> StoreLayout:
    """How a store's inputs are scaled and cast: the book's raw sizes over the touch size."""
    if store == "book":
        if touch_size is None:
            raise ValueError("the book's inputs need a touch size")
        _, columns = STORES[store]
        return StoreLayout(touch_size, tuple(c for c in RAW_SIZE_COLUMNS if c in columns))
    return StoreLayout(1.0, ())


def model_inputs(frame: pd.DataFrame, columns: Sequence[str], scale: StoreLayout) -> pd.DataFrame:
    """The inputs as a fit sees them: scaled, cast to the store's dtypes and back to
    float64, an overflow (infinite) as missing."""
    stored = to_stored(frame[["timestamp", *columns]], scale).drop(columns="timestamp")
    out = stored.astype(np.float64)
    return out.replace([np.inf, -np.inf], np.nan)


def predict(model: Any, frame: pd.DataFrame, scale: StoreLayout, scored_from: int) -> np.ndarray:
    """Class probabilities on every row at or after ``scored_from`` with every input
    present; NaN elsewhere (never scored, never imputed)."""
    columns = list(model.columns_)
    x = model_inputs(frame, columns, scale)
    keep = (stamps_ns(frame) >= scored_from) & x.notna().all(axis=1).to_numpy()
    out = np.full((len(frame), 3), np.nan)
    if keep.any():
        out[keep] = np.asarray(model.predict_proba(x.loc[keep]), dtype=np.float64)
    return out


def tape(inputs: DayInputs, name: str) -> Any:
    """The venue's prints as the print rule's tape."""
    return print_tape(inputs.prints(name))


def grid_of(prints: pd.DataFrame) -> np.ndarray:
    """The print grid's row stamps of a day's prints."""
    from trading_research.spot.grid import grid_stamps_ns

    return grid_stamps_ns(prints)
