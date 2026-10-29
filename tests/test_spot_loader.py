"""The spot day loaders: the access first, the UTC day only, every exclusion by its rule."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.data import fetch_log as fl
from trading_research.data.study_loader import BookDay
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import archives as ar
from trading_research.spot import binance, bybit_book, bybit_prints
from trading_research.spot import loader as ld
from trading_research.spot import synthetic as syn

DAY, GAP, EMPTY, ABSENT = date(2026, 3, 3), date(2026, 3, 4), date(2026, 3, 5), date(2026, 3, 6)
DAYS = (DAY, GAP, EMPTY, ABSENT)
MARKET = syn.SpotMarket(frozenset({DAY, GAP, EMPTY}), step_ms=600_000, rpi_every=4,
                        gap_days=frozenset({GAP}), empty_days=frozenset({EMPTY}))  # fmt: skip


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


class Guard:
    def __init__(self, days: Iterable[date]) -> None:
        self.days = frozenset(days)

    def require(
        self, instrument: str, days: Iterable[date], plane: str, *, what: str = "read"
    ) -> None:
        if any(day not in self.days for day in days):
            raise HeldOutLocked(f"cannot {what} {instrument} {plane}")


OPEN = Guard(DAYS)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("spot")
    venue = syn.SpotVenue(MARKET)
    for day in (DAY, GAP, EMPTY):
        for fetch_day in (bybit_book.fetch_day, bybit_prints.fetch_day, binance.fetch_day):
            assert fetch_day(day, base, base / "a", access=OPEN, opener=venue).outcome == fl.OK
    return base


def _rewrite(
    root: Path, plane: ar.Plane, day: date, edit: Callable[[pd.DataFrame], pd.DataFrame]
) -> Path:
    path = ar.day_path(plane, day, root)
    frame = pd.read_parquet(path)
    attrs = dict(frame.attrs)
    frame = edit(frame)
    frame.attrs = attrs
    out = root / "edited"
    target = ar.day_path(plane, day, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return out


def test_a_book_day_loads_usable_with_its_own_tick_and_a_note(root: Path) -> None:
    book = ld.load_book_day(DAY, root, access=OPEN, max_sequence_gaps=0, spec_tick=0.01)
    assert book.usable and book.symbol == ar.BYBIT_SPOT and book.frame is not None
    assert book.tick == pytest.approx(0.1) and book.sequence_gaps == 0
    assert book.notes == ("own tick 0.1 differs from the specification's 0.01 (a multiple of it)",)


def test_rows_of_the_next_day_are_dropped_and_counted(root: Path) -> None:
    def add_next_day(frame: pd.DataFrame) -> pd.DataFrame:
        late = frame.iloc[[-1]].assign(timestamp=pd.Timestamp(DAY + timedelta(days=1), tz="UTC"))
        return pd.concat([frame, late], ignore_index=True)

    edited = _rewrite(root, ar.BYBIT_BOOK, DAY, add_next_day)
    book = ld.load_book_day(DAY, edited, access=OPEN, max_sequence_gaps=0)
    assert book.rows_outside_day == 1 and book.usable
    prints = ld.load_prints_day(
        ar.BINANCE_PRINTS, DAY, _rewrite(root, ar.BINANCE_PRINTS, DAY, add_next_day), access=OPEN
    )
    assert prints.rows_outside_day == 1 and prints.usable


def test_book_exclusions_by_rule(root: Path) -> None:
    gaps = ld.load_book_day(GAP, root, access=OPEN, max_sequence_gaps=2)
    assert gaps.exclusions == (ld.SEQUENCE_GAPS,) and gaps.sequence_gaps == 3
    assert ld.load_book_day(GAP, root, access=OPEN, max_sequence_gaps=None).usable
    assert ld.load_book_day(ABSENT, root, access=OPEN, max_sequence_gaps=0).exclusions == (
        ld.NO_BOOK,
    )
    zero = _rewrite(
        root,
        ar.BYBIT_BOOK,
        DAY,
        lambda f: f.assign(bid_price_9=f["bid_price_9"].where(f.index != 5, 0.0)),
    )
    assert (
        ld.NON_POSITIVE_PRICE
        in ld.load_book_day(DAY, zero, access=OPEN, max_sequence_gaps=0).exclusions
    )
    third = _rewrite(
        root, ar.BYBIT_BOOK, DAY, lambda f: f.assign(ask_price_9=f["ask_price_9"] + 1 / 3)
    )
    assert ld.load_book_day(DAY, third, access=OPEN, max_sequence_gaps=0).exclusions == (
        ld.OFF_GRID_PRICE,
    )


@pytest.mark.parametrize("plane", [ar.BYBIT_PRINTS, ar.BINANCE_PRINTS])
def test_print_days_and_their_exclusions(root: Path, plane: ar.Plane) -> None:
    day = ld.load_prints_day(plane, DAY, root, access=OPEN)
    assert day.usable and day.frame is not None and day.tick is not None
    assert day.rpi_rows == (len(day.frame) // 4 if plane is ar.BYBIT_PRINTS else 0)
    assert ld.load_prints_day(plane, EMPTY, root, access=OPEN).exclusions == (ld.NO_PRINT,)
    assert ld.load_prints_day(plane, ABSENT, root, access=OPEN).exclusions == (ld.NO_PRINTS_FILE,)
    third = _rewrite(root, plane, DAY, lambda f: f.assign(price=f["price"] + 1 / 3))
    assert ld.load_prints_day(plane, DAY, third, access=OPEN).exclusions == (ld.OFF_GRID_PRICE,)


def test_the_access_is_asked_before_any_file_is_opened(root: Path) -> None:
    with pytest.raises(HeldOutLocked):
        ld.load_book_day(DAY, root, access=Guard(()), max_sequence_gaps=0)
    with pytest.raises(HeldOutLocked):
        ld.load_prints_day(ar.BYBIT_PRINTS, DAY, root, access=Guard(()))
    with pytest.raises(ValueError, match="not a print plane"):
        ld.load_prints_day(ar.BYBIT_BOOK, DAY, root, access=OPEN)


def test_fetch_ends_venue_days_the_gap_threshold_and_an_unreadable_block(tmp_path: Path) -> None:
    log = fl.FetchLog(tmp_path / "log.json")
    now = pd.Timestamp("2026-10-25", tz="UTC").to_pydatetime()
    log.get("bybit_prints", ABSENT).add(fl.MISSING, now=now)
    log.get("bybit_book", DAY).add(fl.REPLAY_FAILED, now=now)
    assert ld.fetch_exclusions(log, ar.BYBIT_PRINTS) == {ABSENT: "missing_plane"}
    book = [BookDay(ar.BYBIT_SPOT, d, None, g, None, exclusions=e)
            for d, g, e in ((DAY, 0, ()), (GAP, 5, (ld.SEQUENCE_GAPS,)), (EMPTY, 1, ()), (ABSENT, None, (ld.NO_BOOK,)))]  # fmt: skip
    venue = ld.excluded_days(
        book, [ld.fetch_exclusions(log, p) for p in (ar.BYBIT_BOOK, ar.BYBIT_PRINTS)]
    )
    assert venue == {
        DAY: ("unreadable",),
        GAP: (ld.SEQUENCE_GAPS,),
        ABSENT: ("missing_plane", ld.NO_BOOK),
    }
    assert (
        ld.gap_threshold(book) == 5
    )  # the 95th percentile of 0, 5 and 1, rounded up; the absent day not counted
    assert not ld.unreadable(venue, 12) and ld.unreadable(venue, 11)


def test_a_walk_depth_book_day_with_empty_deep_levels_is_usable(root: Path) -> None:
    """At a 200-level walk, a level the archive did not hold is empty, not a fault;
    a deep price present and not positive still excludes the day."""

    def deepen(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for k in range(10, 12):
            out[f"bid_price_{k}"] = out["bid_price_9"] - 0.1 * (k - 9)
            out[f"ask_price_{k}"] = out["ask_price_9"] + 0.1 * (k - 9)
            out[f"bid_size_{k}"] = out[f"ask_size_{k}"] = 0.5
        out.loc[out.index[:3], ["bid_price_11", "bid_size_11"]] = np.nan
        return out

    deep = _rewrite(root, ar.BYBIT_BOOK, DAY, deepen)
    book = ld.load_book_day(DAY, deep, access=OPEN, max_sequence_gaps=0, depth=12)
    assert book.usable and book.tick == pytest.approx(0.1)
    bad = _rewrite(root, ar.BYBIT_BOOK, DAY,
                   lambda f: deepen(f).assign(ask_price_11=lambda g: g["ask_price_11"] * 0 - 1))  # fmt: skip
    assert (
        ld.NON_POSITIVE_PRICE
        in ld.load_book_day(DAY, bad, access=OPEN, max_sequence_gaps=0, depth=12).exclusions
    )
