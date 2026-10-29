"""Where the spot study's parts meet: the study's access guards the data's fetch and
loaders, the loader's days feed both executors, the fee table excludes days."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from trading_research.data import fetch_log as fl
from trading_research.market_making.events import NS_PER_S
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import access as acc
from trading_research.spot import archives as ar
from trading_research.spot import fetch_stage as st
from trading_research.spot import loader as ld
from trading_research.spot import synthetic as syn
from trading_research.spot.costs import (
    FEE_NOT_SOURCED,
    book_settings,
    execute_book_day,
    load_fee_table,
    taker_book,
)
from trading_research.spot.grid import grid_stamps_ns
from trading_research.spot.print_taker import execute_print_day, print_settings, print_tape

DAY, H_DAY, P_DAY = date(2026, 3, 3), date(2026, 8, 10), date(2026, 11, 2)
MARKET = syn.SpotMarket(frozenset({DAY}), step_ms=2_000, rpi_every=5)
TABLE = load_fee_table(Path(__file__).resolve().parents[1] / "configs" / "spot_2026.yaml")


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``data/`` with the reader-test day fetched under the study's own access, the
    book replayed at all twelve of the synthetic venue's levels."""
    base = tmp_path_factory.mktemp("data")
    with syn.no_network():
        log = st.run_stage(
            0, st.every_plane([DAY]), st.SpotRoots.under(base / "spot"),
            fl.FetchLog(base / "log.json"), access=acc.reader_test_access("readers' proof"),
            disk_record=base / "disk.json", opener=syn.SpotVenue(MARKET),
            book_depth=syn.LEVELS, measure=lambda _: 100.0,
        )  # fmt: skip
    assert all(log.get(name, DAY).status == fl.ON_DISK for name in st.ORDER)
    return base


def test_the_study_access_guards_the_fetch_and_the_loaders(data: Path) -> None:
    root, access = data / "spot", acc.development_access()
    for plane in ar.PLANES.values():
        path = ar.day_path(plane, DAY, root)
        assert path.parent == acc.data_root(plane.instrument, plane.kind, data)
        assert acc.require_spot_path(path, data) == path and path.exists()
    assert len(acc.files_of_days([DAY], data)) == 3 and not acc.files_of_days([H_DAY], data)
    with pytest.raises(HeldOutLocked):
        ld.load_book_day(H_DAY, root, access=access, max_sequence_gaps=None)
    with pytest.raises(HeldOutLocked):
        ld.load_prints_day(ar.BINANCE_PRINTS, H_DAY, root, access=access)
    with pytest.raises(HeldOutLocked):
        st.run_stage(2, st.every_plane([H_DAY]), st.SpotRoots.under(root),
                     fl.FetchLog(data / "h.json"), access=access, disk_record=data / "h_disk.json",
                     measure=lambda _: 100.0)  # fmt: skip


def test_a_loaded_book_day_runs_the_book_executor_at_the_walk_depth(data: Path) -> None:
    book_day = ld.load_book_day(DAY, data / "spot", access=acc.development_access(),
                                max_sequence_gaps=None)  # fmt: skip
    assert book_day.usable
    book = taker_book(book_day, depth=syn.LEVELS)
    assert book.bid_px.shape[1] == syn.LEVELS and len(book.funding_ts) == 0
    rows = np.arange(100, len(book.ts) - 100, 97)
    settings = book_settings(TABLE, ar.BYBIT_SPOT, DAY)
    day = execute_book_day(book, book.ts[rows], np.where(rows % 2, 1, -1), 50 * NS_PER_S,
                           settings, TABLE)  # fmt: skip
    assert len(day.trades) > 10 and (day.trades["fees"] > 0).all()
    with pytest.raises(ValueError, match="not executed"):
        taker_book(ld.BookDay(ar.BYBIT_SPOT, DAY, None, None, None, exclusions=(ld.NO_BOOK,)))


@pytest.mark.parametrize("plane", [ar.BINANCE_PRINTS, ar.BYBIT_PRINTS])
def test_a_loaded_print_day_runs_the_print_executor_on_the_grid(
    data: Path, plane: ar.Plane
) -> None:
    prints = ld.load_prints_day(plane, DAY, data / "spot", access=acc.development_access())
    tape = print_tape(prints)
    assert tape.rpi.sum() == prints.rpi_rows and len(tape) == len(prints.frame)
    rows = grid_stamps_ns(prints.frame)
    assert np.all(np.diff(rows) > 0) and np.isin(rows, tape.ts).all()
    pick = rows[100:-100:13]
    settings = print_settings(TABLE, plane.instrument)
    day = execute_print_day(tape, rows, pick, np.where(np.arange(len(pick)) % 2, 1, -1),
                            50 * NS_PER_S, settings)  # fmt: skip
    assert len(day.trades) > 10 and day.counters["decisions"] == len(pick)


def test_a_day_the_fee_table_does_not_cover_is_excluded_on_its_venue() -> None:
    for instrument in (ar.BINANCE_SPOT, ar.BYBIT_SPOT):
        found = ld.fee_exclusions(TABLE, instrument, [DAY, H_DAY, P_DAY])
        assert found == {P_DAY: FEE_NOT_SOURCED}
        assert ld.excluded_days([], [found]) == {P_DAY: (FEE_NOT_SOURCED,)}
