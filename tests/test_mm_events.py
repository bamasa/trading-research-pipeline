"""The merged event stream: what order a participant could have seen things in."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.events import (
    BOOK,
    FUNDING,
    TRADE,
    DayUnavailable,
    InstrumentSpec,
    OffGridPrice,
    assemble,
    day_start_ns,
    derive_lot,
    derive_tick,
    load_day,
    sweep_order,
    to_ticks,
)
from trading_research.market_making.synthetic import hand_built_market

DAY = date(2020, 1, 6)


def _levels(best: int, step: int, size: float = 5.0, depth: int = 3) -> list[tuple[int, float]]:
    return [(best + step * i, size) for i in range(depth)]


def test_events_are_time_ordered_with_trades_before_books() -> None:
    """At one timestamp: funding, then prints, then the snapshot that reflects them."""
    events = hand_built_market(
        [(1000.0, _levels(100, -1), _levels(102, 1)), (2000.0, _levels(100, -1), _levels(102, 1))],
        [(2000.0, 100, 1.0, -1), (1500.0, 102, 1.0, 1)],
        funding=[(0.0, 1e-4)],
    )
    assert np.all(np.diff(events.ts) >= 0)
    at_2000 = events.kind[events.ts == events.ts[-1]]
    assert list(at_2000) == [TRADE, BOOK]
    assert events.kind[0] == FUNDING


def test_prints_sharing_a_timestamp_are_ordered_as_a_sweep() -> None:
    """Buys lift the lowest ask first, sells hit the highest bid first; larger
    prints first at one price; sells before buys at one timestamp."""
    ts = np.array([5, 5, 5, 5, 5, 3], dtype=np.int64)
    price = np.array([101, 103, 102, 99, 100, 50])
    size = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 1.0])
    aggressor = np.array([1, 1, 1, -1, -1, 1])
    order = sweep_order(ts, price, size, aggressor)
    assert list(order) == [5, 4, 3, 0, 2, 1]

    ties = sweep_order(
        np.array([7, 7, 7]), np.array([10, 10, 10]), np.array([1.0, 4.0, 1.0]), np.array([1, 1, 1])
    )
    assert list(ties) == [1, 0, 2]  # larger first, then the input order


def test_the_assembled_stream_holds_prints_in_sweep_order() -> None:
    events = hand_built_market(
        [(0.0, _levels(100, -1), _levels(102, 1))],
        [(10.0, 104, 1.0, 1), (10.0, 102, 1.0, 1), (10.0, 103, 1.0, 1)],
    )
    assert list(events.trade_px) == [102, 103, 104]


def test_prices_off_the_tick_grid_are_refused() -> None:
    assert list(to_ticks(np.array([0.41, 0.4123, 1.0]), 0.0001)) == [4100, 4123, 10000]
    with pytest.raises(OffGridPrice):
        to_ticks(np.array([0.41, 0.41235]), 0.0001)


def test_the_tick_and_lot_are_derived_from_the_day() -> None:
    assert derive_tick(np.array([0.4123, 0.4125, 0.4124, 0.4130])) == pytest.approx(0.0001)
    assert derive_tick(np.array([52000.1, 52000.3, 52000.0])) == pytest.approx(0.1)
    assert derive_lot(np.array([0.3, 1.2, 12.0, 0.1])) == pytest.approx(0.1)
    with pytest.raises(OffGridPrice):
        derive_tick(np.array([1.0]))


def test_an_instrument_spec_refuses_nonsense() -> None:
    with pytest.raises(ValueError):
        InstrumentSpec("X", 0.0, 1.0)
    with pytest.raises(ValueError):
        InstrumentSpec("X", 0.01, 1.0, funding_interval_h=5)


def test_a_funding_settlement_off_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="off the"):
        hand_built_market(
            [(0.0, _levels(100, -1), _levels(102, 1))],
            [(10.0, 100, 1.0, -1)],
            funding=[(3_600_000.0, 1e-4)],
        )


def test_a_day_missing_either_plane_cannot_be_assembled() -> None:
    with pytest.raises(DayUnavailable, match="no trade prints"):
        hand_built_market([(0.0, _levels(100, -1), _levels(102, 1))], [])
    spec = InstrumentSpec("X", 0.01, 1.0)
    empty = np.empty((0, 3))
    with pytest.raises(DayUnavailable, match="no book"):
        assemble(
            spec,
            DAY,
            book_ts=np.empty(0, dtype=np.int64),
            bid_px=empty,
            bid_sz=empty,
            ask_px=empty,
            ask_sz=empty,
            trade_ts=np.array([1]),
            trade_px=np.array([100]),
            trade_sz=np.array([1.0]),
            trade_aggressor=np.array([1]),
        )


def test_the_stream_is_read_only() -> None:
    events = hand_built_market([(0.0, _levels(100, -1), _levels(102, 1))], [(10.0, 100, 1.0, -1)])
    with pytest.raises(ValueError):
        events.bid_px[0, 0] = 1


# ---------------------------------------------------------------------------
# Reading one day from disk
# ---------------------------------------------------------------------------


def _write_book(root: Path, symbol: str, day: date, depth: int = 3, gaps: int = 0) -> None:
    rows = []
    for i in range(5):
        row: dict[str, object] = {
            "timestamp": pd.to_datetime(day_start_ns(day) + i * 100_000_000, unit="ns", utc=True)
        }
        for level in range(depth):
            row[f"bid_price_{level}"] = round(0.4120 - 0.0001 * level, 4)
            row[f"bid_size_{level}"] = 10.0
            row[f"ask_price_{level}"] = round(0.4122 + 0.0001 * level, 4)
            row[f"ask_size_{level}"] = 10.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["sequence_gaps"] = gaps
    (root / symbol).mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / symbol / f"{day.isoformat()}.parquet", index=False)


def _write_trades(root: Path, symbol: str, day: date) -> None:
    start = pd.Timestamp(day.isoformat(), tz="UTC")
    frame = pd.DataFrame(
        {
            # Float-second timestamps carry nanosecond noise; the loader rounds it.
            "timestamp": pd.to_datetime(
                [start.timestamp() + 0.1234, start.timestamp() + 0.2], unit="s", utc=True
            ),
            "price": [0.4120, 0.4122],
            "size": [1.5, 0.3],
            "aggressor": [-1, 1],
        }
    )
    (root / symbol).mkdir(parents=True, exist_ok=True)
    frame.to_parquet(root / symbol / f"{day.isoformat()}.parquet", index=False)


def test_one_day_is_read_from_its_own_files(tmp_path: Path) -> None:
    books, fresh, trades = tmp_path / "book", tmp_path / "book_fresh", tmp_path / "trades"
    _write_book(fresh, "BICOUSDT", DAY, gaps=2)
    _write_trades(trades, "BICOUSDT", DAY)
    # Another day on disk must not be touched: it is unreadable on purpose.
    (trades / "BICOUSDT" / "2020-01-07.parquet").write_bytes(b"not parquet")

    events = load_day(
        "BICOUSDT", DAY, book_roots=[books, fresh], trades_root=trades, funding_root=None, depth=3
    )
    assert events.spec.tick == pytest.approx(0.0001)
    assert events.spec.lot == pytest.approx(0.3)
    assert list(events.bid_px[0]) == [4120, 4119, 4118]
    assert list(events.trade_px) == [4120, 4122]
    assert events.sequence_gaps == 2
    assert not events.has_funding
    start = day_start_ns(DAY)
    assert list(events.trade_ts - start) == [123_400_000, 200_000_000]


def test_rows_stamped_after_the_days_midnight_are_never_read(tmp_path: Path) -> None:
    """The archives spill a few rows past midnight; a day reads its own rows
    only, so the last day of a block never reads the next block."""
    books, trades = tmp_path / "book", tmp_path / "trades"
    _write_book(books, "BICOUSDT", DAY, gaps=1)
    _write_trades(trades, "BICOUSDT", DAY)
    path = books / "BICOUSDT" / f"{DAY.isoformat()}.parquet"
    frame = pd.read_parquet(path)
    spill = frame.iloc[[-1]].copy()
    spill["timestamp"] = pd.Timestamp(DAY.isoformat(), tz="UTC") + pd.Timedelta(
        days=1, milliseconds=7
    )
    spill["bid_price_0"] = 9.0  # off the day's grid: reading it would refuse the day
    both = pd.concat([frame, spill], ignore_index=True)
    both.attrs["sequence_gaps"] = 1
    both.to_parquet(path, index=False)

    events = load_day(
        "BICOUSDT", DAY, book_roots=[books], trades_root=trades, funding_root=None, depth=3
    )
    assert len(events.book_ts) == 5
    assert events.rows_outside_day == 1
    assert events.sequence_gaps == 1
    assert int(events.book_ts.max()) < day_start_ns(DAY) + 86_400_000_000_000


def test_a_day_missing_a_plane_on_disk_is_reported(tmp_path: Path) -> None:
    books, trades = tmp_path / "book", tmp_path / "trades"
    _write_book(books, "BICOUSDT", DAY)
    with pytest.raises(DayUnavailable, match="no trades"):
        load_day("BICOUSDT", DAY, book_roots=[books], trades_root=trades, funding_root=None)
    _write_trades(trades, "BICOUSDT", DAY)
    with pytest.raises(DayUnavailable, match="no book"):
        load_day(
            "BICOUSDT", date(2020, 1, 7), book_roots=[books], trades_root=trades, funding_root=None
        )
    with pytest.raises(DayUnavailable, match="no funding"):
        load_day(
            "BICOUSDT",
            DAY,
            book_roots=[books],
            trades_root=trades,
            funding_root=tmp_path / "funding",
            depth=3,
        )


def test_funding_on_disk_joins_the_stream(tmp_path: Path) -> None:
    books, trades, funding = tmp_path / "book", tmp_path / "trades", tmp_path / "funding"
    _write_book(books, "BICOUSDT", DAY)
    _write_trades(trades, "BICOUSDT", DAY)
    (funding / "BICOUSDT").mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [day_start_ns(DAY) + h * 3600 * 10**9 for h in (0, 8, 16)], unit="ns", utc=True
            ),
            "rate": [1e-4, 2e-4, -1e-4],
        }
    ).to_parquet(funding / "BICOUSDT" / f"{DAY.isoformat()}.parquet", index=False)
    events = load_day(
        "BICOUSDT", DAY, book_roots=[books], trades_root=trades, funding_root=funding, depth=3
    )
    assert events.has_funding
    assert list(events.funding_rate) == [1e-4, 2e-4, -1e-4]
    with pytest.raises(DayUnavailable, match="every 8 h"):
        load_day(
            "BICOUSDT",
            DAY,
            book_roots=[books],
            trades_root=trades,
            funding_root=funding,
            depth=3,
            funding_interval_h=4,
        )


def test_prints_far_outside_the_visible_book_are_not_eligible_to_fill() -> None:
    """Within 25 bp of the deepest visible levels a print may be a deep sweep and
    is kept; beyond that it could not have swept this book."""
    from trading_research.market_making.events import OFF_BOOK_BP, off_book_eligible

    book_ts = np.array([0, 100], dtype=np.int64)
    bid_px = np.array([[10_000, 9_999, 9_998]] * 2, dtype=np.int64)  # deepest bid 9998
    ask_px = np.array([[10_002, 10_003, 10_004]] * 2, dtype=np.int64)  # deepest ask 10004
    trade_ts = np.array([-5, 50, 50, 50, 50, 50], dtype=np.int64)
    trade_px = np.array([5_000, 9_997, 9_970, 10_030, 10_040, 10_001], dtype=np.int64)
    eligible = off_book_eligible(book_ts, bid_px, ask_px, trade_ts, trade_px)
    # Before any book: kept. 1 tick (1 bp) beyond: kept. 28 bp below and 36 bp
    # above: off-book. 26 bp above: off-book. Inside: kept.
    assert eligible.tolist() == [True, True, False, False, False, True]
    assert OFF_BOOK_BP == 25.0
