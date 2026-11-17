"""The replication's two ways of holding a position, on a book with known prices."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.taker import TakerSettings
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.replication_eth.execution import (
    close_by_time,
    round_trips,
    signal_to_signal,
    taker_book,
)
from trading_research.replication_eth.grid import GRID_NS, level_columns

DAY = date(2025, 2, 5)
START = day_start_ns(DAY)


def _frame(n: int = 3000) -> pd.DataFrame:
    """A book whose mid rises one tick a second, 0.01 wide, ten levels of 5 ETH."""
    ts = START + 60 * NS_PER_S + GRID_NS * np.arange(n)
    mid = 2000.0 + 0.01 * (ts - ts[0]) // NS_PER_S
    data: dict[str, np.ndarray] = {"ts": ts}
    for column in level_columns(10):
        side, kind, level = column.split("_")
        step = 0.01 * int(level)
        if kind == "size":
            data[column] = np.full(n, 5.0)
        else:
            data[column] = mid + 0.005 + step if side == "ask" else mid - 0.005 - step
    return pd.DataFrame(data)


def _at(seconds: float) -> int:
    return START + 60 * NS_PER_S + int(seconds * NS_PER_S)


def test_round_trips_open_from_flat_and_close_on_the_opposite_signal() -> None:
    t = np.array([_at(1), _at(2), _at(5), _at(6), _at(9)])
    side = np.array([1, 1, -1, -1, 1])
    trips = round_trips(t, side, DAY, TakerSettings())
    assert trips == [(_at(1), 1, _at(5)), (_at(6), -1, _at(9))]
    early = np.array([START + 30 * NS_PER_S])
    assert round_trips(early, np.array([1]), DAY, TakerSettings()) == []
    assert round_trips(t[:1], side[:1], DAY, TakerSettings()) == [(_at(1), 1, -1)]


def test_signal_to_signal_prices_and_fees() -> None:
    book = taker_book(_frame(), DAY)
    t, side = np.array([_at(10), _at(110)]), np.array([1, -1])
    gross = signal_to_signal(book, t, side, 0.0)
    assert len(gross) == 1
    # Bought at the ask 0.1 s after 10 s, sold at the bid 0.1 s after 110 s: 100 ticks of
    # drift less one spread, on 0.01 ETH.
    assert gross["gross"].iloc[0] == pytest.approx(0.01 * (1.00 - 0.01))
    taxed = signal_to_signal(book, t, side, 2.2)
    turnover = float(((taxed["entry_price"] + taxed["exit_price"]) * taxed["size"]).sum())
    assert taxed["fees"].sum() == pytest.approx(2.2e-4 * turnover)
    assert taxed["net"].sum() == pytest.approx(gross["net"].sum() - 2.2e-4 * turnover)


def test_close_by_time_holds_and_skips_while_open() -> None:
    book = taker_book(_frame(), DAY)
    t, side = np.array([_at(10), _at(50), _at(200)]), np.array([1, 1, -1])
    done = close_by_time(book, t, side, 100.0, 0.0)
    assert len(done.trades) == 2 and done.counters["entries_busy"] == 1
    assert done.trades["gross"].iloc[0] == pytest.approx(0.01 * (1.00 - 0.01))
