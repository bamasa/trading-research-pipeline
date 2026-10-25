"""The print grid and the touch proxies: hand cases and the prints cut inside a row interval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test_spot_synthetic_prints import DAY, make_prints, no_network
from trading_research.features.prints import bucket_prints
from trading_research.features.timed import TimedFeature, stamps_ns
from trading_research.spot.grid import (
    PROXY_COLUMNS,
    grid_stamps_ns,
    print_grid,
    touch_proxies,
)
from trading_research.validation.leakage_timed import check_streaming_parity, check_timed_causal

MS = 1_000_000
T0 = pd.Timestamp(DAY, tz="UTC").value


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


def frame(rows: list[tuple[int, float, float, int, bool]]) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["ms", "price", "size", "aggressor", "rpi"])
    out.insert(0, "timestamp", pd.to_datetime(T0 + out.pop("ms") * MS, unit="ns", utc=True))
    return out


HAND = frame(
    [
        (0, 100.0, 1.0, 1, False),
        (50, 99.0, 1.0, -1, False),  # same interval, later: the next row's
        (150, 101.0, 1.0, 1, False),
        (150, 102.0, 1.0, 1, True),  # flagged: still a proxy
        (150, 98.0, 1.0, -1, False),
        (420, 97.0, 1.0, -1, False),
    ]
)


def test_one_row_per_occupied_interval_at_its_first_print() -> None:
    rows = print_grid(HAND)
    assert (stamps_ns(rows) - T0).tolist() == [0, 150 * MS, 420 * MS]
    owned = bucket_prints(HAND, rows.assign(bid_price_0=1.0, ask_price_0=1.0))
    assert owned["trade_count"].tolist() == [1.0, 4.0, 1.0]
    assert owned["sell_volume"].tolist() == [0.0, 2.0, 1.0]


def test_the_proxies_carry_the_state_after_every_print_of_the_stamp() -> None:
    rows = print_grid(HAND)
    assert list(rows.columns) == ["timestamp", *PROXY_COLUMNS]
    assert np.isnan(rows["bid_proxy"].iloc[0]) and np.isnan(rows["mid_proxy"].iloc[0])
    assert rows["ask_proxy"].tolist() == [100.0, 102.0, 102.0]
    assert rows["bid_proxy"].tolist()[1:] == [98.0, 97.0]
    assert rows["mid_proxy"].tolist()[1:] == [100.0, 99.5]


def test_an_empty_day_has_no_rows_and_the_grid_must_be_positive() -> None:
    assert len(grid_stamps_ns(HAND.iloc[:0])) == 0
    with pytest.raises(ValueError, match="positive"):
        grid_stamps_ns(HAND, grid_ms=0)


def _proxy(column: str) -> TimedFeature:
    def fn(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
        assert prints is not None
        values = touch_proxies(prints, stamps_ns(rows))[column].to_numpy()
        return pd.Series(values, index=rows.index)

    # A day's state: the declaration holds the whole day, which the parity
    # check then gives it.
    return TimedFeature(column, "trades", 86_400.0, 0, fn)


@pytest.mark.parametrize("column", PROXY_COLUMNS)
def test_the_proxies_read_nothing_after_a_cut_inside_a_row_interval(column: str) -> None:
    prints = make_prints(1_500, seed=3)
    rows = print_grid(prints)
    for cut in (0.3, 0.6, 0.9):
        assert not check_timed_causal(_proxy(column), rows, prints, cut=cut).leaks
    assert not check_streaming_parity(_proxy(column), rows, prints, range(0, len(rows), 37)).leaks


def test_a_proxy_read_at_its_intervals_end_is_caught() -> None:
    """Reading the whole 100 ms interval gives a row its interval's later prints."""
    prints = make_prints(1_500, seed=3)
    rows = print_grid(prints)

    def forward(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
        assert prints is not None
        ends = stamps_ns(rows) // (100 * MS) * (100 * MS) + 100 * MS - 1
        return pd.Series(touch_proxies(prints, ends)["mid_proxy"].to_numpy(), index=rows.index)

    ahead = TimedFeature("ahead", "trades", 86_400.0, 0, forward)
    assert any(check_timed_causal(ahead, rows, prints, cut=c).leaks for c in (0.2, 0.4, 0.6, 0.8))
