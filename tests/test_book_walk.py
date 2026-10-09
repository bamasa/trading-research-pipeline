"""The book walk every taker price comes from."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.walk import book_arrays, round_trip_cost_bp, walk_book
from trading_research.data.schema import ask_price_col, ask_size_col, bid_price_col, bid_size_col


def _loop(prices: np.ndarray, sizes: np.ndarray, size: float) -> tuple[float, float]:
    """One snapshot walked level by level, as a loop over Python floats would."""
    left = size
    cost = 0.0
    for level in range(len(prices)):
        take = min(left, float(sizes[level]))
        cost += take * float(prices[level])
        left -= take
        if left <= 0.0:
            break
    beyond = left if left > 0.0 else 0.0
    if left > 0.0:
        cost += left * float(prices[-1])
    return cost / size, beyond


def test_a_walk_inside_the_visible_book() -> None:
    walk = walk_book(np.array([100.0, 101.0, 102.0]), np.array([1.0, 2.0, 3.0]), 2.0)
    assert walk.vwap[0] == pytest.approx(100.5)
    assert walk.beyond[0] == 0.0
    assert walk.visible[0] == 6.0


def test_size_beyond_the_last_level_is_charged_at_its_price_and_reported() -> None:
    walk = walk_book(np.array([100.0, 101.0, 102.0]), np.array([1.0, 2.0, 3.0]), 8.0)
    assert walk.vwap[0] == pytest.approx((100 + 202 + 306 + 2 * 102) / 8)
    assert walk.beyond[0] == pytest.approx(2.0)


def test_rows_are_walked_to_the_last_bit_as_a_loop_would() -> None:
    rng = np.random.default_rng(0)
    rows, depth = 500, 10
    prices = 10_000 + np.cumsum(rng.integers(1, 3, (rows, depth)), axis=1)
    sizes = rng.integers(0, 15, (rows, depth)) * 0.001 + rng.random((rows, depth)) * 0.01
    wanted = rng.random(rows) * 0.2 + 0.001
    walk = walk_book(prices, sizes, wanted)
    for row in range(rows):
        vwap, beyond = _loop(prices[row], sizes[row], float(wanted[row]))
        assert walk.vwap[row] == vwap
        assert walk.beyond[row] == beyond


def test_missing_levels_are_empty_and_the_deepest_visible_price_is_charged() -> None:
    prices = np.array([[100.0, 101.0, np.nan], [100.0, np.nan, np.nan]])
    sizes = np.array([[1.0, 1.0, np.nan], [1.0, np.nan, np.nan]])
    walk = walk_book(prices, sizes, 3.0)
    assert walk.vwap[0] == pytest.approx((100 + 101 + 101) / 3)
    assert walk.vwap[1] == pytest.approx(100.0)
    assert walk.beyond.tolist() == pytest.approx([1.0, 2.0])
    assert walk.visible.tolist() == [2.0, 1.0]


def test_a_walk_refuses_a_non_positive_size_and_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="positive"):
        walk_book(np.array([1.0]), np.array([1.0]), 0.0)
    with pytest.raises(ValueError, match="shape"):
        walk_book(np.ones((2, 3)), np.ones((2, 2)), 1.0)


def test_book_arrays_and_the_round_trip_cost_from_a_frame() -> None:
    frame = pd.DataFrame(
        {
            bid_price_col(0): [99.0],
            bid_size_col(0): [1.0],
            bid_price_col(1): [98.0],
            bid_size_col(1): [5.0],
            ask_price_col(0): [101.0],
            ask_size_col(0): [1.0],
            ask_price_col(1): [102.0],
            ask_size_col(1): [5.0],
        }
    )
    bid_px, bid_sz, ask_px, ask_sz = book_arrays(frame, depth=2)
    assert bid_px.shape == (1, 2)
    cost = round_trip_cost_bp(bid_px, bid_sz, ask_px, ask_sz, 2.0)
    # Buy at 101.5, sell at 98.5, on a mid of 100: 300 bp.
    assert cost[0] == pytest.approx(300.0)
    with pytest.raises(KeyError, match="lacks"):
        book_arrays(frame, depth=3)
