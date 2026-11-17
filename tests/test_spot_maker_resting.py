"""The resting-order fill rule: trade-through, touch and delayed, partial by print size."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import resting as rs

DAY = date(2026, 5, 4)
T0 = int(np.datetime64("2026-05-04T00:00:00", "ns").astype(np.int64))
S = 1_000_000_000


def _tape() -> PrintTape:
    ts = T0 + np.array([1, 2, 3, 4, 5, 6], dtype=np.int64) * S
    price = np.array([100.00, 99.99, 100.00, 99.98, 100.05, 99.97])
    size = np.array([0.004, 0.003, 0.002, 0.009, 0.020, 0.001])
    side = np.array([-1, -1, -1, -1, 1, -1])
    return PrintTape(DAY, ts, price, size, side)


def test_trade_through_fills_strictly_beyond_partial_by_size() -> None:
    idx, size = rs.fill(_tape(), 1, 100.00, 0.010, T0, T0 + 10 * S, "trade_through")
    assert list(idx) == [1, 3] and list(size) == pytest.approx([0.003, 0.007])


def test_touch_fills_at_the_price_too_and_delayed_skips_the_first() -> None:
    idx, size = rs.fill(_tape(), 1, 100.00, 0.010, T0, T0 + 10 * S, "touch")
    assert list(idx) == [0, 1, 2, 3] and size.sum() == pytest.approx(0.010)
    idx, size = rs.fill(_tape(), 1, 100.00, 0.010, T0, T0 + 10 * S, "delayed")
    assert list(idx) == [3, 5] and list(size) == pytest.approx([0.009, 0.001])


def test_life_bounds_side_and_sell_orders() -> None:
    idx, _ = rs.fill(_tape(), 1, 100.00, 0.010, T0 + 2 * S + 1, T0 + 4 * S, "trade_through")
    assert len(idx) == 0  # the print at 4 s is at the end bound, excluded
    idx, size = rs.fill(_tape(), -1, 100.01, 0.010, T0, T0 + 10 * S, "trade_through")
    assert list(idx) == [4] and size[0] == pytest.approx(0.010)  # a buyer print above 100.01
    with pytest.raises(ValueError):
        rs.fill(_tape(), 1, 100.0, 0.01, T0, T0 + S, "queue")


def test_flatten_uses_the_spot_taker_rule_and_falls_back() -> None:
    tape = _tape()
    ts, price, back = rs.flatten(tape, -1, 0.002, T0 + 3 * S, 10_000_000)
    assert (ts, price, back) == (T0 + 4 * S, 99.98, False)
    ts, price, back = rs.flatten(tape, 1, 0.002, T0 + 7 * S, 10_000_000)
    assert back and price == 100.05 and ts == T0 + 7 * S + 10_000_000
