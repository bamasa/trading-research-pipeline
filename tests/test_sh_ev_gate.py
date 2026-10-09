"""The EV gate, its units check, events and the entry thinning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _sh_synthetic import make_rows
from trading_research.backtest.costs import BYBIT_BASE, BYBIT_LINEAR_TIERS
from trading_research.backtest.walk import walk_cost_bp
from trading_research.features.normalise import RollingNormaliser
from trading_research.strategies.ev_gate import (
    EVGate,
    UnitsError,
    gate_events,
    thin_entries,
)


def test_the_gate_compares_the_bound_with_every_cost() -> None:
    gate = EVGate(k=1.0, beta_bp=1.0)
    assert gate.cost_bp == 11.0
    mu = np.array([20.0, -14.0, 12.0, np.nan, 13.0])
    sigma = np.array([4.0, 2.0, 0.0, 1.0, 0.0])
    walk = np.array([0.5, 0.5, 0.5, 0.5, 1.5])
    assert gate.sides(mu, sigma, walk).tolist() == [1, 0, 0, 0, 0]
    assert EVGate(k=0.5, beta_bp=1.0).sides(mu, sigma, walk).tolist() == [1, -1, 0, 0, 0]


def test_cost_follows_the_tier() -> None:
    assert EVGate.at_tier(BYBIT_BASE, k=0, beta_bp=0).cost_bp == 11.0
    cheapest = next(t for t in BYBIT_LINEAR_TIERS if t.name == "pro6_g1")
    assert EVGate.at_tier(cheapest, k=0, beta_bp=0).cost_bp == 3.0


def test_a_normalised_column_is_refused_as_a_gate_input() -> None:
    rows = make_rows(300, seed=2)
    frame = pd.DataFrame({"walk_cost_bp": walk_cost_bp(rows, 0.01)})
    normalised = RollingNormaliser(window=50).transform(frame, ["walk_cost_bp"])
    gate = EVGate(k=0.5, beta_bp=0.0)
    mu = np.zeros(len(frame))
    with pytest.raises(UnitsError, match="normalised"):
        gate.sides(mu, np.ones(len(frame)), normalised["walk_cost_bp"])
    gate.sides(mu, np.ones(len(frame)), frame["walk_cost_bp"])
    with pytest.raises(UnitsError, match="sigma"):
        gate.sides(mu, -np.ones(len(frame)), frame["walk_cost_bp"])


def test_a_known_book_gives_the_expected_walk_cost() -> None:
    rows = make_rows(3, seed=1)
    for side, price in (("bid", 50_000.0), ("ask", 50_000.1)):
        rows[f"{side}_price_0"] = price
        rows[f"{side}_size_0"] = 1.0
    rows["ask_price_1"] = 50_000.3
    expected = 0.1 / 50_000.05 * 1e4
    np.testing.assert_allclose(walk_cost_bp(rows, 0.01), expected, rtol=1e-9)
    rows["ask_size_0"] = 0.005
    rows["ask_size_1"] = 1.0
    walked = (0.005 * 50_000.1 + 0.005 * 50_000.3) / 0.01 - 50_000.0
    np.testing.assert_allclose(walk_cost_bp(rows, 0.01), walked / 50_000.05 * 1e4, atol=1e-8)


def test_events_are_runs_of_open_rows() -> None:
    sides = np.array([0, 1, 1, 0, -1, -1, 1, 0, 1])
    assert gate_events(sides).tolist() == [-1, 0, 0, -1, 1, 1, 1, -1, 2]


def test_one_entry_per_event_and_one_position_at_a_time() -> None:
    sides = np.array([1, 1, 1, 0, 1, 1, 0, 0, -1, -1, 0, 1])
    events = gate_events(sides)
    busy = np.arange(len(sides)) + 5
    # Row 0 enters and holds to row 5; the event of rows 4-5 opened while busy
    # and enters at its first free row, 6? No: 6 is closed, so it never enters.
    assert thin_entries(sides, busy, events).tolist() == [0]
    busy = np.arange(len(sides)) + 2
    assert thin_entries(sides, busy, events).tolist() == [0, 4, 8]
    assert thin_entries(sides, busy).tolist() == [0, 4, 8, 11][:3]


def _reference(sides: np.ndarray, busy: np.ndarray, events: np.ndarray | None) -> list[int]:
    out: list[int] = []
    free, used = 0, -1
    for row in range(len(sides)):
        if row < free or sides[row] == 0:
            continue
        if events is not None and events[row] == used:
            continue
        if busy[row] >= len(sides):
            break
        out.append(row)
        free = busy[row] + 1
        used = -1 if events is None else events[row]
    return out


@pytest.mark.parametrize("seed", range(5))
def test_thinning_equals_a_row_by_row_loop(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sides = rng.choice([-1, 0, 0, 1], size=400)
    busy = np.arange(400) + rng.integers(1, 15, 400)
    events = gate_events(sides)
    assert thin_entries(sides, busy, events).tolist() == _reference(sides, busy, events)
    assert thin_entries(sides, busy).tolist() == _reference(sides, busy, None)
