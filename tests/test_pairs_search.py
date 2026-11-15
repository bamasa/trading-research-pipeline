"""The pairs search's selection, placebos, funding and portfolio accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.search_engine import pairs as ps
from trading_research.strategies.carry import Market


def windows() -> pd.DataFrame:
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    rows = []
    for w in range(2):
        for k, (a, b) in enumerate(pairs):
            p = [1e-6, 0.5, 0.9][k] if w == 0 else 0.5
            rows.append({"window": w, "a": a, "b": b, "p_value": p, "eg_t": -5.0 + k, "beta": 1.0})
    return pd.DataFrame(rows)


def test_only_pairs_surviving_holm_are_chosen() -> None:
    assert ps.chosen(windows()) == {(0, "A", "B")}


def test_placebos_draw_as_many_pairs_each_week() -> None:
    w = windows()
    picked = ps.chosen(w)
    for seed in range(5):
        drawn = ps.random_pairs(w, picked, seed)
        assert len(drawn) == 1 and next(iter(drawn))[0] == 0


def test_funding_is_paid_long_and_received_short() -> None:
    hours = pd.date_range("2023-01-01", periods=48, freq="1h", tz="UTC")
    flat = pd.Series(1.0, index=hours)
    rates = pd.Series(0.0001, index=hours[::8])  # 1 bp a settlement
    markets = {"A": Market(flat, flat, rates), "B": Market(flat, flat, rates * 0)}
    trades = pd.DataFrame({"a": ["A"], "b": ["B"], "side": [1], "beta": [1.0],
                           "entry_fill": [hours[1]], "exit_fill": [hours[25]]})  # fmt: skip
    # long A over three settlements (08, 16, 24h): pays 3 bp
    assert np.isclose(ps.funding_bp(markets, trades)[0], -3.0)


def test_portfolio_weights_a_slot_over_both_legs() -> None:
    span = (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-03-01", tz="UTC"))
    trades = pd.DataFrame({"window": [0, 0], "a": ["A", "A"], "b": ["B", "C"], "beta": [1.0, 1.0],
                           "net_with_funding_bp": [100.0, -100.0], "gross_bp": [120.0, -80.0],
                           "exit_fill": [pd.Timestamp("2023-01-10", tz="UTC")] * 2})  # fmt: skip
    frame = ps.portfolio(trades, {(0, "A", "B")}, span)
    assert list(frame["month"]) == ["2023-01", "2023-02"]
    assert np.isclose(frame["return_pct"].iloc[0], 100 / ps.SLOTS / 2 / 1e4 * 100)
    assert frame.attrs["trades"] == 1


def test_the_grid_is_the_registrations() -> None:
    assert {(r.entry_z, r.formation) for r in ps.GRID} == {
        (2.0, 720),
        (2.0, 1440),
        (2.5, 720),
        (2.5, 1440),
    }
    assert len(ps.UNIVERSE) == 15 and "BTCUSDT" not in ps.UNIVERSE
