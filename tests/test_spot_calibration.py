"""FILL-S, its thinned form and the capacity ratio, on synthetic books and prints."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from _sh_synthetic import make_prints, make_rows
from test_spot_synthetic_prints import no_network
from trading_research.backtest.costs import FeeTier
from trading_research.backtest.taker import TakerBook, TakerSettings, execute_day
from trading_research.features.timed import stamps_ns
from trading_research.spot.calibration import (
    book_orders,
    capacity_median,
    capacity_ratio,
    k_fill_charges,
    k_fill_fee_bp,
    price_orders,
    side_summary,
    thin_prints,
    thinned_calibration,
    trade_fills,
)
from trading_research.spot.fills import BRACKETS, PrintTape, fill_orders

DAY = date(2026, 3, 3)
S = 1_000_000_000
#: A test fee; a study run builds its settings from the fee table (spot/costs.py).
SETTINGS = TakerSettings(fees=FeeTier("test", 10.0, 10.0))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


@pytest.fixture(scope="module")
def market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = make_rows(2_500, start="2026-03-03 00:01:00", quiet_at=1_500)
    prints = make_prints(rows, rate=2.0).assign(rpi=False)
    stamps = stamps_ns(rows)
    rng = np.random.default_rng(0)
    decisions = np.sort(rng.choice(stamps[10:-50], 40, replace=False))
    sides = np.where(rng.random(40) < 0.5, 1, -1)
    run = execute_day(TakerBook.from_frame(rows, DAY), decisions, sides, 5 * S, SETTINGS)
    return rows, prints, run.trades


def test_every_entry_and_exit_is_an_order(market) -> None:
    _, _, trades = market
    orders = book_orders(trades)
    assert len(orders) == 2 * len(trades) > 10
    entries = orders[orders["kind"] == "entry"]
    assert (entries["side"].to_numpy() == trades["side"].to_numpy()).all()
    assert (np.diff(orders["arrival_ns"].to_numpy()) >= 0).all()


def test_orders_are_priced_by_the_shared_fill_function(market) -> None:
    _, prints, trades = market
    tape = PrintTape.from_frame(prints, DAY)
    priced = price_orders(tape, book_orders(trades))
    exits = priced[priced["kind"] == "exit"]
    direct = fill_orders(tape, exits["side"].to_numpy(), exits["arrival_ns"].to_numpy(),
                         exits["decision_ns"].to_numpy(), until_ns=tape.end_ns, fallback=True)  # fmt: skip
    for name in BRACKETS:
        np.testing.assert_array_equal(exits[name].to_numpy(), getattr(direct, name))
    buys = priced["side"] == 1
    diff = (priced["rule"] - priced["book_price"]) / priced["book_mid"] * 1e4
    np.testing.assert_allclose(priced.loc[buys, "adv_rule_bp"], -diff[buys])
    np.testing.assert_allclose(priced.loc[~buys, "adv_rule_bp"], diff[~buys])
    summary = side_summary(priced, "worse_of_two")
    assert list(summary.index) == ["buy", "sell"] and (summary["orders"] > 0).all()
    with pytest.raises(ValueError, match="bracket"):
        side_summary(priced, "best")


def test_thinning_keeps_archive_order_and_depends_only_on_the_seed(market) -> None:
    _, prints, _ = market
    one, again = thin_prints(prints, 300, 3), thin_prints(prints, 300, 3)
    assert len(one) == 300 and one.index.equals(again.index)
    assert one.index.is_monotonic_increasing
    assert not one.index.equals(thin_prints(prints, 300, 4).index)
    assert thin_prints(prints, len(prints) + 5, 0) is prints
    with pytest.raises(ValueError, match="non-negative"):
        thin_prints(prints, -1, 0)


def test_k_fill_reads_the_worse_of_two_mean_over_every_seed(market) -> None:
    _, prints, trades = market
    days = [thinned_calibration(prints, 400, trades, DAY)]
    assert sorted(days[0]["seed"].unique()) == list(range(10))
    charges = k_fill_charges(days)
    every = days[0][days[0]["worse_of_two"].notna()]
    expected = every.loc[every["side"] == 1, "adv_worse_of_two_bp"].mean()
    assert charges["a_buy"] == pytest.approx(expected)
    assert k_fill_fee_bp({"a_buy": -2.0, "a_sell": 1.5}, 1) == 0.0
    assert k_fill_fee_bp({"a_buy": -2.0, "a_sell": 1.5}, -1) == 1.5
    # Thin prints wait longer: the share of late prints cannot fall.
    full = price_orders(PrintTape.from_frame(prints, DAY), book_orders(trades))
    assert days[0]["late"].mean() >= full["late"].mean()


def test_capacity_is_the_larger_side_share_of_the_venues_taker_volume() -> None:
    prints = pd.DataFrame({"aggressor": [1, 1, -1, -1], "size": [1.0, 3.0, 0.5, 1.5]})
    trades = pd.DataFrame({"side": [1, -1], "size": [0.4, 0.4]})
    side, size = trade_fills(trades)
    assert side.tolist() == [1, -1, -1, 1]
    assert capacity_ratio(side, size, prints) == pytest.approx(0.8 / 2.0)
    assert capacity_ratio(np.zeros(0), np.zeros(0), prints) == 0.0
    assert capacity_ratio(np.array([1]), np.array([0.1]), prints.iloc[2:]) == float("inf")
    assert capacity_median([0.0, 0.01, 0.03]) == 0.01
    assert np.isnan(capacity_median([]))
