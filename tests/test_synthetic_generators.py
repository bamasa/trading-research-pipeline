"""The generators: valid data, reproducible, causal, and planting what truth.json says."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.data import bybit_funding
from trading_research.data.validate import validate_bars, validate_book, validate_trades
from trading_research.features.book import queue_imbalance
from trading_research.synthetic.book_days import BookDaysSpec, generate
from trading_research.synthetic.break_days import BreakDaysSpec, generate_breaks
from trading_research.synthetic.cross_section import CrossSectionSpec
from trading_research.synthetic.cross_section import generate as generate_panel
from trading_research.synthetic.funding import FundingSpec, estimate_persistence
from trading_research.synthetic.funding import generate as generate_funding
from trading_research.synthetic.order_flow import ADD, CANCEL, TRADE, OrderFlowSpec, true_fills
from trading_research.synthetic.order_flow import generate as generate_flow

SMALL = BookDaysSpec(days=3, session_s=600, warmup_s=30.0)
runner = CliRunner()


def test_book_days_pass_the_validators_and_are_stamped_synthetic() -> None:
    data = generate(SMALL)
    for day in data.days:
        assert validate_book(data.book[day], depth=10).ok
        assert validate_trades(data.trades[day]).ok
        assert set(data.book[day]["source"]) == {"synthetic"}
        assert set(data.trades[day]["source"]) == {"synthetic"}
    assert data.truth.planted["size_bp"] == SMALL.size_bp


def test_the_same_seed_gives_the_same_days_and_another_seed_does_not() -> None:
    a, b = generate(SMALL), generate(SMALL)
    day = a.days[1]
    pd.testing.assert_frame_equal(a.book[day], b.book[day])
    pd.testing.assert_frame_equal(a.trades[day], b.trades[day])
    other = generate(BookDaysSpec(days=3, session_s=600, warmup_s=30.0, seed=7))
    assert not a.book[day]["bid_price_0"].equals(other.book[day]["bid_price_0"])


def test_the_feature_shows_at_the_onset_and_the_drift_starts_after_it() -> None:
    data = generate(SMALL)
    day = data.days[0]
    latent = data.latent[day]
    drift = latent["drift_log"].to_numpy()
    imbalance = queue_imbalance(data.book[day]).to_numpy()
    episodes = data.episodes[data.episodes["day"] == day.isoformat()]
    assert len(episodes) > 0
    for row, direction in zip(episodes["row"], episodes["feature_direction"], strict=True):
        # Nothing has moved at the onset; the feature already shows the direction.
        assert drift[row] == pytest.approx(drift[row - 1])
        assert np.sign(imbalance[row]) == direction
        moved = (drift[row + 300] - drift[row]) * 1e4 * direction
        assert moved == pytest.approx(SMALL.size_bp, rel=1e-9)


def test_the_null_keeps_the_feature_and_unhooks_the_drift_from_it() -> None:
    spec = BookDaysSpec(days=6, session_s=900, warmup_s=30.0, null=True)
    data = generate(spec)
    episodes = data.episodes
    agree = (episodes["feature_direction"] == np.sign(episodes["drift_direction"])).mean()
    assert 0.3 < agree < 0.7
    day = data.days[0]
    drift = data.latent[day]["drift_log"].to_numpy()
    for row in episodes[episodes["day"] == day.isoformat()]["row"]:
        # The whole move is in the step after the onset: nothing is left after it.
        assert drift[row + 1] != pytest.approx(drift[row])
        assert drift[row + 5] == pytest.approx(drift[row + 1])


def test_flow_episodes_lean_the_aggressor_and_arrive_in_a_burst() -> None:
    spec = BookDaysSpec(days=2, session_s=900, warmup_s=30.0, feature="flow")
    data = generate(spec)
    day = data.days[0]
    prints = data.prints(day)
    signal = data.latent[day]["signal"].to_numpy()
    stamps = data.book[day]["timestamp"].to_numpy(dtype="datetime64[ns]").view(np.int64)
    rows = np.searchsorted(
        stamps, prints["timestamp"].to_numpy(dtype="datetime64[ns]").view(np.int64)
    )
    lean = signal[rows] * prints["aggressor"].to_numpy()
    assert (lean[signal[rows] != 0] > 0).mean() > 0.75
    assert (signal[rows] != 0).mean() > 3 * (signal != 0).mean()


def test_a_written_dataset_passes_validate_data(tmp_path) -> None:
    out = generate(SMALL).write(tmp_path / "book_days")
    result = runner.invoke(app, ["validate-data", "--input", str(out)])
    assert result.exit_code == 0, result.output
    truth = json.loads((out / "truth.json").read_text())
    assert truth["source"] == "synthetic" and truth["spec"]["seed"] == SMALL.seed


def test_the_break_stream_changes_at_the_break_and_its_null_does_not() -> None:
    spec = BreakDaysSpec(break_at_s=36 * 3600).with_book(step_ms=1000)
    data, null = (
        generate_breaks(spec),
        generate_breaks(BreakDaysSpec(null=True, break_at_s=36 * 3600).with_book(step_ms=1000)),
    )
    mid = np.concatenate([np.log(data.latent[d]["mid_latent"].to_numpy()) for d in data.days])
    returns = np.diff(mid)
    at = 36 * 3600
    ratio = returns[at:].std() / returns[:at].std()
    assert 1.4 < ratio < 1.8
    day = data.days[1]
    after = data.book[day].iloc[-1]
    assert after["ask_price_0"] - after["bid_price_0"] >= 0.4 - 1e-9
    assert data.truth.planted["break_ns"] == spec.break_ns
    assert null.truth.planted["break_ns"] is None
    pd.testing.assert_frame_equal(data.book[data.days[0]], null.book[null.days[0]])


def test_the_cross_section_bars_validate_and_the_transient_reverts() -> None:
    data = generate_panel(CrossSectionSpec(days=3, n_instruments=8))
    for symbol, frames in data.bars.items():
        for frame in frames.values():
            assert validate_bars(frame).ok, symbol
    z = data.transient.to_numpy()
    hold = 60
    owed = data.spec.expected_move(z, hold)
    future = data.log_prices.shift(-hold).to_numpy() - data.log_prices.to_numpy()
    ok = np.isfinite(future).all(axis=1)
    slope = np.polyfit(owed[ok].ravel(), future[ok].ravel(), 1)[0]
    assert 0.7 < slope < 1.3


def test_the_cross_section_null_has_no_autocorrelation_left() -> None:
    spec = CrossSectionSpec(days=3, n_instruments=8, null=True)
    data = generate_panel(spec)
    assert float(np.abs(data.transient.to_numpy()).max()) == 0.0
    returns = data.log_prices.diff(60).to_numpy()[60:]
    lagged = returns[:-60].ravel()
    ahead = returns[60:].ravel()
    assert abs(np.corrcoef(lagged, ahead)[0, 1]) < 0.03


def test_order_flow_validates_and_its_markout_is_the_half_spread_less_the_planted() -> None:
    data = generate_flow(OrderFlowSpec(days=1, session_s=1800))
    day = data.days[0]
    assert validate_book(data.book_frame(day), depth=10).ok
    assert validate_trades(data.trades_frame(day)).ok
    from trading_research.market_making.analysis import market_wide_markouts

    markout = market_wide_markouts(data.events(day), (5.0,))["markout_bp"].iloc[0]
    expected = data.truth.planted["half_spread_bp"] - data.truth.planted["adverse_bp"]
    assert markout == pytest.approx(expected, abs=0.15)


def _log(rows: list[tuple[int, int, int, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts", "kind", "side", "price", "size", "added_ns"])


def test_true_fills_follow_the_queue_order_by_order() -> None:
    log = _log(
        [
            (10, ADD, 1, 100, 5, -1),  # ahead of our order
            (20, ADD, 1, 100, 3, -1),  # ahead of our order
            (40, ADD, 1, 100, 4, -1),  # behind it
            (50, CANCEL, 1, 100, 3, 20),  # ahead: the queue in front shrinks to 5
            (60, CANCEL, 1, 100, 4, 40),  # behind: nothing changes
            (70, TRADE, 1, 100, 4, -1),  # 4 of the 5 in front
            (80, TRADE, 1, 100, 3, -1),  # 1 more in front, 2 for us
            (90, TRADE, -1, 102, 9, -1),  # the other side: nothing
        ]
    )
    ours = pd.DataFrame(
        {"side": [1], "price": [100], "size": [3.0], "live_ns": [30], "end_ns": [1000]}
    )
    out = true_fills(log, ours)
    assert out["filled"].iloc[0] == 2.0
    assert out["first_fill_ns"].iloc[0] == 80


def test_true_fills_through_the_price_and_ties_go_to_the_market() -> None:
    log = _log(
        [
            (10, ADD, -1, 105, 6, -1),
            (30, TRADE, -1, 105, 6, -1),  # at our arrival stamp: before us
            (40, TRADE, -1, 106, 2, -1),  # a buyer through our ask: we fill
            (50, TRADE, -1, 106, 9, -1),  # after our end: nothing
        ]
    )
    ours = pd.DataFrame(
        {"side": [-1], "price": [105], "size": [5.0], "live_ns": [30], "end_ns": [45]}
    )
    out = true_fills(log, ours)
    assert out["filled"].iloc[0] == 2.0


def test_funding_recovers_its_persistence_and_reads_as_the_venues(tmp_path) -> None:
    planted = generate_funding(FundingSpec(days=730))
    null = generate_funding(FundingSpec(days=730, null=True))
    assert estimate_persistence(planted.series["rate"].to_numpy()) == pytest.approx(0.9, abs=0.03)
    assert abs(estimate_persistence(null.series["rate"].to_numpy())) < 0.08
    root = planted.write(tmp_path / "funding")
    day = bybit_funding.load_day("SYNTHUSDT", date(2020, 1, 7), root / "funding")
    assert len(day) == 3
    assert bybit_funding.settlement_interval_h(planted.series["timestamp"]) == 8
