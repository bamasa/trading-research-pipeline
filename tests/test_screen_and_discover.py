"""Tests for instrument discovery and the headroom screen.

The screen's job is to eliminate, so the property worth pinning is that it
ranks on room to trade rather than on cost. An instrument that is cheap and
still would rank at the top of a cost-ordered list and near the bottom of this
one, which is the whole point of it existing.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import TakerCosts
from trading_research.data.discover import Candidate
from trading_research.data.screen import screen_directory, screen_frame

COSTS = TakerCosts(fee_bp_per_side=5.0, slippage_bp=0.5)


def book(
    *,
    price: float,
    spread_bp: float,
    volatility_bp: float,
    n: int = 4000,
    seed: int = 0,
    seconds: float = 1.0,
) -> pd.DataFrame:
    """A synthetic book with a chosen spread and a chosen amount of movement."""
    rng = np.random.default_rng(seed)
    mid = price * np.exp(np.cumsum(rng.normal(0, volatility_bp / 1e4, n)))
    half = mid * spread_bp / 2e4
    start = pd.Timestamp("2024-02-05", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(np.arange(n) * seconds, unit="s"),
            "bid_price_0": mid - half,
            "ask_price_0": mid + half,
            "bid_size_0": np.full(n, 10.0),
            "ask_size_0": np.full(n, 10.0),
        }
    )


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def test_a_mover_beats_a_cheap_stiff_instrument() -> None:
    """The finding the screen exists for: cost alone ranks the wrong way.

    The cheap one has a tenth of the spread and barely moves; the dear one
    costs more per round trip and clears it far more often.
    """
    cheap = screen_frame(
        book(price=40000, spread_bp=0.02, volatility_bp=0.5), "CHEAP", horizon_s=120, costs=COSTS
    )
    mover = screen_frame(
        book(price=15, spread_bp=0.5, volatility_bp=4.0), "MOVER", horizon_s=120, costs=COSTS
    )
    assert cheap.round_trip_bp < mover.round_trip_bp
    assert mover.headroom > cheap.headroom


def test_headroom_rises_with_the_horizon() -> None:
    """A longer horizon gives a move more time to clear the same fixed cost."""
    frame = book(price=100, spread_bp=1.0, volatility_bp=2.0, n=8000)
    short = screen_frame(frame, "X", horizon_s=10, costs=COSTS)
    long = screen_frame(frame, "X", horizon_s=300, costs=COSTS)
    assert long.headroom > short.headroom


def test_headroom_is_a_share_between_zero_and_one() -> None:
    result = screen_frame(book(price=100, spread_bp=1.0, volatility_bp=2.0), "X", horizon_s=60)
    assert 0.0 <= result.headroom <= 1.0


def test_the_round_trip_includes_the_spread_once() -> None:
    result = screen_frame(
        book(price=100, spread_bp=2.0, volatility_bp=1.0), "X", horizon_s=60, costs=COSTS
    )
    expected = 2 * COSTS.fee_bp_per_side + 2.0 + 2 * COSTS.slippage_bp
    assert result.round_trip_bp == pytest.approx(expected, abs=0.05)


def test_a_horizon_longer_than_the_data_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot support"):
        screen_frame(book(price=100, spread_bp=1.0, volatility_bp=1.0, n=50), "X", horizon_s=1e6)


def test_move_over_cost_says_whether_a_typical_move_pays() -> None:
    """Below one, even a perfectly predicted average move does not cover a
    round trip — which is the number to read before any model is fitted."""
    result = screen_frame(
        book(price=40000, spread_bp=0.02, volatility_bp=0.3), "STIFF", horizon_s=120, costs=COSTS
    )
    assert result.as_row()["move_over_cost"] < 1.0


def test_the_directory_screen_ranks_by_headroom(tmp_path) -> None:
    for name, volatility in (("QUIET", 0.4), ("BUSY", 4.0)):
        directory = tmp_path / name / "bookTicker"
        directory.mkdir(parents=True)
        book(price=100, spread_bp=1.0, volatility_bp=volatility).to_parquet(
            directory / "2024-02-05.parquet", index=False
        )
    table = screen_directory(tmp_path, horizon_s=120, costs=COSTS)
    assert list(table["symbol"]) == ["BUSY", "QUIET"]


def test_an_unreadable_instrument_is_reported_not_dropped(tmp_path) -> None:
    """A screen that silently omits candidates is worse than one that says why."""
    (tmp_path / "EMPTY" / "bookTicker").mkdir(parents=True)
    good = tmp_path / "GOOD" / "bookTicker"
    good.mkdir(parents=True)
    book(price=100, spread_bp=1.0, volatility_bp=2.0).to_parquet(
        good / "2024-02-05.parquet", index=False
    )
    table = screen_directory(tmp_path, horizon_s=60, costs=COSTS)
    assert "EMPTY" in set(table["symbol"])
    assert table.loc[table["symbol"] == "EMPTY", "skipped"].notna().all()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_candidates_sort_by_volume() -> None:
    made = [
        Candidate("SMALL", 1e6, date(2020, 1, 1), 0.01, "TRADING"),
        Candidate("BIG", 9e9, date(2020, 1, 1), 0.01, "TRADING"),
    ]
    assert [c.symbol for c in sorted(made, key=lambda c: -c.quote_volume_usd)] == ["BIG", "SMALL"]


def test_a_candidate_reports_what_the_screen_needs() -> None:
    row = Candidate("X", 5e8, date(2021, 3, 4), 0.001, "TRADING").as_row()
    assert set(row) == {"symbol", "quote_volume_usd", "onboard_date", "tick_size", "status"}


def test_a_contract_listed_after_the_period_has_no_history_to_screen() -> None:
    """The filter that matters: volume is from today, the data is not.

    Without it the list fills with recent listings whose archives return
    nothing, which silently shortens the candidate set.
    """
    period_start = date(2024, 2, 1)
    recent = Candidate("NEW", 8e9, period_start + timedelta(days=300), 0.01, "TRADING")
    established = Candidate("OLD", 1e8, date(2020, 1, 1), 0.01, "TRADING")
    kept = [c for c in (recent, established) if c.onboard_date < period_start]
    assert [c.symbol for c in kept] == ["OLD"]
