"""Tests for the horizon diagnostics.

These two measurements are what the project's central finding rests on, so the
tests check they measure what they claim: that the information coefficient
picks up a planted signal with the right sign and decays when the signal does,
and that the breakeven share counts moves against the cost rather than against
each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobml.reporting.diagnostics import (
    breakeven_by_horizon,
    format_horizon_summary,
    forward_log_return,
    horizon_summary,
    information_coefficient_by_horizon,
)


def book_with_known_drift(n: int = 5000, strength: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A book whose next return is driven by the current queue imbalance.

    The imbalance at row *t* moves the price between *t* and *t+1*, so a
    correctly implemented information coefficient must find it at horizon 1 and
    see it dilute as the horizon lengthens and unrelated noise accumulates.
    """
    rng = np.random.default_rng(seed)
    imbalance = rng.uniform(-1, 1, n)

    step = rng.normal(0, 1e-4, n)
    step[1:] += strength * 1e-4 * imbalance[:-1]
    mid = 30_000 * np.exp(np.cumsum(step))

    # Sizes chosen so that (bid - ask) / (bid + ask) reproduces `imbalance`.
    total = 10.0
    bid = total * (1 + imbalance) / 2
    ask = total * (1 - imbalance) / 2

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n, freq="100ms", tz="UTC"),
            "symbol": pd.array(["X"] * n, dtype="string"),
            "sequence_id": np.arange(1, n + 1, dtype="int64"),
            "source": pd.array(["test"] * n, dtype="string"),
            "bid_price_0": mid - 0.05,
            "bid_size_0": bid,
            "ask_price_0": mid + 0.05,
            "ask_size_0": ask,
        }
    )


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------


def test_forward_return_looks_the_right_distance_ahead() -> None:
    book = book_with_known_drift(200)
    got = forward_log_return(book, 10)
    mid = (book["bid_price_0"] + book["ask_price_0"]) / 2
    assert got[0] == pytest.approx(np.log(mid.iloc[10] / mid.iloc[0]))


def test_the_tail_has_no_forward_return() -> None:
    got = forward_log_return(book_with_known_drift(200), 10)
    assert np.isnan(got[-10:]).all()
    assert not np.isnan(got[:-10]).any()


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------


def test_a_planted_signal_is_found_with_the_right_sign() -> None:
    book = book_with_known_drift(n=20_000, strength=3.0)
    table = information_coefficient_by_horizon(book, ["queue_imbalance"], horizons=(1, 5))
    assert table.loc[1, "queue_imbalance"] > 0.1


def test_no_signal_means_no_information_coefficient() -> None:
    """The control: with the drift switched off, nothing should be found."""
    book = book_with_known_drift(n=20_000, strength=0.0)
    table = information_coefficient_by_horizon(book, ["queue_imbalance"], horizons=(1, 5))
    assert abs(table.loc[1, "queue_imbalance"]) < 0.05


def test_the_coefficient_decays_with_horizon() -> None:
    """A one-step effect dilutes as unrelated noise accumulates around it."""
    book = book_with_known_drift(n=40_000, strength=3.0)
    table = information_coefficient_by_horizon(book, ["queue_imbalance"], horizons=(1, 50, 500))
    values = table["queue_imbalance"].abs()
    assert values.loc[1] > values.loc[50] > values.loc[500]


def test_horizons_index_the_result() -> None:
    book = book_with_known_drift(2000)
    table = information_coefficient_by_horizon(book, ["spread_bp"], horizons=(1, 10, 100))
    assert list(table.index) == [1, 10, 100]


# ---------------------------------------------------------------------------
# Breakeven share
# ---------------------------------------------------------------------------


def test_breakeven_counts_moves_larger_than_the_cost() -> None:
    book = book_with_known_drift(n=20_000, strength=0.0)
    shares = breakeven_by_horizon(book, cost_bp=0.0, horizons=(10,))
    # Against a zero cost every non-zero move qualifies.
    assert shares.loc[10] > 0.95


def test_an_impossible_cost_leaves_nothing_tradeable() -> None:
    book = book_with_known_drift(n=20_000)
    shares = breakeven_by_horizon(book, cost_bp=1e6, horizons=(10, 100))
    assert (shares == 0.0).all()


def test_longer_horizons_clear_the_cost_more_often() -> None:
    """The arithmetic behind the central finding: moves grow, the fee does not."""
    book = book_with_known_drift(n=40_000)
    shares = breakeven_by_horizon(book, cost_bp=5.0, horizons=(10, 100, 1000))
    assert shares.loc[10] < shares.loc[100] < shares.loc[1000]


# ---------------------------------------------------------------------------
# Combined summary
# ---------------------------------------------------------------------------


def test_summary_puts_prediction_and_tradeability_side_by_side() -> None:
    book = book_with_known_drift(n=20_000, strength=3.0)
    table = horizon_summary(book, ["queue_imbalance"], cost_bp=5.0, horizons=(10, 100, 1000))
    assert "queue_imbalance" in table.columns
    assert "breakeven_share" in table.columns
    assert "seconds" in table.columns


def test_summary_converts_horizons_to_seconds() -> None:
    book = book_with_known_drift(5000)
    table = horizon_summary(book, ["spread_bp"], cost_bp=5.0, horizons=(10, 600), grid_ms=100)
    assert table.loc[10, "seconds"] == pytest.approx(1.0)
    assert table.loc[600, "seconds"] == pytest.approx(60.0)


def test_summary_renders_without_a_plotting_dependency() -> None:
    book = book_with_known_drift(n=20_000, strength=2.0)
    table = horizon_summary(book, ["queue_imbalance"], cost_bp=5.0, horizons=(10, 100))
    text = format_horizon_summary(table)
    assert "queue_imbalance" in text
    assert "%" in text
