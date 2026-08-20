"""The frozen reversion candidate: its parameters, and the line that matters."""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.strategies.reversion import (
    GRID_SECONDS,
    HELD_OUT_RESULT,
    ReversionConfig,
    decide,
    index_level,
    signal,
    threshold_for_rate,
)


def test_the_frozen_parameters_are_what_was_measured() -> None:
    """A candidate whose parameters drift is a different candidate.

    These values are what the search block chose and what the held-out block was
    then read with. Changing them is allowed; changing them silently is how a
    candidate quietly becomes a re-tuned result.
    """
    config = ReversionConfig()
    assert config.lookback == 120
    assert config.hold == 120
    assert config.lookback_seconds == 600
    assert config.hold_seconds == 600
    assert config.trades_per_day == 60.0
    assert config.exclude == ("BTCUSDT", "ETHUSDT")
    assert GRID_SECONDS == 5


def test_the_recorded_result_keeps_its_caveat() -> None:
    """The number and the reason not to trust it travel together."""
    assert HELD_OUT_RESULT["positive_instruments"] == 20
    assert HELD_OUT_RESULT["instruments"] == 26
    assert "not independent" in HELD_OUT_RESULT["caveat"]
    # The decay is the main reason this is a candidate rather than a result, so
    # it must stay visible beside the held-out figure.
    assert (
        HELD_OUT_RESULT["search_block_median_net_bp"] > HELD_OUT_RESULT["median_net_bp_per_trade"]
    )


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------


def test_the_index_excludes_the_column_being_predicted() -> None:
    """The one line where a mistake manufactures a result rather than losing money."""
    panel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(index_level(panel, exclude=0), [2.5, 5.5])
    assert np.allclose(index_level(panel, exclude=2), [1.5, 4.5])
    assert np.allclose(index_level(panel), [2.0, 5.0])


def test_excluding_the_only_column_is_refused() -> None:
    with pytest.raises(ValueError, match="no index"):
        index_level(np.ones((10, 1)), exclude=0)


def test_a_column_outside_the_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="outside a panel"):
        index_level(np.ones((10, 3)), exclude=7)


def test_a_one_dimensional_input_is_refused() -> None:
    with pytest.raises(ValueError, match="axes"):
        index_level(np.ones(10), exclude=0)


# ---------------------------------------------------------------------------
# Signal and decision
# ---------------------------------------------------------------------------


def test_the_signal_is_the_index_return_in_basis_points() -> None:
    level = np.log(np.linspace(100.0, 101.0, 500))
    values = signal(level, ReversionConfig(lookback=100))
    assert np.isnan(values[:100]).all(), "no signal before the window is full"
    expected = (level[100] - level[0]) * 1e4
    assert values[100] == pytest.approx(expected)


def test_the_decision_is_against_the_move() -> None:
    """Reversion. Getting this backwards produces a plausible mirror image."""
    values = np.array([50.0, -50.0, 1.0, np.nan])
    out = decide(values, threshold=10.0)
    assert out[0] == -1, "the index rose, so sell"
    assert out[1] == +1, "the index fell, so buy"
    assert out[2] == 0, "too small to bother with"
    assert out[3] == 0, "no signal, no trade"


def test_a_negative_or_missing_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        decide(np.zeros(10), threshold=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        decide(np.zeros(10), threshold=float("nan"))


def test_the_threshold_delivers_roughly_the_requested_rate() -> None:
    rng = np.random.default_rng(0)
    rows_per_day = 17_280
    values = rng.normal(0, 20, rows_per_day * 10)
    config = ReversionConfig(trades_per_day=60.0)
    threshold = threshold_for_rate(values, rows_per_day, config)

    signalled = (np.abs(values) >= threshold).sum()
    # Signalled rows, not trades: a hold and a cooldown thin these down. What
    # matters is that the quantile is in the right place.
    assert signalled == pytest.approx(60 * 10, rel=0.1)


def test_a_higher_rate_means_a_lower_threshold() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0, 20, 100_000)
    rare = threshold_for_rate(values, 17_280, ReversionConfig(trades_per_day=5.0))
    often = threshold_for_rate(values, 17_280, ReversionConfig(trades_per_day=200.0))
    assert rare > often


def test_too_few_rows_to_set_a_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="usable rows"):
        threshold_for_rate(np.zeros(100), 17_280, ReversionConfig())
