"""The frozen reversion candidate: its parameters, and the line that matters."""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.strategies.reversion import (
    GRID_SECONDS,
    HELD_OUT_RESULT,
    ROWS_PER_DAY,
    ReversionConfig,
    decide,
    index_level,
    reversion_tape,
    signal,
    threshold_for_rate,
    trailing_autocorrelation,
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


def test_the_refutation_is_recorded_beside_the_result() -> None:
    """A killed candidate must carry the number that killed it.

    The temptation, once a finding dies, is to delete it. Keeping the frozen
    parameters next to the result on data that chose nothing is what makes the
    record useful to anyone who finds the same thing later.
    """
    from trading_research.strategies.reversion import FRESH_SPAN_RESULT

    assert FRESH_SPAN_RESULT["positive_instruments"] == 0
    assert FRESH_SPAN_RESULT["median_net_bp_per_trade"] < 0
    # The gross edge changed sign, which is the part that distinguishes an
    # absent effect from a decayed one.
    assert FRESH_SPAN_RESULT["median_gross_bp_per_trade"] < 0
    assert "killed" in FRESH_SPAN_RESULT["verdict"]


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


def test_the_trailing_autocorrelation_known_at_the_open_reads_nothing_of_its_day() -> None:
    """With ``known_at_open`` a day's value is fixed before the day starts:
    rewriting the day (and every later one) leaves it unchanged. Without it,
    the window's last forward returns reach ``lag`` rows into the day, as in
    the regime-gate experiment of section 27."""
    rng = np.random.default_rng(1)
    level = np.cumsum(rng.normal(0, 1e-4, 4 * ROWS_PER_DAY))
    shocked = level.copy()
    shocked[2 * ROWS_PER_DAY :] += np.cumsum(rng.normal(0, 5e-4, 2 * ROWS_PER_DAY))
    for at_open, same in ((True, True), (False, False)):
        a = trailing_autocorrelation(level, 120, ROWS_PER_DAY, known_at_open=at_open)
        b = trailing_autocorrelation(shocked, 120, ROWS_PER_DAY, known_at_open=at_open)
        day_two = slice(2 * ROWS_PER_DAY, 3 * ROWS_PER_DAY)
        assert np.isfinite(a[day_two]).all()
        assert np.array_equal(a[day_two], b[day_two]) is same
    first = trailing_autocorrelation(level, 120, ROWS_PER_DAY, known_at_open=True)
    assert np.isnan(first[:ROWS_PER_DAY]).all()
    assert np.isfinite(first[ROWS_PER_DAY : 2 * ROWS_PER_DAY]).all()


def test_the_tape_needs_bin_end_labels_and_its_instrument() -> None:
    import pandas as pd

    index = pd.date_range("2024-02-01 00:00:05", periods=300, freq="5s", tz="UTC")
    rng = np.random.default_rng(2)
    panel = pd.DataFrame(
        np.cumsum(rng.normal(0, 1e-4, (300, 3)), axis=0), index=index, columns=["A", "B", "C"]
    )
    tape = reversion_tape(panel, "A", ReversionConfig())
    assert tape.index.equals(index)
    level = panel[["B", "C"]].to_numpy().mean(axis=1)
    assert tape.iloc[200] == pytest.approx((level[200] - level[80]) * 1e4)
    assert np.isnan(tape.iloc[:120]).all()
    with pytest.raises(KeyError):
        reversion_tape(panel, "Z", ReversionConfig())
    with pytest.raises(ValueError, match="DatetimeIndex"):
        reversion_tape(panel.reset_index(drop=True), "A", ReversionConfig())
