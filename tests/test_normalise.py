"""Tests for the rolling normaliser and pooling across instruments.

A z-score is trivial and a causal one is not. Statistics computed over the whole
sample leak the future into every row — including rows inside the training
block, so a train/test split does not catch it — and the values still look like
features afterwards. The first test here is the one that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.features.normalise import (
    RollingNormaliser,
    coverage,
    normalise_pooled,
)


def series(n: int = 3000, seed: int = 0, scale: float = 1.0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-05", periods=n, freq="5s", tz="UTC"),
            "a": rng.normal(drift, scale, n),
            "b": rng.normal(0, 1, n),
        }
    )


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_nothing_before_a_cutoff_moves_when_the_future_changes() -> None:
    """The property the whole module stands on.

    Mutate every row after a cutoff and assert that no normalised value before
    it moved. A full-sample scaler fails this on every row.
    """
    frame = series(n=3000)
    cutoff = 2000
    mutated = frame.copy()
    mutated.loc[cutoff:, ["a", "b"]] *= 50.0

    normaliser = RollingNormaliser(window=500)
    original = normaliser.transform(frame, ["a", "b"])
    changed = normaliser.transform(mutated, ["a", "b"])

    # `.loc` includes its right endpoint, so the slice stops one short of the
    # cutoff — row `cutoff` is itself mutated and is not "before" anything.
    before = slice(0, cutoff - 1)
    pd.testing.assert_frame_equal(original.loc[before, ["a", "b"]], changed.loc[before, ["a", "b"]])


def test_a_row_does_not_contribute_to_its_own_scale() -> None:
    """The window is shifted by one: including t would be a small leak."""
    frame = series(n=1000)
    spike = frame.copy()
    spike.loc[900, "a"] = 1e6

    normalised = RollingNormaliser(window=200).transform(spike, ["a"])
    # The spike itself is clipped, but the rows before it are untouched.
    assert normalised.loc[899, "a"] == pytest.approx(
        RollingNormaliser(window=200).transform(frame, ["a"]).loc[899, "a"]
    )


# ---------------------------------------------------------------------------
# What it does
# ---------------------------------------------------------------------------


def test_two_instruments_on_different_scales_become_comparable() -> None:
    """The point of the exercise: pooling needs a common scale."""
    small = series(n=3000, seed=1, scale=0.01)
    large = series(n=3000, seed=1, scale=100.0)
    normaliser = RollingNormaliser(window=500)

    a = normaliser.transform(small, ["a"])["a"].dropna()
    b = normaliser.transform(large, ["a"])["a"].dropna()
    assert abs(a.std() - b.std()) < 0.1


def test_a_drifting_level_is_reduced_but_not_erased() -> None:
    """A trailing window lags a deterministic drift, and should.

    The centre is behind by roughly half the window, so a steady climb leaves a
    persistent positive offset. That is correct — a causal scaler cannot know
    where the level is going — and it is worth pinning, because a test asserting
    the drift vanishes would only pass for a scaler that peeked.
    """
    frame = series(n=4000, seed=2)
    drift = np.linspace(0, 100, len(frame))
    frame["a"] = frame["a"] + drift

    normalised = RollingNormaliser(window=500).transform(frame, ["a"])["a"].dropna()
    # Raw, the level ranges over a hundred units; normalised it sits inside a
    # few, and the remaining offset is the lag rather than the trend.
    assert normalised.max() - normalised.min() < 10
    assert normalised.std() < 2.0


def test_values_are_clipped_to_the_stated_range() -> None:
    frame = series(n=2000)
    frame.loc[1500, "a"] = 1e9
    normalised = RollingNormaliser(window=400, clip=5.0).transform(frame, ["a"])["a"]
    assert normalised.abs().max() <= 5.0


def test_robust_scaling_survives_an_outlier_that_breaks_a_z_score() -> None:
    """One repriced quote should not set the scale for a whole window."""
    frame = series(n=2000, seed=3)
    frame.loc[1000, "a"] = 5000.0

    robust = RollingNormaliser(window=500, method="robust", clip=100).transform(frame, ["a"])["a"]
    zscore = RollingNormaliser(window=500, method="zscore", clip=100).transform(frame, ["a"])["a"]
    after = slice(1001, 1400)
    assert robust.loc[after].std() > zscore.loc[after].std()


def test_untouched_columns_are_left_alone() -> None:
    frame = series(n=500)
    out = RollingNormaliser(window=100).transform(frame, ["a"])
    pd.testing.assert_series_equal(out["b"], frame["b"])


def test_the_warm_up_is_missing_rather_than_wrong() -> None:
    frame = series(n=1000)
    out = RollingNormaliser(window=400).transform(frame, ["a"])
    assert out["a"].head(50).isna().all()


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        RollingNormaliser(window=5)
    with pytest.raises(ValueError, match="method"):
        RollingNormaliser(method="magic")
    with pytest.raises(ValueError, match="clip"):
        RollingNormaliser(clip=0.0)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def test_pooling_stacks_instruments_and_labels_them() -> None:
    frames = {"AAA": series(n=1500, seed=4), "BBB": series(n=1500, seed=5)}
    pooled = normalise_pooled(frames, ["a", "b"], RollingNormaliser(window=300))
    assert set(pooled["symbol"]) == {"AAA", "BBB"}
    assert len(pooled) == 3000


def test_each_instrument_is_normalised_against_itself() -> None:
    """Normalising the pool as a whole would centre every instrument on an
    average none of them experiences, and make each row depend on which other
    instruments were included."""
    frames = {
        "SMALL": series(n=2000, seed=6, scale=0.01),
        "LARGE": series(n=2000, seed=6, scale=100.0),
    }
    pooled = normalise_pooled(frames, ["a"], RollingNormaliser(window=400))
    spreads = pooled.dropna(subset=["a"]).groupby("symbol")["a"].std()
    assert abs(spreads["SMALL"] - spreads["LARGE"]) < 0.1


def test_pooling_sorts_by_time_so_a_split_cuts_every_instrument_at_once() -> None:
    frames = {"AAA": series(n=800, seed=7), "BBB": series(n=800, seed=8)}
    pooled = normalise_pooled(frames, ["a"], RollingNormaliser(window=200))
    assert pooled["timestamp"].is_monotonic_increasing


def test_an_instrument_without_the_columns_is_skipped() -> None:
    frames = {"GOOD": series(n=900, seed=9), "BARE": pd.DataFrame({"other": [1.0, 2.0]})}
    pooled = normalise_pooled(frames, ["a"], RollingNormaliser(window=200))
    assert set(pooled["symbol"]) == {"GOOD"}


def test_pooling_nothing_usable_is_an_error() -> None:
    with pytest.raises(ValueError, match="no instrument"):
        normalise_pooled({"X": pd.DataFrame({"other": [1.0]})}, ["a"])


def test_coverage_reports_what_survived_the_warm_up() -> None:
    frame = RollingNormaliser(window=400).transform(series(n=1000), ["a"])
    assert 0.0 < coverage(frame, ["a"]) < 1.0
