"""Tests for positioning features built from metrics and funding.

These arrive every five minutes against a grid of a tenth of a second, so the
property that matters is that nothing here pretends to be timely — and that the
funding join reaches backwards, since a rate published at 16:00 is not knowable
at 15:59.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.features.positioning import build_positioning_features


def metrics(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n, freq="5min", tz="UTC"),
            "sum_open_interest": 70000 + np.cumsum(rng.normal(0, 50, n)),
            "count_long_short_ratio": np.exp(rng.normal(0.4, 0.1, n)),
            "sum_toptrader_long_short_ratio": np.exp(rng.normal(0.2, 0.1, n)),
            "count_toptrader_long_short_ratio": np.exp(rng.normal(0.3, 0.1, n)),
            "sum_taker_long_short_vol_ratio": np.exp(rng.normal(0.0, 0.2, n)),
        }
    )


def funding(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n, freq="8h", tz="UTC"),
            "funding_rate": np.linspace(1e-4, 5e-4, n),
        }
    )


def test_open_interest_is_logged_so_a_change_means_the_same_at_any_level() -> None:
    frame = build_positioning_features(metrics())
    assert "log_open_interest" in frame
    assert frame["log_open_interest"].between(10, 12).all()


def test_the_crowd_and_the_large_accounts_are_compared() -> None:
    """The one quantity here that is hard to get anywhere else."""
    frame = build_positioning_features(metrics())
    assert "crowd_minus_size" in frame
    assert frame["crowd_minus_size"].std() > 0


def test_ratios_are_logged_so_opposite_positions_are_equal_and_opposite() -> None:
    """2.0 and 0.5 are twice as many longs and twice as many shorts; raw they
    are 1.0 and 0.5 from parity, logged they are symmetric."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=2, freq="5min", tz="UTC"),
            "count_long_short_ratio": [2.0, 0.5],
        }
    )
    out = build_positioning_features(frame, windows=(2,))
    assert out["log_crowd_ratio"].iloc[0] == pytest.approx(-out["log_crowd_ratio"].iloc[1])


def test_nothing_differences_two_adjacent_observations() -> None:
    """At five-minute resolution that would measure the sampling grid."""
    frame = build_positioning_features(metrics(), windows=(12, 72))
    changes = [c for c in frame.columns if "change" in c]
    assert changes
    assert all(int(c.rsplit("_", 1)[-1]) >= 12 for c in changes)


def test_a_row_is_excluded_from_its_own_reference_distribution() -> None:
    frame = metrics(n=300)
    spiked = frame.copy()
    spiked.loc[250, "sum_open_interest"] *= 3

    plain = build_positioning_features(frame, windows=(72,))
    changed = build_positioning_features(spiked, windows=(72,))
    # Rows before the spike are untouched; the spike itself scores against a
    # window that does not contain it.
    pd.testing.assert_series_equal(
        plain["open_interest_z72"].iloc[:250], changed["open_interest_z72"].iloc[:250]
    )
    assert abs(changed["open_interest_z72"].iloc[250]) > 5


def test_funding_joins_backwards() -> None:
    """A rate published at 16:00 is not knowable at 15:59."""
    frame = build_positioning_features(metrics(n=200), funding=funding())
    assert "funding_bp" in frame
    # The first rate applies from 00:00, so nothing before it is filled from a
    # later publication.
    early = frame[frame["timestamp"] < pd.Timestamp("2024-02-01 08:00", tz="UTC")]
    assert (early["funding_bp"] == early["funding_bp"].iloc[0]).all()


def test_funding_is_reported_in_basis_points() -> None:
    # Long enough for every published rate to apply to something: six rates
    # eight hours apart span forty hours, or 480 five-minute observations.
    frame = build_positioning_features(metrics(n=500), funding=funding())
    assert frame["funding_bp"].max() == pytest.approx(5.0, abs=0.1)


def test_missing_columns_are_skipped_rather_than_faked() -> None:
    bare = pd.DataFrame(
        {"timestamp": pd.date_range("2024-02-01", periods=50, freq="5min", tz="UTC")}
    )
    frame = build_positioning_features(bare)
    assert list(frame.columns) == ["timestamp"]


def test_a_frame_without_timestamps_is_refused() -> None:
    with pytest.raises(KeyError, match="timestamp"):
        build_positioning_features(pd.DataFrame({"sum_open_interest": [1.0]}))


def test_three_scales_are_produced() -> None:
    frame = build_positioning_features(metrics(), windows=(12, 72, 288))
    for window in (12, 72, 288):
        assert f"open_interest_z{window}" in frame
        assert f"crowd_ratio_z{window}" in frame
