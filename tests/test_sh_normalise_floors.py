"""Calibration-window floors and the normalised-column tag of RollingNormaliser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.features.normalise import (
    NORMALISED_ATTR,
    RollingNormaliser,
    fit_scale_floors,
    normalised_columns,
)


def _sizes(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "bid_size_0": rng.exponential(0.5, n),
            "ask_size_0": rng.exponential(0.5, n),
            "spread_bp": np.full(n, 0.02),
        }
    )


def test_the_tag_names_every_rewritten_column() -> None:
    frame = _sizes(300, 0)
    out = RollingNormaliser(window=50).transform(frame, ["bid_size_0"])
    assert out.attrs[NORMALISED_ATTR] == ["bid_size_0"]
    assert normalised_columns(out["bid_size_0"]) == {"bid_size_0"}
    assert normalised_columns(out["ask_size_0"]) == frozenset()
    again = RollingNormaliser(window=50).transform(out, ["ask_size_0"])
    assert normalised_columns(again) == {"bid_size_0", "ask_size_0"}
    assert normalised_columns(frame) == frozenset()


def test_floors_are_low_quantiles_of_the_calibration_scale() -> None:
    normaliser = RollingNormaliser(window=50)
    calibration = _sizes(2_000, 1)
    floors = fit_scale_floors(normaliser, calibration, ["bid_size_0", "ask_size_0"])
    rolling = calibration["bid_size_0"].rolling(50, min_periods=normaliser.warmup)
    scale = ((rolling.quantile(0.75) - rolling.quantile(0.25)) / 1.349).dropna().to_numpy()
    expected = max(np.quantile(scale, 0.10), 0.05 * np.median(scale))
    assert floors["bid_size_0"] == pytest.approx(expected)
    assert all(value > 0 for value in floors.values())


def test_a_quiet_stretch_is_divided_by_the_floor_not_by_nothing() -> None:
    normaliser = RollingNormaliser(window=50, clip=1e9)
    calibration = _sizes(2_000, 3)
    floors = fit_scale_floors(normaliser, calibration, ["bid_size_0"])
    quiet = _sizes(400, 4)
    quiet.loc[100:299, "bid_size_0"] = 0.5
    quiet.loc[300, "bid_size_0"] = 0.6
    plain = normaliser.transform(quiet, ["bid_size_0"])
    floored = normaliser.transform(quiet, ["bid_size_0"], column_floors=floors)
    assert abs(floored["bid_size_0"].iloc[300]) == pytest.approx(0.1 / floors["bid_size_0"])
    # Without the floor the constant stretch has no scale, and the jump out of
    # it is read as no information at all.
    assert plain["bid_size_0"].iloc[300] == 0.0


def test_floored_normalising_stays_causal() -> None:
    normaliser = RollingNormaliser(window=50)
    floors = fit_scale_floors(normaliser, _sizes(1_000, 5), ["ask_size_0"])
    frame = _sizes(500, 6)
    full = normaliser.transform(frame, ["ask_size_0"], column_floors=floors)
    part = normaliser.transform(frame.iloc[:300], ["ask_size_0"], column_floors=floors)
    np.testing.assert_allclose(full["ask_size_0"].iloc[:300], part["ask_size_0"])
