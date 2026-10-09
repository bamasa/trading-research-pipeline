"""The ten-level tensor: W calibration, values, and the leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _sh_synthetic import make_rows
from trading_research.features.lob_tensor import (
    TensorCalibration,
    TensorCalibrator,
    fit_tensor_calibration,
    lob_tensor,
    mid_log_change_std,
    register_tensor,
    tensor_columns,
)
from trading_research.features.timed import TimedRegistry, build_timed, stamps_ns
from trading_research.validation.leakage_timed import assert_timed_honest


@pytest.fixture(scope="module")
def warm_up() -> list[pd.DataFrame]:
    return [make_rows(300, seed=s, start=f"2024-12-0{s + 1}") for s in range(3)]


@pytest.fixture(scope="module")
def calibration(warm_up) -> TensorCalibration:
    return fit_tensor_calibration(warm_up)


def test_day_by_day_calibration_equals_the_pooled_one(warm_up, calibration) -> None:
    pooled = pd.concat(warm_up, ignore_index=True)
    spread = (pooled["ask_price_0"] - pooled["bid_price_0"]).to_numpy()
    assert calibration.spread_floor == pytest.approx(np.median(spread))
    sizes = np.log1p(pooled[[f"ask_size_{i}" for i in range(10)]].to_numpy())
    np.testing.assert_allclose(calibration.size_mean[:10], sizes.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(calibration.size_std[:10], sizes.std(axis=0), rtol=1e-10)


def test_spread_median_counts_an_even_split() -> None:
    calibrator = TensorCalibrator()
    calibrator.spread_counts = {0.1: 2, 0.2: 2}
    assert calibrator.spread_median() == pytest.approx(0.15)


def test_calibration_round_trips_through_the_fitted_values_file(calibration) -> None:
    assert TensorCalibration.from_dict(calibration.to_dict()) == calibration
    with pytest.raises(ValueError, match="positive"):
        TensorCalibration(0.0, calibration.size_mean, calibration.size_std)


def test_tensor_values_follow_the_registered_definition(calibration) -> None:
    rows = make_rows(400, seed=11)
    out = lob_tensor(rows, calibration)
    assert list(out.columns) == tensor_columns()
    assert np.allclose(out["lob_ask_dist_0"].dropna(), 0.0)
    stamps = stamps_ns(rows)
    mid = ((rows["bid_price_0"] + rows["ask_price_0"]) / 2).to_numpy()
    for t in (5, 120, 350):
        inside = (stamps > stamps[t] - 60_000_000_000) & (np.arange(len(rows)) <= t)
        idx = np.flatnonzero(inside)
        idx = idx[idx >= 1]
        changes = np.log(mid[idx] / mid[idx - 1])
        sigma = changes.std(ddof=1)
        scale = max(sigma * mid[t], calibration.spread_floor)
        expected = (rows["ask_price_3"].iloc[t] - rows["ask_price_0"].iloc[t]) / scale
        assert out["lob_ask_dist_3"].iloc[t] == pytest.approx(expected, rel=1e-9)
        z = (np.log1p(rows["bid_size_2"].iloc[t]) - calibration.size_mean[12]) / (
            calibration.size_std[12]
        )
        assert out["lob_bid_size_2"].iloc[t] == pytest.approx(z, rel=1e-12)


def test_the_floor_holds_when_the_mid_stands_still(calibration) -> None:
    rows = make_rows(200, seed=5)
    for side in ("bid", "ask"):
        for level in range(10):
            rows[f"{side}_price_{level}"] = rows[f"{side}_price_{level}"].iloc[0]
    out = lob_tensor(rows, calibration)
    gap = rows["ask_price_1"].iloc[50] - rows["ask_price_0"].iloc[50]
    assert out["lob_ask_dist_1"].iloc[50] == pytest.approx(gap / calibration.spread_floor)


def test_sigma_keeps_its_digits_at_btc_prices() -> None:
    rows = make_rows(300, seed=9)
    for side in ("bid", "ask"):
        for level in range(10):
            rows[f"{side}_price_{level}"] = rows[f"{side}_price_{level}"] + 70_000.0
    sigma = mid_log_change_std(rows, 60.0)
    mid = ((rows["bid_price_0"] + rows["ask_price_0"]) / 2).to_numpy()
    two_pass = np.diff(np.log(mid))[: len(mid) - 1].std(ddof=1)
    assert sigma[-1] == pytest.approx(two_pass, rel=1e-6)


def test_tensor_passes_truncation_and_streaming_parity(calibration) -> None:
    rows = make_rows(240, seed=21)
    registry = TimedRegistry()
    features = register_tensor(calibration, registry)
    assert len(features) == 40
    assert registry.longest_lookback_minutes() == 1
    names = ["lob_ask_dist_4", "lob_bid_dist_9", "lob_ask_size_0", "lob_bid_size_7"]
    assert_timed_honest(registry, rows, names=names)
    built = build_timed(rows, names, registry=registry)
    pd.testing.assert_series_equal(
        built["lob_bid_dist_9"], lob_tensor(rows, calibration)["lob_bid_dist_9"]
    )
