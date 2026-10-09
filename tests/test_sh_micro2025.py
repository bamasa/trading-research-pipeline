"""The 2025 additions and the study's microstructure set: values and leakage."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from _sh_synthetic import make_prints, make_rows
from trading_research.features.micro2025 import REGISTRY_2025, turn_count
from trading_research.features.study_set import (
    ADDED_2025,
    RAW_SIZE_COLUMNS,
    fit_touch_size_scale,
    microstructure_registry,
    scale_raw_sizes,
    winsorise_trailing,
)
from trading_research.features.timed import build_timed, stamps_ns
from trading_research.validation.leakage_timed import assert_timed_honest

S = 1_000_000_000


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return make_rows(360, seed=4)


@pytest.fixture(scope="module")
def prints(rows: pd.DataFrame) -> pd.DataFrame:
    return make_prints(rows, seed=5)


@pytest.fixture(scope="module")
def built(rows, prints) -> pd.DataFrame:
    return build_timed(rows, list(ADDED_2025), prints=prints, registry=REGISTRY_2025)


def _window(stamps: np.ndarray, t: int, seconds: float) -> np.ndarray:
    return np.flatnonzero((stamps > stamps[t] - seconds * S) & (stamps <= stamps[t]))


def test_book_side_additions_equal_loops(rows, built) -> None:
    stamps = stamps_ns(rows)
    mid = ((rows["bid_price_0"] + rows["ask_price_0"]) / 2).to_numpy()
    ret = np.r_[np.nan, (mid[1:] / mid[:-1] - 1) * 1e4]
    for t in (3, 77, 200, 359):
        inside = _window(stamps, t, 1.0)
        values = ret[inside][~np.isnan(ret[inside])]
        expected = values.std(ddof=1) if len(values) > 1 else np.nan
        np.testing.assert_allclose(built["volatility_1s_bp"].iloc[t], expected, rtol=1e-9)
        products = [np.sign(ret[j]) * np.sign(ret[j - 1]) for j in inside if j >= 2]
        expected = np.mean(products) if products else np.nan
        np.testing.assert_allclose(built["return_sign_autocorr_1s"].iloc[t], expected, atol=1e-12)
        before = np.flatnonzero(stamps <= stamps[t] - 5 * S)
        if len(before):
            change = np.log(mid[t] / mid[before[-1]]) * 1e4
            assert built["mid_log_change_bp_5s"].iloc[t] == pytest.approx(change, abs=1e-9)


def test_turn_count_counts_reversals_inside_the_window(rows, built) -> None:
    stamps = stamps_ns(rows)
    mid = ((rows["bid_price_0"] + rows["ask_price_0"]) / 2).to_numpy()
    for t in (10, 150, 330):
        inside = _window(stamps, t, 5.0)
        moves = [(j, np.sign(mid[j] - mid[j - 1])) for j in inside if j >= 1]
        moves = [(j, d) for j, d in moves if d != 0]
        turns = sum(1 for (_, a), (_, b) in pairwise(moves) if a == -b)
        assert built["turn_count_5s"].iloc[t] == turns
    flat = rows.copy()
    flat["bid_price_0"] = 100.0
    flat["ask_price_0"] = 100.1
    assert turn_count(flat, 5.0).max() == 0


def test_trade_side_additions_equal_loops(rows, prints, built) -> None:
    stamps = stamps_ns(rows)
    times = stamps_ns(prints)
    side = prints["aggressor"].to_numpy()
    for t in (0, 50, 299, 359):
        span = np.flatnonzero((times > stamps[t] - 60 * S) & (times <= stamps[t]))
        if len(span) == 0:
            assert built["trade_run_length_60s"].iloc[t] == 0
            continue
        last = span[-1]
        run = 1
        while last - run >= span[0] and side[last - run] == side[last]:
            run += 1
        assert built["trade_run_length_60s"].iloc[t] == run * side[last]
        distinct = np.unique(times[span])
        expected = (distinct[-1] - distinct[-2]) / 1e9 if len(distinct) > 1 else np.nan
        np.testing.assert_allclose(built["inter_trade_time_s"].iloc[t], expected)
        since = min((stamps[t] - times[last]) / 1e9, 60.0)
        assert built["time_since_trade_s"].iloc[t] == pytest.approx(since)


def test_winsorising_clips_at_trailing_quantiles() -> None:
    rows = make_rows(200, seed=8)
    stamps = stamps_ns(rows)
    values = np.random.default_rng(0).standard_t(2, len(rows))
    clipped = winsorise_trailing(values, stamps, window_s=3.0, quantiles=(0.1, 0.9))
    for t in (5, 60, 199):
        inside = values[_window(stamps, t, 3.0)]
        low, high = np.quantile(inside, [0.1, 0.9])
        assert clipped[t] == pytest.approx(np.clip(values[t], low, high))


def test_touch_size_scale_is_the_pooled_median() -> None:
    days = [make_rows(100, seed=s) for s in range(3)]
    pooled = pd.concat(days)
    expected = np.median(np.r_[pooled["bid_size_0"].to_numpy(), pooled["ask_size_0"].to_numpy()])
    scale = fit_touch_size_scale(days)
    assert scale == pytest.approx(expected)
    frame = pd.DataFrame({"ofi_1": [2.0 * scale], "spread_bp": [1.0]})
    scaled = scale_raw_sizes(frame, scale)
    assert scaled["ofi_1"].iloc[0] == pytest.approx(2.0)
    assert scaled["spread_bp"].iloc[0] == 1.0
    assert set(RAW_SIZE_COLUMNS) == {"ofi_1", "ofi_20"}


def test_the_set_declares_n_of_61_minutes() -> None:
    registry = microstructure_registry()
    assert registry.longest_lookback_minutes() == 61
    assert microstructure_registry(winsorise=False).longest_lookback_minutes() == 1
    assert {"depth_imbalance", "flow_imbalance_120", "ofi_20", *ADDED_2025} <= set(registry.names())


@pytest.mark.parametrize("winsorise", [False, True])
def test_every_feature_of_the_set_passes_truncation_and_streaming(rows, prints, winsorise) -> None:
    registry = microstructure_registry(winsorise=winsorise)
    positions = [*range(0, len(rows), 9), len(rows) - 1]
    assert_timed_honest(registry, rows, prints, positions=positions)
