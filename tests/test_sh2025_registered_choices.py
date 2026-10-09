"""The values the study code fixed, against the registration that records them.

Each value below was left open by the registration's text and chosen while the
code was written, before any data was read; the registration now states it,
in the document and in the configuration file. These tests hold the code, the
configuration and the document to one value each.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from trading_research.data import feature_store as fs
from trading_research.data import fetch_log, study_fetch, study_loader
from trading_research.features import REGISTRY, study_set
from trading_research.features.lob_tensor import fit_tensor_calibration, lob_tensor, register_tensor
from trading_research.features.micro2025 import REGISTRY_2025
from trading_research.labels.taker_vwap import control_move_bp
from trading_research.models import windows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "short_horizon_2025.yaml"
DOCUMENT = ROOT / "docs" / "preregistration" / "short_horizon_2025.md"


def _registered() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _document() -> str:
    return " ".join(DOCUMENT.read_text(encoding="utf-8").split())


def _choices() -> str:
    text = _document()
    start = text.index("**Values the study code fixed.**")
    return text[start : text.index("The configuration as registered", start)]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_the_gap_threshold_is_the_registered_percentile() -> None:
    rule = _registered()["data"]["max_sequence_gaps_percentile"]
    assert rule["q"] == 95 and rule["interpolation"].startswith("linear")
    assert study_loader.max_sequence_gaps(list(range(20))) == 19  # 18.05 rounded up
    assert study_loader.max_sequence_gaps([3] * 30) == 3
    with mock.patch.object(study_loader.np, "percentile", return_value=3.0000000000000004):
        assert study_loader.max_sequence_gaps([3, 3]) == 3  # noise adds no gap
    assert "rounded to nine decimals before it is rounded up" in _document()


def test_the_backoff_and_the_unfetchable_rule_meet_at_the_eleventh_failure() -> None:
    fetch = _registered()["fetch"]
    assert fetch["retry_backoff_s"]["max"] == fetch_log.MAX_DELAY_S == 43_200
    assert [fetch_log.backoff_s(k) for k in (1, 2, 3)] == [60.0, 180.0, 540.0]
    assert fetch_log.backoff_s(30) == fetch_log.MAX_DELAY_S
    waited = np.cumsum([0.0] + [fetch_log.backoff_s(k) for k in range(1, 11)])
    span = fetch_log.UNFETCHABLE_SPAN.total_seconds()
    attempts = fetch_log.UNFETCHABLE_ATTEMPTS
    assert waited[attempts - 1] < span <= waited[attempts]  # the 10th is too early
    assert 50 * 3600 < waited[attempts] < 60 * 3600
    assert "about 54 hours" in _choices()


def test_every_download_times_out_per_socket_read_at_the_registered_value() -> None:
    timeout = _registered()["fetch"]["timeout_s_per_socket_read"]
    for fetch in (study_fetch.fetch_book_day, study_fetch.fetch_prints_day):
        assert inspect.signature(fetch).parameters["timeout"].default == timeout


def test_the_touch_size_pools_both_sides_over_every_row() -> None:
    days = [
        pd.DataFrame({"bid_size_0": [1.0, 2.0], "ask_size_0": [3.0, 9.0]}),
        pd.DataFrame({"bid_size_0": [4.0], "ask_size_0": [5.0]}),
    ]
    pooled = float(np.median([1.0, 2.0, 3.0, 9.0, 4.0, 5.0]))
    assert fs.median_touch_size(days) == pooled == 3.5  # the two middle values' mean
    assert study_set.fit_touch_size_scale(days) == pooled
    assert "ask_size_0 pooled over every W row" in _registered()["features"]["touch_size"]


def test_the_raw_size_columns_are_the_registered_two() -> None:
    assert tuple(_registered()["features"]["raw_size_columns"]) == study_set.RAW_SIZE_COLUMNS
    flow_and_depth = set(study_set.FLOW_COLUMNS) | set(study_set.DEPTH_COLUMNS)
    assert not flow_and_depth & set(study_set.RAW_SIZE_COLUMNS)


def test_the_finiteness_test_is_the_registered_one() -> None:
    rule = _registered()["features"]["finiteness"]
    assert rule["limit_abs"] == fs.FLOAT16_MAX == 65_504
    frame = pd.DataFrame(
        {
            "wide": [1.0, -70_000.0],
            "edge": [65_504.0, np.nan],
            "broken": [np.inf, 1.0],
        }
    )
    found = fs.finiteness_failures(frame, (), 1.0)
    assert found.overflow == ("wide",)
    assert found.infinite_at_source == ("broken",)  # reported, never widened
    layout, _ = fs.fit_layout([frame], touch_size=1.0, raw_size_columns=())
    assert layout.float32_columns == ("wide",)


def test_every_tenth_row_is_the_same_positions_in_the_store_and_the_fits() -> None:
    assert fs.STRIDES[fs.EVERY_TENTH_ROW] == 10
    assert windows.every_nth_row(25).tolist() == [0, 10, 20]
    assert "positions 0, 10, 20" in _registered()["features"]["every_tenth_row"]


def test_the_document_states_the_data_values() -> None:
    text = _document()
    for phrase in (
        "after the k-th failure, at most 12 hours",
        "The study's book download uses the same 120 s per socket read",
        "`zipfile.testzip`",
        "**A day after a hole starts cold the same way**",
        "In this set those are `ofi_1` and `ofi_20` alone",
        "exceeds 65,504 (float16's largest)",
    ):
        assert phrase in text, phrase
    assert math.isclose(float(_registered()["fetch"]["retry_backoff_s"]["max"]) / 3600, 12)


def test_the_blocks_carry_the_cold_rule_after_a_hole() -> None:
    assert "starts cold" in _registered()["blocks"]["cold_after_hole"]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _book(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """A small ten-level book on 100 ms stamps, prices in USDT."""
    rng = np.random.default_rng(seed)
    mid = 60_000 + np.cumsum(rng.normal(0, 0.5, n)).round(1)
    frame: dict[str, object] = {
        "timestamp": pd.date_range("2024-02-11", periods=n, freq="100ms", tz="UTC")
    }
    for i in range(10):
        frame[f"ask_price_{i}"] = mid + 0.05 + 0.1 * i
        frame[f"bid_price_{i}"] = mid - 0.05 - 0.1 * i
        frame[f"ask_size_{i}"] = rng.uniform(0.001, 2.0, n)
        frame[f"bid_size_{i}"] = rng.uniform(0.001, 2.0, n)
    return pd.DataFrame(frame)


def test_the_2025_windows_are_the_registered_ones() -> None:
    windows_s = _registered()["features"]["added_2025_windows_s"]
    declared = {name: REGISTRY_2025.get(name).lookback_s for name in REGISTRY_2025.names()}
    expected = {
        "volatility_1s_bp": windows_s["volatility_1s"],
        "return_sign_autocorr_1s": windows_s["return_sign_autocorrelation"],
        "trade_run_length_60s": windows_s["run_length"],
        "inter_trade_time_s": windows_s["inter_trade_time"],
        "time_since_trade_s": windows_s["time_since_trade"],
    }
    expected |= {f"turn_count_{w}s": w for w in windows_s["turn_count"]}
    for level in ("ask0", "bid0", "mid"):
        expected |= {f"{level}_log_change_bp_{w}s": w for w in windows_s["level0_lag_logs"]}
    assert declared == expected


def test_winsorising_is_the_registered_trailing_clip() -> None:
    rule = _registered()["features"]["winsorising"]
    assert rule["window_s"] == study_set.WINSOR_WINDOW_S
    assert list(study_set.WINSOR_QUANTILES) == rule["quantiles"]
    stamps = pd.date_range("2025-01-01", periods=6, freq="20min").as_unit("ns").asi8
    values = np.array([1.0, 2.0, 3.0, 100.0, 4.0, np.nan])
    clipped = study_set.winsorise_trailing(values, stamps)
    # Row 3 sees rows 1..3 (60 minutes back is excluded): its bound includes itself.
    upper = float(pd.Series([2.0, 3.0, 100.0]).quantile(0.999))
    assert clipped[3] == pytest.approx(upper)
    assert clipped[0] == 1.0  # one value in the window is its own bound
    assert np.isnan(clipped[5])


def test_n_is_61_minutes_over_the_whole_set_and_the_tensor() -> None:
    registered = _registered()["features"]["longest_lookback_minutes"]
    micro = study_set.microstructure_registry()
    assert micro.longest_lookback_minutes() == registered == 61
    calibration = fit_tensor_calibration([_book()])
    register_tensor(calibration, micro)
    assert micro.longest_lookback_minutes() == 61
    some = micro.names()[:5]
    assert micro.longest_lookback_minutes(some) <= registered  # any selection is covered
    assert "**`N` is 61 minutes**" in _document()


def test_the_row_declared_windows_are_the_registered_ones() -> None:
    rows = _registered()["features"]["row_declared_windows"]
    raw = study_set.microstructure_registry(winsorise=False)
    registry = {n: raw.get(n).lookback_rows for n in REGISTRY.names() if n in raw.names()}
    assert max(registry.values()) == 70 == registry["ofi_20_norm"]
    assert "70 rows, ofi_20_norm" in rows["registry_book_features"]
    flow = {raw.get(n).lookback_rows for n in study_set.FLOW_COLUMNS}
    assert flow == {study_set.FLOW_LOOKBACK_ROWS} == {rows["flow_builder"]}
    assert raw.longest_lookback_rows() == rows["flow_builder"]
    assert rows["network_window"] == windows.WINDOW_ROWS
    smoothing = inspect.signature(control_move_bp).parameters["smoothing"].default
    assert rows["control_smoothing"] == smoothing


def test_the_tensor_keeps_level_0_at_zero_and_fits_population_moments() -> None:
    rows = _book()
    calibration = fit_tensor_calibration([rows])
    tensor = lob_tensor(rows, calibration)
    level_0 = tensor[["lob_ask_dist_0", "lob_bid_dist_0"]].to_numpy()
    assert np.all((level_0 == 0) | np.isnan(level_0))
    distances = tensor.filter(like="_dist_")
    assert distances.iloc[:2].isna().all(axis=None)  # sigma needs two changes
    assert distances.iloc[3:].notna().all(axis=None)
    assert tensor.filter(like="_size_").notna().all(axis=None)
    spread = rows["ask_price_0"] - rows["bid_price_0"]
    assert calibration.spread_floor == pytest.approx(float(np.median(spread)))
    sizes = np.log1p(rows["ask_size_0"].to_numpy())
    assert calibration.size_std[0] == pytest.approx(float(np.std(sizes)))  # ddof 0


def test_no_rolling_normaliser_reaches_the_study() -> None:
    assert "not applied" in _registered()["features"]["rolling_normaliser"]
    assert "**No rolling normaliser** is applied to any input" in _document()
    text = " ".join(_registered()["search"]["fixed_not_searched"])
    assert "no rolling normaliser" in text
