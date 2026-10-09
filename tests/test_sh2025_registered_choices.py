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

from trading_research.backtest.taker import TakerBook, execute_day
from trading_research.data import feature_store as fs
from trading_research.data import fetch_log, study_fetch, study_loader
from trading_research.features import REGISTRY, study_set
from trading_research.features.lob_tensor import fit_tensor_calibration, lob_tensor, register_tensor
from trading_research.features.micro2025 import REGISTRY_2025
from trading_research.labels.taker_vwap import WalkedTakerLabel, control_move_bp
from trading_research.market_making import fill_rules, signal_entry
from trading_research.models import windows
from trading_research.models.gbm import GradientBoostedBaseline
from trading_research.models.linear import LogisticBaseline
from trading_research.models.tcn import GaussianTCN
from trading_research.strategies.ev_gate import EVGate, gate_events, thin_entries
from trading_research.strategies.hold_first import fit_hold_first, side_threshold

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


# ---------------------------------------------------------------------------
# Labels and models
# ---------------------------------------------------------------------------


def test_a_row_without_a_fillable_entry_has_no_label() -> None:
    rows = _book(200)
    label = WalkedTakerLabel(hold_s=1.0)
    assert label(rows).iloc[:150].notna().all()
    assert label(rows).iloc[-5:].isna().all()  # no exit snapshot: no label
    thin = rows.copy()
    for i in range(10):
        thin[f"ask_size_{i}"] = thin[f"bid_size_{i}"] = 0.0005  # 0.005 visible < 0.010
    assert label(thin).isna().all()  # every entry would be skipped: none is hold
    assert "has no label rather than `hold`" in _document()


def test_every_fit_has_the_registered_seed() -> None:
    seed = _registered()["models"]["seed"]
    for model in (GradientBoostedBaseline, LogisticBaseline, GaussianTCN):
        assert inspect.signature(model.__init__).parameters["seed"].default == seed == 42


def test_training_rows_are_thinned_by_position_over_the_days_in_order() -> None:
    days = [np.arange(0, 60, 10), np.arange(0, 40, 10), np.arange(0, 50, 10)]  # 6 + 4 + 5
    kept = windows.thin_evenly(days, 5)
    positions = np.floor(np.arange(5) * (15 / 5)).astype(int)  # 0, 3, 6, 9, 12
    flat = np.concatenate(days)
    assert np.concatenate(kept).tolist() == flat[positions].tolist()
    assert [len(k) for k in kept] == [2, 2, 1]
    assert "floor(i x total / 4000000)" in _registered()["models"]["training_rows_kept"]


def test_the_network_leaves_a_constant_input_unscaled() -> None:
    torch_tcn = pytest.importorskip("trading_research.models.tcn")
    values = np.column_stack([np.arange(100.0), np.zeros(100), np.full(100, 7.0)])
    day = windows.Day("d", lambda: values.astype(np.float32), windows.every_nth_row(100))
    mean, scale = torch_tcn._input_moments(windows.WindowDataset([day], window=8))
    assert scale[1] == scale[2] == 1.0 and mean[2] == 7.0
    assert scale[0] == pytest.approx(
        np.std(np.arange(10.0, 100.0, 10.0))
    )  # population, at window ends


def test_the_fold_level_test_fits_on_the_reproducible_device() -> None:
    pytest.importorskip("torch")
    rule = _registered()["d_procedure"]["walk_forward"]["fold_level_leakage_test_fits"]
    assert rule["device"] == "cpu" and "epochs 1" in rule["tcn"]
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(600, 3)), columns=["a", "b", "c"])
    y = pd.Series(x["a"].rolling(5).sum().fillna(0.0) + rng.normal(size=600))
    vx, vy = x.iloc[:200].reset_index(drop=True), y.iloc[:200].reset_index(drop=True)
    small = {"window": 8, "channels": 4, "levels": 1, "batch_size": 64, "epochs": 1}

    def fit() -> GaussianTCN:
        return GaussianTCN(**small, device=rule["device"]).fit(x, y, validation=(vx, vy))  # type: ignore[arg-type]

    one, two = fit(), fit()
    assert one.weights_sha256() == two.weights_sha256()
    assert one.history_ == two.history_ and one.checkpoint_epoch_ == 1
    assert "**The test's fits run on the CPU.**" in _document()


# ---------------------------------------------------------------------------
# Decision rules and execution
# ---------------------------------------------------------------------------


def test_theta_takes_half_the_rate_per_side_on_the_validation_rows() -> None:
    rule = _registered()["search"]["hold_first_theta"]
    assert rule["rows"] == "every row of the inner validation"
    p_long = np.linspace(0.30, 0.90, 200)
    proba = np.column_stack([np.full(200, 0.05), 1 - p_long - 0.05, p_long])  # SELL, HOLD, BUY
    fitted = fit_hold_first(proba, days=7, trades_per_day=25)  # 87.5 entries, half to even
    assert fitted.theta_long == np.sort(p_long)[::-1][88 - 1]
    assert fitted.theta_short == np.inf  # no row has the short side above the long
    few = side_threshold(np.array([0.9, 0.2, 0.8]), np.array([0.1, 0.5, 0.1]), 5)
    assert few == 0.8  # fewer qualifying rows than entries: the smallest of them


def test_an_event_that_opens_during_a_position_enters_at_its_first_free_row() -> None:
    sides = np.array([1, 0, 0, 1, -1, 1, 0, 1, 0, 0])
    events = gate_events(sides)
    assert events.tolist() == [0, -1, -1, 1, 1, 1, -1, 2, -1, -1]  # 3..5 is one event
    busy_until = np.array([3, 9, 9, 9, 6, 9, 9, 8, 9, 9])
    assert thin_entries(sides, busy_until, events).tolist() == [0, 4, 7]
    gate = EVGate(k=0.0, beta_bp=0.0)
    assert gate.sides(np.array([np.nan, 50.0]), np.ones(2), np.array([0.0, np.nan])).tolist() == [
        0,
        0,
    ]


def _taker_frame(rows: int = 1200, size: float = 0.002) -> pd.DataFrame:
    stamps = pd.date_range("2024-02-11 00:05", periods=rows, freq="100ms", tz="UTC")
    frame: dict[str, object] = {"timestamp": stamps}
    for i in range(10):
        frame[f"ask_price_{i}"] = 60_000.1 + 0.1 * i
        frame[f"bid_price_{i}"] = 60_000.0 - 0.1 * i
        frame[f"ask_size_{i}"] = size
        frame[f"bid_size_{i}"] = size
    return pd.DataFrame(frame)


def test_the_taker_exit_and_the_next_decision_are_the_registered_ones() -> None:
    frame = _taker_frame()
    book = TakerBook.from_frame(frame, frame["timestamp"].iloc[0].date())
    stamps = book.ts
    hold = 1_000_000_000
    k = 100
    # Exit decided at row k + 10 (t + 1 s), walked at row k + 11 (+ 10 ms).
    decisions = np.array([stamps[k], stamps[k + 10], stamps[k + 11]])
    day = execute_day(book, decisions, np.array([1, -1, 1]), hold)
    assert day.counters["entries"] == 2 and day.counters["entries_busy"] == 1
    timing = _registered()["common"]["taker_fill"]["timing"]
    assert "a decision on the exit's own row may enter" in timing["next_decision"]
    thin = TakerBook.from_frame(_taker_frame(size=0.0009), book.day)  # 0.009 visible
    skipped = execute_day(thin, decisions[:1], np.array([1]), hold)
    assert skipped.counters["entries_skipped"] == 1
    assert timing["entry_skip_beyond"] == 1e-9


def test_the_quoter_and_fill_choices_are_the_code_s() -> None:
    quoter = signal_entry.__doc__ or ""
    assert "``hold_s`` after the post" in " ".join(quoter.split())
    assert "``expired_unseen``" in quoter
    assert "same decisions, not the same orders" in (fill_rules.__doc__ or "")
    timing = _registered()["common"]["passive_fill"]["quoter_timing"]
    assert "after its post, not after the decision" in timing["entry_cancel"]
    assert "same decisions" in _registered()["measurements"]["FILL"]["same_orders"]
    choices = _choices()
    for phrase in ("θ on every row of the inner validation", 'FILL\'s "same orders"'):
        assert phrase in choices, phrase
