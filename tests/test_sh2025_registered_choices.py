"""The values the study code fixed, against the registration that records them.

Each value below was left open by the registration's text and chosen while the
code was written, before any data was read; the registration now states it,
in the document and in the configuration file. These tests hold the code, the
configuration and the document to one value each.
"""

from __future__ import annotations

import inspect
import math
import re
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import yaml

from trading_research.backtest.taker import TRADE_COLUMNS, TakerBook, execute_day
from trading_research.data import feature_store as fs
from trading_research.data import fetch_log, study_fetch, study_loader
from trading_research.evaluation.family import slot_p_value
from trading_research.evaluation.overfitting import relative_rank_logit
from trading_research.evaluation.significance import NEWEY_WEST_LAG
from trading_research.features import REGISTRY, study_set
from trading_research.features.lob_tensor import fit_tensor_calibration, lob_tensor, register_tensor
from trading_research.features.micro2025 import REGISTRY_2025
from trading_research.labels.taker_vwap import WalkedTakerLabel, control_move_bp
from trading_research.market_making import fill_rules, prereg, signal_entry
from trading_research.market_making.verdicts import day_t
from trading_research.models import windows
from trading_research.models.gbm import GradientBoostedBaseline
from trading_research.models.linear import LogisticBaseline
from trading_research.models.tcn import GaussianTCN
from trading_research.pipeline.regime import random_flag_sets, usable_hours
from trading_research.reporting.regime_map import regime_map
from trading_research.strategies.ev_gate import EVGate, gate_events, thin_entries
from trading_research.strategies.hold_first import fit_hold_first, side_threshold
from trading_research.validation.choice import (
    CARRY_FORWARD_EMBARGO_DAYS,
    CellDays,
    cell_day_t,
    choose,
)
from trading_research.validation.placebo_search import null_a_signs
from trading_research.validation.regime_flags import STREAM_START, FlagError, hourly_stream

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


# ---------------------------------------------------------------------------
# Statistics and the regime layer
# ---------------------------------------------------------------------------


def test_a_degenerate_cell_has_a_t_of_g_of_zero() -> None:
    values = np.array(
        [[1.0, np.nan, np.nan], [2.0, 2.0, 2.0], [1.0, 2.0, 3.0]],
    )
    t = cell_day_t(values)
    assert t[0] == 0.0 and t[1] == 0.0  # one day; no dispersion
    assert t[2] == pytest.approx(day_t([1.0, 2.0, 3.0]).t)
    assert "fewer than two days" in _registered()["search"]["nulls"]["t_of_G_degenerate"]


def test_ties_after_the_own_mean_go_to_the_first_key() -> None:
    days = 4
    net = np.ones((3, days))
    cells = CellDays(np.full((3, days), 6.0), net, net.copy())
    index = np.arange(3).reshape(3, 1)  # each cell its own neighbourhood
    assert choose(cells, [0, 1, 2], index).cell == 0
    assert choose(cells, [2, 1], index).cell == 1  # the earlier row of the table
    assert _registered()["search"]["choice"]["ties_after_own_mean"].startswith("the key first")


def test_the_carry_forward_embargo_and_the_lag_are_registered() -> None:
    assert CARRY_FORWARD_EMBARGO_DAYS == 5
    assert "days 1 to 5" in _registered()["search"]["admission"]["signal"]["iii_details"]["embargo"]
    newey_west = _registered()["metrics"]["newey_west"]
    assert NEWEY_WEST_LAG == newey_west["lag_days"] == 5
    assert newey_west["lag_counts"] == "positions in the daily series"


def test_a_median_choice_counts_as_overfit() -> None:
    assert relative_rank_logit(1, np.array([1.0, 2.0, 3.0])) == 0.0  # at the median
    assert relative_rank_logit(2, np.array([1.0, 2.0, 3.0])) > 0
    assert "at or below 0 counts as overfit" in _registered()["search"]["pbo_rule"]


def test_null_a_signs_are_the_registered_draw() -> None:
    for draw in (0, 7, 998):
        expected = 1 - 2 * np.random.default_rng(draw).integers(0, 2, size=50)
        assert null_a_signs(draw, 50).tolist() == expected.tolist()


def test_a_constant_positive_series_has_p_zero_and_a_single_day_p_one() -> None:
    assert slot_p_value([2.0, 2.0, 2.0]) == 0.0
    assert slot_p_value([2.0]) == 1.0
    assert _registered()["metrics"]["p_value_constant_positive_series"] == 0


def test_the_stream_needs_its_first_bar() -> None:
    closes = pd.date_range("2024-12-01 01:00", periods=6, freq="h", tz="UTC")
    assert closes[0].floor("D") == STREAM_START  # the first bar closes an hour in
    bars = pd.DataFrame({"timestamp": closes, "close": np.arange(100.0, 106.0)})
    end = closes[-1]
    assert len(hourly_stream(bars.drop(index=2), end=end).missing) == 1  # carried
    with pytest.raises(FlagError, match="first bar"):
        hourly_stream(bars.drop(index=0), end=end)
    assert "must be present" in _registered()["detector"]["first_bar"]


def test_a_segment_starts_on_the_first_day_after_its_alarm() -> None:
    days = pd.date_range("2025-02-01", periods=6, freq="D", tz="UTC")
    daily = pd.Series(np.arange(1.0, 7.0), index=days)
    alarm = pd.Timestamp("2025-02-03 12:00", tz="UTC")
    segments = regime_map(daily, [alarm])
    assert len(segments) == 2
    assert pd.Timestamp(segments["end"].iloc[0]).day == 3  # the alarm's own day
    assert pd.Timestamp(segments["start"].iloc[1]).day == 4
    blocks = {d.date(): ("D" if d.day <= 2 else "H") for d in days}
    assert len(regime_map(daily, [alarm], blocks=blocks)) == 3  # a block change cuts


def test_the_network_shuffles_shards_of_four_days() -> None:
    shards = _registered()["models"]["tcn_batches"]["days_per_shard"]
    assert inspect.signature(GaussianTCN.__init__).parameters["days_per_shard"].default == shards
    batches = inspect.signature(windows.WindowDataset.training_batches).parameters
    assert batches["days_per_shard"].default == shards == 4


def test_random_flag_sets_draw_distinct_hours_from_their_own_seed() -> None:
    days = [date(2025, 7, 8) + timedelta(days=i) for i in range(3)]
    flag_sets = random_flag_sets(days, 10, sets=3)
    hours = usable_hours(days)
    for k, flags in enumerate(flag_sets):
        assert len(np.unique(flags)) == 10
        expected = np.sort(np.random.default_rng(k).choice(hours, size=10, replace=False))
        assert flags.tolist() == expected.tolist()
    assert (
        "without replacement"
        in (_registered()["search"]["regime_source"]["attribution_flag_hours"])
    )


def test_a_trade_s_null_a_hour_is_its_entry_snapshot_s() -> None:
    assert "entry_snapshot_ns" in TRADE_COLUMNS
    assert _registered()["search"]["nulls"]["null_A_trade_hour"].endswith("entry snapshot")
    assert "A trade belongs to the UTC hour of its entry snapshot" in _document()


# ---------------------------------------------------------------------------
# The registered hash
# ---------------------------------------------------------------------------


def test_the_document_states_the_configuration_s_null_placeholder_hash() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    computed = prereg.sha256_bytes(prereg.unfrozen_text(text).encode())
    stated = re.search(r"with every placeholder `null`, has sha256 `([0-9a-f]{64})`", _document())
    assert stated is not None and stated.group(1) == computed
    # the ob200 reader test, the fits lacking a class and the second detector on D, 2026-10-12
    assert len(prereg.placeholder_lines(text)) == 27
    assert ("features", "longest_lookback_minutes") not in prereg.placeholder_lines(text)


def test_every_new_value_is_fixed_not_a_placeholder() -> None:
    marked = {".".join(path) for path in prereg.placeholder_lines(CONFIG.read_text("utf-8"))}
    for key in (
        "data.max_sequence_gaps_percentile",
        "features.longest_lookback_minutes",
        "features.row_declared_windows",
        "models.seed",
        "d_procedure.walk_forward.fold_level_leakage_test_fits",
        "common.taker_fill.timing",
        "search.hold_first_theta",
    ):
        assert key not in marked, key


# ---------------------------------------------------------------------------
# The clarifications the fixes of the study code called for (2026-10-12)
# ---------------------------------------------------------------------------


def test_the_registration_states_the_clarifications_the_fixes_called_for() -> None:
    text, tree = _document(), _registered()
    for phrase in (
        "07-08 takes the tape of the last usable day at or before 07-31, never one from 08-01",
        "The slot has one shifted series on H",
        "a day the shift's own run flags stale is left out in turn",
        "with each earlier run's on that run's own dates only",
        "E's seven days enter no comparison",
        "A result void on F or P keeps the status `void`",
        "a fault in any run of the slot's read",
        "K-cap reads the true run only",
        "the share is the one the frozen \u03c3_D and MDE imply",
        "P's first day, 2025-11-17, reads 2025-11-16 when that day is usable under the frozen threshold",
        "the placebo means and the random draws' 95th percentile, with no precedence applied and no status",
        "each read session's first simulated day runs alone to measure that peak",
        "A tier pays a passive slot only when the slot's net is positive at that tier's own maker and taker fees together",
        "an entry is a run of fills from a flat position to the next flat position",
        "the entry notional is the filled part of the entry",
        "a fill at the price is floored to the instrument's lot",
        "only its taker exit is honoured, so an exit due before 23:59 is crossed when due",
        "A day enters its 1 s stream when its book is on disk and its only exclusion is sequence gaps",
        "D's run is made by stage C under the development access and frozen in the amendment",
        "with the variance each is deflated against",
        "the second detector's run over D under `detector.second_detector_D`",
    ):
        assert phrase in text, phrase
    assert "steps_over" in tree["placebos"]["one_day_shift"]
    assert "one shifted series per slot" in tree["placebos"]["one_day_shift"]["circle"]
    for key in ("void_on_F_or_P", "R_void_scope"):
        assert key in tree["statuses"], key
    assert "measured_inside_reports" in tree["statuses"]["S25_signal_slots"]
    for key in ("K-acct_simulator_slots", "K-cap_simulator_slots"):
        assert key in tree["kill_conditions"], key
    assert "P_first_day_history" in tree["blocks"]
    assert "on_F_and_P_days_and_share" in tree["metrics"]["mde"]
    for key in ("fill_at_the_price", "between_stop_and_flatten", "decisions_outside_hours"):
        assert key in tree["common"]["passive_fill"], key
    for key in ("passive_entry", "per_trade_denominator"):
        assert key in tree["metrics"]["day_series"], key
    assert "tier_pays_passive" in tree["measurements"]["fee_break_even"]
    for key in ("day_enters_stream", "alarm_stamp", "runs_by", "within_one_day"):
        assert key in tree["detector"]["reported_beside"]["changepoint"], key
    assert "first simulated day runs alone" in tree["compute"]["machine"]["simulator_workers_max"]
    marked = prereg.placeholder_lines(CONFIG.read_text("utf-8"))
    assert ("models", "fits_lacking_a_class") in marked and (
        "detector",
        "second_detector_D",
    ) in marked
    assert "second detector" in " ".join(tree["amendment"]["records"])
