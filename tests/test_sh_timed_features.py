"""Timed features, backward print alignment and their leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _sh_synthetic import make_prints, make_rows
from trading_research.features.flow import align_to_grid
from trading_research.features.prints import bucket_prints, print_rows
from trading_research.features.timed import (
    TimedFeature,
    TimedRegistry,
    at_or_after,
    at_or_before,
    build_timed,
    register_builder,
    rolling_time,
    stamps_ns,
    timed_feature,
    trailing_start,
)
from trading_research.validation.leakage_timed import (
    check_streaming_parity,
    check_timed_causal,
)


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return make_rows(400)


@pytest.fixture(scope="module")
def prints(rows: pd.DataFrame) -> pd.DataFrame:
    return make_prints(rows)


def test_time_helpers_follow_the_registered_conventions() -> None:
    stamps = np.array([0, 100, 250, 1000], dtype=np.int64)
    assert at_or_after(stamps, np.array([0, 1, 250, 1001])).tolist() == [0, 1, 2, 4]
    assert at_or_before(stamps, np.array([-1, 99, 250])).tolist() == [-1, 0, 2]
    # (s_t - L, s_t]: a row exactly L before is outside the window.
    window = trailing_start(stamps * 1_000_000, 0.15)
    assert window.tolist() == [0, 0, 2, 3]


def test_stamps_must_not_go_backwards() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime([2, 1], unit="s", utc=True)})
    with pytest.raises(ValueError, match="non-decreasing"):
        stamps_ns(frame)


def test_a_print_belongs_to_the_first_row_at_or_after_it(rows, prints) -> None:
    stamps = stamps_ns(rows)
    owner = print_rows(prints, stamps)
    times = stamps_ns(prints)
    for t, j in zip(times, owner, strict=True):
        assert stamps[j] >= t
        assert j == 0 or stamps[j - 1] < t
    buckets = bucket_prints(prints, rows)
    sizes = prints["size"].to_numpy()
    expected = np.zeros(len(rows))
    np.add.at(expected, owner, sizes * (prints["aggressor"].to_numpy() > 0))
    np.testing.assert_allclose(buckets["buy_volume"], expected)
    assert buckets["trade_count"].sum() == len(prints)


def _per_row_count(aligned: str) -> TimedFeature:
    def fn(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
        assert prints is not None
        if aligned == "backward":
            return bucket_prints(prints, rows)["trade_count"]
        stamps = rows["timestamp"].reset_index(drop=True)
        mid = ((rows["bid_price_0"] + rows["ask_price_0"]) / 2).to_numpy()
        return pd.Series(align_to_grid(prints, stamps, mid)["trade_count"].to_numpy())

    return TimedFeature(f"count_{aligned}", "trades", 0.0, 0, fn)


def test_the_print_cut_catches_the_forward_rule(rows, prints) -> None:
    assert not check_timed_causal(_per_row_count("backward"), rows, prints).leaks
    assert check_timed_causal(_per_row_count("forward"), rows, prints).leaks


def test_streaming_parity_catches_an_understated_window(rows) -> None:
    def two_seconds(frame: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
        stamps = stamps_ns(frame)
        return rolling_time(frame["bid_size_0"], stamps, 2.0).sum()

    honest = TimedFeature("sum_2s", "book", 2.0, 0, two_seconds)
    understated = TimedFeature("sum_2s_said_1s", "book", 1.0, 0, two_seconds)
    assert not check_streaming_parity(honest, rows).leaks
    assert check_streaming_parity(understated, rows).leaks


def test_truncation_catches_a_forward_read(rows) -> None:
    def ahead(frame: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
        return frame["bid_size_0"].shift(-1)

    assert check_timed_causal(TimedFeature("ahead", "book", 0.0, 0, ahead), rows).leaks


def test_registry_derives_n_and_a_row_bound() -> None:
    registry = TimedRegistry()

    @timed_feature(name="a", plane="book", lookback_s=61.0, registry=registry)
    def a(rows: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
        return rows["bid_size_0"]

    @timed_feature(name="b", plane="book", lookback_s=0.0, lookback_rows=20, registry=registry)
    def b(rows: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
        return rows["ask_size_0"]

    assert registry.longest_lookback_minutes() == 2
    assert registry.longest_lookback_rows() == 20
    assert registry.max_rows(["a"]) == 611
    with pytest.raises(ValueError, match="already registered"):
        registry.add(a)
    assert len(registry.describe()) == 2


def test_a_builder_runs_once_for_all_its_columns(rows) -> None:
    registry = TimedRegistry()
    calls = []

    def builder(frame: pd.DataFrame, _: pd.DataFrame | None) -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"x": frame["bid_size_0"], "y": frame["ask_size_0"]})

    register_builder(
        builder,
        ["x", "y"],
        plane="book",
        lookback_s=0.0,
        lookback_rows=0,
        description="two columns",
        prefix="t_",
        registry=registry,
    )
    out = build_timed(rows, ["t_x", "t_y"], registry=registry)
    assert len(calls) == 1
    assert list(out.columns) == ["timestamp", "t_x", "t_y"]
    np.testing.assert_allclose(out["t_y"], rows["ask_size_0"])
