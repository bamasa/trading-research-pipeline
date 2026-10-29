"""The regime area's constants against the registered configuration file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from trading_research.pipeline import regime
from trading_research.validation import regime_flags as rf
from trading_research.validation.structural_breaks import RECORDED_THRESHOLDS

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "short_horizon_2025.yaml"


def _registered() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_the_monitor_is_the_registered_one() -> None:
    detector = _registered()["detector"]
    monitor = detector["monitor"]
    spec = rf.DETECTOR_SPEC
    assert (spec.history_len, spec.online_len, spec.min_history) == (
        monitor["history_len"],
        monitor["online_len"],
        monitor["min_history"],
    )
    assert spec.minimum_gap == monitor["minimum_gap"]
    assert list(spec.statistics) == monitor["statistics"]
    recorded = RECORDED_THRESHOLDS[(spec.history_len, spec.online)]
    assert {s: recorded[s] for s in spec.statistics} == monitor["thresholds"]
    assert detector["mean_channel"]["threshold"] == rf.MEAN_THRESHOLD


def test_the_stream_starts_where_the_bars_do() -> None:
    bars = _registered()["data"]["detector_bars"]
    assert pd.Timestamp(str(bars["from"]), tz="UTC") == rf.STREAM_START
    assert bars["interval"] == "1h"


def test_the_axis_and_the_windows_are_the_registered_ones() -> None:
    search = _registered()["search"]
    source = search["regime_source"]
    assert tuple(source["windows_h"]) == regime.REGIME_WINDOWS_H
    assert max(source["windows_h"]) == rf.LEAD_IN_HOURS
    assert tuple(search["axes"]["regime"]["values"]) == regime.SEARCH_POLICIES
    sets = source["attribution"]["sets"]
    assert (sets["taker"], sets["simulator"]) == (regime.TAKER_SETS, regime.SIMULATOR_SETS)
    numbers = search["choice"]["axes_as_numbers"]["regime"]
    for name in regime.SEARCH_POLICIES:
        assert f"{name} {regime.window_hours(name)}" in numbers


def test_the_folds_are_the_walk_forward_months() -> None:
    months = [f.name for f in rf.walk_forward_months()]
    blocks = _registered()["blocks"]
    first_test = pd.Timestamp(str(blocks["D"]["start"])) + pd.offsets.MonthBegin(1)
    assert months[0] == f"{first_test:%Y-%m}"
    assert months[-1] == f"{pd.Timestamp(str(blocks['D']['end'])):%Y-%m}"
    assert rf.FROZEN_PROFILE_SPAN[0] == pd.Timestamp(str(blocks["D"]["start"]), tz="UTC")
