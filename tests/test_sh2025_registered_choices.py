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
import yaml

from trading_research.data import feature_store as fs
from trading_research.data import fetch_log, study_fetch, study_loader
from trading_research.features import study_set
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
