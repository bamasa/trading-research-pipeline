"""S25-BB's and S25-T's inputs and S25-BB's label on a synthetic spot book, from the 2025 code."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from _sh_synthetic import make_prints, make_rows
from test_spot_synthetic_prints import no_network
from trading_research.features.timed import build_timed, stamps_ns
from trading_research.labels.taker_vwap import WalkedTakerLabel
from trading_research.spot.book_inputs import (
    S25_SELECTED,
    book_day_frame,
    book_registry,
    book_targets,
    scale_inputs,
    spot_touch_size,
    touch_size_2025,
)
from trading_research.spot.print_label import forward_column, label_column
from trading_research.validation.leakage_timed import assert_timed_honest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return make_rows(900, start="2026-03-02 23:58:30", quiet_at=500)


@pytest.fixture(scope="module")
def prints(rows: pd.DataFrame) -> pd.DataFrame:
    return make_prints(rows)


def test_the_inputs_are_the_2025_selection_as_both_configurations_record_it() -> None:
    spot = yaml.safe_load((ROOT / "configs" / "spot_2026.yaml").read_text())
    study_2025 = yaml.safe_load((ROOT / "configs" / "short_horizon_2025.yaml").read_text())
    assert list(S25_SELECTED) == spot["features"]["s25_selected"]
    assert list(S25_SELECTED) == study_2025["features"]["selected"]
    assert len(set(S25_SELECTED)) == 40 and set(S25_SELECTED) <= set(book_registry().names())


def test_the_40_pass_truncation_and_streaming_on_a_spot_like_book(rows, prints) -> None:
    positions = [*range(0, len(rows), 29), len(rows) - 1]
    assert_timed_honest(
        book_registry(), rows, prints, names=list(S25_SELECTED), positions=positions
    )


def test_a_warm_day_equals_both_days_computed_together(rows, prints) -> None:
    midnight = pd.Timestamp("2026-03-03", tz="UTC")
    day = rows["timestamp"] >= midnight
    day_prints = prints["timestamp"] > rows["timestamp"][~day].iloc[-1]
    warm = book_day_frame(rows[day].reset_index(drop=True), prints[day_prints],
                          history=(rows[~day], prints[~day_prints]))  # fmt: skip
    together = build_timed(rows, list(S25_SELECTED), prints=prints, registry=book_registry())
    expected = together[day.to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(warm, expected)
    cold = book_day_frame(rows[day].reset_index(drop=True), prints[day_prints])
    assert len(cold) == len(warm) and not cold.equals(warm)


def test_raw_sizes_are_divided_by_the_slots_touch_size(rows, prints) -> None:
    frame = book_day_frame(rows, prints, names=["ofi_1", "ofi_20", "spread_bp"])
    scale = spot_touch_size([rows.iloc[:300], rows.iloc[300:]])
    pooled = np.concatenate([rows["bid_size_0"], rows["ask_size_0"]])
    assert scale == pytest.approx(float(np.median(pooled)))
    scaled = scale_inputs(frame, scale)
    np.testing.assert_allclose(scaled["ofi_20"], frame["ofi_20"] / scale)
    pd.testing.assert_series_equal(scaled["spread_bp"], frame["spread_bp"])
    assert touch_size_2025({"touch_size": 0.25, "tensor": {}}) == 0.25
    with pytest.raises(ValueError, match="positive"):
        touch_size_2025({"touch_size": 0.0})


@pytest.mark.parametrize("depth", [10, 12])
def test_targets_are_the_2025_walked_label_at_each_fee(depth: int) -> None:
    book = make_rows(700, depth=depth, start="2026-03-03", quiet_at=None)
    at = np.arange(0, len(book), 10)
    targets = book_targets(book, [2.0, 5.0], [0.0, 10.0], depth=depth, at=at)
    for hold in (2.0, 5.0):
        for fee in (0.0, 10.0):
            direct = WalkedTakerLabel(hold, label_fee_bp=fee, depth=depth)(book).to_numpy()[at]
            np.testing.assert_array_equal(targets[label_column(hold, fee)].to_numpy(), direct)
        assert targets[forward_column(hold)].notna().any()
    stamps = stamps_ns(book)
    expected = np.full(len(at), -1, dtype=np.int64)
    for hold in (2.0, 5.0):
        exits = WalkedTakerLabel(hold, depth=depth).outcomes(book)["exit_index"].to_numpy()[at]
        known = exits < len(book)
        expected[known] = np.maximum(expected[known], stamps[exits[known]])
        own = np.where(known, stamps[np.minimum(exits, len(book) - 1)], -1)
        assert (targets[f"reads_until_ns_{hold:g}"].to_numpy() == own).all()
    assert (targets["reads_until_ns"].to_numpy() == expected).all() and (expected == -1).any()
