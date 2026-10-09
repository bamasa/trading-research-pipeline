"""Two float16 feature stores, the raw-size scaling and the finiteness test.

A float16 holds at most 65,504. A depth sum in BTC on a busy day can pass it,
and the cast would store infinity where a number was: the tests pin that the
scaling by the warm-up's touch size prevents it, that a column which still
overflows on the warm-up block is stored at float32 everywhere, and that a
later block's overflow is reported rather than silently stored.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.data import feature_store as fs

DAY = date(2024, 12, 2)


def book(sizes_bid: list[float], sizes_ask: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"bid_size_0": sizes_bid, "ask_size_0": sizes_ask})


def features(n: int = 40, *, depth: float = 1e3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2024-12-02", periods=n, freq="100ms", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "spread_bp": rng.uniform(0.0, 2.0, n),
            "ofi_1": rng.normal(0.0, depth, n),
            "depth_sum_10": rng.uniform(depth, 2 * depth, n),
            "turns": rng.integers(0, 50, n),
        }
    )


def test_the_touch_size_pools_both_sides_of_every_day() -> None:
    days = [book([1.0, 3.0], [5.0, 7.0]), book([2.0], [100.0])]
    assert fs.median_touch_size(days) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="no book rows"):
        fs.median_touch_size([])
    with pytest.raises(ValueError, match="positive scale"):
        fs.median_touch_size([book([0.0, 0.0], [0.0, 0.0])])


def test_raw_size_columns_are_divided_by_the_touch_size_only() -> None:
    frame = features()
    scaled = fs.scale_raw_sizes(frame, ["ofi_1", "depth_sum_10"], 4.0)
    assert np.allclose(scaled["ofi_1"], frame["ofi_1"] / 4.0)
    assert np.allclose(scaled["spread_bp"], frame["spread_bp"])
    with pytest.raises(KeyError, match="absent"):
        fs.scale_raw_sizes(frame, ["ofi_20"], 4.0)


def test_scaling_keeps_a_busy_days_depth_finite_at_float16() -> None:
    frame = features(depth=1e5)  # sums in the hundreds of thousands of BTC
    unscaled = fs.finiteness_failures(frame, [], 1.0)
    assert set(unscaled.overflow) == {"ofi_1", "depth_sum_10"}
    scaled = fs.finiteness_failures(frame, ["ofi_1", "depth_sum_10"], 50.0)
    assert scaled.ok


def test_missing_values_pass_and_infinities_are_reported_at_their_source() -> None:
    frame = features()
    frame.loc[:5, "spread_bp"] = np.nan
    assert fs.finiteness_failures(frame, ["ofi_1"], 1.0).ok
    frame.loc[7, "spread_bp"] = -np.inf
    found = fs.finiteness_failures(frame, ["ofi_1"], 1.0)
    assert found.infinite_at_source == ("spread_bp",) and found.overflow == ()


def test_a_column_that_overflows_on_the_warm_up_is_stored_at_float32() -> None:
    quiet, busy = features(seed=1), features(depth=1e8, seed=2)
    layout, found = fs.fit_layout(
        [quiet, busy], touch_size=10.0, raw_size_columns=["ofi_1", "depth_sum_10"]
    )
    assert layout.float32_columns == ("depth_sum_10", "ofi_1")
    assert found.overflow == layout.float32_columns
    stored = fs.to_stored(busy, layout)
    assert stored["depth_sum_10"].dtype == np.float32
    assert stored["spread_bp"].dtype == np.float16 and stored["turns"].dtype == np.float16
    assert np.isfinite(stored.drop(columns="timestamp").to_numpy(dtype=np.float64)).all()
    assert stored["timestamp"].equals(busy["timestamp"])


def test_a_store_holds_numeric_inputs_only() -> None:
    frame = features().assign(symbol="BTCUSDT")
    with pytest.raises(TypeError, match="numeric model inputs only"):
        fs.finiteness_failures(frame, [], 1.0)


def test_the_layout_round_trips_and_refuses_a_bad_scale() -> None:
    layout = fs.StoreLayout(2.5, ("ofi_1",), ("depth_sum_10",))
    assert fs.StoreLayout.from_dict(layout.to_dict()) == layout
    with pytest.raises(ValueError, match="positive"):
        fs.StoreLayout(0.0, ())


LAYOUT = fs.StoreLayout(10.0, ("ofi_1", "depth_sum_10"))


def test_the_two_stores_keep_every_row_and_every_tenth(tmp_path: Path) -> None:
    stores = fs.open_stores(tmp_path, LAYOUT)
    frame = features(n=95)
    every = stores[fs.EVERY_ROW].write_day(DAY, frame)
    tenth = stores[fs.EVERY_TENTH_ROW].write_day(DAY, frame)
    assert (every.rows, tenth.rows) == (95, 10)
    back = stores[fs.EVERY_TENTH_ROW].read_day(DAY)
    assert back["timestamp"].tolist() == frame["timestamp"].iloc[::10].tolist()
    assert back["spread_bp"].dtype == np.float16
    expected = (frame["ofi_1"].iloc[::10] / 10.0).to_numpy(dtype=np.float16)
    assert np.array_equal(back["ofi_1"].to_numpy(), expected)
    assert stores[fs.EVERY_ROW].read_day(DAY, ["ofi_1"]).columns.tolist() == ["ofi_1"]
    assert every.size > 0 and every.overflow == ()
    assert not list(tmp_path.rglob("*.part"))


def test_a_later_blocks_overflow_is_reported_not_repaired(tmp_path: Path) -> None:
    store = fs.FeatureStore(tmp_path, fs.EVERY_ROW, LAYOUT)
    written = store.write_day(DAY, features(depth=1e7))
    assert written.overflow == ("ofi_1", "depth_sum_10")


def test_a_store_is_never_written_under_two_layouts(tmp_path: Path) -> None:
    fs.FeatureStore(tmp_path, fs.EVERY_ROW, LAYOUT)
    fs.FeatureStore(tmp_path, fs.EVERY_ROW, LAYOUT)  # the same layout reopens
    other = fs.StoreLayout(11.0, ("ofi_1", "depth_sum_10"))
    with pytest.raises(ValueError, match="another layout"):
        fs.FeatureStore(tmp_path, fs.EVERY_ROW, other)
    with pytest.raises(ValueError, match="unknown store"):
        fs.FeatureStore(tmp_path, "every_fifth_row", LAYOUT)


def test_reading_a_day_not_stored_raises(tmp_path: Path) -> None:
    store = fs.FeatureStore(tmp_path, fs.EVERY_ROW, LAYOUT)
    with pytest.raises(FileNotFoundError, match="no every_row features"):
        store.read_day(DAY)


def test_the_warm_up_and_development_features_are_deleted_by_span(tmp_path: Path) -> None:
    stores = fs.open_stores(tmp_path, LAYOUT)
    days = [date(2024, 12, 31), date(2025, 1, 1), date(2025, 6, 30), date(2025, 7, 8)]
    for store in stores.values():
        for day in days:
            store.write_day(day, features())
    before = sum(s.size_bytes() for s in stores.values())
    freed = fs.delete_features(stores.values(), date(2024, 12, 1), date(2025, 6, 30))
    assert freed > 0
    assert sum(s.size_bytes() for s in stores.values()) == before - freed
    for store in stores.values():
        assert store.days() == [date(2025, 7, 8)]
        assert (store.directory / fs.LAYOUT_FILE).exists()
