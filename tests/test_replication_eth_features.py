"""The replication's grid, inputs and labels: values on small cases, and causality."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.events import day_start_ns
from trading_research.replication_eth.features import aux_frame, exec_price, lag_log, zscore
from trading_research.replication_eth.grid import GRID_NS, grid_day, level_columns, trade_aggregates
from trading_research.replication_eth.lob import labels, lob_features, relative_change

DAY = date(2025, 2, 5)


def _book(n: int, depth: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 2700 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], size=n))
    data: dict[str, np.ndarray] = {"ts": day_start_ns(DAY) + GRID_NS * np.arange(1, n + 1)}
    for i in range(depth):
        data[f"ask_price_{i}"] = mid + 0.005 + 0.01 * i
        data[f"ask_size_{i}"] = rng.uniform(0.1, 5.0, n)
    for i in range(depth):
        data[f"bid_price_{i}"] = mid - 0.005 - 0.01 * i
        data[f"bid_size_{i}"] = rng.uniform(0.1, 5.0, n)
    frame = pd.DataFrame(data)
    frame["trade_count"] = rng.integers(0, 3, n).astype(float)
    frame["total_trade_volume"] = frame["trade_count"] * 0.5
    frame["sell_count"] = np.minimum(frame["trade_count"], 1.0)
    frame["signed_volume"] = rng.normal(0, 1, n)
    return frame


def test_lag_log_and_zscore_follow_the_original() -> None:
    x = pd.Series([1.0, 2.0, 0.0, 0.0])
    out = lag_log(x)
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == pytest.approx(np.log((2 + 1e-6) / (1 + 1e-6)), abs=1e-9)
    assert out.iloc[3] == 0.0
    z = zscore(pd.Series([1.0, 1.0, 4.0]))
    assert np.isnan(z.iloc[0]) and z.iloc[1] == pytest.approx(0.0)
    assert z.iloc[2] == pytest.approx((4 - 2) / (np.sqrt(3.0) + 1e-9))


def test_exec_price_walks_one_lot_and_falls_back_to_the_whole_book() -> None:
    prices = np.array([[100.0, 100.01, 100.02], [100.0, 100.01, 100.02]])
    sizes = np.array([[0.4, 0.4, 0.4], [0.1, 0.1, 0.1]])
    out = exec_price(prices, sizes)
    assert out[0] == pytest.approx(100.01)  # 0.4 * 100 + 0.4 * 100.01 + 0.2 * 100.02, on the tick
    assert out[1] == pytest.approx(100.01)  # less than a lot: the weighted price of all three


def test_trade_aggregates_keep_the_original_sign_and_boundaries() -> None:
    start = day_start_ns(DAY)
    prints = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([start + GRID_NS, start + 150_000_000], utc=True),
            "size": [2.0, 3.0],
            "aggressor": [-1, 1],
        }
    )
    agg = trade_aggregates(prints, DAY)
    assert agg.loc[0, "signed_volume"] == 2.0 and agg.loc[0, "sell_count"] == 1.0
    assert agg.loc[1, "signed_volume"] == -3.0 and agg.loc[1, "total_trade_volume"] == 3.0


def test_grid_rows_never_show_a_later_book() -> None:
    start = day_start_ns(DAY)
    book = pd.DataFrame({c: [1.0, 2.0] for c in level_columns(1)})
    book["timestamp"] = pd.to_datetime([start + 150_000_000, start + 250_000_000], utc=True)
    empty = pd.DataFrame({"timestamp": pd.to_datetime([], utc=True), "size": [], "aggressor": []})
    grid = grid_day(book, empty, DAY, depth=1)
    assert grid["ts"].iloc[0] == start + 2 * GRID_NS
    assert list(grid["ask_price_0"].iloc[:2]) == [1.0, 2.0]


def test_inputs_are_trailing() -> None:
    full = _book(1500)
    cut = full.iloc[:900].reset_index(drop=True)
    a, b = aux_frame(full, 20), aux_frame(cut, 20)
    pd.testing.assert_frame_equal(a.iloc[:900], b)
    book = [
        full[[f"{s}_{k}_{i}" for i in range(10)]].to_numpy()
        for s, k in [("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size")]
    ]
    head = [x[:900] for x in book]
    prices = np.r_[0:10, 20:30]  # sizes use a floor read from the whole day, as in the original
    np.testing.assert_array_equal(lob_features(*book)[:800, prices], lob_features(*head)[:, prices])


def test_labels_compare_future_and_past_means() -> None:
    mid = np.r_[np.full(60, 100.0), np.full(60, 100.1)]
    rel = relative_change(mid, 50)
    assert np.isnan(rel[:49]).all() and np.isnan(rel[-50:]).all()
    y = labels(rel, 5e-4)
    assert (y[49:70] == 3).all() and (y[:49] == 2).all()
    assert rel[49] == pytest.approx((100.08 - 100.0) / 100.0)
    assert (labels(-rel, 5e-4)[49:70] == 1).all()
    assert (labels(rel, 2e-3) == 2).all()
