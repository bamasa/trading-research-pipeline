"""A block's days as the study reads them: reasons per venue, warm and cold, inputs, predictions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _spot_day import DAYS, Stub, fetch_days, load
from trading_research.features.timed import stamps_ns
from trading_research.spot import access as sa
from trading_research.spot import days as dy
from trading_research.spot import loader as ld
from trading_research.spot import synthetic as syn


@pytest.fixture(autouse=True)
def _offline():  # type: ignore[no-untyped-def]
    with syn.no_network():
        yield


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, object]:
    found = tmp_path_factory.mktemp("days")
    return found, fetch_days(found, gap_days=frozenset({DAYS[1]}))


def test_follows_reads_the_same_block_or_w_before_d() -> None:
    assert dy.follows(date(2026, 5, 1), date(2026, 4, 30))
    assert dy.follows(date(2026, 5, 2), date(2026, 5, 1))
    assert not dy.follows(date(2026, 8, 8), date(2026, 8, 7))  # E before H: H starts cold
    assert not dy.follows(date(2026, 5, 3), date(2026, 5, 1))


def test_reasons_per_venue_and_the_gap_threshold(base) -> None:  # type: ignore[no-untyped-def]
    root, log = base
    day = load(root, log, DAYS[1], max_gaps=0)
    assert day.usable(sa.BINANCE) and day.reasons[sa.BYBIT] == (ld.SEQUENCE_GAPS,)
    assert day.gap_only(sa.BYBIT)
    assert load(root, log, DAYS[1]).usable(sa.BYBIT)


def test_walk_days_carries_history_between_consecutive_days(base) -> None:  # type: ignore[no-untyped-def]
    root, log = base
    seen = list(dy.walk_days(list(DAYS), lambda d: load(root, log, d)))
    assert [d.day for d in seen] == list(DAYS)
    assert not seen[0].warm[sa.BINANCE] and seen[1].warm[sa.BINANCE] and seen[2].warm[sa.BYBIT]
    assert "binance" in seen[1].history and "book" in seen[2].history
    loaded: list[date] = []

    def counted(day: date) -> dy.DayInputs:
        loaded.append(day)
        return load(root, log, day)

    only_last = list(dy.walk_days([DAYS[2]], counted))
    assert loaded == [DAYS[2], DAYS[1]] and only_last[0].warm[sa.BINANCE]
    warm = dy.store_frame("binance", seen[1])
    cold = dy.store_frame("binance", load(root, log, DAYS[1]))
    assert len(warm) == len(cold) and not warm["flow_imbalance_120"].equals(
        cold["flow_imbalance_120"]
    )


def test_book_rows_feed_the_features_ten_levels(base) -> None:  # type: ignore[no-untyped-def]
    root, log = base
    day = load(root, log, DAYS[0])
    frame = day.book.frame
    deep = frame.assign(bid_price_10=1.0, ask_size_11=2.0)
    assert list(dy.ten_levels(deep).columns) == list(frame.columns)
    tail = dy.book_tail(frame, minutes=10)
    assert stamps_ns(tail)[0] >= stamps_ns(frame)[-1] - 600 * 1_000_000_000


def test_predictions_are_nan_before_scoring_and_on_incomplete_rows() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime([1, 2, 3, 4], unit="s", utc=True),
                          "a": [1.0, -1.0, np.nan, 2.0]})  # fmt: skip
    found = dy.predict(Stub("a", ["a"]), frame, dy.layout("binance", None), 2 * 1_000_000_000)
    assert np.isnan(found[0]).all() and np.isnan(found[2]).all()
    assert found[1, 0] == 0.6 and found[3, 2] == 0.6
    with pytest.raises(ValueError, match="touch size"):
        dy.layout("book", None)
    scale = dy.layout("book", 2.0)
    assert scale.raw_size_columns == ("ofi_1", "ofi_20")
