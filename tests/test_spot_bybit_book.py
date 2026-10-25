"""Bybit's spot ob200 book: the registered replay, its tag, its grid, its message statistics."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.data.bybit import BybitArchiveError
from trading_research.data.validate import Severity, validate_book
from trading_research.spot import bybit_book as bb
from trading_research.spot import synthetic as syn

DAY = date(2026, 3, 3)
MIDNIGHT_MS = 1_772_496_000_000


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


def _write(tmp_path: Path, market: syn.SpotMarket) -> Path:
    path = tmp_path / "day.zip"
    path.write_bytes(market.book_archive(DAY))
    return path


def _message(kind: str, ts: int, u: int, bid: float, ask: float) -> str:
    b = [[f"{bid - k:.1f}", "1.0"] for k in range(10)]
    a = [[f"{ask + k:.1f}", "1.0"] for k in range(10)]
    data = {"s": "BTCUSDT", "b": b, "a": a, "u": u, "seq": u}
    return json.dumps({"topic": "orderbook.200.BTCUSDT", "ts": ts, "type": kind, "data": data})


def test_a_day_replays_at_ten_levels_with_the_spot_tag(tmp_path: Path) -> None:
    frame = bb.replay(_write(tmp_path, syn.SpotMarket(frozenset({DAY}), step_ms=60_000)))
    assert len(frame) == 1440 and frame.attrs["sequence_gaps"] == 0
    assert set(frame["source"]) == {"bybit-spot-ob200"}
    assert set(frame["symbol"]) == {"bybit-spot:BTCUSDT"}
    assert [c for c in frame.columns if c.startswith("bid_price_")][-1] == "bid_price_9"
    report = validate_book(frame, depth=10, sampled=True)
    assert not [f for f in report if f.severity is Severity.ERROR]


def test_one_row_per_100_ms_interval_at_its_first_update(tmp_path: Path) -> None:
    lines = [
        _message("snapshot", MIDNIGHT_MS + 5, 1, 100.0, 100.1),
        _message("delta", MIDNIGHT_MS + 30, 2, 101.0, 101.1),
        _message("delta", MIDNIGHT_MS + 99, 3, 102.0, 102.1),
        _message("delta", MIDNIGHT_MS + 140, 4, 103.0, 103.1),
        '{"topic": "other", "ts": 1772496000150, "type": "status"}',
    ]
    path = tmp_path / "small.data"
    path.write_text("\n".join(lines) + "\n")
    frame = bb.replay(path)
    stamps = (frame["timestamp"].astype("int64") // 1_000_000 - MIDNIGHT_MS).tolist()
    assert stamps == [5, 140]  # the 30 and 99 ms updates show only in the next interval's row
    assert frame["bid_price_0"].tolist() == [100.0, 103.0]
    stats = bb.message_stats(path)
    assert (stats.messages, stats.snapshots, stats.deltas, stats.other) == (4, 1, 3, 1)
    assert (stats.first_ts, stats.last_ts) == (MIDNIGHT_MS + 5, MIDNIGHT_MS + 140)
    assert stats.median_interval_ms == 41.0 and stats.max_interval_ms == 69


def test_gaps_are_counted_as_reconstruct_counts_them(tmp_path: Path) -> None:
    market = syn.SpotMarket(frozenset({DAY}), step_ms=60_000, gap_days=frozenset({DAY}))
    path = _write(tmp_path, market)
    stats = bb.message_stats(path)
    assert stats.sequence_gaps == 3 == bb.replay(path).attrs["sequence_gaps"]
    assert stats.messages == 1440 and stats.median_interval_ms == 60_000.0


def test_the_walk_can_replay_deeper_and_a_too_shallow_book_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, syn.SpotMarket(frozenset({DAY}), step_ms=600_000))
    assert "ask_size_11" in bb.replay(path, depth=syn.LEVELS).columns
    with pytest.raises(BybitArchiveError, match="no usable book rows"):
        bb.replay(path, depth=bb.ARCHIVE_LEVELS)


def test_the_share_of_rows_whose_clip_exceeds_ten_levels() -> None:
    rows: dict[str, object] = {}
    for k in range(10):
        rows[f"bid_size_{k}"] = np.array([0.002, 0.0009, 0.002, 0.002])
        rows[f"ask_size_{k}"] = np.array([0.002, 0.002, 0.0005, 0.0015])
    frame = pd.DataFrame(rows)
    assert bb.beyond_levels_share(frame) == 0.5  # 10 x 0.0009 and 10 x 0.0005 fall short of 0.010
    assert bb.beyond_levels_share(frame, clip=0.0075) == 0.25
    with pytest.raises(ValueError, match="no book rows"):
        bb.beyond_levels_share(frame.iloc[:0])


def test_the_archive_member_is_read_from_the_zip(tmp_path: Path) -> None:
    path = _write(tmp_path, syn.SpotMarket(frozenset({DAY}), step_ms=600_000))
    with zipfile.ZipFile(path) as packed:
        assert packed.namelist() == ["2026-03-03_BTCUSDT_ob200.data"]
    assert bb.message_stats(path).messages == 144
