"""Tests for reconstructing a book from Bybit's delta stream.

Three properties decide whether a reconstructed book is usable, and all three
fail silently if got wrong: a size of zero must remove a level rather than
create an empty one, a sequence gap must be noticed, and a sample must be the
book as it stood rather than an average across updates.
"""

from __future__ import annotations

import json
import zipfile
from datetime import date

import pandas as pd
import pytest

from trading_research.data.bybit import (
    ARCHIVE_DEPTH,
    BookState,
    BybitArchiveError,
    archive_url,
    reconstruct,
)


def message(kind: str, ts: int, bids: list, asks: list, update: int) -> dict:
    return {
        "topic": "orderbook.500.TESTUSDT",
        "type": kind,
        "ts": ts,
        "data": {"s": "TESTUSDT", "b": bids, "a": asks, "u": update, "seq": update},
    }


def levels(base: float, n: int, step: float, size: str = "1.0") -> list[list[str]]:
    return [[f"{base + i * step:.2f}", size] for i in range(n)]


def write(path, messages: list[dict]) -> str:
    raw = path / "book.data"
    raw.write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")
    return str(raw)


@pytest.fixture
def simple(tmp_path) -> str:
    start = 1_707_091_200_000
    first = message("snapshot", start, levels(100.0, 12, -0.1), levels(101.0, 12, 0.1), update=1)
    rest = [
        message("delta", start + 100 * i, [["100.00", f"{2.0 + i}"]], [], update=1 + i)
        for i in range(1, 6)
    ]
    return write(tmp_path, [first, *rest])


# ---------------------------------------------------------------------------
# The book itself
# ---------------------------------------------------------------------------


def test_zero_removes_a_level_rather_than_creating_an_empty_one() -> None:
    """A phantom level at the touch changes every imbalance feature."""
    state = BookState()
    state.apply("b", [["100.0", "1.0"], ["99.9", "2.0"]])
    state.apply("b", [["100.0", "0"]])
    assert 100.0 not in state.bids
    assert state.bids == {99.9: 2.0}


def test_a_later_size_replaces_an_earlier_one() -> None:
    state = BookState()
    state.apply("a", [["101.0", "1.0"]])
    state.apply("a", [["101.0", "5.0"]])
    assert state.asks == {101.0: 5.0}


def test_the_top_is_sorted_inward_from_each_side() -> None:
    state = BookState()
    state.apply("b", [["99.0", "1"], ["100.0", "1"], ["98.0", "1"]])
    state.apply("a", [["103.0", "1"], ["101.0", "1"], ["102.0", "1"]])
    bids, asks = state.top(3)
    assert [p for p, _ in bids] == [100.0, 99.0, 98.0]
    assert [p for p, _ in asks] == [101.0, 102.0, 103.0]


def test_a_book_with_one_empty_side_is_not_ready() -> None:
    state = BookState()
    state.apply("b", [["100.0", "1.0"]])
    assert not state.ready


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_a_stream_reconstructs_into_levels(simple) -> None:
    frame = reconstruct(simple, symbol="TESTUSDT", depth=10)
    assert len(frame) > 0
    for level in range(10):
        assert f"bid_price_{level}" in frame.columns
        assert f"ask_size_{level}" in frame.columns
    assert (frame["bid_price_0"] > frame["bid_price_9"]).all()
    assert (frame["ask_price_0"] < frame["ask_price_9"]).all()


def test_the_touch_never_crosses(simple) -> None:
    frame = reconstruct(simple, symbol="TESTUSDT", depth=10)
    assert (frame["bid_price_0"] < frame["ask_price_0"]).all()


def test_updates_are_applied_in_order(simple) -> None:
    """The last size written to a level is the one sampled."""
    frame = reconstruct(simple, symbol="TESTUSDT", depth=10, grid_ms=100)
    # The deltas write 2.0 + i for i in 1..5, so the final size is 7.0.
    assert frame["bid_size_0"].iloc[-1] == pytest.approx(7.0)


def test_a_sequence_gap_is_counted_not_hidden(tmp_path) -> None:
    """A missed update leaves the book wrong until the next snapshot, and
    nothing downstream can tell."""
    start = 1_707_091_200_000
    messages = [
        message("snapshot", start, levels(100.0, 12, -0.1), levels(101.0, 12, 0.1), update=1),
        message("delta", start + 100, [["100.00", "3.0"]], [], update=2),
        # 3 is missing.
        message("delta", start + 200, [["100.00", "4.0"]], [], update=4),
    ]
    frame = reconstruct(write(tmp_path, messages), symbol="TESTUSDT", depth=10)
    assert frame.attrs["sequence_gaps"] == 1


def test_a_clean_stream_reports_no_gaps(simple) -> None:
    assert reconstruct(simple, symbol="TESTUSDT", depth=10).attrs["sequence_gaps"] == 0


def test_the_grid_samples_at_most_once_per_interval(tmp_path) -> None:
    """A photograph, not an average: several updates inside one interval leave
    one row carrying the state at the end of it."""
    start = 1_707_091_200_000
    messages = [
        message("snapshot", start, levels(100.0, 12, -0.1), levels(101.0, 12, 0.1), update=1)
    ]
    messages += [
        message("delta", start + i, [["100.00", f"{1.0 + i}"]], [], update=1 + i)
        for i in range(1, 10)
    ]
    frame = reconstruct(write(tmp_path, messages), symbol="TESTUSDT", depth=10, grid_ms=1000)
    assert len(frame) == 1


def test_a_thinner_book_than_requested_is_skipped(tmp_path) -> None:
    """Emitting a row with missing levels would hand downstream a NaN it cannot
    distinguish from a quiet market."""
    start = 1_707_091_200_000
    messages = [message("snapshot", start, levels(100.0, 3, -0.1), levels(101.0, 3, 0.1), 1)]
    with pytest.raises(BybitArchiveError, match="no usable"):
        reconstruct(write(tmp_path, messages), symbol="TESTUSDT", depth=10)


def test_a_snapshot_resets_the_book(tmp_path) -> None:
    start = 1_707_091_200_000
    messages = [
        message("snapshot", start, levels(100.0, 12, -0.1), levels(101.0, 12, 0.1), 1),
        message("snapshot", start + 1000, levels(200.0, 12, -0.1), levels(201.0, 12, 0.1), 1),
    ]
    frame = reconstruct(write(tmp_path, messages), symbol="TESTUSDT", depth=10, grid_ms=100)
    assert frame["bid_price_0"].iloc[-1] == pytest.approx(200.0)


def test_a_zipped_archive_reads_the_same_as_a_plain_one(simple, tmp_path) -> None:
    zipped = tmp_path / "book.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.write(simple, arcname="book.data")
    plain = reconstruct(simple, symbol="TESTUSDT", depth=10)
    packed = reconstruct(zipped, symbol="TESTUSDT", depth=10)
    pd.testing.assert_frame_equal(plain, packed)


def test_the_archive_url_names_the_day_and_symbol() -> None:
    url = archive_url("BTCUSDT", date(2024, 2, 5))
    assert "BTCUSDT" in url and "2024-02-05" in url
    assert str(ARCHIVE_DEPTH) in url


# ---------------------------------------------------------------------------
# Trade prints: order within a timestamp, and one day at a time
# ---------------------------------------------------------------------------

_HEADER = (
    "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
    "grossValue,homeNotional,foreignNotional"
)


def _archive(rows: list[tuple[float, str, float, float, str]], *, match_id: bool = True) -> bytes:
    header = _HEADER if match_id else "timestamp,symbol,side,size,price"
    lines = [header]
    for stamp, side, size, price, match in rows:
        tail = f",PlusTick,{match},0,0,0" if match_id else ""
        lines.append(f"{stamp},TESTUSDT,{side},{size},{price}{tail}")
    return "\n".join(lines).encode()


def test_parse_keeps_the_file_order_of_equal_timestamps_and_the_match_id() -> None:
    """Ties are the norm in the tape, and an unstable sort reorders them; with
    two thousand prints over five timestamps it reliably does."""
    import numpy as np

    from trading_research.data.bybit_trades import parse

    rng = np.random.default_rng(0)
    stamps = rng.choice([1707955200.1, 1707955200.2, 1707955200.3, 1707955200.4], 2000)
    rows = [
        (float(stamp), "Buy" if i % 3 else "Sell", 1.0, 0.41, f"m{i:05d}")
        for i, stamp in enumerate(stamps)
    ]
    frame = parse(_archive(rows))
    assert frame["timestamp"].is_monotonic_increasing
    assert "match_id" in frame.columns
    for _, group in frame.groupby("timestamp"):
        assert list(group["match_id"]) == sorted(group["match_id"])
    assert set(frame["aggressor"]) == {-1, 1}


def test_parse_reads_an_archive_without_a_match_id() -> None:
    from trading_research.data.bybit_trades import parse

    frame = parse(_archive([(1707955200.5, "Sell", 2.0, 0.41, "")], match_id=False))
    assert "match_id" not in frame.columns
    assert frame["aggressor"].tolist() == [-1]


def test_one_day_of_prints_is_read_alone(tmp_path) -> None:
    from trading_research.data.bybit_trades import BybitTradesError, load_day

    directory = tmp_path / "TESTUSDT"
    directory.mkdir()
    pd.DataFrame({"price": [1.0], "size": [2.0]}).to_parquet(directory / "2024-02-15.parquet")
    (directory / "2024-02-16.parquet").write_bytes(b"unreadable on purpose")
    assert load_day("TESTUSDT", date(2024, 2, 15), tmp_path)["size"].tolist() == [2.0]
    with pytest.raises(BybitTradesError, match="no trades"):
        load_day("TESTUSDT", date(2024, 2, 17), tmp_path)


# ---------------------------------------------------------------------------
# The grid's label
# ---------------------------------------------------------------------------


def _observed_book() -> pd.DataFrame:
    """A bid equal to the second it was observed at: 0.0, 4.9, 7.0, 12.3."""
    seconds = [0.0, 4.9, 7.0, 12.3]
    stamps = pd.to_datetime([1_706_745_600 + s for s in seconds], unit="s", utc=True)
    return pd.DataFrame({"timestamp": stamps, "bid_price_0": seconds, "ask_price_0": seconds})


def test_a_left_labelled_grid_is_five_seconds_ahead_when_joined_at_its_label() -> None:
    """The look-ahead the right label removes, pinned: the row labelled
    00:00:00 carries the value observed at 00:00:04.9."""
    from trading_research.data.grid import to_grid

    grid = to_grid(_observed_book(), 5)
    label = (grid["timestamp"] - grid["timestamp"].iloc[0]).dt.total_seconds()
    assert grid["bid_price_0"].iloc[0] == 4.9
    assert (grid["bid_price_0"] - label).max() == pytest.approx(4.9)


def test_a_right_labelled_grid_never_carries_a_value_from_after_its_label() -> None:
    from trading_research.data.grid import to_grid

    left = to_grid(_observed_book(), 5)
    right = to_grid(_observed_book(), 5, label="right")
    origin = left["timestamp"].iloc[0]
    label = (right["timestamp"] - origin).dt.total_seconds()
    assert (right["bid_price_0"] < label).all()
    assert list(label) == [5.0, 10.0, 15.0]
    # The same values, one bin later.
    assert right["bid_price_0"].tolist() == left["bid_price_0"].tolist()
    assert ((right["timestamp"] - left["timestamp"]).dt.total_seconds() == 5.0).all()
    with pytest.raises(ValueError, match="label"):
        to_grid(_observed_book(), 5, label="middle")  # type: ignore[arg-type]
