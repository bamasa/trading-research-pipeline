"""The two archive names of Bybit's book, and the stream statistics beside a replay.

The venue moved its linear book archive from five hundred levels to two hundred
on one day, with no day published under both. The name is chosen by date, the
replay reads either the same way, and the stream statistics count what the
replayed book cannot show.
"""

from __future__ import annotations

import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from trading_research.data import bybit
from trading_research.data.bybit import (
    ARCHIVE_CHANGE,
    ARCHIVE_DEPTHS,
    archive_kind,
    archive_message_stats,
    archive_url,
    reconstruct,
    source_tag,
)
from trading_research.data.validate import validate_book

START = 1_756_000_000_000


def _message(kind: str, ts: int, bids: list, asks: list, update: int, depth: int) -> dict:
    return {
        "topic": f"orderbook.{depth}.TESTUSDT",
        "type": kind,
        "ts": ts,
        "data": {"s": "TESTUSDT", "b": bids, "a": asks, "u": update, "seq": update + 7},
        "cts": ts - 3,
    }


def _levels(base: float, n: int, step: float) -> list[list[str]]:
    return [[f"{base + i * step:.1f}", "1.5"] for i in range(n)]


def _stream(depth: int = 200, *, gap_at: int | None = None) -> list[dict]:
    """A snapshot at ``depth`` levels and thirty deltas, one every 40 ms."""
    out = [
        _message(
            "snapshot", START, _levels(100.0, depth, -0.1), _levels(100.1, depth, 0.1), 10, depth
        )
    ]
    update = 10
    for i in range(1, 31):
        update += 2 if i == gap_at else 1
        out.append(_message("delta", START + 40 * i, [["100.0", f"{1 + i}"]], [], update, depth))
    return out


def _archive(directory: Path, day: date, messages: list[dict], kind: str = "ob200") -> Path:
    """A zip shaped like the venue's: one ``.data`` member of JSON lines."""
    name = f"{day.isoformat()}_TESTUSDT_{kind}.data"
    path = directory / f"{name}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, "\n".join(json.dumps(m) for m in messages) + "\n")
    return path


def test_the_name_changes_on_the_registered_day() -> None:
    assert archive_kind(date(2025, 8, 20)) == "ob500"
    assert archive_kind(ARCHIVE_CHANGE) == "ob200"
    assert archive_kind(date(2026, 1, 5)) == "ob200"
    assert archive_kind(date(2024, 12, 1)) == "ob500"


def test_each_url_names_its_archive() -> None:
    before = archive_url("BTCUSDT", date(2025, 8, 20))
    after = archive_url("BTCUSDT", date(2025, 8, 21))
    assert before.endswith("/BTCUSDT/2025-08-20_BTCUSDT_ob500.data.zip")
    assert after.endswith("/BTCUSDT/2025-08-21_BTCUSDT_ob200.data.zip")
    assert "ob200" in archive_url("BTCUSDT", date(2024, 2, 1), "ob200")


def test_an_unknown_archive_name_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown archive"):
        archive_url("BTCUSDT", date(2025, 1, 1), "ob50")
    with pytest.raises(ValueError, match="unknown archive"):
        source_tag("ob1000")
    assert ARCHIVE_DEPTHS == {"ob500": 500, "ob200": 200}


def test_an_ob200_archive_replays_into_ten_levels(tmp_path: Path) -> None:
    path = _archive(tmp_path, ARCHIVE_CHANGE, _stream())
    frame = reconstruct(path, symbol="TESTUSDT", depth=10, source=source_tag("ob200"))
    assert len(frame) == 13  # 31 messages 40 ms apart: one row per 100 ms interval
    assert set(frame["source"]) == {"bybit-linear-ob200"}
    assert frame["bid_size_0"].iloc[-1] == pytest.approx(31.0)
    assert frame.attrs["sequence_gaps"] == 0
    report = validate_book(frame, depth=10, sampled=True)
    assert not [f for f in report.errors if f.check == "positive_prices"]
    assert report.ok


def test_the_ob200_replay_matches_the_same_stream_named_ob500(tmp_path: Path) -> None:
    """The name changes the URL and the source tag, never the replay."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = _archive(tmp_path / "a", ARCHIVE_CHANGE, _stream(), "ob200")
    two = _archive(tmp_path / "b", ARCHIVE_CHANGE, _stream(), "ob500")
    left = reconstruct(one, symbol="TESTUSDT").drop(columns="source")
    right = reconstruct(two, symbol="TESTUSDT").drop(columns="source")
    pd.testing.assert_frame_equal(left, right)


def test_a_day_after_the_change_is_fetched_and_tagged_as_ob200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache name and the source column follow the day's archive."""
    day = date(2026, 1, 5)
    fixture = _archive(tmp_path, day, _stream())
    requested: list[str] = []

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return fixture.read_bytes()

    def opener(url: str, timeout: float) -> _Response:
        requested.append(url)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", opener)
    out, cache = tmp_path / "out", tmp_path / "cache"
    out.mkdir()
    job = ("TESTUSDT", day, out, cache, 10, 100, True, False)
    result = bybit._one_day(job)
    assert result["rows"] == 13
    assert requested == [archive_url("TESTUSDT", day)]
    assert (cache / "TESTUSDT-2026-01-05-ob200.zip").exists()
    written = pd.read_parquet(out / "2026-01-05.parquet")
    assert set(written["source"]) == {"bybit-linear-ob200"}


def test_message_stats_count_what_the_replay_cannot_show(tmp_path: Path) -> None:
    path = _archive(tmp_path, ARCHIVE_CHANGE, _stream())
    stats = archive_message_stats(path)
    assert (stats.messages, stats.snapshots, stats.deltas) == (31, 1, 30)
    assert stats.median_interval_ms == pytest.approx(40.0)
    assert stats.sequence_gaps == 0


def test_message_stats_count_gaps_as_the_replay_does(tmp_path: Path) -> None:
    path = _archive(tmp_path, ARCHIVE_CHANGE, _stream(gap_at=12))
    stats = archive_message_stats(path)
    assert stats.sequence_gaps == 1
    assert stats.sequence_gaps == reconstruct(path, symbol="TESTUSDT").attrs["sequence_gaps"]


def test_message_stats_read_an_unusual_layout_in_full(tmp_path: Path) -> None:
    """A line the patterns miss is decoded rather than skipped."""
    messages = _stream(depth=12)
    messages[3]["ts"] = str(messages[3]["ts"])  # a quoted stamp the pattern does not fit
    lines = [json.dumps(m) for m in messages]
    lines.append(json.dumps({"topic": "other", "type": "status", "ts": START}))
    plain = tmp_path / "plain.data"
    plain.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats = archive_message_stats(plain)
    assert (stats.messages, stats.deltas) == (31, 30)
