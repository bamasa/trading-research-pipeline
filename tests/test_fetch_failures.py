"""Failed downloads: only a 404 is a missing plane, and two rules end the retrying.

Each rule is a statement about which days a study keeps. A failure taken for a
missing day would drop busy days first; a retry that never ends would hold a
stage open forever; a terminal rule that fired early would exclude a day the
venue does have.
"""

from __future__ import annotations

import io
import urllib.error
from datetime import UTC, date, datetime, timedelta
from email.message import Message
from pathlib import Path

import pytest

from trading_research.data import fetch_log as fl
from trading_research.data.failures import (
    IncompleteDownload,
    download_to,
    http_status,
    is_missing_plane,
)

T0 = datetime(2026, 10, 9, tzinfo=UTC)


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "status", Message(), io.BytesIO(b""))


class _Response(io.BytesIO):
    def __init__(self, body: bytes, length: int | None = None) -> None:
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}


def test_only_a_404_is_a_missing_plane() -> None:
    def wrapped(code: int) -> RuntimeError:
        try:
            raise RuntimeError("download failed") from _http(code)
        except RuntimeError as exc:
            return exc

    assert is_missing_plane(wrapped(404))
    assert http_status(wrapped(503)) == 503
    assert not is_missing_plane(wrapped(503))
    assert not is_missing_plane(wrapped(403))
    assert not is_missing_plane(TimeoutError("timed out"))
    assert not is_missing_plane(ConnectionResetError())
    assert http_status(OSError("reset")) is None


def test_a_download_is_hashed_and_renamed_only_when_whole(tmp_path: Path) -> None:
    body = b"x" * 3_000_000

    def opener(url: str, timeout: float) -> _Response:
        return _Response(body, len(body))

    result = download_to("https://x/a.zip", tmp_path / "a.zip", opener=opener)
    assert result.path.read_bytes() == body
    assert result.size == len(body) and result.complete
    assert len(result.sha256) == 64
    assert not list(tmp_path.glob("*.part"))


def test_a_short_body_is_a_transfer_failure_and_leaves_nothing(tmp_path: Path) -> None:
    def opener(url: str, timeout: float) -> _Response:
        return _Response(b"abc", 10)

    with pytest.raises(IncompleteDownload):
        download_to("https://x/a.zip", tmp_path / "a.zip", opener=opener)
    assert list(tmp_path.iterdir()) == []


def test_a_dropped_connection_leaves_no_partial_file(tmp_path: Path) -> None:
    class _Dropping(_Response):
        def read(self, size: int | None = -1) -> bytes:
            raise ConnectionResetError("reset by peer")

    def opener(url: str, timeout: float) -> _Response:
        return _Dropping(b"", None)

    with pytest.raises(ConnectionResetError):
        download_to("https://x/a.zip", tmp_path / "a.zip", opener=opener)
    assert list(tmp_path.iterdir()) == []


def test_backoff_grows_and_is_capped() -> None:
    delays = [fl.backoff_s(n) for n in range(1, 12)]
    assert delays[0] == 60.0
    assert delays == sorted(delays)
    assert max(delays) == fl.MAX_DELAY_S


def test_a_404_is_recorded_at_once_and_never_retried() -> None:
    record = fl.DayRecord(plane="book", day="2025-01-01")
    assert record.add(fl.MISSING, now=T0, detail="HTTP 404") == fl.MISSING_PLANE
    assert record.resolved


def test_ten_failures_inside_two_days_do_not_end_the_retrying() -> None:
    record = fl.DayRecord(plane="book", day="2025-01-01")
    for i in range(12):
        status = record.add(fl.TRANSIENT, now=T0 + timedelta(hours=i), detail="timeout")
    assert status == fl.PENDING
    assert not record.resolved


def test_ten_failures_over_two_days_make_a_day_unfetchable() -> None:
    record = fl.DayRecord(plane="book", day="2025-01-01")
    statuses = [
        record.add(fl.TRANSIENT, now=T0 + timedelta(hours=6 * i), detail="reset") for i in range(10)
    ]
    assert statuses[8] == fl.PENDING  # nine attempts over 48 hours: one short
    assert statuses[9] == fl.UNFETCHABLE  # the tenth, 54 hours after the first


def test_the_backoff_schedule_reaches_the_unfetchable_rule() -> None:
    """Retrying on schedule, a permanent outage ends in the terminal rule."""
    record = fl.DayRecord(plane="prints", day="2025-01-01")
    now = T0
    while not record.resolved:
        record.add(fl.TRANSIENT, now=now, detail="timeout")
        now = record.due(now)
    assert record.status == fl.UNFETCHABLE
    assert len(record.failures()) >= fl.UNFETCHABLE_ATTEMPTS
    span = record.failures()[-1].when - record.failures()[0].when
    assert span >= fl.UNFETCHABLE_SPAN


def test_one_corrupt_archive_is_retried_and_two_identical_ones_are_unreadable() -> None:
    record = fl.DayRecord(plane="book", day="2025-01-01")
    assert record.add(fl.CORRUPT, now=T0, sha256="a" * 64) == fl.PENDING
    assert record.add(fl.CORRUPT, now=T0, sha256="b" * 64) == fl.PENDING
    assert record.add(fl.CORRUPT, now=T0, sha256="a" * 64) == fl.UNREADABLE


def test_a_replay_that_raises_on_a_complete_archive_is_unreadable() -> None:
    record = fl.DayRecord(plane="book", day="2025-01-01")
    assert record.add(fl.REPLAY_FAILED, now=T0, sha256="c" * 64) == fl.UNREADABLE


def test_success_records_the_file_and_its_download_time() -> None:
    record = fl.DayRecord(plane="book", day="2025-01-01")
    record.add(fl.TRANSIENT, now=T0, detail="reset")
    assert record.due(T0) == T0 + timedelta(seconds=60)
    record.add(fl.OK, now=T0 + timedelta(minutes=2), sha256="d" * 64, size=12)
    assert record.status == fl.ON_DISK
    assert (record.sha256, record.size) == ("d" * 64, 12)
    assert record.downloaded_at == (T0 + timedelta(minutes=2)).isoformat()


def test_an_unknown_outcome_is_refused_before_it_is_recorded() -> None:
    record = fl.DayRecord(plane="book", day="2025-01-01")
    with pytest.raises(ValueError, match="unknown attempt outcome"):
        record.add("lost", now=T0)
    assert record.attempts == []


def test_the_log_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "fetch.json"
    log = fl.FetchLog(path)
    log.get("book", date(2025, 1, 1)).add(fl.TRANSIENT, now=T0, detail="reset")
    log.get("book", date(2025, 1, 2)).add(fl.MISSING, now=T0)
    log.get("prints", date(2025, 1, 1)).add(fl.OK, now=T0, sha256="e" * 64, size=5)
    log.save()
    again = fl.FetchLog(path)
    assert len(again.get("book", date(2025, 1, 1)).attempts) == 1
    assert again.counts("book") == {fl.PENDING: 1, fl.MISSING_PLANE: 1}
    assert again.excluded() == {"book/2025-01-02": fl.MISSING_PLANE}
    assert again.complete("prints") and not again.complete("book")
    assert not list(tmp_path.glob("*.part"))
