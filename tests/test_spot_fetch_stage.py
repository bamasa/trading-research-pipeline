"""A spot stage's fetch: access first, the disk check with its margin, retries to a terminal rule."""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_research.data import fetch_log as fl
from trading_research.data.disk import InsufficientDisk
from trading_research.data.fetch_stage import FOUND_PRESENT_NOTE
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import archives as ar
from trading_research.spot import fetch
from trading_research.spot import fetch_stage as st
from trading_research.spot import synthetic as syn

DAY, MISSING = date(2026, 3, 3), date(2026, 3, 4)
MARKET = syn.SpotMarket(frozenset({DAY}), step_ms=600_000)
ALL = dict.fromkeys(st.ORDER, (DAY, MISSING))


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


class Guard:
    def __init__(self, days: Iterable[date]) -> None:
        self.days = frozenset(days)

    def require(
        self, instrument: str, days: Iterable[date], plane: str, *, what: str = "read"
    ) -> None:
        if any(day not in self.days for day in days):
            raise HeldOutLocked(f"cannot {what} {instrument} {plane}")


class Clock:
    """A clock that moves only when the stage sleeps; the fetchers read it too."""

    def __init__(self) -> None:
        self.now, self.slept = datetime(2026, 10, 25, tzinfo=UTC), []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


class _Body(io.BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}


class Failing:
    """The venue, with every URL in ``fail`` failing its first ``times`` requests."""

    def __init__(self, fail: set[str], times: int) -> None:
        self.venue, self.fail, self.left = syn.SpotVenue(MARKET), fail, dict.fromkeys(fail, times)

    def __call__(self, url: str, timeout: float | None = None) -> Any:
        if url in self.fail and self.left[url] > 0:
            self.left[url] -= 1
            raise urllib.error.URLError("connection reset")
        return self.venue(url, timeout)


def _stage(
    tmp_path: Path, opener: Any, clock: Clock, *, free: float = 40.0, **extra: Any
) -> fl.FetchLog:
    log = fl.FetchLog(tmp_path / "log.json")
    return st.run_stage(
        0, extra.pop("days", ALL), st.SpotRoots.under(tmp_path / "spot"), log,
        access=extra.pop("access", Guard((DAY, MISSING))), disk_record=tmp_path / "disk.json",
        opener=opener, clock=clock, sleep=clock.sleep, measure=lambda _: free, **extra,
    )  # fmt: skip


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(fetch, "now", fake)
    return fake


def test_a_stage_fetches_every_day_and_a_404_on_a_listed_day_is_noted(
    tmp_path: Path, clock: Clock
) -> None:
    present = {name: frozenset({DAY, MISSING}) for name in st.ORDER}
    log = _stage(tmp_path, syn.SpotVenue(MARKET), clock, expected_present=present)
    for name in st.ORDER:
        assert log.get(name, DAY).status == fl.ON_DISK
        missing = log.get(name, MISSING)
        assert missing.status == fl.MISSING_PLANE and missing.stats["note"] == FOUND_PRESENT_NOTE
    assert not any((tmp_path / "spot" / "_archives_in_flight").iterdir())
    assert st.pending(ALL, log) == {name: [] for name in st.ORDER}
    assert len(st.files_on_disk(ALL, tmp_path / "spot")) == 3


def test_a_failed_transfer_is_retried_with_backoff_until_on_disk(
    tmp_path: Path, clock: Clock
) -> None:
    opener = Failing({ar.BYBIT_BOOK.url(DAY)}, times=2)
    log = _stage(tmp_path, opener, clock, days={"bybit_book": (DAY,)})
    record = log.get("bybit_book", DAY)
    assert [a.outcome for a in record.attempts] == [fl.TRANSIENT, fl.TRANSIENT, fl.OK]
    assert clock.slept == [60.0, 180.0]


def test_a_day_that_never_arrives_ends_unfetchable(tmp_path: Path, clock: Clock) -> None:
    opener = Failing({ar.BINANCE_PRINTS.url(DAY)}, times=10**6)
    log = _stage(tmp_path, opener, clock, days={"binance_prints": (DAY,)})
    record = log.get("binance_prints", DAY)
    assert record.status == fl.UNFETCHABLE and len(record.attempts) >= fl.UNFETCHABLE_ATTEMPTS
    assert record.attempts[-1].when - record.attempts[0].when >= fl.UNFETCHABLE_SPAN


def test_the_same_bad_bytes_twice_make_a_day_unreadable(tmp_path: Path, clock: Clock) -> None:
    class Garbled(Failing):
        def __call__(self, url: str, timeout: float | None = None) -> Any:
            if url == ar.BYBIT_PRINTS.url(DAY):
                return _Body(b"not gzip")
            return self.venue(url, timeout)

    log = _stage(tmp_path, Garbled(set(), 0), clock, days={"bybit_prints": (DAY,)})
    assert log.get("bybit_prints", DAY).status == fl.UNREADABLE


def test_the_stage_waits_for_disk_above_its_budget_and_records_each_measurement(
    tmp_path: Path, clock: Clock
) -> None:
    readings = iter([29.0, 29.5])
    venue = syn.SpotVenue(MARKET)
    log = fl.FetchLog(tmp_path / "log.json")
    st.run_stage(1, {"bybit_prints": (DAY,)}, st.SpotRoots.under(tmp_path / "spot"), log,
                 access=Guard((DAY,)), disk_record=tmp_path / "disk.json", opener=venue,
                 clock=clock, sleep=clock.sleep, measure=lambda _: next(readings), poll_s=900)  # fmt: skip
    kept = json.loads((tmp_path / "disk.json").read_text())
    assert [(k["stage"], k["free_gb"], k["passed"]) for k in kept] == [
        (1, 29.0, False),
        (1, 29.5, True),
    ]
    assert kept[0]["budget_gb"] == 19.0 and kept[0]["margin_gb"] == 10.0 and clock.slept == [900]


def test_without_disk_or_access_nothing_is_asked(tmp_path: Path, clock: Clock) -> None:
    venue = syn.SpotVenue(MARKET)
    with pytest.raises(InsufficientDisk):
        _stage(tmp_path, venue, clock, free=20.0, max_disk_checks=1)
    with pytest.raises(HeldOutLocked):
        _stage(tmp_path / "x", venue, clock, access=Guard((DAY,)))
    assert venue.asked == [] and not (tmp_path / "x" / "disk.json").exists()
    with pytest.raises(ValueError, match="book workers"):
        _stage(tmp_path, venue, clock, book_workers=5)
    with pytest.raises(ValueError, match="unknown planes"):
        _stage(tmp_path, venue, clock, days={"book": (DAY,)})


def test_archives_left_in_flight_are_deleted_and_a_resumed_stage_asks_nothing_again(
    tmp_path: Path, clock: Clock
) -> None:
    left = tmp_path / "spot" / "_archives_in_flight" / "bybit_book-2026-03-01.zip"
    left.parent.mkdir(parents=True)
    left.write_bytes(b"half")
    _stage(tmp_path, syn.SpotVenue(MARKET), clock)
    assert not left.exists()
    again = syn.SpotVenue(MARKET)
    _stage(tmp_path, again, clock)
    assert again.asked == []
