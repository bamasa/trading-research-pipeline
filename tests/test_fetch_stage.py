"""A stage's plane runs until every day is on disk or ended by a terminal rule.

The loop is driven here by a fake clock, so two days of retries take no time:
the outcomes are scripted per day, and the test reads back what the log kept.
"""

from __future__ import annotations

import functools
import io
import json
import operator
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading_research.data import bybit
from trading_research.data import fetch_log as fl
from trading_research.data import fetch_stage as fs
from trading_research.data import study_fetch as sf
from trading_research.data.failures import DirectoryOpener

T0 = datetime(2026, 10, 9, tzinfo=UTC)
DAYS = [date(2024, 12, d) for d in range(1, 6)]


class Clock:
    def __init__(self) -> None:
        self.now = T0
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


class Script:
    """Per-day outcomes in order; the last one repeats."""

    def __init__(self, clock: Clock, plan: dict[date, list[str]]) -> None:
        self.clock, self.plan, self.calls = clock, plan, []

    def __call__(self, day: date) -> sf.DayOutcome:
        self.calls.append(day)
        outcomes = self.plan[day]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        sha = "a" * 64 if outcome in (fl.CORRUPT, fl.OK) else None
        stats = {"rows": 7} if outcome == fl.OK else {}
        return sf.DayOutcome(sf.BOOK, day, outcome, self.clock(), outcome, sha, 1, stats)


def test_replay_workers_follow_the_measured_peak() -> None:
    assert fs.replay_workers(5_500_000_000) == 4
    assert fs.replay_workers(7_000_000_000) == 3
    assert fs.replay_workers(13_000_000_000) == 1
    assert fs.replay_workers(30_000_000_000) == 1
    assert fs.replay_workers(1_000_000_000, max_workers=2) == 2
    with pytest.raises(ValueError, match="not a measurement"):
        fs.replay_workers(0)


def test_peak_memory_is_measured_in_a_fresh_process() -> None:
    # b"x" * n writes every byte, so the pages are resident on any platform.
    small = fs.measure_peak_rss(operator.mul, b"x", 1_000)
    large = fs.measure_peak_rss(operator.mul, b"x", 300_000_000)
    assert large - small > 250_000_000


def test_every_day_ends_on_disk_or_under_a_terminal_rule(tmp_path: Path) -> None:
    clock = Clock()
    script = Script(
        clock,
        {
            DAYS[0]: [fl.OK],
            DAYS[1]: [fl.TRANSIENT, fl.TRANSIENT, fl.OK],
            DAYS[2]: [fl.MISSING],
            DAYS[3]: [fl.TRANSIENT],  # a permanent outage
            DAYS[4]: [fl.CORRUPT],  # the same bytes every time
        },
    )
    seen: list[str] = []
    log = fs.run_plane(
        sf.BOOK,
        DAYS,
        script,
        fl.FetchLog(tmp_path / "log.json"),
        expected_present=[DAYS[2]],
        clock=clock,
        sleep=clock.sleep,
        on_record=lambda r: seen.append(r.status),
    )
    statuses = [log.get(sf.BOOK, d).status for d in DAYS]
    assert statuses == [fl.ON_DISK, fl.ON_DISK, fl.MISSING_PLANE, fl.UNFETCHABLE, fl.UNREADABLE]
    assert log.get(sf.BOOK, DAYS[0]).stats == {"rows": 7}
    assert log.get(sf.BOOK, DAYS[2]).stats == {"note": fs.FOUND_PRESENT_NOTE}
    assert clock.now - T0 >= fl.UNFETCHABLE_SPAN
    assert clock.slept[0] == fl.backoff_s(1)
    assert len(seen) == len(script.calls)
    assert fl.FetchLog(tmp_path / "log.json").complete(sf.BOOK)


def test_a_resumed_stage_fetches_only_what_is_pending(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "log.json"
    first = Script(clock, {d: [fl.OK] for d in DAYS[:2]})
    fs.run_plane(sf.BOOK, DAYS[:2], first, fl.FetchLog(path), clock=clock, sleep=clock.sleep)
    again = Script(clock, {d: [fl.OK] for d in DAYS})
    fs.run_plane(sf.BOOK, DAYS, again, fl.FetchLog(path), clock=clock, sleep=clock.sleep)
    assert again.calls == DAYS[2:]


def test_the_book_replay_is_capped_at_four_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most 4"):
        fs.run_plane(sf.BOOK, DAYS, Script(Clock(), {}), fl.FetchLog(tmp_path / "l"), workers=5)


def _book_archive(path: Path, day: date) -> None:
    start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)

    def message(kind: str, ts: int, bids: list, asks: list, update: int) -> str:
        data = {"s": "BTCUSDT", "b": bids, "a": asks, "u": update, "seq": update}
        return json.dumps({"topic": "orderbook.500.BTCUSDT", "type": kind, "ts": ts, "data": data})

    bids = [[f"{100.0 - 0.1 * i:.1f}", "2"] for i in range(12)]
    asks = [[f"{100.1 + 0.1 * i:.1f}", "2"] for i in range(12)]
    lines = [message("snapshot", start, bids, asks, 1)]
    lines += [
        message("delta", start + 100 * i, [["100.0", str(i)]], [], 1 + i) for i in range(1, 50)
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("book.data", "\n".join(lines) + "\n")
    path.write_bytes(buffer.getvalue())


def test_four_workers_replay_real_archives_and_report_their_peaks(tmp_path: Path) -> None:
    """The book plane on four processes, archives served from a directory."""
    served = tmp_path / "served"
    served.mkdir()
    for day in DAYS[:4]:  # the fifth day answers 404
        _book_archive(served / bybit.archive_url("BTCUSDT", day).rsplit("/", 1)[-1], day)
    fetch = functools.partial(
        sf.fetch_book_day,
        "BTCUSDT",
        root=tmp_path / "book",
        archive_dir=tmp_path / "archives",
        opener=DirectoryOpener(served),
    )
    log = fs.run_plane(sf.BOOK, DAYS, fetch, fl.FetchLog(tmp_path / "log.json"), workers=4)
    assert [log.get(sf.BOOK, d).status for d in DAYS] == [fl.ON_DISK] * 4 + [fl.MISSING_PLANE]
    peaks = [log.get(sf.BOOK, d).stats["peak_rss_bytes"] for d in DAYS[:4]]
    assert min(peaks) > 0
    assert len(pd.read_parquet(sf.book_path("BTCUSDT", DAYS[0], tmp_path / "book"))) == 50
    assert not list((tmp_path / "archives").glob("*"))  # no archive outlives its replay


def test_the_replay_peak_of_one_archive_is_measured(tmp_path: Path) -> None:
    archive = tmp_path / "day.zip"
    _book_archive(archive, DAYS[0])
    assert fs.replay_peak_rss(archive, symbol="BTCUSDT") > 10_000_000


def test_funding_retries_the_span_until_every_day_is_resolved(tmp_path: Path) -> None:
    clock = Clock()
    answers = [fl.TRANSIENT, fl.TRANSIENT, "answered"]

    def span(days: list[date]) -> list[sf.DayOutcome]:
        answer = answers.pop(0)
        out = []
        for day in days:
            outcome = answer if answer != "answered" else (fl.OK if day != DAYS[1] else fl.MISSING)
            out.append(sf.DayOutcome(sf.FUNDING, day, outcome, clock(), sha256="b" * 64))
        return out

    log = fs.run_funding(
        DAYS[:3], span, fl.FetchLog(tmp_path / "f.json"), clock=clock, sleep=clock.sleep
    )
    assert [log.get(sf.FUNDING, d).status for d in DAYS[:3]] == [
        fl.ON_DISK,
        fl.MISSING_PLANE,
        fl.ON_DISK,
    ]
    assert clock.slept == [fl.backoff_s(1), fl.backoff_s(2)]
