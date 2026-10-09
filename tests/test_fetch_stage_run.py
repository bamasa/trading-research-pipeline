"""A whole stage on a fake venue: disk checked first, every plane fetched, nothing left over."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from datetime import UTC, date, datetime
from email.message import Message
from pathlib import Path

import pytest

from trading_research.data import fetch_log as fl
from trading_research.data import fetch_stage as fs
from trading_research.data import study_fetch as sf
from trading_research.data.disk import InsufficientDisk, StageBudget

DAYS = (date(2024, 12, 2), date(2024, 12, 3), date(2024, 12, 4))
SMALL = StageBudget(archives_in_flight_gb=0.001, books_gb=0.001, prints_gb=0.001)


def _ms(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp() * 1000)


def _zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def _book(day: date) -> bytes:
    def message(kind: str, ts: int, bids: list, asks: list, update: int) -> str:
        data = {"s": "TESTUSDT", "b": bids, "a": asks, "u": update, "seq": update}
        return json.dumps({"type": kind, "ts": ts, "data": data})

    bids = [[f"{100.0 - 0.1 * i:.1f}", "2"] for i in range(12)]
    asks = [[f"{100.1 + 0.1 * i:.1f}", "2"] for i in range(12)]
    lines = [message("snapshot", _ms(day), bids, asks, 1)]
    lines += [
        message("delta", _ms(day) + 100 * i, [["100.0", str(i)]], [], 1 + i) for i in range(1, 9)
    ]
    return _zip("book.data", "\n".join(lines) + "\n")


def _prints(day: date) -> bytes:
    import gzip

    rows = ["timestamp,symbol,side,size,price"]
    rows += [f"{_ms(day) / 1000 + i},TESTUSDT,Buy,0.01,100.0" for i in range(4)]
    return gzip.compress(("\n".join(rows) + "\n").encode())


def _klines(day: date) -> bytes:
    rows = [f"{_ms(day, h)},1,2,0.5,1.5,10,{_ms(day, h) + 3_599_999},0,0,0,0,0" for h in range(24)]
    return _zip("k.csv", "\n".join(rows) + "\n")


def _funding(days: tuple[date, ...]) -> bytes:
    rows = [
        {"fundingRateTimestamp": str(_ms(d, h)), "fundingRate": "0.0001"}
        for d in days
        for h in (0, 8, 16)
    ]
    return json.dumps({"retCode": 0, "result": {"list": rows[::-1]}}).encode()


class Venue:
    """Answers by URL, as the three sources would; a day not listed is a 404."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, url: str, timeout: float | None = None) -> io.BytesIO:
        self.asked.append(url)
        body = self.body(url)
        if body is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), io.BytesIO())
        response = io.BytesIO(body)
        response.headers = {"Content-Length": str(len(body))}  # type: ignore[attr-defined]
        return response

    def body(self, url: str) -> bytes | None:
        if "api.bybit.com" in url:
            return _funding(DAYS)
        if url.endswith(".CHECKSUM") or "/monthly/" in url:
            return None
        for day in DAYS[:2]:
            stamp = day.isoformat()
            if "quote-saver" in url and f"/{stamp}_" in url:
                return _book(day)
            if "public.bybit.com" in url and url.endswith(f"{stamp}.csv.gz"):
                return _prints(day)
        for day in DAYS:
            if "/daily/klines/" in url and url.endswith(f"-1h-{day.isoformat()}.zip"):
                return _klines(day)
        return None


@pytest.fixture
def venue(monkeypatch: pytest.MonkeyPatch) -> Venue:
    fake = Venue()
    monkeypatch.setattr("urllib.request.urlopen", fake)
    return fake


PLAN = fs.StagePlan(book=DAYS, prints=DAYS, bars=DAYS, funding=DAYS)


def test_a_stage_fetches_every_plane_and_resolves_every_day(tmp_path: Path, venue: Venue) -> None:
    roots = fs.StudyRoots.under(tmp_path / "data")
    log = fl.FetchLog(tmp_path / "log.json")
    present = {"ob500": frozenset(DAYS), "ob200": frozenset(), "prints": frozenset(DAYS[:2])}
    fs.run_stage(
        "TESTUSDT",
        PLAN,
        roots,
        log,
        budget=SMALL,
        book_workers=1,
        print_workers=1,
        expected_present=present,
    )
    status = {(r.plane, r.day): r.status for r in log.of()}
    for plane in (sf.BOOK, sf.PRINTS):
        assert [status[plane, d.isoformat()] for d in DAYS] == [fl.ON_DISK] * 2 + [fl.MISSING_PLANE]
    assert {status[sf.BARS, d.isoformat()] for d in DAYS} == {fl.ON_DISK}
    assert {status[sf.FUNDING, d.isoformat()] for d in DAYS} == {fl.ON_DISK}
    assert log.get(sf.BOOK, DAYS[2]).stats["note"] == fs.FOUND_PRESENT_NOTE
    assert "note" not in log.get(sf.PRINTS, DAYS[2]).stats  # not listed present
    assert not [u for u in venue.asked if "/monthly/" in u]
    assert not list(roots.archives.glob("*"))
    assert len(fs.files_on_disk("TESTUSDT", PLAN, roots)) == 2 + 2 + 3 + 3
    assert log.complete()


def test_a_stage_refuses_before_fetching_when_the_disk_is_short(
    tmp_path: Path, venue: Venue
) -> None:
    huge = StageBudget(archives_in_flight_gb=1e9, books_gb=0, prints_gb=0)
    roots = fs.StudyRoots.under(tmp_path)
    with pytest.raises(InsufficientDisk):
        fs.run_stage(
            "TESTUSDT", PLAN, roots, fl.FetchLog(tmp_path / "l.json"), budget=huge, book_workers=1
        )
    assert venue.asked == []


def test_leftovers_of_an_interrupted_run_are_cleared_first(tmp_path: Path, venue: Venue) -> None:
    roots = fs.StudyRoots.under(tmp_path)
    roots.archives.mkdir(parents=True)
    (roots.archives / "TESTUSDT-2024-12-01-ob500.zip").write_bytes(b"left over")
    plan = fs.StagePlan(book=DAYS[:1])
    fs.run_stage(
        "TESTUSDT", plan, roots, fl.FetchLog(tmp_path / "l.json"), budget=SMALL, book_workers=1
    )
    assert not list(roots.archives.glob("*"))


def test_the_head_check_table_gives_the_days_found_present(tmp_path: Path) -> None:
    table = tmp_path / "head.csv"
    table.write_text(
        "day,ob500_status,ob500_bytes,ob200_status,ob200_bytes,prints_status,prints_bytes\n"
        "2025-08-20,200,1,404,,200,1\n"
        "2025-08-21,404,,200,2,200,2\n",
        encoding="utf-8",
    )
    present = fs.head_check_present(table)
    assert present["ob500"] == {date(2025, 8, 20)}
    assert present["ob200"] == {date(2025, 8, 21)}
    assert len(present["prints"]) == 2


def test_the_registered_head_check_table_reads() -> None:
    table = (
        Path(__file__).resolve().parent.parent
        / "experiments/results/short_horizon_2025_head_check.csv"
    )
    present = fs.head_check_present(table)
    assert len(present["ob500"]) == 263 and len(present["ob200"]) == 133
    assert len(present["prints"]) == 396
    assert max(present["ob500"]) < min(present["ob200"]) == date(2025, 8, 21)
