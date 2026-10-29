"""One plane of one day: on disk, or the reason it is not, and never a guess.

Every test serves synthetic files through a fake opener; nothing touches the
network. The outcomes are what decide exclusions, so each failure class is
pinned: a 404 is missing, a failed transfer is transient, an archive that does
not decompress is corrupt, and one whose replay raises is unreadable at once.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import zipfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from email.message import Message
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_research.data import binance, bybit, bybit_trades
from trading_research.data import fetch_log as fl
from trading_research.data import study_fetch as sf

DAY = date(2024, 12, 3)
START = int(datetime(2024, 12, 3, tzinfo=UTC).timestamp() * 1000)


def book_archive(
    day: date = DAY, *, deltas: int = 20, depth: int = 12, bad_line: bool = False
) -> bytes:
    """A venue-shaped zip: a snapshot and ``deltas`` deltas 100 ms apart."""

    def message(kind: str, ts: int, bids: list, asks: list, update: int) -> dict:
        data = {"s": "BTCUSDT", "b": bids, "a": asks, "u": update, "seq": update}
        return {"topic": "orderbook.500.BTCUSDT", "type": kind, "ts": ts, "data": data}

    bids = [[f"{100.0 - 0.1 * i:.1f}", "2"] for i in range(depth)]
    asks = [[f"{100.1 + 0.1 * i:.1f}", "2"] for i in range(depth)]
    lines = [json.dumps(message("snapshot", START, bids, asks, 1))]
    lines += [
        json.dumps(message("delta", START + 100 * i, [["100.0", f"{3 + i}"]], [], 1 + i))
        for i in range(1, deltas + 1)
    ]
    if bad_line:
        lines.insert(3, '{"type": "delta", "ts": 17')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{day}_BTCUSDT_ob500.data", "\n".join(lines) + "\n")
    return buffer.getvalue()


def prints_archive() -> bytes:
    rows = ["timestamp,symbol,side,size,price,trdMatchID"]
    rows += [
        f"{START / 1000 + i},BTCUSDT,{'Buy' if i % 2 else 'Sell'},0.01,100.0,m{i}" for i in range(5)
    ]
    return gzip.compress(("\n".join(rows) + "\n").encode())


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "status", Message(), io.BytesIO(b""))


class Response(io.BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}


def serving(answer: bytes | BaseException) -> Callable[..., Any]:
    def opener(url: str, timeout: float) -> Response:
        if isinstance(answer, BaseException):
            raise answer
        return Response(answer)

    return opener


def run_book(tmp_path: Path, answer: bytes | BaseException) -> sf.DayOutcome:
    return sf.fetch_book_day(
        "BTCUSDT", DAY, tmp_path / "book", tmp_path / "archives", opener=serving(answer)
    )


def test_a_book_day_is_replayed_written_and_its_archive_deleted(tmp_path: Path) -> None:
    body = book_archive()
    result = run_book(tmp_path, body)
    assert result.outcome == fl.OK
    assert result.sha256 == sha256(body).hexdigest() and result.size == len(body)
    frame = pd.read_parquet(sf.book_path("BTCUSDT", DAY, tmp_path / "book"))
    assert len(frame) == result.stats["rows"] == 21
    assert set(frame["source"]) == {"bybit-linear-ob500"}
    assert result.stats["messages"] == 21 and result.stats["sequence_gaps"] == 0
    assert result.stats["median_interval_ms"] == 100.0
    assert result.stats["archive"] == "ob500" and result.stats["peak_rss_bytes"] > 0
    assert not list((tmp_path / "archives").iterdir())


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        (http_error(404), fl.MISSING),
        (http_error(503), fl.TRANSIENT),
        (http_error(403), fl.TRANSIENT),
        (ConnectionResetError("reset by peer"), fl.TRANSIENT),
        (TimeoutError("read timed out"), fl.TRANSIENT),
    ],
)
def test_only_a_404_makes_a_missing_book_day(
    tmp_path: Path, answer: BaseException, outcome: str
) -> None:
    result = run_book(tmp_path, answer)
    assert result.outcome == outcome
    assert not (tmp_path / "book").exists()
    assert not list((tmp_path / "archives").glob("*"))


def test_an_archive_that_does_not_decompress_is_corrupt_with_its_hash(tmp_path: Path) -> None:
    body = book_archive()[:-200] + b"\x00" * 200
    result = run_book(tmp_path, body)
    assert result.outcome == fl.CORRUPT
    assert result.sha256 == sha256(body).hexdigest()
    assert not list((tmp_path / "archives").glob("*"))


def test_a_line_the_decoder_cannot_parse_makes_the_day_unreadable(tmp_path: Path) -> None:
    result = run_book(tmp_path, book_archive(bad_line=True))
    assert result.outcome == fl.REPLAY_FAILED
    record = fl.DayRecord(plane=sf.BOOK, day=DAY.isoformat())
    assert record.add(result.outcome, now=result.finished_at, sha256=result.sha256) == fl.UNREADABLE


def test_a_book_without_ten_levels_is_a_replay_failure(tmp_path: Path) -> None:
    result = run_book(tmp_path, book_archive(depth=4))
    assert result.outcome == fl.REPLAY_FAILED
    assert "no usable book rows" in result.detail


def test_a_day_after_the_change_asks_for_the_ob200_archive(tmp_path: Path) -> None:
    asked: list[str] = []

    def opener(url: str, timeout: float) -> Response:
        asked.append(url)
        return Response(book_archive())

    day = date(2026, 1, 6)
    result = sf.fetch_book_day("BTCUSDT", day, tmp_path, tmp_path / "a", opener=opener)
    assert asked == [bybit.archive_url("BTCUSDT", day, "ob200")]
    assert result.stats["archive"] == "ob200"


def test_a_print_day_is_parsed_and_written(tmp_path: Path) -> None:
    result = sf.fetch_prints_day(
        "BTCUSDT", DAY, tmp_path / "trades", tmp_path / "a", opener=serving(prints_archive())
    )
    assert result.outcome == fl.OK and result.stats == {"rows": 5}
    prints = bybit_trades.load_day("BTCUSDT", DAY, tmp_path / "trades")
    assert sorted(prints["aggressor"].unique()) == [-1, 1]
    assert not list((tmp_path / "a").glob("*"))


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        (b"not gzip at all", fl.CORRUPT),
        (gzip.compress(b"a,b\n1,2\n"), fl.REPLAY_FAILED),
        (http_error(404), fl.MISSING),
        (http_error(500), fl.TRANSIENT),
    ],
)
def test_print_failures_are_classified(tmp_path: Path, answer: Any, outcome: str) -> None:
    result = sf.fetch_prints_day(
        "BTCUSDT", DAY, tmp_path / "trades", tmp_path / "a", opener=serving(answer)
    )
    assert result.outcome == outcome
    assert not (tmp_path / "trades").exists()


def _klines(day: date) -> bytes:
    start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
    rows = [
        f"{start + h * 3_600_000},1,2,0.5,1.5,10,{start + (h + 1) * 3_600_000 - 1},0,0,0,0,0"
        for h in range(24)
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("k.csv", "\n".join(rows) + "\n")
    return buffer.getvalue()


def _binance_failure(code: int) -> binance.BinanceArchiveError:
    try:
        raise http_error(code)
    except urllib.error.HTTPError as exc:
        try:
            raise binance.BinanceArchiveError(f"HTTP {code}") from exc
        except binance.BinanceArchiveError as wrapped:
            return wrapped


@pytest.mark.parametrize(
    ("code", "outcome"), [(None, fl.OK), (404, fl.MISSING), (502, fl.TRANSIENT)]
)
def test_bars_come_from_the_day_alone_and_only_a_404_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int | None, outcome: str
) -> None:
    asked: list[str] = []

    def fetch(url: str, *, verify: bool = True) -> bytes:
        asked.append(url)
        if code is not None:
            raise _binance_failure(code)
        return _klines(DAY)

    monkeypatch.setattr(binance, "fetch_archive", fetch)
    result = sf.fetch_bars_day("BTCUSDT", DAY, tmp_path)
    assert result.outcome == outcome
    assert all("/daily/" in url for url in asked) and len(asked) == 1
    if code is None:
        assert result.stats == {"rows": 24}
        assert result.sha256 == sha256(_klines(DAY)).hexdigest()
    assert not list(tmp_path.rglob("*.zip"))


def _funding_payload(days: list[date]) -> bytes:
    rows = [
        {
            "fundingRateTimestamp": str(
                int(datetime(d.year, d.month, d.day, h, tzinfo=UTC).timestamp() * 1000)
            ),
            "fundingRate": "0.0001",
        }
        for d in days
        for h in (0, 8, 16)
    ]
    rows.reverse()  # newest first, as the venue answers
    return json.dumps({"retCode": 0, "result": {"list": rows}}).encode()


def test_funding_days_without_a_settlement_are_missing(tmp_path: Path) -> None:
    days = [date(2024, 12, 3), date(2024, 12, 4), date(2024, 12, 5)]
    opener = serving(_funding_payload([days[0], days[2]]))
    results = sf.fetch_funding_span("BTCUSDT", days, tmp_path, opener=opener)
    assert [r.outcome for r in results] == [fl.OK, fl.MISSING, fl.OK]
    assert results[0].stats == {"rows": 3} and results[0].sha256 is not None


@pytest.mark.parametrize(("code", "outcome"), [(404, fl.MISSING), (500, fl.TRANSIENT)])
def test_a_failed_funding_request_applies_to_every_day(
    tmp_path: Path, code: int, outcome: str
) -> None:
    days = [date(2024, 12, 3), date(2024, 12, 4)]
    results = sf.fetch_funding_span("BTCUSDT", days, tmp_path, opener=serving(http_error(code)))
    assert [r.outcome for r in results] == [outcome, outcome]
    assert sf.fetch_funding_span("BTCUSDT", [], tmp_path) == []
