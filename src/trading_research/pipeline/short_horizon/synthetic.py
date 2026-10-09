"""A synthetic venue for the dry run: every request answered locally, nothing from the network.

The dry run fetches through the study's own fetch stage, so the venue answers
the requests that code makes, by URL, in the formats the venue publishes:

* book archives (``ob500`` before 2025-08-21, ``ob200`` from it): a zip of
  JSON lines, one snapshot and then deltas every few seconds over the whole UTC
  day, which :func:`trading_research.data.bybit.reconstruct` replays;
* print archives: the gzipped CSV of the day's prints at the touch;
* Binance hourly klines: one daily zip per day, for every day, so the
  detector's stream has no hole;
* funding: the REST history, a settlement every eight hours;
* the instrument specification (tick and lot).

Books and prints exist only on the days of the dry run's calendar; any other
day answers 404, a missing plane. The synthetic instrument trades near 1,000
USDT on a 0.1 tick, so one tick is about a basis point and a hold of tens of
seconds moves the walked label across the 5.5 bp fee. Everything is drawn from
generators seeded by the day, so the venue answers the same bytes every time.

:func:`serving` patches ``urllib.request.urlopen`` for the duration of a dry
run, so a request the venue does not recognise fails rather than reaching the
network.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.message import Message
from unittest import mock
from urllib.parse import parse_qs, urlparse

import numpy as np

from trading_research.data import bybit

SYMBOL = "BTCUSDT"
TICK = 0.1
LOT = 0.001
LEVELS = 12
#: The book's price near the start, in ticks (about 1,000 USDT).
START_TICKS = 10_000
MS_PER_DAY = 86_400_000
STREAM_START = date(2024, 12, 1)


def _ms(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp() * 1000)


def _zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def _price(ticks: int) -> str:
    return f"{ticks * TICK:.1f}"


@dataclass(frozen=True)
class DayPath:
    """One day's market: message stamps (ms), each message's best bid in ticks."""

    stamps_ms: np.ndarray
    bid0: np.ndarray
    rng_seed: int


@dataclass(frozen=True)
class SyntheticMarket:
    """The days a book and prints exist on, and how often the book updates."""

    days: frozenset[date]
    step_ms: int = 4_000
    seed: int = 2025

    def path(self, day: date) -> DayPath:
        """A random walk of the best bid with a fat tail, one message per step."""
        seed = self.seed * 100_000 + day.toordinal()
        rng = np.random.default_rng(seed)
        n = MS_PER_DAY // self.step_ms
        jitter = rng.integers(1, 50, size=n)
        stamps = _ms(day) + np.arange(n, dtype=np.int64) * self.step_ms + jitter
        moves = np.rint(rng.standard_t(4, size=n) * 1.5).astype(np.int64)
        moves[0] = 0
        level = START_TICKS + int(rng.integers(-500, 500))
        return DayPath(stamps, level + np.cumsum(moves), seed)

    def book_lines(self, day: date, *, hours: int | None = None) -> list[str]:
        """The archive's JSON lines: a snapshot, then deltas of what changed."""
        p = self.path(day)
        rng = np.random.default_rng(p.rng_seed + 1)
        end = len(p.stamps_ms) if hours is None else hours * 3_600_000 // self.step_ms

        def sizes(k: int) -> np.ndarray:
            return np.round(rng.exponential(0.6, size=k) + 0.02, 3)

        bids = {int(p.bid0[0]) - i: float(s) for i, s in enumerate(sizes(LEVELS))}
        asks = {int(p.bid0[0]) + 1 + i: float(s) for i, s in enumerate(sizes(LEVELS))}

        def side(levels: Mapping[int, float]) -> list[list[str]]:
            return [[_price(t), f"{s:.3f}"] for t, s in levels.items()]

        def message(kind: str, stamp: int, b: list[list[str]], a: list[list[str]], u: int) -> str:
            data = {"s": SYMBOL, "b": b, "a": a, "u": u, "seq": u}
            return json.dumps({"type": kind, "ts": int(stamp), "data": data})

        lines = [message("snapshot", p.stamps_ms[0], side(bids), side(asks), 1)]
        for i in range(1, end):
            top = int(p.bid0[i])
            new_bids = {top - k: bids.get(top - k, 0.0) for k in range(LEVELS)}
            new_asks = {top + 1 + k: asks.get(top + 1 + k, 0.0) for k in range(LEVELS)}
            fresh = sizes(4)
            for k, key in enumerate((top, top - 1)):
                new_bids[key] = float(fresh[k])
            for k, key in enumerate((top + 1, top + 2)):
                new_asks[key] = float(fresh[2 + k])
            for new in (new_bids, new_asks):
                for key in new:
                    if new[key] == 0.0:
                        new[key] = float(sizes(1)[0])
            delta_b = [[_price(t), "0"] for t in bids if t not in new_bids]
            delta_b += [[_price(t), f"{s:.3f}"] for t, s in new_bids.items() if bids.get(t) != s]
            delta_a = [[_price(t), "0"] for t in asks if t not in new_asks]
            delta_a += [[_price(t), f"{s:.3f}"] for t, s in new_asks.items() if asks.get(t) != s]
            bids, asks = new_bids, new_asks
            lines.append(message("delta", p.stamps_ms[i], delta_b, delta_a, i + 1))
        return lines

    def book_archive(self, day: date, *, hours: int | None = None) -> bytes:
        return _zip(
            f"{day.isoformat()}_{SYMBOL}.data", "\n".join(self.book_lines(day, hours=hours))
        )

    def prints_archive(self, day: date) -> bytes:
        """Prints between consecutive messages, at the touch the earlier one shows."""
        p = self.path(day)
        rng = np.random.default_rng(p.rng_seed + 2)
        counts = rng.poisson(0.6, size=len(p.stamps_ms) - 1)
        at = np.repeat(np.arange(len(counts)), counts)
        gap = p.stamps_ms[at + 1] - p.stamps_ms[at]
        times = p.stamps_ms[at] + 1 + np.floor(rng.random(len(at)) * (gap - 1)).astype(np.int64)
        side = rng.choice(np.array([-1, 1]), size=len(at))
        price = np.where(side > 0, p.bid0[at] + 1, p.bid0[at])
        size = np.round(rng.exponential(0.05, size=len(at)) + 0.001, 3)
        rows = ["timestamp,symbol,side,size,price"]
        rows += [
            f"{t / 1000:.3f},{SYMBOL},{'Buy' if s > 0 else 'Sell'},{q:.3f},{_price(int(x))}"
            for t, s, q, x in zip(times, side, size, price, strict=True)
        ]
        return gzip.compress(("\n".join(rows) + "\n").encode())

    def hourly_closes(self) -> tuple[int, np.ndarray]:
        """Every hourly close from the stream's start to 2026-01-31: (first open ms, closes)."""
        first = _ms(STREAM_START)
        hours = (date(2026, 2, 1) - STREAM_START).days * 24
        rng = np.random.default_rng(self.seed)
        closes = 1_000.0 * np.exp(np.cumsum(rng.standard_t(5, size=hours) * 0.004))
        return first, closes

    def klines_archive(self, day: date) -> bytes:
        first, closes = self.hourly_closes()
        start = (_ms(day) - first) // 3_600_000
        rows = []
        for h in range(24):
            close = closes[start + h]
            open_ = closes[start + h - 1] if start + h > 0 else close
            high, low = max(open_, close) * 1.001, min(open_, close) * 0.999
            t = _ms(day, h)
            rows.append(
                f"{t},{open_:.2f},{high:.2f},{low:.2f},{close:.2f},10,{t + 3_599_999},"
                f"{10 * close:.2f},100,5,{5 * close:.2f},0"
            )
        return _zip(f"{SYMBOL}-1h-{day.isoformat()}.csv", "\n".join(rows) + "\n")

    def funding_page(self, start_ms: int, end_ms: int, limit: int) -> bytes:
        """Settlements every eight hours in ``[start_ms, end_ms]``, newest first."""
        first, last = _ms(date(2024, 1, 1)), _ms(date(2026, 2, 1))
        stamps = np.arange(first, last, 8 * 3_600_000, dtype=np.int64)
        inside = stamps[(stamps >= start_ms) & (stamps <= end_ms)][::-1][:limit]
        rows = [{"fundingRateTimestamp": str(t), "fundingRate": "0.0001"} for t in inside]
        return json.dumps({"retCode": 0, "result": {"list": rows}}).encode()

    def instrument_info(self) -> bytes:
        row = {
            "symbol": SYMBOL,
            "priceFilter": {"tickSize": "0.1"},
            "lotSizeFilter": {"qtyStep": "0.001"},
        }
        return json.dumps({"retCode": 0, "result": {"list": [row]}}).encode()

    def present(self) -> dict[str, frozenset[date]]:
        """The existence check's answer, per archive, as ``fetch_stage.head_check_present``."""
        ob500 = frozenset(d for d in self.days if bybit.archive_kind(d) == "ob500")
        ob200 = frozenset(d for d in self.days if bybit.archive_kind(d) == "ob200")
        return {"ob500": ob500, "ob200": ob200, "prints": frozenset(self.days)}


class _Response(io.BytesIO):
    """What ``urlopen`` returns, with the length header the downloaders read."""

    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}
        self.status = 200


def _not_found(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 404, "Not Found", Message(), io.BytesIO())


@dataclass
class Venue:
    """Answers every request by URL; a request it does not recognise is a 404,
    kept in :attr:`unknown` so a dry run can refuse to call itself complete."""

    market: SyntheticMarket
    reader_test_days: frozenset[date] = frozenset()

    def __post_init__(self) -> None:
        self.asked: list[str] = []
        self.unknown: list[str] = []

    def __call__(self, url: str, timeout: float | None = None) -> _Response:
        del timeout  # answered at once
        self.asked.append(url)
        body = self.body(url)
        if body is None:
            raise _not_found(url)
        return _Response(body)

    def _day(self, text: str) -> date | None:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None

    def body(self, url: str) -> bytes | None:
        parsed = urlparse(url)
        path = parsed.path
        if parsed.netloc == "quote-saver.bycsi.com":
            day = self._day(path.rsplit("/", 1)[-1][:10])
            if day is None or not path.endswith(f"_{bybit.archive_kind(day)}.data.zip"):
                return None
            if day in self.reader_test_days:
                return self.market.book_archive(day, hours=2)
            return self.market.book_archive(day) if day in self.market.days else None
        if parsed.netloc == "public.bybit.com":
            day = self._day(path.rsplit(SYMBOL, 1)[-1][:10])
            return self.market.prints_archive(day) if day in self.market.days else None
        if parsed.netloc == "data.binance.vision":
            if "/daily/klines/" not in path or not path.endswith(".zip"):
                return None  # monthly archives and checksums do not exist here
            day = self._day(path[-14:-4])
            return None if day is None else self.market.klines_archive(day)
        if parsed.netloc == "api.bybit.com":
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if path.endswith("/funding/history"):
                start, end = int(query["startTime"]), int(query["endTime"])
                return self.market.funding_page(start, end, int(query.get("limit", 200)))
            if path.endswith("/instruments-info"):
                return self.market.instrument_info()
        self.unknown.append(url)
        return None


def _no_network(*_: object, **__: object) -> None:
    raise OSError("the dry run has no network: every request goes to the synthetic venue")


@contextmanager
def serving(venue: Venue) -> Iterator[Venue]:
    """Route every ``urllib.request.urlopen`` to ``venue`` and refuse every socket
    connection in this process while the block runs. A worker process is not
    covered, so the dry run passes the venue as the opener and fetches in-process."""
    with (
        mock.patch("urllib.request.urlopen", venue),
        mock.patch("socket.socket.connect", _no_network),
        mock.patch("socket.create_connection", _no_network),
    ):
        yield venue
