"""A local spot venue for the tests and the dry run: every spot URL answered from fixtures, no socket.

The venue answers exactly the registered URLs (:mod:`trading_research.spot.archives`,
:data:`trading_research.spot.instrument.SPEC_URLS`), in the layouts the format
probe of 2026-10-25 found:

* Bybit's spot ``ob200`` book: a zip of JSON lines, a snapshot and then a
  delta every ``step_ms``, ``{"topic", "ts", "type", "data": {"s", "b", "a",
  "u", "seq"}, "cts"}``, which :func:`trading_research.data.bybit.reconstruct`
  replays;
* Bybit's spot prints: a gzipped CSV ``id,timestamp,price,volume,side,rpi``,
  stamps in milliseconds, sides in lower case, at the touch of the message
  before them;
* Binance's daily BTCFDUSD ``aggTrades``: a zip of headerless eight-field
  rows, stamps in microseconds from 2025-01-01 and milliseconds before (as the
  venue writes them), and the archive's ``.CHECKSUM``;
* both venues' spot instrument specifications.

``HEAD`` is answered with headers only. A day outside the market's days
answers 404, a missing plane; so does any URL the venue does not know, which
is also kept in :attr:`SpotVenue.unknown`. Everything is drawn from
generators seeded by the day, so the venue answers the same bytes every time.

:func:`serving` routes ``urllib.request.urlopen`` to the venue and refuses
every socket connection in the process while it runs; :func:`no_network`
refuses sockets alone, for tests that need no venue. A worker process is not
covered, so the dry run passes the venue as the opener and fetches in-process.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.message import Message
from unittest import mock

import numpy as np

from trading_research.spot.archives import (
    BINANCE_PRINTS,
    BINANCE_SPOT,
    BYBIT_BOOK,
    BYBIT_PRINTS,
    BYBIT_SPOT,
    checksum_url,
)
from trading_research.spot.instrument import SPEC_URLS

SYMBOL_BYBIT = "BTCUSDT"
SYMBOL_BINANCE = "BTCFDUSD"
#: Levels a side in the generated book (the archive has 200; ten are read).
LEVELS = 12
BYBIT_TICK, BYBIT_LOT = 0.1, 0.000001
BINANCE_TICK, BINANCE_LOT = 0.01, 0.00001
#: The book's price near the start, in Bybit ticks (about 68,000 USDT).
START_TICKS = 680_000
#: Binance writes spot stamps in microseconds from this day, in milliseconds before.
MICROSECONDS_FROM = date(2025, 1, 1)
MS_PER_DAY = 86_400_000


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def _zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as packed:
        packed.writestr(entry, text)
    return buffer.getvalue()


@dataclass(frozen=True)
class SpotMarket:
    """The days both venues publish, and how the generated market behaves.

    ``gap_days`` skip three update ids in the book; ``empty_days`` publish a
    print archive without a row on both venues; every ``rpi_every``-th Bybit
    print is flagged ``rpi`` (none when 0); ``binance_unit`` forces Binance's
    stamps into ``ms`` or ``us`` (None: the venue's rule by date).
    """

    days: frozenset[date]
    step_ms: int = 10_000
    seed: int = 2026
    gap_days: frozenset[date] = frozenset()
    empty_days: frozenset[date] = frozenset()
    rpi_every: int = 0
    binance_unit: str | None = None
    print_rate: float = 0.8

    def _rng(self, day: date, salt: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, day.toordinal(), salt])

    def path(self, day: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Message stamps (ms), the best bid and the spread, in Bybit ticks."""
        rng = self._rng(day, 0)
        n = MS_PER_DAY // self.step_ms
        stamps = _ms(day) + np.arange(n, dtype=np.int64) * self.step_ms + rng.integers(1, 50, n)
        moves = rng.choice(np.array([-2, -1, 0, 0, 1, 2]), size=n)
        moves[0] = 0
        return stamps, START_TICKS + np.cumsum(moves), 1 + (rng.random(n) < 0.3)

    def book_lines(self, day: date) -> list[str]:
        stamps, bid0, spread = self.path(day)
        rng = self._rng(day, 1)
        skips = set(np.linspace(len(stamps) // 4, 3 * len(stamps) // 4, 3).astype(int))
        if day not in self.gap_days:
            skips = set()

        def side(levels: dict[int, float]) -> list[list[str]]:
            return [[f"{t * BYBIT_TICK:.1f}", f"{s:.6f}"] for t, s in levels.items()]

        def levels(top: int, sign: int) -> dict[int, float]:
            sizes = np.round(rng.exponential(0.05, LEVELS) + 0.002, 6)
            return {top + sign * k: float(sizes[k]) for k in range(LEVELS)}

        lines: list[str] = []
        update = 1_000
        bids: dict[int, float] = {}
        asks: dict[int, float] = {}
        for i, stamp in enumerate(stamps):
            new_bids = levels(int(bid0[i]), -1)
            new_asks = levels(int(bid0[i] + spread[i]), 1)
            b = side(new_bids) + [[f"{t * BYBIT_TICK:.1f}", "0"] for t in bids if t not in new_bids]
            a = side(new_asks) + [[f"{t * BYBIT_TICK:.1f}", "0"] for t in asks if t not in new_asks]
            update += 2 if i in skips else 1
            data = {"s": SYMBOL_BYBIT, "b": b, "a": a, "u": update, "seq": 10 * update}
            kind = "snapshot" if i == 0 else "delta"
            message = {"topic": "orderbook.200.BTCUSDT", "ts": int(stamp), "type": kind}
            lines.append(json.dumps({**message, "data": data, "cts": int(stamp) - 2}))
            bids, asks = new_bids, new_asks
        return lines

    def book_archive(self, day: date) -> bytes:
        name = f"{day.isoformat()}_{SYMBOL_BYBIT}_ob200.data"
        return _zip(name, "\n".join(self.book_lines(day)) + "\n")

    def bybit_prints_csv(self, day: date) -> bytes:
        """Prints between consecutive messages, at the touch the earlier one shows."""
        rows = ["id,timestamp,price,volume,side,rpi"]
        if day not in self.empty_days:
            stamps, bid0, spread = self.path(day)
            rng = self._rng(day, 2)
            counts = rng.poisson(self.print_rate, size=len(stamps) - 1)
            at = np.repeat(np.arange(len(counts)), counts)
            gap = stamps[at + 1] - stamps[at]
            times = np.sort(stamps[at] + 1 + (rng.random(len(at)) * (gap - 1)).astype(np.int64))
            times[1::7] = times[:-1:7]  # some prints share a stamp
            buy = rng.random(len(at)) < 0.5
            price = np.where(buy, bid0[at] + spread[at], bid0[at])
            size = np.round(rng.exponential(0.004, len(at)) + 0.000001, 6)
            for k, (t, b, q, x) in enumerate(zip(times, buy, size, price, strict=True), start=1):
                rpi = int(self.rpi_every > 0 and k % self.rpi_every == 0)
                side = "buy" if b else "sell"
                rows.append(f"{k},{t},{x * BYBIT_TICK:.1f},{q:.6f},{side},{rpi}")
        return ("\n".join(rows) + "\n").encode()

    def bybit_prints_archive(self, day: date) -> bytes:
        return gzip.compress(self.bybit_prints_csv(day), mtime=0)

    def binance_unit_for(self, day: date) -> str:
        if self.binance_unit is not None:
            return self.binance_unit
        return "us" if day >= MICROSECONDS_FROM else "ms"

    def binance_rows(self, day: date) -> list[str]:
        """A thinner market of aggregated prints on a 0.01 tick, near the book's price."""
        if day in self.empty_days:
            return []
        rng = self._rng(day, 3)
        stamps, bid0, _ = self.path(day)
        counts = rng.poisson(self.print_rate / 2, size=len(stamps) - 1)
        at = np.repeat(np.arange(len(counts)), counts)
        times = np.sort(
            stamps[at] + 1 + (rng.random(len(at)) * (self.step_ms - 2)).astype(np.int64)
        )
        scale = 1000 if self.binance_unit_for(day) == "us" else 1
        micros = rng.integers(0, scale, len(at)) if scale > 1 else np.zeros(len(at), dtype=np.int64)
        cents = bid0[at] * 10 + rng.integers(-3, 4, len(at))
        sizes = np.round(rng.exponential(0.003, len(at)) + 0.00001, 5)
        maker = rng.random(len(at)) < 0.5
        rows = []
        for k, (t, u, c, q, m) in enumerate(zip(times, micros, cents, sizes, maker, strict=True)):
            agg = 5_000_000 + k
            rows.append(
                f"{agg},{c * BINANCE_TICK:.2f},{q:.5f},{3 * agg},{3 * agg + 2},"
                f"{t * scale + u},{'True' if m else 'False'},True"
            )
        return rows

    def binance_archive(self, day: date) -> bytes:
        name = f"{SYMBOL_BINANCE}-aggTrades-{day.isoformat()}.csv"
        rows = self.binance_rows(day)
        return _zip(name, "\n".join(rows) + "\n" if rows else "")

    def binance_checksum(self, day: date) -> bytes:
        digest = hashlib.sha256(self.binance_archive(day)).hexdigest()
        return f"{digest}  {SYMBOL_BINANCE}-aggTrades-{day.isoformat()}.zip\n".encode()

    def present(self) -> dict[str, frozenset[date]]:
        """The existence check's answer, per plane, as ``archives.listed_present`` gives it."""
        return {
            plane.name: frozenset(self.days) for plane in (BYBIT_BOOK, BYBIT_PRINTS, BINANCE_PRINTS)
        }


def bybit_spec() -> bytes:
    row = {
        "symbol": SYMBOL_BYBIT,
        "priceFilter": {"tickSize": f"{BYBIT_TICK}"},
        "lotSizeFilter": {"basePrecision": f"{BYBIT_LOT:.6f}", "minOrderQty": "0.000048"},
    }
    return json.dumps({"retCode": 0, "result": {"category": "spot", "list": [row]}}).encode()


def binance_spec() -> bytes:
    filters = [
        {"filterType": "PRICE_FILTER", "tickSize": f"{BINANCE_TICK:.2f}"},
        {"filterType": "LOT_SIZE", "stepSize": f"{BINANCE_LOT:.5f}"},
    ]
    row = {"symbol": SYMBOL_BINANCE, "filters": filters}
    return json.dumps({"symbols": [row]}).encode()


class _Response(io.BytesIO):
    """What ``urlopen`` returns, with the headers the downloaders and ``HEAD`` read."""

    def __init__(self, body: bytes, *, length: int | None = None) -> None:
        super().__init__(body)
        self.status = 200
        self.headers = {
            "Content-Length": str(len(body) if length is None else length),
            "Last-Modified": "Thu, 01 Jan 2026 00:00:00 GMT",
        }


def _not_found(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 404, "Not Found", Message(), io.BytesIO())


@dataclass
class SpotVenue:
    """Answers every request by URL; one it does not recognise is a 404, kept in
    :attr:`unknown` so a dry run can refuse to call itself complete."""

    market: SpotMarket

    def __post_init__(self) -> None:
        self.asked: list[tuple[str, str]] = []
        self.unknown: list[str] = []

    def __call__(
        self, request: str | urllib.request.Request, timeout: float | None = None
    ) -> _Response:
        del timeout  # answered at once
        if isinstance(request, urllib.request.Request):
            url, method = request.full_url, request.get_method()
        else:
            url, method = request, "GET"
        self.asked.append((method, url))
        body = self.body(url)
        if body is None:
            raise _not_found(url)
        return _Response(b"", length=len(body)) if method == "HEAD" else _Response(body)

    def body(self, url: str) -> bytes | None:
        """The bytes the venue publishes at ``url``, or None for a 404."""
        if url == SPEC_URLS[BYBIT_SPOT]:
            return bybit_spec()
        if url == SPEC_URLS[BINANCE_SPOT]:
            return binance_spec()
        found = _DAY.search(url)
        day = date.fromisoformat(found.group(0)) if found else None
        if day is None or url not in self._urls(day):
            self.unknown.append(url)
            return None
        if day not in self.market.days:
            return None
        return self._urls(day)[url](day)

    def _urls(self, day: date) -> dict[str, Callable[[date], bytes]]:
        market = self.market
        return {
            BYBIT_BOOK.url(day): market.book_archive,
            BYBIT_PRINTS.url(day): market.bybit_prints_archive,
            BINANCE_PRINTS.url(day): market.binance_archive,
            checksum_url(BINANCE_PRINTS, day): market.binance_checksum,
        }


_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _no_network(*_: object, **__: object) -> None:
    raise OSError("no network here: every request goes to the synthetic spot venue")


@contextmanager
def no_network() -> Iterator[None]:
    """Refuse every socket connection in this process while the block runs."""
    with (
        mock.patch("socket.socket.connect", _no_network),
        mock.patch("socket.socket.connect_ex", _no_network),
        mock.patch("socket.create_connection", _no_network),
    ):
        yield


@contextmanager
def serving(venue: SpotVenue) -> Iterator[SpotVenue]:
    """Route every ``urllib.request.urlopen`` to ``venue`` and refuse every socket."""
    with no_network(), mock.patch("urllib.request.urlopen", venue):
        yield venue
