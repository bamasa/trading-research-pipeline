"""The spot study's archives: their URLs, their coverage by ``HEAD``, and where each lands.

Three planes, each keyed by venue and market, never by symbol alone (Bybit's
linear perpetual is also ``BTCUSDT``):

==================  ==========================  ==============================================
Plane               Instrument                  Archive, one file a UTC day
==================  ==========================  ==============================================
``bybit_book``      ``bybit-spot:BTCUSDT``      ``ob200`` zip of JSON lines
``bybit_prints``    ``bybit-spot:BTCUSDT``      gzipped CSV ``id,timestamp,price,volume,side,rpi``
``binance_prints``  ``binance-spot:BTCFDUSD``   daily ``aggTrades`` zip, with its ``.CHECKSUM``
==================  ==========================  ==============================================

Binance is asked for daily files only: no URL here names a monthly archive, so
no request can put another day of a month on disk. Coverage is learned by
HTTP ``HEAD`` (:func:`head`), which reads no byte of a file, and the days the
registered head check of 2026-10-22 found present are read from its committed
CSV (:func:`listed_present`).

Every file lands under ``data/spot/<venue>/<plane>/<symbol>/<day>.parquet``
(:func:`day_path`), never under ``data/{book,trades,bars,funding}/BTCUSDT``.
"""

from __future__ import annotations

import csv
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

#: The study's data root, relative to the repository.
DATA_ROOT = Path("data/spot")
#: The registered head check: one row per day, 2025-01-01 to 2026-10-22.
HEAD_CHECK = Path("experiments/results/spot_2026_head_check.csv")

BYBIT_SPOT = "bybit-spot:BTCUSDT"
BINANCE_SPOT = "binance-spot:BTCFDUSD"
INSTRUMENTS: tuple[str, ...] = (BINANCE_SPOT, BYBIT_SPOT)

#: The planes a study access is asked about, per instrument.
BOOK, PRINTS = "book", "prints"

Opener = Callable[..., Any]


@dataclass(frozen=True)
class Plane:
    """One archive of one instrument, and where its converted days live."""

    name: str
    instrument: str
    kind: str
    url_template: str
    #: The first day the archive holds, as the registration lists it.
    first_day: date
    #: The head check's column that says a day is present.
    head_column: str

    @property
    def venue(self) -> str:
        return self.instrument.split("-", 1)[0]

    @property
    def symbol(self) -> str:
        return self.instrument.split(":", 1)[1]

    @property
    def directory(self) -> Path:
        """``<venue>/<plane>/<symbol>``, below the study's data root."""
        return Path(self.venue) / self.kind / self.symbol

    def url(self, day: date) -> str:
        return self.url_template.format(symbol=self.symbol, day=day.isoformat())


BYBIT_BOOK = Plane(
    "bybit_book",
    BYBIT_SPOT,
    BOOK,
    "https://quote-saver.bycsi.com/orderbook/spot/{symbol}/{day}_{symbol}_ob200.data.zip",
    date(2025, 4, 29),
    "ob200_status",
)
BYBIT_PRINTS = Plane(
    "bybit_prints",
    BYBIT_SPOT,
    PRINTS,
    "https://public.bybit.com/spot/{symbol}/{symbol}_{day}.csv.gz",
    date(2022, 11, 10),
    "prints_status",
)
BINANCE_PRINTS = Plane(
    "binance_prints",
    BINANCE_SPOT,
    PRINTS,
    "https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip",
    date(2023, 8, 4),
    "fdusd_aggtrades_bytes",
)
PLANES: dict[str, Plane] = {p.name: p for p in (BYBIT_BOOK, BYBIT_PRINTS, BINANCE_PRINTS)}


def checksum_url(plane: Plane, day: date) -> str:
    """The published sha256 of a Binance archive (Bybit publishes none)."""
    if plane.venue != "binance":
        raise ValueError(f"{plane.name} has no published checksum")
    return f"{plane.url(day)}.CHECKSUM"


def day_path(plane: Plane, day: date, root: Path | str = DATA_ROOT) -> Path:
    """Where one converted day of ``plane`` lives."""
    return Path(root) / plane.directory / f"{day.isoformat()}.parquet"


@dataclass(frozen=True)
class Head:
    """What a ``HEAD`` request learned about one file, without reading it."""

    url: str
    status: int
    size: int | None
    last_modified: str | None


def _length(headers: Any) -> int | None:
    value = headers.get("Content-Length") if headers is not None else None
    return int(value) if value is not None and str(value).isdigit() else None


def head(url: str, *, opener: Opener = urllib.request.urlopen, timeout: float = 30.0) -> Head:
    """Ask for ``url``'s headers only. An HTTP error is a status, not an exception;
    a failure of the connection itself propagates."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with opener(request, timeout=timeout) as response:
            headers = getattr(response, "headers", None)
            status = int(getattr(response, "status", 200))
            modified = headers.get("Last-Modified") if headers is not None else None
            return Head(url, status, _length(headers), modified)
    except urllib.error.HTTPError as exc:
        return Head(url, int(exc.code), None, None)


def coverage(
    plane: Plane, days: Iterable[date], *, opener: Opener = urllib.request.urlopen
) -> dict[date, Head]:
    """``HEAD`` on every day of ``plane``: which files exist, and their sizes."""
    return {day: head(plane.url(day), opener=opener) for day in days}


def listed_present(path: Path | str = HEAD_CHECK) -> dict[str, frozenset[date]]:
    """Days the registered head check found present, per plane name.

    Bybit's planes are present on a ``HEAD`` status of 200; Binance's when the
    archive's S3 listing gave the day's file a size.
    """
    present: dict[str, set[date]] = {name: set() for name in PLANES}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["day"])
            for name, plane in PLANES.items():
                value = row[plane.head_column].strip()
                if (value == "200") if plane.venue == "bybit" else value.isdigit():
                    present[name].add(day)
    return {name: frozenset(days) for name, days in present.items()}
