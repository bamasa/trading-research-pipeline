"""A stored day fed through the same interface as the live stream.

Four kinds of stored day, each turned into the same events a live session
sees (:mod:`trading_research.live.events`):

* **a raw log** written by the recorder (``*.jsonl.gz``): every message goes
  back through the parser the live session used, in the order it arrived, so
  the replay is the live session's event sequence, gaps included;
* **a Bybit book archive** (``*.zip`` or ``*.data``, the venue's own format):
  the messages through the same parser, which keeps
  :func:`trading_research.data.bybit.reconstruct`'s rows; prints from an
  optional ``--prints`` file in the Bybit print contract;
* **a dataset directory** of whole planes (``book.parquet`` and
  ``trades.parquet``, as ``generate-demo-data`` writes them);
* **a staged day** (``.../bookTicker/<day>.parquet`` with its sibling
  ``aggTrades`` day, the layout ``download`` and the recorder write).

Stored frames carry no receipt time, so their events have ``recv_ns`` equal
to the stamp, and rows and prints are merged by stamp with a print stamped at
a row's stamp first (it belongs to that row). :func:`paced` replays at wall
clock (``speed=1``), faster (``speed=10``) or as fast as the loop runs
(``speed=0``).
"""

from __future__ import annotations

import time
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from trading_research.data import store
from trading_research.data.schema import infer_depth
from trading_research.live.events import (
    DEPTH,
    BookRow,
    Event,
    Gap,
    Print,
    frame_to_prints,
    frame_to_rows,
    merge_by_time,
    trades_to_prints,
)
from trading_research.live.record import DEFAULT_ROOT, raw_log_path, read_raw_log
from trading_research.live.stream import BybitParser, make_parser

try:  # pragma: no cover
    from orjson import loads as _loads
except ImportError:  # pragma: no cover
    from json import loads as _loads


class ReplayError(ValueError):
    """The day asked for cannot be found or read."""


@dataclass(frozen=True)
class StoredDay:
    """A stored day ready to replay: its events and what it is."""

    kind: str
    path: Path
    symbol: str
    events: tuple[Event, ...]
    #: ``venue`` when rows carry the venue's stamp, ``receipt`` when local receipt.
    stamps: str = "venue"
    depth: int = DEPTH

    @property
    def rows(self) -> list[BookRow]:
        return [e for e in self.events if isinstance(e, BookRow)]

    @property
    def prints(self) -> list[Print]:
        return [e for e in self.events if isinstance(e, Print)]


def _raw_log(path: Path, venue: str, market: str, symbol: str) -> StoredDay:
    parser = make_parser(venue, market)
    events: list[Event] = []
    for recv_ns, connection, text in read_raw_log(path):
        if connection.startswith("#"):
            events.append(Gap(recv_ns, recv_ns, connection[1:], text))
            continue
        events.extend(parser.parse(text, recv_ns))
    stamps = "receipt" if (venue, market) == ("binance", "spot") else "venue"
    return StoredDay("raw_log", path, symbol, tuple(events), stamps)


def _archive_lines(path: Path) -> Iterator[bytes]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as packed, packed.open(packed.namelist()[0]) as binary:
            yield from (raw for raw in binary if raw.strip())
        return
    with path.open("rb") as binary:
        yield from (raw for raw in binary if raw.strip())


def _bybit_archive(path: Path, symbol: str, prints_path: Path | None) -> StoredDay:
    parser = BybitParser()
    rows: list[Event] = []
    for raw in _archive_lines(path):
        message = _loads(raw)
        stamp = int(message["ts"]) * 1_000_000
        rows.extend(parser.book_message(message, stamp))
    prints = frame_to_prints(pd.read_parquet(prints_path)) if prints_path is not None else []
    book = [e for e in rows if isinstance(e, BookRow)]
    gaps = [e for e in rows if isinstance(e, Gap)]
    merged = merge_by_time(book, prints)
    events: list[Event] = sorted([*merged, *gaps], key=lambda e: e.ts_ns)
    return StoredDay("bybit_archive", path, symbol, tuple(events))


def _from_frames(kind: str, path: Path, book: pd.DataFrame, prints: pd.DataFrame) -> StoredDay:
    depth = min(infer_depth(book), DEPTH)
    rows = frame_to_rows(book, depth=depth)
    events = merge_by_time(rows, frame_to_prints(prints))
    symbol = str(book["symbol"].iloc[0]) if len(book) else ""
    return StoredDay(kind, path, symbol, tuple(events), depth=depth)


def open_day(
    day: str | Path,
    *,
    venue: str | None = None,
    market: str | None = None,
    symbol: str | None = None,
    root: Path | str = DEFAULT_ROOT,
    prints: Path | str | None = None,
) -> StoredDay:
    """Find and read a stored day: a path of any kind above, or a date of a recording."""
    text = str(day)
    path = Path(text)
    if not path.exists():
        try:
            when = date.fromisoformat(text)
        except ValueError as exc:
            raise ReplayError(f"{text!r} is neither a file, a directory nor a date") from exc
        if not (venue and market and symbol):
            raise ReplayError("a date needs --venue, --market and --symbol to find its recording")
        path = raw_log_path(root, venue, market, symbol, when)
        if not path.exists():
            raise ReplayError(f"no recording for {venue}-{market} {symbol} on {when}: {path}")
    name = path.name
    if name.endswith(".jsonl.gz"):
        if not (venue and market):
            # The recorder's layout names both: <root>/<venue>-<market>/<SYMBOL>/raw/<day>.
            pair = path.parent.parent.parent.name
            venue, market = pair.split("-", 1) if "-" in pair else (venue, market)
        if not (venue and market):
            raise ReplayError(f"cannot tell the venue and market of {path}; pass them")
        return _raw_log(path, venue, market, (symbol or path.parent.parent.name).upper())
    if path.is_dir():
        if not store.has_plane(path, "book"):
            raise ReplayError(f"{path} holds no book.parquet")
        book = store.read_book(path)
        trades = store.read_trades(path) if store.has_plane(path, "trades") else None
        printed = trades_to_prints(trades) if trades is not None else _empty_prints()
        return _from_frames("dataset", path, book, printed)
    if path.suffix in (".zip", ".data"):
        return _bybit_archive(path, (symbol or "").upper(), Path(prints) if prints else None)
    if path.suffix == ".parquet":
        book = pd.read_parquet(path)
        sibling = path.parent.parent / "aggTrades" / path.name
        if prints is not None:
            printed = pd.read_parquet(prints)
        elif sibling.exists():
            printed = trades_to_prints(pd.read_parquet(sibling))
        else:
            printed = _empty_prints()
        if "aggressor" not in printed.columns:
            printed = trades_to_prints(printed)
        return _from_frames("staged", path, book, printed)
    raise ReplayError(f"do not know how to replay {path}")


def _empty_prints() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.Series([], dtype="int64"), utc=True),
            "price": pd.Series([], dtype="float64"),
            "size": pd.Series([], dtype="float64"),
            "aggressor": pd.Series([], dtype="int64"),
        }
    )


def paced(events: Iterable[Event], speed: float = 0.0) -> Iterator[Event]:
    """The events, released no faster than ``speed`` times the recorded clock (0: at once)."""
    if speed <= 0:
        yield from events
        return
    start_wall: float | None = None
    start_ts = 0
    for event in events:
        if start_wall is None:
            start_wall, start_ts = time.monotonic(), event.recv_ns
        due = start_wall + (event.recv_ns - start_ts) / 1e9 / speed
        wait = due - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        yield event
