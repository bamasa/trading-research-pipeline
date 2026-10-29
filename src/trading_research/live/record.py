"""Recording a public stream to day files the batch pipeline reads.

Two kinds of file, both under the gitignored ``data/`` and never committed:

**The raw log**, ``<root>/<venue>-<market>/<SYMBOL>/raw/<day>.jsonl.gz``: every
data message as it arrived, one line each, ``recv_ns<TAB>connection<TAB>message``.
It is what a replay feeds back through the same parser the live run used, so a
replayed recording reproduces the live session's events exactly. A session
appends a gzip member, so several sessions on one day make one valid file.

**The archive layouts**, written from a raw log by :func:`export_day`:

* Bybit: the venue's own archive file,
  ``archives/<SYMBOL>-<day>-ob200.zip`` holding ``<day>_<SYMBOL>_ob200.data``
  (the book messages, line for line), which
  :func:`trading_research.data.bybit.reconstruct` replays and which
  ``data.bybit.download_range`` takes from its cache instead of downloading;
  the replayed book ``book/<SYMBOL>/<day>.parquet`` (ten levels, 100 ms); and
  the prints ``trades/<SYMBOL>/<day>.parquet`` through
  :func:`trading_research.data.bybit_trades.parse`, from a CSV in the linear
  archive's columns.
* Both venues: the staged pipeline's layout,
  ``staged/<SYMBOL>/bookTicker/<day>.parquet`` (the ten-level rows; ``prepare``
  reads the touch) and ``staged/<SYMBOL>/aggTrades/<day>.parquet`` (the trade
  contract), each directory with a manifest, so
  ``trading-research prepare --raw <root>/<venue>-<market>/<SYMBOL>/staged`` runs on a
  recorded day.

A day is a UTC day; a message is filed under the day of its receipt.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import IO, Any

from trading_research import __version__
from trading_research.data.schema import SCHEMA_VERSION
from trading_research.live.events import (
    BookRow,
    Print,
    prints_to_frame,
    prints_to_trades,
    rows_to_frame,
)
from trading_research.live.stream import make_parser, source_tag

#: Where recordings go by default: below the gitignored data directory.
DEFAULT_ROOT = Path("data/paper")


def instrument_dir(root: Path | str, venue: str, market: str, symbol: str) -> Path:
    return Path(root) / f"{venue}-{market}" / symbol.upper()


def raw_log_path(root: Path | str, venue: str, market: str, symbol: str, day: date) -> Path:
    return instrument_dir(root, venue, market, symbol) / "raw" / f"{day.isoformat()}.jsonl.gz"


def utc_day(ns: int) -> date:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).date()


@dataclass
class RawRecorder:
    """Appends every message to the day's raw log, rolling over at midnight UTC."""

    root: Path
    venue: str
    market: str
    symbol: str
    flush_every: int = 500
    _day: date | None = field(default=None, repr=False)
    _handle: IO[str] | None = field(default=None, repr=False)
    _since_flush: int = field(default=0, repr=False)
    written: dict[str, int] = field(default_factory=dict)

    def __call__(self, connection: str, recv_ns: int, text: str) -> None:
        day = utc_day(recv_ns)
        if day != self._day:
            self.close()
            path = raw_log_path(self.root, self.venue, self.market, self.symbol, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = gzip.open(path, "at", encoding="utf-8")  # noqa: SIM115 - kept open across calls
            self._day = day
        assert self._handle is not None
        self._handle.write(f"{recv_ns}\t{connection}\t{text.strip()}\n")
        key = day.isoformat()
        self.written[key] = self.written.get(key, 0) + 1
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._handle.flush()
            self._since_flush = 0

    def note(self, recv_ns: int, kind: str, detail: str) -> None:
        """A line that is not a message (a reconnect), kept so a replay shows it too."""
        self(f"#{kind}", recv_ns, json.dumps({"detail": detail}))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._day = None

    def days(self) -> list[date]:
        return [date.fromisoformat(d) for d in sorted(self.written)]


def read_raw_log(path: Path | str) -> Iterator[tuple[int, str, str]]:
    """``(recv_ns, connection, message)`` for every line of a raw log (gzipped or plain)."""
    name = str(path)
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            recv, connection, text = line.split("\t", 2)
            yield int(recv), connection, text


def parse_raw_log(path: Path | str, venue: str, market: str) -> tuple[list[BookRow], list[Print]]:
    """The rows and prints a raw log holds, through the live parser."""
    parser = make_parser(venue, market)
    rows: list[BookRow] = []
    prints: list[Print] = []
    for recv_ns, connection, text in read_raw_log(path):
        if connection.startswith("#"):
            continue
        for event in parser.parse(text, recv_ns):
            if isinstance(event, BookRow):
                rows.append(event)
            elif isinstance(event, Print):
                prints.append(event)
    return rows, prints


def _manifest(directory: Path, payload: dict[str, Any]) -> None:
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_day(
    raw_log: Path | str, venue: str, market: str, symbol: str, root: Path | str
) -> dict[str, str]:
    """Write one recorded day in the archive layouts; returns ``{what: path}``."""
    raw = Path(raw_log)
    day = date.fromisoformat(raw.name.split(".")[0])
    base = instrument_dir(root, venue, market, symbol)
    symbol = symbol.upper()
    written: dict[str, str] = {}
    rows, prints = parse_raw_log(raw, venue, market)
    tag = source_tag(venue, market)
    if venue == "bybit":
        written.update(_export_bybit(raw, base, symbol, day, tag))
    stage = base / "staged" / symbol
    if rows:
        book = rows_to_frame(rows, symbol=symbol, source=tag)
        target = stage / "bookTicker" / f"{day.isoformat()}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        book.to_parquet(target, index=False)
        _manifest(
            target.parent,
            {
                "source": tag,
                "plane": "book",
                "symbol": symbol,
                "depth": 10,
                "resample_grid": "100ms",
                "schema_version": SCHEMA_VERSION,
                "trading_research_version": __version__,
                "notes": "Recorded from the public stream: ten levels a side, at most one row "
                "per 100 ms interval at its first update. Named bookTicker because prepare reads "
                "the touch from this directory; the levels behind it are carried as well.",
            },
        )
        written["staged_book"] = str(target)
    if prints:
        trades = prints_to_trades(prints_to_frame(prints), symbol=symbol, source=tag)
        target = stage / "aggTrades" / f"{day.isoformat()}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(target, index=False)
        _manifest(
            target.parent,
            {
                "source": tag,
                "plane": "trades",
                "symbol": symbol,
                "schema_version": SCHEMA_VERSION,
                "trading_research_version": __version__,
                "notes": "Recorded from the public stream. is_buyer_maker is True when the "
                "seller crossed; Bybit's string ids are replaced by the print's position.",
            },
        )
        written["staged_trades"] = str(target)
    return written


def _export_bybit(raw: Path, base: Path, symbol: str, day: date, tag: str) -> dict[str, str]:
    from trading_research.data import bybit, bybit_trades

    written: dict[str, str] = {}
    book_lines: list[str] = []
    csv_rows: list[str] = []
    for _, connection, text in read_raw_log(raw):
        if connection.startswith("#"):
            continue
        head = text[:60]
        if '"orderbook.' in head:
            book_lines.append(text)
        elif '"publicTrade.' in head:
            for trade in json.loads(text).get("data", []):
                csv_rows.append(
                    f"{int(trade['T']) / 1000:.3f},{symbol},{trade['S']},{trade['v']},"
                    f"{trade['p']},{trade.get('i', '')}"
                )
    if book_lines:
        archive = base / "archives" / f"{symbol}-{day.isoformat()}-ob200.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as packed:
            packed.writestr(f"{day.isoformat()}_{symbol}_ob200.data", "\n".join(book_lines) + "\n")
        written["archive"] = str(archive)
        frame = bybit.reconstruct(archive, symbol=symbol, depth=10, grid_ms=100, source=tag)
        target = base / "book" / symbol / f"{day.isoformat()}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        written["book"] = str(target)
    if csv_rows:
        text = "timestamp,symbol,side,size,price,trdMatchID\n" + "\n".join(csv_rows) + "\n"
        frame = bybit_trades.parse(text.encode("utf-8"))
        target = bybit_trades.day_path(symbol, day, base / "trades")
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        written["trades"] = str(target)
    return written
