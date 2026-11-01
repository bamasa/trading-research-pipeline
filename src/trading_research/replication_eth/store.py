"""Fetching a day from the public archives and keeping what the study reads.

Layout under one root (gitignored):

* ``_archives/``: the venue's book archive while it is replayed, deleted after;
* ``prints/ETHUSDT/<day>.parquet``: the day's prints, as the pipeline parses them;
* ``books/<day>.parquet``: the gridded book's first ten levels a side, which the
  executor walks;
* ``days/<day>.npz``: the arrays of :mod:`.days`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from trading_research.data import bybit, bybit_trades
from trading_research.replication_eth.days import build_day, load_day, save_day
from trading_research.replication_eth.grid import DEPTH, grid_day, level_columns

SYMBOL = "ETHUSDT"


def day_path(root: Path | str, day: date) -> Path:
    return Path(root) / "days" / f"{day.isoformat()}.npz"


def book_path(root: Path | str, day: date) -> Path:
    return Path(root) / "books" / f"{day.isoformat()}.parquet"


def span(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def fetch_day(root: Path | str, day: date) -> dict[str, object]:
    """Download, replay, grid and store one day; a stored day is not redone."""
    base = Path(root)
    target = day_path(base, day)
    if target.exists() and book_path(base, day).exists():
        return {"day": day, "status": "stored"}
    kind = bybit.archive_kind(day)
    archive = bybit.download_day(SYMBOL, day, base / "_archives")
    book = bybit.reconstruct(archive, symbol=SYMBOL, depth=DEPTH, source=bybit.source_tag(kind))
    gaps = int(book.attrs.get("sequence_gaps", 0))
    archive.unlink(missing_ok=True)
    prints = pd.read_parquet(bybit_trades.download_day(SYMBOL, day, base / "prints"))
    frame = grid_day(book, prints, day)
    del book
    ten = ["ts", *level_columns(10)]
    book_file = book_path(base, day)
    book_file.parent.mkdir(parents=True, exist_ok=True)
    frame[ten].to_parquet(book_file, compression="zstd", index=False)
    save_day(build_day(frame, depth=DEPTH), target)
    return {
        "day": day,
        "status": "fetched",
        "archive": kind,
        "rows": len(frame),
        "prints": len(prints),
        "sequence_gaps": gaps,
    }


def load_span(root: Path | str, days: list[date]) -> dict[str, list]:
    """Each stored day's arrays, in day order, as lists per key."""
    loaded = [load_day(day_path(root, d)) for d in days]
    return {key: [arrays[key] for arrays in loaded] for key in loaded[0]}
