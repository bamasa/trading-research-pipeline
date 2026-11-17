"""Bybit's daily print files: the URL, the reader, and one day's fetch.

A file is one gzip CSV per UTC day with ``timestamp`` (float seconds, as text),
``symbol``, ``side`` (the aggressor: ``Buy`` lifted an offer, ``Sell`` hit a
bid), ``size`` (BTC), ``price`` (USDT) and more columns not read here except
``trdMatchID``. Older files were written newest first, so the reader looks at
the file's direction and reverses a newest-first file before a stable sort by
stamp: prints sharing a stamp keep the venue's order. Stamps are converted from
their text exactly (seconds and a decimal fraction), never through a float.

The frame written is the spot study's print contract (``spot/prints.py``), with
``trade_id`` the print's position in the venue's order, ``rpi`` false and the
source :data:`SOURCE`. Rows stamped outside the day are dropped and counted.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_DAY, day_start_ns
from trading_research.mm_bybit.access import Access
from trading_research.spot import prints as contract

URL = "https://public.bybit.com/trading/{symbol}/{symbol}{day}.csv.gz"
SYMBOL = "BTCUSDT"
SOURCE = "bybit-linear-prints"

Opener = Callable[..., Any]


class UnreadableFile(ValueError):
    """A file that does not parse into the print contract."""


class MissingFile(LookupError):
    """The archive answered 404 for the day."""


def url(day: date, symbol: str = SYMBOL) -> str:
    return URL.format(symbol=symbol, day=day.isoformat())


def stamps_ns(text: pd.Series) -> np.ndarray:
    """Exact nanoseconds from decimal-second strings such as ``1609459200.1234``."""
    parts = text.astype(str).str.strip().str.split(".", n=1, expand=True)
    seconds = parts[0].astype(np.int64).to_numpy(dtype=np.int64)
    fraction = np.zeros(len(seconds), dtype=np.int64)
    if parts.shape[1] > 1:
        digits = parts[1].fillna("").str.ljust(9, "0").str.slice(0, 9)
        fraction = np.asarray(digits.astype(np.int64).to_numpy(), dtype=np.int64)
    return np.asarray(seconds * 1_000_000_000 + fraction, dtype=np.int64)


def parse(raw: bytes, day: date) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One day's CSV into the print contract, and what the reader found."""
    try:
        frame = pd.read_csv(
            io.BytesIO(raw),
            usecols=lambda c: c in {"timestamp", "side", "size", "price", "trdMatchID"},
            dtype={"timestamp": str, "side": str, "trdMatchID": str},
        )
        ts = stamps_ns(frame["timestamp"])
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        raise UnreadableFile(f"{day}: {exc}") from exc
    if not {"side", "size", "price"} <= set(frame.columns):
        raise UnreadableFile(f"{day}: columns {list(frame.columns)}")
    if not frame["side"].isin(("Buy", "Sell")).all():
        raise UnreadableFile(f"{day}: a side other than Buy or Sell")
    descending = len(ts) > 1 and ts[0] > ts[-1]
    order = np.arange(len(ts))[::-1] if descending else np.arange(len(ts))
    ts = ts[order]
    frame = frame.iloc[order].reset_index(drop=True)
    decimals = frame["timestamp"].str.split(".", n=1).str[1].fillna("").str.len()
    resolution = int(decimals.max()) if len(decimals) else 0
    start = day_start_ns(day)
    inside = (ts >= start) & (ts < start + NS_PER_DAY)
    backwards = int((np.diff(ts) < 0).sum()) if len(ts) > 1 else 0
    duplicates = (
        int(frame["trdMatchID"].duplicated().sum()) if "trdMatchID" in frame.columns else None
    )
    found: dict[str, Any] = {
        "rows": len(ts),
        "direction": "descending" if descending else "ascending",
        "stamps_ordered": backwards == 0,
        "rows_out_of_order": backwards,
        "stamp_decimals": resolution,
        "rows_outside_day": int((~inside).sum()),
        "duplicate_match_ids": duplicates,
        "ids_unique": duplicates == 0 if duplicates is not None else None,
    }
    ts, frame = ts[inside], frame.loc[inside].reset_index(drop=True)
    aggressor = np.where(frame["side"].to_numpy() == "Buy", 1, -1).astype(np.int8)
    try:
        out = contract.assemble(
            pd.Series(pd.to_datetime(ts, unit="ns", utc=True)),
            frame["price"].astype(np.float64),
            frame["size"].astype(np.float64),
            aggressor,
            pd.Series(np.arange(len(ts), dtype=np.int64)),
            np.zeros(len(ts), dtype=bool),
            SOURCE,
        )
    except contract.PrintContractError as exc:
        raise UnreadableFile(f"{day}: {exc}") from exc
    found["sides"] = {"buy": int((aggressor > 0).sum()), "sell": int((aggressor < 0).sum())}
    return out, found


def prints_path(root: Path | str, day: date, symbol: str = SYMBOL) -> Path:
    return Path(root) / "prints" / symbol / f"{day.isoformat()}.parquet"


def status_path(root: Path | str, day: date, symbol: str = SYMBOL) -> Path:
    """Where a day the fetch could not convert records why (404 or unreadable)."""
    return Path(root) / "prints" / symbol / f"{day.isoformat()}.status.json"


def write_status(root: Path | str, day: date, status: str, detail: str = "") -> None:
    target = status_path(root, day)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"day": day.isoformat(), "status": status, "detail": detail}))


def read_status(root: Path | str, day: date) -> str | None:
    """The recorded reason a fetched day has no prints file, or None."""
    target = status_path(root, day)
    if not target.exists():
        return None
    return str(json.loads(target.read_text())["status"])


def fetch_day(
    day: date,
    root: Path | str,
    access: Access,
    *,
    opener: Opener = urllib.request.urlopen,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Download, parse and write one day; the raw file never stays on disk."""
    access.require(day)
    target = prints_path(root, day)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with opener(url(day), timeout=timeout) as response:
            gz = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise MissingFile(f"{day}: 404") from exc
        raise
    try:
        raw = gzip.decompress(gz)
    except (OSError, EOFError) as exc:
        raise UnreadableFile(f"{day}: {exc}") from exc
    frame, found = parse(raw, day)
    partial = target.with_suffix(f".{os.getpid()}.part")
    frame.to_parquet(partial, index=False, compression="zstd")
    partial.replace(target)
    found.update(day=day.isoformat(), gz_bytes=len(gz), gz_sha256=hashlib.sha256(gz).hexdigest())
    return found
