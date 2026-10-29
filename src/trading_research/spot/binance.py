"""Binance BTCFDUSD aggregated prints: ``data/binance.py``'s spot parser and an adapter.

The archive is read by :func:`trading_research.data.binance.parse_agg_trades`,
unchanged, and its output is adapted to the spot print contract
(:mod:`trading_research.spot.prints`). Only daily archives are fetched
(:data:`trading_research.spot.archives.BINANCE_PRINTS`).

**The side flag**, as the venue documents it: ``is_buyer_maker`` true means
the buyer was resting and the seller crossed, a seller-initiated print
(aggressor -1); false a buyer-initiated one (+1).

**Two units of time.** Binance writes spot times in milliseconds before
2025-01-01 and in microseconds from it (the format probe found 16 digits on
2026-03-03). ``parse_agg_trades`` takes each file's unit by magnitude
(:func:`trading_research.data.binance.epoch_unit`); this adapter also refuses a
file whose smallest and largest stamps are in different units, or whose stamps
fall outside the file's own day by more than a day, so a file that mixed units
raises instead of being cut silently to an empty day.

**Checksums.** Each archive is checked against its published ``.CHECKSUM``
(:func:`verify`) before it is read.

The archive carries no flag of prints an order cannot reach; its
``is_best_match`` field is counted for stage 0's record (:class:`AggChecks`)
and sets nothing.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data import binance as archive
from trading_research.data import fetch_log as fl
from trading_research.data.failures import Opener
from trading_research.data.study_fetch import DayOutcome
from trading_research.spot import fetch, prints
from trading_research.spot.archives import (
    BINANCE_PRINTS,
    BINANCE_SPOT,
    PRINTS,
    checksum_url,
    day_path,
)

SPEC = archive.ArchiveSpec(market="spot", kind="aggTrades", symbol="BTCFDUSD")


class BinanceChecksumError(ValueError):
    """An archive's sha256 is not the one its ``.CHECKSUM`` publishes."""


class BinancePrintsError(prints.PrintContractError):
    """The archive's times do not belong to its day in one unit."""


def published_sha256(checksum: bytes | str) -> str:
    """The hex digest in a ``.CHECKSUM`` file (``<sha256>  <file name>``)."""
    text = checksum.decode() if isinstance(checksum, bytes) else checksum
    fields = text.split()
    if not fields or len(fields[0]) != 64:
        raise BinanceChecksumError(f"not a checksum file: {text[:80]!r}")
    return fields[0].lower()


def verify(payload: bytes, checksum: bytes | str) -> str:
    """The payload's sha256, if it equals the published one; raise otherwise."""
    actual = hashlib.sha256(payload).hexdigest()
    expected = published_sha256(checksum)
    if actual != expected:
        raise BinanceChecksumError(
            f"sha256 {actual[:12]}... is not the published {expected[:12]}..."
        )
    return actual


#: A published checksum unlike the archive's, or an archive that does not decompress.
_CORRUPT: tuple[type[BaseException], ...] = (BinanceChecksumError, *fetch.UNREADABLE_ARCHIVE)


@dataclass(frozen=True)
class AggChecks:
    """What a day's archive showed, for stage 0's record."""

    rows: int
    #: The epoch unit the stamps were read in (``ms`` or ``us``).
    unit: str
    #: Rows whose ``is_best_match`` is false.
    not_best_match: int
    #: Every aggregate id one more than the row before it, in the archive's order.
    ids_consecutive: bool


def adapt(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    """``parse_agg_trades``' output as a print-contract frame, its stamps checked against ``day``."""
    start = pd.Timestamp(datetime(day.year, day.month, day.day, tzinfo=UTC))
    stamps = frame["timestamp"]
    if len(frame) and (
        stamps.min() < start - timedelta(days=1) or stamps.max() >= start + timedelta(days=2)
    ):
        raise BinancePrintsError(f"stamps {stamps.min()} to {stamps.max()} are not around {day}")
    aggressor = np.where(frame["is_buyer_maker"].to_numpy(dtype=bool), -1, 1)
    return prints.assemble(
        stamps,
        frame["price"],
        frame["quantity"],
        aggressor,
        frame["trade_id"],
        np.zeros(len(frame), dtype=bool),
        prints.BINANCE_SOURCE,
    )


def _column(raw: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    """The first of ``names`` the archive has (its header has changed over the years)."""
    return next((raw[name] for name in names if name in raw.columns), None)


def _false_count(values: pd.Series) -> int:
    text = values.astype("string").str.strip().str.lower()
    return int((text == "false").sum())


def _no_rows(payload: bytes) -> bool:
    with zipfile.ZipFile(io.BytesIO(payload)) as packed:
        members = [n for n in packed.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            return False  # read_archive_csv refuses it
        with packed.open(members[0]) as text:
            for line in text:
                if line.strip() and line.strip()[:1].isdigit():
                    return False  # a data row
    return True  # nothing, or a header alone


def read(payload: bytes, day: date) -> tuple[pd.DataFrame, AggChecks]:
    """Parse one day's archive (already verified) into the print contract, with its checks.

    An archive whose one CSV holds no row (or only a header) is a day without a
    print, which the loader excludes, not an unreadable file.
    """
    if _no_rows(payload):
        return prints.empty(prints.BINANCE_SOURCE), AggChecks(0, "none", 0, True)
    raw = archive.read_archive_csv(payload)
    times = _column(raw, ("transact_time", "transaction_time", "timestamp", "time", "T"))
    if times is None:
        raise BinancePrintsError(f"no time column among {list(raw.columns)}")
    unit = archive.epoch_unit(times)
    ends = {archive.epoch_unit(pd.Series([v])) for v in (times.min(), times.max())}
    if ends != {unit}:
        raise BinancePrintsError(f"stamps in more than one unit: {sorted(ends | {unit})}")
    frame = archive.parse_agg_trades(payload, SPEC)
    best = _column(raw, ("is_best_match", "M"))
    ids = frame["trade_id"].to_numpy(dtype=np.int64)
    checks = AggChecks(
        rows=len(frame),
        unit=unit,
        not_best_match=0 if best is None else _false_count(best),
        ids_consecutive=bool(np.all(np.diff(ids) == 1)),
    )
    return adapt(frame, day), checks


def parse(payload: bytes, day: date) -> pd.DataFrame:
    """One day's archive as a print-contract frame."""
    return read(payload, day)[0]


def fetch_day(
    day: date,
    root: Path | str,
    archive_dir: Path | str,
    *,
    access: fetch.AccessGuard,
    opener: Opener | None = None,
    timeout: float = 120.0,
) -> DayOutcome:
    """Fetch one daily archive and its ``.CHECKSUM``, verify, convert, write; delete both after.

    A 404 on the archive is a missing plane. A checksum that cannot be fetched
    (a 404 included) is a transient failure: the archive is not read unchecked.
    A sha256 unlike the published one is ``corrupt`` (unreadable once the same
    bytes fail twice).
    """
    access.require(BINANCE_SPOT, [day], PRINTS, what="fetch")
    name = BINANCE_PRINTS.name
    target = Path(archive_dir) / f"{name}-{day.isoformat()}.zip"
    checksum_file = target.with_name(f"{target.name}.CHECKSUM")
    try:
        got, missed = fetch.download(
            name, day, BINANCE_PRINTS.url(day), target, opener=opener, timeout=timeout
        )
        if got is None:
            assert missed is not None
            return missed
        published, unchecked = fetch.download(
            name,
            day,
            checksum_url(BINANCE_PRINTS, day),
            checksum_file,
            opener=opener,
            timeout=timeout,
        )
        if published is None:
            assert unchecked is not None
            detail = f"checksum not fetched: {unchecked.detail}"
            return DayOutcome(name, day, fl.TRANSIENT, fetch.now(), detail, got.sha256, got.size)
        payload = target.read_bytes()
        try:
            verify(payload, checksum_file.read_bytes())
            fetch.check_zip(target)
        except _CORRUPT as exc:
            return fetch.failed(name, day, fl.CORRUPT, exc, got)
        try:
            frame, checks = read(payload, day)
        except (
            archive.BinanceArchiveError,
            prints.PrintContractError,
            ValueError,
            KeyError,
        ) as exc:
            return fetch.failed(name, day, fl.REPLAY_FAILED, exc, got)
        fetch.write_parquet(frame, day_path(BINANCE_PRINTS, day, root))
        return fetch.done(name, day, got, asdict(checks))
    finally:
        target.unlink(missing_ok=True)
        checksum_file.unlink(missing_ok=True)
