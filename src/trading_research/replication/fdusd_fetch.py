"""Binance spot BTCFDUSD daily aggTrades for the 2025 replication, in a root of its own.

The spot study's access refuses every 2025 day by design and is not used here.
Each daily archive is downloaded with its published ``.CHECKSUM``, verified
(:func:`trading_research.spot.binance.verify`), parsed by the spot reader's
lower-level function (:func:`trading_research.spot.binance.read`, which maps the
side flag and takes each file's epoch unit), and written to
``data/replication/binance/prints/BTCFDUSD/<day>.parquet``: the spot layout's
:func:`~trading_research.spot.archives.day_path` under this root, never under
``data/spot/`` or ``data/{book,trades,bars,funding}/``. The archive and its
checksum are deleted after the conversion, whatever happened.

:class:`ReplicationAccess` is the replication's own guard for the spot loader:
it allows Binance BTCFDUSD prints on the replication's days and refuses anything else
(:class:`OutsideReplication`).
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from trading_research.data.failures import Opener, download_to
from trading_research.spot.archives import (
    BINANCE_PRINTS,
    BINANCE_SPOT,
    PRINTS,
    checksum_url,
    day_path,
)
from trading_research.spot.binance import read, verify
from trading_research.spot.fetch import check_zip

ROOT = Path("data/replication")
#: The days fetched: the author's 2025 split with four warm-up days, and May to June 2025.
SPANS: tuple[tuple[date, date], ...] = (
    (date(2024, 11, 1), date(2025, 2, 26)),
    (date(2025, 5, 1), date(2025, 6, 30)),
)


class OutsideReplication(LookupError):
    """A plane, a day or a data root that is not the replication's."""


def span_days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def replication_days() -> list[date]:
    return [d for first, last in SPANS for d in span_days(first, last)]


class ReplicationAccess:
    """The loader's guard here: Binance BTCFDUSD prints on the replication's days only."""

    def __init__(self) -> None:
        self.days = frozenset(replication_days())

    def require(
        self, instrument: str, days: Iterable[date], plane: str, *, what: str = "read"
    ) -> None:
        if instrument != BINANCE_SPOT or plane != PRINTS:
            raise OutsideReplication(f"the replication reads {BINANCE_SPOT} prints only, not {what} "
                                  f"of {instrument} {plane}")  # fmt: skip
        outside = sorted(d for d in days if d not in self.days)
        if outside:
            raise OutsideReplication(f"{what}: {outside[0]} is not a replication day")


def target_path(day: date, root: Path | str = ROOT) -> Path:
    """Where one converted day lives; refused anywhere but a replication root."""
    path = day_path(BINANCE_PRINTS, day, root)
    parts = Path(root).resolve().parts
    if "replication" not in parts or "spot" in parts:
        raise OutsideReplication(f"{root} is not a replication data root")
    return path


def fetch_day(
    day: date, root: Path | str = ROOT, *, opener: Opener | None = None, timeout: float = 120.0
) -> dict[str, Any]:
    """Fetch, verify, convert and write one day; a day already on disk is not fetched."""
    ReplicationAccess().require(BINANCE_SPOT, [day], PRINTS, what="fetch")
    target = target_path(day, root)
    if target.exists():
        return {"day": day.isoformat(), "status": "on_disk"}
    archive = Path(root) / "archives" / f"BTCFDUSD-aggTrades-{day.isoformat()}.zip"
    checksum = archive.with_name(f"{archive.name}.CHECKSUM")
    open_ = opener or urllib.request.urlopen
    try:
        got = download_to(BINANCE_PRINTS.url(day), archive, timeout=timeout, opener=open_)
        download_to(checksum_url(BINANCE_PRINTS, day), checksum, timeout=timeout, opener=open_)
        payload = archive.read_bytes()
        sha256 = verify(payload, checksum.read_bytes())
        check_zip(archive)
        frame, checks = read(payload, day)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.part")
        frame.to_parquet(partial, compression="zstd", index=False)
        partial.replace(target)
        return {"day": day.isoformat(), "status": "ok", "sha256": sha256, "bytes": got.size,
                "rows": checks.rows, "unit": checks.unit,
                "ids_consecutive": checks.ids_consecutive,
                "not_best_match": checks.not_best_match}  # fmt: skip
    finally:
        archive.unlink(missing_ok=True)
        checksum.unlink(missing_ok=True)
