"""Bybit's spot prints: the parser of ``public.bybit.com/spot/<symbol>/<symbol>_<day>.csv.gz``.

New code, never :func:`trading_research.data.bybit_trades.parse`: the spot
archive's layout is not the linear one. The format probe of 2026-10-25
(``docs/spot_2026_format_probe.md``) found a header
``id,timestamp,price,volume,side,rpi``:

- ``id``: the day's print number, an integer;
- ``timestamp``: integer **milliseconds** (the linear archive writes float seconds);
- ``price``, ``volume``: the price and the size in BTC;
- ``side``: the aggressor's side, ``buy`` or ``sell``, in lower case;
- ``rpi``: 1 for a print against a retail-price-improvement order, which no
  order of the study can reach; 0 otherwise.

The parser holds the archive to that contract and raises
:class:`BybitPrintsError` on anything else (another header, a side string it
does not know, a stamp that is not a millisecond of 2020 to 2099, an ``rpi``
other than 0 or 1), so a change of layout makes a day unreadable instead of
silently inverting a side or scaling a time. :func:`read` also returns what
stage 0 records about the day (:class:`PrintChecks`).
"""

from __future__ import annotations

import gzip
import io
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data import fetch_log as fl
from trading_research.data.failures import Opener
from trading_research.data.study_fetch import DayOutcome
from trading_research.spot import fetch, prints
from trading_research.spot.archives import BYBIT_PRINTS, BYBIT_SPOT, PRINTS, day_path

HEADER: tuple[str, ...] = ("id", "timestamp", "price", "volume", "side", "rpi")
#: The aggressor's side as the archive writes it, and its sign.
SIDES: dict[str, int] = {"buy": 1, "sell": -1}
#: Milliseconds of 2020-01-01 and 2100-01-01: a stamp outside is not in milliseconds.
MS_BOUNDS: tuple[int, int] = (1_577_836_800_000, 4_102_444_800_000)

_DTYPES = {
    "id": "int64",
    "timestamp": "int64",
    "price": "float64",
    "volume": "float64",
    "side": "string",
    "rpi": "int64",
}


class BybitPrintsError(prints.PrintContractError):
    """The archive does not have the spot print layout the parser was written for."""


@dataclass(frozen=True)
class PrintChecks:
    """What a day's archive showed, for stage 0's record."""

    rows: int
    sides: dict[str, int]
    rpi: int
    #: Every ``id`` one more than the row before it, in the archive's order.
    ids_consecutive: bool
    #: Stamps never decrease in the archive's order.
    stamps_ordered: bool

    @property
    def rpi_share(self) -> float:
        return self.rpi / self.rows if self.rows else 0.0


def read(raw: bytes) -> tuple[pd.DataFrame, PrintChecks]:
    """Parse one day's decompressed CSV into the print contract, with its checks."""
    head = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
    if tuple(head.split(",")) != HEADER:
        raise BybitPrintsError(f"header {head!r}, expected {','.join(HEADER)!r}")
    try:
        frame = pd.read_csv(io.BytesIO(raw), dtype=_DTYPES)
    except ValueError as exc:  # a value of the wrong type
        raise BybitPrintsError(f"a column does not parse as its type: {exc}") from exc
    if frame.empty:  # a header and no row: a day without a print, which the loader excludes
        return prints.empty(prints.BYBIT_SOURCE), PrintChecks(
            0, dict.fromkeys(SIDES, 0), 0, True, True
        )
    unknown = sorted(set(frame["side"].dropna().unique()) - set(SIDES))
    if unknown or frame["side"].isna().any():
        raise BybitPrintsError(f"side strings {unknown or ['<missing>']} are not {sorted(SIDES)}")
    stamps = frame["timestamp"].to_numpy()
    if stamps.min() < MS_BOUNDS[0] or stamps.max() >= MS_BOUNDS[1]:
        raise BybitPrintsError("a timestamp is not in milliseconds of 2020 to 2099")
    if not frame["rpi"].isin((0, 1)).all():
        raise BybitPrintsError("an rpi flag is neither 0 nor 1")
    ids = frame["id"].to_numpy()
    checks = PrintChecks(
        rows=len(frame),
        sides={side: int((frame["side"] == side).sum()) for side in SIDES},
        rpi=int(frame["rpi"].sum()),
        ids_consecutive=bool(np.all(np.diff(ids) == 1)),
        stamps_ordered=bool(np.all(np.diff(stamps) >= 0)),
    )
    out = prints.assemble(
        pd.to_datetime(frame["timestamp"], unit="ms", utc=True),
        frame["price"],
        frame["volume"],
        frame["side"].map(SIDES).to_numpy(dtype=np.int8),
        frame["id"],
        frame["rpi"].to_numpy() == 1,
        prints.BYBIT_SOURCE,
    )
    return out, checks


def parse(raw: bytes) -> pd.DataFrame:
    """One day's decompressed CSV as a print-contract frame."""
    return read(raw)[0]


def read_archive(path: Path | str) -> tuple[pd.DataFrame, PrintChecks]:
    """Decompress and parse one day's ``.csv.gz``."""
    return read(gzip.decompress(Path(path).read_bytes()))


def fetch_day(
    day: date,
    root: Path | str,
    archive_dir: Path | str,
    *,
    access: fetch.AccessGuard,
    opener: Opener | None = None,
    timeout: float = 120.0,
) -> DayOutcome:
    """Fetch, decompress, parse and write one day of prints; delete the archive after."""
    access.require(BYBIT_SPOT, [day], PRINTS, what="fetch")
    name = BYBIT_PRINTS.name
    archive = Path(archive_dir) / f"{name}-{day.isoformat()}.csv.gz"
    got, missed = fetch.download(
        name, day, BYBIT_PRINTS.url(day), archive, opener=opener, timeout=timeout
    )
    if got is None:
        assert missed is not None
        return missed
    try:
        try:
            raw = gzip.decompress(archive.read_bytes())
        except fetch.UNREADABLE_ARCHIVE as exc:
            return fetch.failed(name, day, fl.CORRUPT, exc, got)
        try:
            frame, checks = read(raw)
        except (prints.PrintContractError, KeyError, TypeError) as exc:
            return fetch.failed(name, day, fl.REPLAY_FAILED, exc, got)
        del raw
        fetch.write_parquet(frame, day_path(BYBIT_PRINTS, day, root))
        return fetch.done(name, day, got, {**asdict(checks), "rpi_share": checks.rpi_share})
    finally:
        archive.unlink(missing_ok=True)
