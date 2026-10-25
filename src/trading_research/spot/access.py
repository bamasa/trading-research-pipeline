"""Who may read which day of the spot study: its blocks, planes, instruments and access.

The registration (``docs/preregistration/spot_2026.md``, "Blocks" and "Access")
fixes this study's own calendar, one for both venues; nothing here is shared
with the 2025 study's access. Every reader of the study asks a
:class:`SpotAccess` (through :func:`require_day`) before it opens or fetches any
file of a day, so a refused day costs nothing and reveals nothing.

* **Instruments** are keyed by venue and market, never by symbol alone, since
  Bybit's linear perpetual is also ``BTCUSDT``: ``binance-spot:BTCFDUSD``
  (prints only: the venue publishes no spot book) and ``bybit-spot:BTCUSDT``
  (book and prints).
* **The development access** (:func:`development_access`) permits the
  reader-test days, W and D on both instruments' planes, and refuses every
  other day of 2025 and 2026 alike.
* **The reader-test days**, 2026-03-03 to 05, lie outside every block; they
  serve the spot readers' proof, stage 0's timing and the dry run, and nothing
  else (:data:`READER_TEST_PURPOSES`, :func:`reader_test_access`).
* **H and P** are granted only by the study's ledger
  (:mod:`trading_research.spot.ledger`), which alone carries the seal an opened access needs.
* **E** is never fetched and never granted; **no day of 2025** is ever read
  (:data:`FIRST_READABLE_DAY`), and neither are the 2025 study's reader-test days.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from trading_research.market_making.prereg import HeldOutLocked

#: The planes a day is read in.
BOOK, PRINTS = "book", "prints"
PLANES: tuple[str, ...] = (BOOK, PRINTS)


@dataclass(frozen=True)
class Instrument:
    """One venue-qualified instrument and the planes it publishes."""

    id: str
    key: str  # the configuration's key under ``instruments``
    venue: str
    market: str
    symbol: str
    quote: str
    planes: tuple[str, ...]


BINANCE = "binance-spot:BTCFDUSD"
BYBIT = "bybit-spot:BTCUSDT"
INSTRUMENTS: dict[str, Instrument] = {
    BINANCE: Instrument(BINANCE, "binance_spot_BTCFDUSD", "binance", "spot", "BTCFDUSD", "FDUSD",
                        (PRINTS,)),
    BYBIT: Instrument(BYBIT, "bybit_spot_BTCUSDT", "bybit", "spot", "BTCUSDT", "USDT",
                      (BOOK, PRINTS)),
}  # fmt: skip

#: The registered blocks, inclusive, UTC dates, in calendar order.
BLOCKS: dict[str, tuple[date, date]] = {
    "W": (date(2026, 4, 1), date(2026, 4, 30)),
    "D": (date(2026, 5, 1), date(2026, 7, 31)),
    "E": (date(2026, 8, 1), date(2026, 8, 7)),
    "H": (date(2026, 8, 8), date(2026, 10, 21)),
    "P": (date(2026, 11, 1), date(2026, 12, 31)),
}
BLOCK_DAYS: dict[str, int] = {"W": 30, "D": 92, "E": 7, "H": 75, "P": 61}
#: The embargo before P, which holds the registration day.
P_EMBARGO: tuple[date, date] = (date(2026, 10, 22), date(2026, 10, 31))

READER_TEST = "reader_test"
READER_TEST_DAYS: tuple[date, ...] = (date(2026, 3, 3), date(2026, 3, 4), date(2026, 3, 5))
#: What the reader-test days may serve, and nothing else; their labels, fits and
#: scores are discarded and never reported.
READER_TEST_PURPOSES: tuple[str, ...] = ("readers' proof", "stage 0 timing", "dry run")

#: Nothing before this day is ever read: no day of 2025, on either venue.
FIRST_READABLE_DAY = date(2026, 1, 1)
#: The 2025 study's reader-test days, which no successor may put in a block.
NEVER_READ_DAYS: tuple[date, ...] = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))

#: What the development access permits, and the blocks the ledger opens, in order.
DEVELOPMENT_BLOCKS: tuple[str, ...] = (READER_TEST, "W", "D")
HELD_OUT: tuple[str, ...] = ("H", "P")
_GRANTABLE = frozenset({*DEVELOPMENT_BLOCKS, *HELD_OUT})


def instrument(name: str) -> Instrument:
    """The instrument with this venue-qualified id; a bare symbol is refused."""
    if name not in INSTRUMENTS:
        raise ValueError(
            f"unknown instrument {name!r}: instruments are keyed by venue and market "
            f"({sorted(INSTRUMENTS)}), never by symbol alone"
        )
    return INSTRUMENTS[name]


def block_of(day: date) -> str | None:
    """The block a day belongs to, ``reader_test``, or None."""
    for name, (start, end) in BLOCKS.items():
        if start <= day <= end:
            return name
    return READER_TEST if day in READER_TEST_DAYS else None


def block_days(name: str) -> list[date]:
    """Every day of a block (or the reader-test days), in order."""
    if name == READER_TEST:
        return list(READER_TEST_DAYS)
    start, end = BLOCKS[name]
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def opened_blocks(block: str) -> tuple[str, ...]:
    """Every block an access opened at ``block`` permits: development's, then H, then P."""
    if block not in HELD_OUT:
        raise ValueError(f"the ledger opens {HELD_OUT}, not {block!r}")
    return (*DEVELOPMENT_BLOCKS, *HELD_OUT[: HELD_OUT.index(block) + 1])


def grants_for(blocks: Iterable[str]) -> frozenset[tuple[str, str, str]]:
    """(instrument, block, plane) for every instrument and each plane it publishes."""
    return frozenset(
        (i.id, block, plane) for i in INSTRUMENTS.values() for block in blocks for plane in i.planes
    )


class _Seal:
    """The ledger's seal; pickled by reference, so it survives a worker process."""

    def __reduce__(self) -> str:
        return "_SEAL"


#: Applied only by the ledger: an access to H or P must carry it.
_SEAL = _Seal()


@dataclass(frozen=True)
class SpotAccess:
    """Which planes of which blocks of which instruments may be read.

    ``grants`` are (instrument, block, plane) triples. ``stamp`` travels with
    everything produced under it: ``development``, ``reader test (<purpose>)``,
    ``first read`` or ``second read``. ``session`` is the read session an opened
    access belongs to: (block, commit, configuration sha256, fingerprint).
    """

    grants: frozenset[tuple[str, str, str]]
    stamp: str
    session: tuple[str, ...] = ()
    seal: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        unknown = {
            g
            for g in self.grants
            if g[0] not in INSTRUMENTS
            or g[1] not in _GRANTABLE
            or g[2] not in INSTRUMENTS[g[0]].planes
        }
        if unknown:
            raise ValueError(f"unknown instruments, blocks or planes {sorted(unknown)}")
        if any(block in HELD_OUT for _, block, _ in self.grants) and self.seal is not _SEAL:
            raise HeldOutLocked(
                "access to H or P is issued only by the spot study's ledger, which checks "
                "the frozen configuration and records the read"
            )

    def allows(self, name: str, day: date, plane: str) -> bool:
        if day < FIRST_READABLE_DAY or day in NEVER_READ_DAYS:
            return False
        return (name, block_of(day), plane) in self.grants

    def require(self, name: str, days: Iterable[date], plane: str, *, what: str = "read") -> None:
        """Raise :class:`HeldOutLocked` unless every day is permitted in ``plane``.

        An unknown instrument (a bare symbol among them) or a plane the venue
        does not publish is a :class:`ValueError`.
        """
        if plane not in instrument(name).planes:
            raise ValueError(f"{name} publishes {instrument(name).planes}, not {plane!r}")
        for day in days:
            if day < FIRST_READABLE_DAY:
                raise HeldOutLocked(
                    f"cannot {what} {name} {plane} on {day}: no day of 2025 is read"
                )
            if not self.allows(name, day, plane):
                block = block_of(day) or "no block"
                permitted = sorted({b for i, b, p in self.grants if i == name and p == plane})
                raise HeldOutLocked(
                    f"cannot {what} {name} {plane} on {day}: it is in {block}, and this access "
                    f"({self.stamp}) permits {name} {plane} only in {permitted}"
                )


def development_access() -> SpotAccess:
    """The reader-test days, W and D on both instruments' planes; nothing else."""
    return SpotAccess(grants_for(DEVELOPMENT_BLOCKS), "development")


def reader_test_access(purpose: str) -> SpotAccess:
    """The reader-test days only, stamped with one of their registered purposes."""
    if purpose not in READER_TEST_PURPOSES:
        raise ValueError(f"the reader-test days serve {READER_TEST_PURPOSES}, not {purpose!r}")
    return SpotAccess(grants_for((READER_TEST,)), f"reader test ({purpose})")


def resolve(access: SpotAccess | None) -> SpotAccess:
    """The access a reader uses: the one given, or the development access."""
    return development_access() if access is None else access


def require_day(
    access: SpotAccess | None, name: str, day: date, plane: str, *, what: str = "read"
) -> SpotAccess:
    """What every reader calls before it opens or fetches a file of ``day``."""
    found = resolve(access)
    found.require(name, (day,), plane, what=what)
    return found


# ---------------------------------------------------------------------------
# Where the study's files live
# ---------------------------------------------------------------------------

#: The study's data root, under the repository's ``data/``.
SPOT_DATA = Path("spot")
_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2})(-\d{2})?(?!\d)")


def data_root(name: str, plane: str, base: Path | str = Path("data")) -> Path:
    """``data/spot/<venue>/<plane>/<symbol>/``: every spot file lives under it."""
    found = instrument(name)
    if plane not in found.planes:
        raise ValueError(f"{name} publishes {found.planes}, not {plane!r}")
    return Path(base) / SPOT_DATA / found.venue / plane / found.symbol


def require_spot_path(path: Path | str, base: Path | str = Path("data")) -> Path:
    """``path`` if it lies under the study's data root; refuse anything else, the
    2025 study's ``data/{book,trades,bars,funding}/BTCUSDT`` among them."""
    root = (Path(base) / SPOT_DATA).resolve()
    target = Path(path).resolve()
    if target != root and root not in target.parents:
        raise HeldOutLocked(f"{path} lies outside the spot study's data root {root}")
    return Path(path)


def files_of_days(days: Iterable[date], base: Path | str = Path("data")) -> list[Path]:
    """Files under the study's data root whose path names one of ``days`` (an ISO
    date) or a month holding one of them (``YYYY-MM`` without a day)."""
    root = Path(base) / SPOT_DATA
    wanted = set(days)
    if not root.exists() or not wanted:
        return []
    named = {d.isoformat() for d in wanted}
    months = {d.isoformat()[:7] for d in wanted}
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        for month, day in _DATE.findall(path.relative_to(root).as_posix()):
            if (day and month + day in named) or (not day and month in months):
                found.append(path)
                break
    return found
