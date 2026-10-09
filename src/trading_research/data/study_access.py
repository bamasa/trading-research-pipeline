"""Who may read which day of the 2025 study: its blocks, its planes, and the access object.

The study's registration (``docs/preregistration/short_horizon_2025.md``) fixes
its own blocks; round one's and round two's ``block_of`` are not used. Every
reader of the study (:mod:`trading_research.data.study_loader`,
:mod:`trading_research.data.study_fetch`) asks a :class:`StudyAccess` before it
opens or fetches any file of a day, so a refused day costs nothing and reveals
nothing.

Access is granted per **instrument, block and plane**:

* the **development access** (:func:`development_access`) permits W and D in
  every plane (book, prints, funding and hourly bars), the three ``ob200``
  reader-test days and the two 2024 days stage 0 is timed on, and nothing
  else: every 2025 day after 2025-06-30 is refused in every plane, the daily
  bars already on disk included;
* every other access is issued only by the study's ledger
  (:mod:`trading_research.pipeline.short_horizon_ledger`) when a block is
  opened: H with E's hourly bars, then F, then P, each adding to the blocks
  opened before it.

A monthly kline archive holds every day of its month, so a cached one whose
month runs past the last day of the highest opened block puts days on disk
that no access permits: :meth:`StudyAccess.refuse_monthly_bars` refuses while
any is there.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from trading_research.market_making.prereg import HeldOutLocked

#: The study's one instrument.
INSTRUMENTS: tuple[str, ...] = ("BTCUSDT",)

#: The planes a day is read in, named as the fetch log names them.
BOOK, PRINTS, BARS, FUNDING = "book", "prints", "bars", "funding"
PLANES: tuple[str, ...] = (BOOK, PRINTS, BARS, FUNDING)

#: The registered blocks, inclusive, UTC dates, in calendar order.
BLOCKS: dict[str, tuple[date, date]] = {
    "W": (date(2024, 12, 1), date(2024, 12, 31)),
    "D": (date(2025, 1, 1), date(2025, 6, 30)),
    "E": (date(2025, 7, 1), date(2025, 7, 7)),
    "H": (date(2025, 7, 8), date(2025, 8, 20)),
    "F": (date(2025, 8, 21), date(2025, 11, 16)),
    "P": (date(2025, 11, 17), date(2025, 12, 31)),
}
#: Registered day counts, checked against the dates above.
BLOCK_DAYS: dict[str, int] = {"W": 31, "D": 181, "E": 7, "H": 44, "F": 88, "P": 45}

#: O, the descriptive window inside D (A0 and FILL); read as D.
O_WINDOW: tuple[date, date] = (date(2025, 2, 4), date(2025, 2, 26))

#: The ``ob200`` reader-test days, outside every block.
READER_TEST = "reader_test"
READER_TEST_DAYS: tuple[date, ...] = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))

#: The two 2024 days already on disk that stage 0 times the code on.
TIMING = "timing"
TIMING_DAYS: tuple[date, ...] = (date(2024, 2, 11), date(2024, 3, 5))

#: The book archive changes on this day (``ob500`` before, ``ob200`` from it).
ARCHIVE_CHANGE = date(2025, 8, 21)
#: Days whose rolling state is rebuilt from the block's own rows.
COLD_START_DAYS: tuple[date, ...] = (date(2025, 7, 8), date(2025, 8, 21))

#: The blocks the ledger opens, in order, and the (block, plane) pairs each adds.
OPENS: dict[str, frozenset[tuple[str, str]]] = {
    "H": frozenset({("E", BARS)} | {("H", plane) for plane in PLANES}),
    "F": frozenset(("F", plane) for plane in PLANES),
    "P": frozenset(("P", plane) for plane in PLANES),
}
HELD_OUT: tuple[str, ...] = tuple(OPENS)

#: The development access's (block, plane) pairs.
DEVELOPMENT_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (block, plane) for block in ("W", "D", READER_TEST, TIMING) for plane in PLANES
)


def block_of(day: date) -> str | None:
    """The block a day belongs to, ``reader_test``, ``timing``, or None."""
    for name, (start, end) in BLOCKS.items():
        if start <= day <= end:
            return name
    if day in READER_TEST_DAYS:
        return READER_TEST
    if day in TIMING_DAYS:
        return TIMING
    return None


def block_days(name: str) -> list[date]:
    """Every day of a block (or of the reader-test or timing days), in order."""
    if name == READER_TEST:
        return list(READER_TEST_DAYS)
    if name == TIMING:
        return list(TIMING_DAYS)
    start, end = BLOCKS[name]
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def opened_pairs(block: str) -> frozenset[tuple[str, str]]:
    """Every (block, plane) an access opened at ``block`` permits: development's,
    and every opening up to and including ``block``'s."""
    if block not in OPENS:
        raise ValueError(f"the ledger opens {HELD_OUT}, not {block!r}")
    pairs = set(DEVELOPMENT_PAIRS)
    for name in HELD_OUT[: HELD_OUT.index(block) + 1]:
        pairs |= OPENS[name]
    return frozenset(pairs)


def opening_days(block: str) -> dict[str, tuple[date, ...]]:
    """The days of each plane that opening ``block`` adds, which must not be on
    disk before it opens (for H, E's hourly bars too)."""
    out: dict[str, list[date]] = {plane: [] for plane in PLANES}
    for name, plane in sorted(OPENS[block]):
        out[plane] += block_days(name)
    return {plane: tuple(sorted(days)) for plane, days in out.items()}


class _Issuer:
    """The ledger's token; pickled by reference, so it survives a worker process."""

    def __reduce__(self) -> str:
        return "_ISSUER"


#: Issued only by the ledger: an access beyond development must carry it.
_ISSUER = _Issuer()


@dataclass(frozen=True)
class StudyAccess:
    """Which planes of which blocks of which instruments may be read.

    ``grants`` are (instrument, block, plane) triples. ``stamp`` travels with
    everything produced under it: ``development``, ``first read`` or ``second
    read``. ``session`` is the read session an opened access belongs to:
    (block, commit, config sha256, environment fingerprint).
    """

    grants: frozenset[tuple[str, str, str]]
    stamp: str
    session: tuple[str, ...] = ()
    issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        known = set(BLOCKS) | {READER_TEST, TIMING}
        unknown = {
            g
            for g in self.grants
            if g[0] not in INSTRUMENTS or g[1] not in known or g[2] not in PLANES
        }
        if unknown:
            raise ValueError(f"unknown instruments, blocks or planes {sorted(unknown)}")
        beyond = {(block, plane) for _, block, plane in self.grants} - DEVELOPMENT_PAIRS
        if beyond and self.issuer is not _ISSUER:
            raise HeldOutLocked(
                "access beyond W, D, the reader-test and the timing days is issued only by "
                "the study's ledger, which checks the frozen configuration and records the read"
            )

    @property
    def open_through(self) -> date:
        """The last day of the highest block this access opens in any plane (the day
        before W when it opens none)."""
        before_w = BLOCKS["W"][0] - timedelta(days=1)
        ends = [BLOCKS[block][1] for _, block, _ in self.grants if block in BLOCKS]
        return max(ends, default=before_w)

    def allows(self, symbol: str, day: date, plane: str) -> bool:
        return (symbol, block_of(day), plane) in self.grants

    def require(self, symbol: str, days: Iterable[date], plane: str, *, what: str = "read") -> None:
        """Raise :class:`HeldOutLocked` unless every day is permitted in ``plane``."""
        if plane not in PLANES:
            raise ValueError(f"unknown plane {plane!r}; the planes are {PLANES}")
        for day in days:
            if not self.allows(symbol, day, plane):
                block = block_of(day) or "no block"
                permitted = sorted({b for s, b, p in self.grants if s == symbol and p == plane})
                raise HeldOutLocked(
                    f"cannot {what} {symbol} {plane} on {day}: it is in {block}, and this "
                    f"access ({self.stamp}) permits {symbol} {plane} only in {permitted}"
                )

    def refuse_monthly_bars(self, symbol: str, root: Path | str, interval: str = "1h") -> None:
        """Raise while a cached monthly kline archive runs past :attr:`open_through`."""
        from trading_research.data.binance import ArchiveSpec, monthly_archives_beyond

        spec = ArchiveSpec(market="futures-um", kind="klines", symbol=symbol, interval=interval)
        beyond = monthly_archives_beyond(spec, root, self.open_through)
        if beyond:
            raise HeldOutLocked(
                f"monthly kline archives past {self.open_through} are on disk "
                f"({[p.name for p in beyond]}): they hold days no opened block permits"
            )


def grants_for(pairs: Iterable[tuple[str, str]]) -> frozenset[tuple[str, str, str]]:
    """(instrument, block, plane) for every instrument of the study."""
    return frozenset((symbol, block, plane) for symbol in INSTRUMENTS for block, plane in pairs)


def development_access() -> StudyAccess:
    """W and D in every plane, the reader-test and the timing days; nothing else."""
    return StudyAccess(grants_for(DEVELOPMENT_PAIRS), "development")


def resolve(access: StudyAccess | None) -> StudyAccess:
    """The access a reader uses: the one given, or the development access."""
    return development_access() if access is None else access
