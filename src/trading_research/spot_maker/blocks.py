"""Who may read which day of the maker study: its blocks, its one plane and its access.

The calendar is the spot study's (``spot/access.py``), reused unchanged and
checked against this study's configuration; the access is this study's own, so
that the spot study's ledger never opens H for it and this study's ledger never
opens H for the spot study. Every reader asks a :class:`MakerAccess` before it
copies or opens a file of a day (``require``, the spot loader's ``AccessGuard``).

* One instrument and one plane: Binance spot BTCFDUSD prints.
* **The development access** permits the reader-test days, W and D; the
  reader-test days alone are also granted for one registered purpose.
* **H and P** are granted only by this study's ledger, which alone applies the
  seal an opened access needs; E, every day of 2025 and the 2025 study's
  reader-test days are never granted.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import access as sa
from trading_research.spot.archives import BINANCE_PRINTS, day_path

INSTRUMENT = sa.BINANCE
PLANE = sa.PRINTS
BLOCKS = sa.BLOCKS
BLOCK_DAYS = sa.BLOCK_DAYS
READER_TEST = sa.READER_TEST
READER_TEST_DAYS = sa.READER_TEST_DAYS
READER_TEST_PURPOSES = sa.READER_TEST_PURPOSES
DEVELOPMENT_BLOCKS: tuple[str, ...] = (READER_TEST, "W", "D")
HELD_OUT: tuple[str, ...] = ("H", "P")
_GRANTABLE = frozenset({*DEVELOPMENT_BLOCKS, *HELD_OUT})
#: The data root under a checkout: ``data/spot``, as the spot study's.
DATA_SUBDIR = Path("data") / "spot"


class _Seal:
    """This study's ledger's seal; pickled by reference, so a worker keeps it."""

    def __reduce__(self) -> str:
        return "_SEAL"


#: Applied only by :mod:`trading_research.spot_maker.ledger`.
_SEAL = _Seal()


def block_of(day: date) -> str | None:
    return sa.block_of(day)


def block_days(name: str) -> list[date]:
    return sa.block_days(name)


def opened_blocks(block: str) -> frozenset[str]:
    """Every block an access opened at ``block`` permits."""
    if block not in HELD_OUT:
        raise ValueError(f"the ledger opens {HELD_OUT}, not {block!r}")
    return frozenset({*DEVELOPMENT_BLOCKS, *HELD_OUT[: HELD_OUT.index(block) + 1]})


@dataclass(frozen=True)
class MakerAccess:
    """Which blocks may be read; ``stamp`` travels with what is made under it."""

    blocks: frozenset[str]
    stamp: str
    session: tuple[str, ...] = ()
    seal: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        unknown = set(self.blocks) - _GRANTABLE
        if unknown:
            raise ValueError(f"unknown blocks {sorted(unknown)}")
        if set(self.blocks) & set(HELD_OUT) and self.seal is not _SEAL:
            raise HeldOutLocked("access to H or P is issued only by the maker study's ledger")
        if self.seal is _SEAL and (
            not self.session
            or self.session[0] not in HELD_OUT
            or self.blocks != opened_blocks(self.session[0])
        ):
            raise HeldOutLocked("a sealed access is a read session's, for exactly its blocks")

    def allows(self, day: date) -> bool:
        if day < sa.FIRST_READABLE_DAY or day in sa.NEVER_READ_DAYS:
            return False
        return block_of(day) in self.blocks

    def require(self, name: str, days: Iterable[date], plane: str, *, what: str = "read") -> None:
        """Raise :class:`HeldOutLocked` unless every day may be read on this plane."""
        if name != INSTRUMENT or plane != PLANE:
            raise ValueError(f"the maker study reads {INSTRUMENT} {PLANE} only, not {name} {plane}")
        for day in days:
            if not self.allows(day):
                raise HeldOutLocked(
                    f"cannot {what} {name} {plane} on {day}: it is in {block_of(day) or 'no block'}"
                    f", and this access ({self.stamp}) permits {sorted(self.blocks)}"
                )


def development_access() -> MakerAccess:
    return MakerAccess(frozenset(DEVELOPMENT_BLOCKS), "development")


def reader_test_access(purpose: str) -> MakerAccess:
    if purpose not in READER_TEST_PURPOSES:
        raise ValueError(f"the reader-test days serve {READER_TEST_PURPOSES}, not {purpose!r}")
    return MakerAccess(frozenset({READER_TEST}), f"reader test ({purpose})")


def data_root(checkout: Path) -> Path:
    """``<checkout>/data/spot``: the root the spot loader reads day files under."""
    return Path(checkout) / DATA_SUBDIR


def day_file(checkout: Path, day: date) -> Path:
    return day_path(BINANCE_PRINTS, day, data_root(checkout))


def files_of_days(checkout: Path, days: Iterable[date]) -> list[Path]:
    """Files under the checkout's data root naming one of ``days`` or a month holding one."""
    return sa.files_of_days(days, Path(checkout) / "data")
