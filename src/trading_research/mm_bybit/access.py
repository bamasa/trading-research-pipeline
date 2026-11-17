"""The blocks, and who may read which day.

Every function that opens or fetches a day of market data passes the day
through :meth:`Access.require` first. :func:`development_access` permits the
reader-test days, W and D; E is never permitted; H only through the ledger
(:mod:`trading_research.mm_bybit.ledger`), which is the one place an access for
H is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

#: The registered blocks, inclusive (``configs/mm_bybit_2021.yaml``).
BLOCKS: dict[str, tuple[date, date]] = {
    "W": (date(2020, 12, 15), date(2020, 12, 31)),
    "D": (date(2021, 1, 1), date(2021, 3, 31)),
    "E": (date(2021, 4, 1), date(2021, 4, 7)),
    "H": (date(2021, 4, 8), date(2021, 6, 30)),
}
READER_TEST_DAYS: tuple[date, ...] = (date(2020, 12, 15), date(2020, 12, 16))
STRESS_DAYS: tuple[date, ...] = (date(2021, 5, 19),)


class AccessRefused(PermissionError):
    """A day the current access does not permit."""


def block_of(day: date) -> str | None:
    for name, (first, last) in BLOCKS.items():
        if first <= day <= last:
            return name
    return None


def block_days(name: str) -> list[date]:
    first, last = BLOCKS[name]
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


@dataclass(frozen=True)
class Access:
    """Which blocks may be read, and why (the stamp carried into results)."""

    blocks: frozenset[str]
    purpose: str
    second_read: bool = False

    def permits(self, day: date) -> bool:
        if "E" in self.blocks:
            return False
        name = block_of(day)
        return name is not None and name in self.blocks

    def require(self, day: date) -> None:
        if not self.permits(day):
            raise AccessRefused(f"{day} ({block_of(day) or 'outside every block'}) under {self}")


def development_access() -> Access:
    """The reader-test days, W and D; nothing else."""
    return Access(frozenset({"W", "D"}), "development")


def _heldout_access(*, second_read: bool = False) -> Access:
    """H's access. Called only by :func:`trading_research.mm_bybit.ledger.open_h`,
    after its checks pass and its entry is committed."""
    return Access(frozenset({"H"}), "held-out read", second_read)
