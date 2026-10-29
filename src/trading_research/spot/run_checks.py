"""The void conditions on every run of a row's day, not on its true run alone.

Registered ("Kill conditions per slot"): K-units on every fill of every run,
K-acct in any run of the slot's read, K-cap and K-inv at any event of every
run. A row's day runs its executor many times (the true run, the 110 ms re-run,
the brackets, the long-only variant, the neighbourhood cells) and looks up
more runs in walk tables (the placebos, K-drift's runs, Null A's flips, the
one-day shift). :class:`Voids` gathers the conditions every one of them fires:

* :meth:`Voids.run` calls an executor run and maps a fee that differs from the
  table (``FeeUnitsError``) to K-units and a broken identity
  (``IdentityError``) to K-acct, returning None for that run;
* :meth:`Voids.account` replays a run's fills on the spot account
  (:func:`trading_research.spot.account.account_day`) and adds its K-acct,
  K-cap and K-inv;
* :func:`print_table_fills` and :func:`book_table_fills` give a walk-table
  run's fills, each fill's fee checked against a fresh look-up of the fee
  table at the fill's stamp (K-units), so the account is replayed on them too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np

from trading_research.market_making.accounting import IdentityError
from trading_research.search_engine.walk_tables import HoldTable, TableTrades
from trading_research.spot import account as acc
from trading_research.spot.costs import (
    FeeTable,
    FeeUnitsError,
    book_fill_fee_bp,
    check_fill_fees,
    fill_fee,
)
from trading_research.spot.fills import PrintTape
from trading_research.spot.walk_tables import PrintHoldTable, PrintTableTrades, side_index

T = TypeVar("T")


@dataclass
class Voids:
    """The void conditions a row's day fired in any of its runs."""

    tape: PrintTape
    fired: set[str] = field(default_factory=set)

    def run(self, call: Callable[[], T]) -> T | None:
        """``call()``, or None with K-units or K-acct recorded when it raises one."""
        try:
            return call()
        except FeeUnitsError:
            self.fired.add("K-units")
        except IdentityError:
            self.fired.add("K-acct")
        return None

    def account(self, fills: acc.Fills | None, net: float) -> None:
        """The spot account's K-acct, K-cap and K-inv on a run's fills."""
        if fills is None or not len(fills.ts):
            return
        try:
            found = acc.account_day(fills, net, *acc.first_last_print(self.tape))
        except ValueError:  # fills out of time order: positions overlapped
            self.fired.add("K-cap")
            return
        self.fired.update(found.void)

    def sorted(self) -> list[str]:
        from trading_research.spot.statuses import VOID_CONDITIONS

        return sorted(self.fired, key=VOID_CONDITIONS.index)


def _interleave(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    out = np.empty(2 * len(entry), dtype=np.result_type(entry, exit_))
    out[0::2], out[1::2] = entry, exit_
    return out


def _fills(
    side: np.ndarray, entry_ns: np.ndarray, exit_ns: np.ndarray, entry_px: np.ndarray,
    exit_px: np.ndarray, clip: float, entry_bp: np.ndarray, exit_bp: np.ndarray,
) -> acc.Fills:  # fmt: skip
    size = np.full(len(side), clip)
    return acc.Fills(_interleave(entry_ns, exit_ns), _interleave(side, -side),
                     _interleave(entry_px, exit_px), _interleave(size, size),
                     _interleave(entry_bp, exit_bp))  # fmt: skip


def print_table_fills(
    table: PrintHoldTable, found: PrintTableTrades, fees: FeeTable
) -> tuple[acc.Fills, float]:
    """A print walk-table run's fills and net, each fill's fee checked (K-units)."""
    own, rows = side_index(found.side), found.rows
    settings = table.settings
    entry_ns, exit_ns = table.entry_print_ns[own, rows], table.free_ns[own, rows]
    entry_px, exit_px = table.entry_price[own, rows], table.exit_price[own, rows]
    entry_bp = fees.taker_bp(settings.instrument, settings.tier, entry_ns)
    exit_bp = fees.taker_bp(settings.instrument, settings.tier, exit_ns)
    clip = settings.clip
    expected = fill_fee(entry_px * clip, entry_bp) + fill_fee(exit_px * clip, exit_bp)
    check_fill_fees(expected, table.fees[own, rows])
    fills = _fills(found.side, entry_ns, exit_ns, entry_px, exit_px, clip, entry_bp, exit_bp)
    return fills, float(found.net.sum())


def book_table_fills(
    table: HoldTable, found: TableTrades, fees: FeeTable, instrument: str, tier: str
) -> tuple[acc.Fills, float]:
    """A book walk-table run's fills and net (no funding on spot), each fill's fee
    checked against the day's fee a fresh look-up gives (K-units)."""
    walk, own, rows = table.walk, side_index(found.side), found.rows
    entry_row, exit_row = table.entry_row[rows], table.exit_row[rows]
    long = found.side > 0
    entry_px = np.where(long, walk.buy_vwap[entry_row], walk.sell_vwap[entry_row])
    exit_px = np.where(long, walk.sell_vwap[exit_row], walk.buy_vwap[exit_row])
    entry_ns, exit_ns = walk.ts[entry_row], table.free_ns[rows]
    entry_bp = book_fill_fee_bp(fees, instrument, tier, entry_ns)
    exit_bp = book_fill_fee_bp(fees, instrument, tier, exit_ns)
    clip = walk.clip
    expected = fill_fee(entry_px * clip, entry_bp) + fill_fee(exit_px * clip, exit_bp)
    gross = found.side * (exit_px - entry_px) * clip
    check_fill_fees(expected, gross - table.net[own, rows])
    fills = _fills(found.side, entry_ns, exit_ns, entry_px, exit_px, clip, entry_bp, exit_bp)
    return fills, float(found.net.sum())
