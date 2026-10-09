"""Null B's walk-table route: each source day's entries executed on the target day's tables.

Null B re-pairs days: under shift ``s`` the source day ``i`` of the circular
calendar order of D's test days executes its accepted entries on target day
``(i + s) mod n``. An entry is its time of day and side, after its own day's
gate, guard and thinning (stage B's accepted entries, :func:`entries_of`). On
the target day it is decided at the first row stamped at or after the same
time of day (never earlier than the time it stands for) and run through the
executor's decision loop on that day's hold table
(:func:`~trading_research.search_engine.walk_tables.execute_rows`): an entry
that falls while the mapped previous position is still open is skipped and
counted (``entries_busy``), as is one the target's book cannot fill. A pair
whose source or target day is excluded contributes nothing and is counted
(``validation.placebo_search.null_b_pairs``).

The work is organised by target day, so each day's walk tables are built once
for all 137 shifts: :meth:`NullBRoute.add_target_day` fills every shift's
column for that target, and :meth:`NullBRoute.shifted` is the
``shifted(s) -> CellDays`` callable ``placebo_search.null_b`` takes, its
columns the contributing pairs in target order. With ``directory`` the arrays
live in ``.npy`` files there, so target days can be filled by separate
processes and a stopped run resumed.

:func:`stage0_parity` is the registered parity test: decisions scored from
walk tables equal the executor's, and at ``s = 0`` the route equals the
executor entry by entry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trading_research.backtest.taker import TakerBook, TakerSettings, execute_day
from trading_research.search_engine.walk_tables import (
    HoldTable,
    Parity,
    TableTrades,
    WalkTable,
    entry_parity,
    execute_rows,
    hold_table,
)
from trading_research.validation.choice import CellDays, ChoiceError
from trading_research.validation.placebo_search import null_b_pairs, null_b_shifts


@dataclass(frozen=True)
class Entries:
    """One cell's accepted entries on one day: time of day (ns) and side."""

    tod_ns: np.ndarray
    side: np.ndarray

    def __post_init__(self) -> None:
        if self.tod_ns.shape != self.side.shape:
            raise ValueError("one side per entry")
        if len(self.tod_ns) and np.any(np.diff(self.tod_ns) < 0):
            raise ValueError("entries must be in time order")


def entries_of(table: HoldTable, trades: TableTrades) -> Entries:
    """A cell-day's accepted entries as Null B carries them to another day."""
    tod = table.walk.ts[trades.rows] - table.walk.start_ns
    return Entries(tod.astype(np.int64), trades.side.astype(np.int64))


def replay(table: HoldTable, entries: Entries) -> TableTrades:
    """Execute another day's entries on ``table``'s day at the same times of day.

    An entry after the target's last row cannot be decided there and counts as
    ``entries_no_snapshot``."""
    ts = table.walk.ts
    rows = np.searchsorted(ts, table.walk.start_ns + entries.tod_ns, side="left")
    inside = rows < table.rows
    result = execute_rows(table, rows[inside], entries.side[inside])
    result.counters["decisions"] += int((~inside).sum())
    result.counters["entries_no_snapshot"] += int((~inside).sum())
    return result


def _array(
    directory: Path | None, name: str, shape: tuple[int, ...], dtype: type, fill: object
) -> np.ndarray:
    if directory is None:
        return np.full(shape, fill, dtype=dtype)
    path = Path(directory) / f"{name}.npy"
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"{path.name} holds {array.shape} {array.dtype}, not {shape} {dtype}")
        return array
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[...] = fill
    return array


class NullBRoute:
    """Every shift's cell-by-day outcomes, filled one target day at a time."""

    def __init__(
        self,
        cells: Sequence[str],
        holds_ns: Sequence[int],
        usable: np.ndarray,
        *,
        shifts: range | None = None,
        directory: Path | None = None,
    ) -> None:
        self.cells = tuple(cells)
        self.holds_ns = np.asarray(holds_ns, dtype=np.int64)
        self.usable = np.asarray(usable, dtype=bool)
        if self.holds_ns.shape != (len(self.cells),):
            raise ValueError("one hold per cell")
        days = len(self.usable)
        self.shifts = shifts if shifts is not None else null_b_shifts(days)
        shape = (len(self.shifts), len(self.cells), days)
        self.trades = _array(directory, "trades", shape, np.int32, 0)
        self.net_bp = _array(directory, "net_bp", shape, np.float64, np.nan)
        self.g_bp = _array(directory, "g_bp", shape, np.float64, np.nan)
        #: Entries skipped because the mapped previous position was open, per
        #: shift and target day, summed over cells.
        self.busy = _array(directory, "busy", (len(self.shifts), days), np.int64, 0)
        self.done = _array(directory, "done", (days,), np.bool_, False)
        self._entries: dict[int, Mapping[int, Entries]] = {}

    def add_source_day(self, day: int, entries: Mapping[int, Entries]) -> None:
        """A source day's accepted entries by cell position; a cell without an
        entry that day is absent or empty."""
        if not 0 <= day < len(self.usable):
            raise ValueError(f"day {day} is not a position among {len(self.usable)} days")
        self._entries[day] = dict(entries)

    def excluded(self, shift: int) -> int:
        """Pairs under ``shift`` with an excluded source or target day."""
        return null_b_pairs(self.usable, shift).excluded

    def add_target_day(self, target: int, tables: Mapping[int, HoldTable]) -> None:
        """Fill every shift's column for ``target`` from its tables (by hold in ns);
        filling a target again replaces what it held."""
        n = len(self.usable)
        self.trades[:, :, target] = 0
        self.net_bp[:, :, target] = np.nan
        self.g_bp[:, :, target] = np.nan
        self.busy[:, target] = 0
        if self.usable[target]:
            for si, shift in enumerate(self.shifts):
                source = (target - shift) % n
                if not self.usable[source]:
                    continue
                if source not in self._entries:
                    raise ValueError(f"source day {source}'s entries were not added")
                for cell, entries in self._entries[source].items():
                    if len(entries.tod_ns) == 0:
                        continue
                    result = replay(tables[int(self.holds_ns[cell])], entries)
                    values = result.day_values()
                    self.trades[si, cell, target] = values["trades"]
                    if result.trades:
                        self.net_bp[si, cell, target] = values["net_bp"]
                        self.g_bp[si, cell, target] = values["G_bp"]
                    self.busy[si, target] += result.counters["entries_busy"]
        self.done[target] = True
        for array in (self.trades, self.net_bp, self.g_bp, self.busy, self.done):
            if isinstance(array, np.memmap):
                array.flush()

    def shifted(self, shift: int) -> CellDays:
        """``placebo_search.null_b``'s callable: one column per contributing pair."""
        pairs = null_b_pairs(self.usable, shift)
        if not self.done[pairs.target].all():
            raise ChoiceError(f"shift {shift}: some target days are not filled yet")
        si = self.shifts.index(shift)
        cols = pairs.target
        return CellDays(
            np.asarray(self.trades[si][:, cols], dtype=np.int64),
            np.asarray(self.net_bp[si][:, cols]),
            np.asarray(self.g_bp[si][:, cols]),
        )


def stage0_parity(
    book: TakerBook,
    rows: np.ndarray,
    side: np.ndarray,
    hold_ns: int,
    settings: TakerSettings | None = None,
) -> Parity:
    """The registered parity test on one day's book.

    Decisions at ``rows`` (time order) with sides +1 / -1 are scored from walk
    tables and by ``backtest.taker.execute_day``; then the entries taken are
    carried through Null B's route at ``s = 0`` (to their own day) and must
    equal the executor's trades on them, entry by entry, to the last bit."""
    settings = settings or TakerSettings()
    table = hold_table(WalkTable.from_book(book, settings.clip), hold_ns, settings)
    ordinary = entry_parity(book, rows, side, table)
    taken = execute_rows(table, rows, side)
    routed = replay(table, entries_of(table, taken))
    trades = execute_day(book, book.ts[taken.rows], taken.side, hold_ns, settings).trades
    out = [f"ordinary: {m}" for m in ordinary.mismatches]
    if routed.trades != len(trades) or not np.array_equal(routed.rows, taken.rows):
        out.append(f"s = 0: {routed.trades} trades routed, executor {len(trades)}")
    else:
        for column, values in (("net", routed.net), ("notional", routed.notional),
                               ("net_bp", routed.net_bp), ("side", routed.side)):  # fmt: skip
            if not np.array_equal(trades[column].to_numpy(), values):
                out.append(f"s = 0: {column} differs")
    if routed.counters["entries_busy"] or routed.counters["entries_skipped"]:
        out.append("s = 0: an accepted entry was skipped on its own day")
    return Parity(ordinary.trades, tuple(out))
