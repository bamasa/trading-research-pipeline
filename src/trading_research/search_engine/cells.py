"""The per-cell driver of stage B: a cell's entries thinned from its tape, its trades looked up.

For one cell on one test day:

1. **Decide.** On the configuration's candidate tape, the cell's own rule
   picks its rows: hold-first at the cell's rate's thresholds (C), or the EV
   gate at the cell's k and beta (R). An EV-gate **event** is a run of
   consecutive book rows open under the cell's own gate, found before the
   market gate and the guard act (``strategies/ev_gate.gate_events``' rule).
   The market gate and the regime guard then remove rows.
2. **Thin.** ``strategies/ev_gate.thin_entries`` takes one position at a time
   (and, in R, one entry per event), jumping from entry to entry. Its
   ``busy_until`` is the hold table's: the next decision may be taken from the
   first row stamped at or after the previous exit, as the executor rules.
   Rows the executor would not fill (outside 00:01 to 23:58, no snapshot to
   enter on, the clip beyond the visible book) are removed first, so they
   neither enter nor block, as in the executor.
3. **Look up.** Each entry's long and short walked P&L comes from the day's
   :class:`~trading_research.search_engine.walk_tables.HoldTable`.

The counters follow the executor's order (outside hours, busy, no snapshot,
beyond the book); in R a free row of an event that already entered is counted
as ``entries_same_event``. Without events the counters and trades equal
``execute_rows`` over the cell's candidate rows. A cell-day without a trade
writes a ledger row of zeros, as the registration's choice reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.search_engine.catalogue import CellKey
from trading_research.search_engine.ledger import LEDGER_COLUMNS
from trading_research.search_engine.tapes import CandidateTape, TapeError
from trading_research.search_engine.walk_tables import (
    HoldTable,
    TableTrades,
    side_index,
    take,
)
from trading_research.strategies.ev_gate import EVGate, thin_entries
from trading_research.strategies.hold_first import HoldFirstRule

NS_PER_S = 1_000_000_000
CELL_COUNTERS: tuple[str, ...] = (
    "decisions",
    "entries",
    "entries_outside_hours",
    "entries_busy",
    "entries_no_snapshot",
    "entries_skipped",
    "entries_same_event",
)


def tape_events(rows: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """An event id per tape row: runs of consecutive book rows open on either
    side; -1 where closed. Rows absent from the tape are closed."""
    rows = np.asarray(rows, dtype=np.int64)
    open_ = np.asarray(sides) != 0
    previous_open = np.r_[False, open_[:-1]]
    adjacent = np.r_[False, np.diff(rows) == 1]
    starts = open_ & ~(previous_open & adjacent)
    return np.where(open_, np.cumsum(starts) - 1, -1).astype(np.int64)


def cell_sides(
    tape: CandidateTape, cell: CellKey, rule: HoldFirstRule | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """The cell's side on each tape row (0 where it does not decide) after its
    market gate and guard, and, in R, each row's event before them."""
    if tape.configuration != cell.configuration:
        raise TapeError(f"{cell.key!r} is not scored from {tape.configuration!r}'s tape")
    events: np.ndarray | None = None
    if cell.stratum == "C":
        if rule is None:
            raise ValueError("a stratum C cell needs its rate's hold-first thresholds")
        if (
            rule.theta_long < tape.loosest["theta_long"]
            or rule.theta_short < tape.loosest["theta_short"]
        ):
            raise TapeError("the cell's thresholds are looser than its tape's")
        side = rule.decide(tape.scores["p_long"], tape.scores["p_short"]).astype(np.int64)
    else:
        gate = EVGate(cell.k, cell.beta_bp, tape.loosest["cost_bp"])
        scores = tape.scores
        side = gate.sides(scores["mu_bp"], scores["sigma_bp"], scores["walk_cost_bp"])
        side = side.astype(np.int64)
        events = tape_events(tape.rows, side)
    if np.any(side * tape.side < 0):
        raise TapeError("a stricter cell turned a tape row's side")
    if cell.gate != "open":
        side[~tape.gates[cell.gate]] = 0
    if cell.regime_h > 0:
        side[tape.guards[cell.regime_h]] = 0
    return side, events


def score_cell(
    tape: CandidateTape, table: HoldTable, cell: CellKey, rule: HoldFirstRule | None = None
) -> TableTrades:
    """One cell's trades on the tape's day, from the day's table at the cell's hold."""
    if tape.day != table.walk.day:
        raise TapeError(f"the tape is {tape.day}'s, the table {table.walk.day}'s")
    if table.hold_ns != round(cell.hold_s * NS_PER_S):
        raise ValueError(f"{cell.key!r} holds {cell.hold_s} s, the table {table.hold_ns} ns")
    side, events = cell_sides(tape, cell, rule)
    rows = tape.rows
    if len(rows) and rows[-1] >= table.rows:
        raise TapeError("the tape's rows run past the day's book")
    return score_sides(table, rows, side, events)


def score_sides(
    table: HoldTable, rows: np.ndarray, side: np.ndarray, events: np.ndarray | None = None
) -> TableTrades:
    """Steps 2 and 3 on decided ``side`` at table ``rows`` (in time order, repeats
    allowed: a repeated row is busy after its first entry), one entry per event
    where ``events`` are given: what :func:`score_cell` does after deciding."""
    rows = np.asarray(rows, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    decided = side != 0
    in_hours = decided & table.allowed[rows]
    fills = table.has_entry[rows]
    within = ~table.skip[side_index(side), rows]
    takeable = np.where(in_hours & fills & within, side, 0)
    busy_until = np.searchsorted(rows, table.busy_until[rows], side="right") - 1
    accepted = thin_entries(takeable, busy_until, events)
    position = np.arange(len(rows))
    last = np.searchsorted(accepted, position, side="left") - 1
    before = accepted[np.maximum(last, 0)] if len(accepted) else np.zeros_like(position)
    has_before = last >= 0
    busy = has_before & (busy_until[before] >= position)
    same_event = np.zeros(len(rows), dtype=bool)
    if events is not None:
        same_event = has_before & (events[before] == events) & (events >= 0)
    taken = np.zeros(len(rows), dtype=bool)
    taken[accepted] = True
    free = in_hours & ~taken & ~busy
    counters = {
        "decisions": int(decided.sum()),
        "entries": len(accepted),
        "entries_outside_hours": int((decided & ~in_hours).sum()),
        "entries_busy": int((in_hours & ~taken & busy).sum()),
        "entries_no_snapshot": int((free & ~fills).sum()),
        "entries_skipped": int((free & fills & ~within & ~same_event).sum()),
        "entries_same_event": int((free & fills & same_event).sum()),
    }
    return take(table, rows[accepted], side[accepted], counters)


def score_configuration(
    tape: CandidateTape,
    table: HoldTable,
    cells: list[CellKey],
    rules: Mapping[float, HoldFirstRule] | None = None,
) -> dict[str, TableTrades]:
    """Every cell of one configuration on the tape's day; ``rules`` maps a C
    cell's trades a day to its fold's hold-first thresholds."""
    out: dict[str, TableTrades] = {}
    for cell in cells:
        rule = None
        if cell.stratum == "C":
            if rules is None or cell.trades_per_day not in rules:
                raise ValueError(f"no hold-first thresholds for {cell.trades_per_day} trades a day")
            rule = rules[cell.trades_per_day]
        out[cell.key] = score_cell(tape, table, cell, rule)
    return out


def ledger_row(run: int, key: str, day: date, trades: TableTrades) -> dict[str, Any]:
    """The trial ledger's row for one cell-day; zeros for a cell without a trade."""
    return {"run": int(run), "key": key, "day": day, **trades.day_values()}


def trade_rows(key: str, table: HoldTable, trades: TableTrades) -> dict[str, np.ndarray]:
    """Per-trade columns Null A sums by hour (``HourCells.from_trades``): ``time``
    is the entry snapshot's stamp in ns (UTC), the executor's ``entry_snapshot_ns``,
    since a trade belongs to the hour of its entry snapshot; ``notional`` is the
    entry's in USDT."""
    return {
        "key": np.full(trades.trades, key, dtype=object),
        "time": table.walk.ts[table.entry_row[trades.rows]],
        "side": trades.side,
        "notional": trades.notional,
        "pnl_long_bp": trades.pnl_long_bp,
        "pnl_short_bp": trades.pnl_short_bp,
    }


def ledger_frame(run: int, day: date, scored: Mapping[str, TableTrades]) -> pd.DataFrame:
    """A day's ledger rows for every scored cell, in the ledger's column order."""
    rows = [ledger_row(run, key, day, trades) for key, trades in scored.items()]
    return pd.DataFrame(rows, columns=list(LEDGER_COLUMNS))
