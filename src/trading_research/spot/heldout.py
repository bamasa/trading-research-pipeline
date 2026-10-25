"""A block's read: every row's runs on each usable day, the one-day shift, the calibration.

Everything is read inside the access the read was given (the ledger's sealed
access on H and P; the development access on D's test months and on the
rehearsal) with only the values the plan carries: each row's fits and
thresholds (:class:`~trading_research.spot.runs.RowModels`), the gap threshold,
the walk depth and the instruments' specification. For each day
(:func:`read_day`):

* on Binance's usable days, S25-PF and S25-PFT on Binance's print grid
  (:func:`trading_research.spot.runs_prints.print_day`);
* on Bybit's usable days, S25-BB and S25-T on its book
  (:func:`trading_research.spot.runs_book.book_day`), S25-PB on its prints, and
  the calibration: S25-PB's decisions priced by the book walk, FILL-S (S25-PB's
  and S25-BB's book-walk orders priced by the print rule and every bracket on
  the full prints), FILL-S thinned (S25-PB's orders on Bybit's prints thinned to
  Binance's count that day, seeds 0 to 9, when Binance's day is usable too), its
  capacity on the thinned prints, and S25-BB's decisions by the print rule;
* a Bybit day that any book run flags ``flatten_stale`` is excluded on Bybit;
* on a day Bybit's book is excluded for its sequence gaps alone, S25-BB's and
  S25-T's true runs with the threshold lifted, kept apart (the verdict with the
  gap-excluded days put back, a measurement).

**The one-day shift**: each kept day takes the previous kept day's signal tape
of the same row at the same times of day; the block's first kept day takes its
last kept day's (circularly over the kept days), computed once every day is
read. Each day's record is written under ``reads/<label>/days`` as it is
finished, so a read that stops resumes at the next day.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.features.timed import stamps_ns
from trading_research.labels.taker_vwap import forward_move_bp as book_forward
from trading_research.spot import access as sa
from trading_research.spot import days as dy
from trading_research.spot import loader as ld
from trading_research.spot.calibration import (
    book_orders,
    capacity_ratio,
    price_orders,
    thin_prints,
    thinned_calibration,
    trade_fills,
)
from trading_research.spot.context import SpotStudy
from trading_research.spot.costs import FeeTable, book_settings, execute_book_day, taker_book
from trading_research.spot.plan import OWN_CELL, ROWS
from trading_research.spot.print_label import forward_move_bp as print_forward
from trading_research.spot.print_taker import execute_print_day, print_settings
from trading_research.spot.runs import RowModels, tier_values
from trading_research.spot.runs_book import book_day
from trading_research.spot.runs_book import shifted as book_shifted
from trading_research.spot.runs_prints import print_day
from trading_research.spot.runs_prints import shifted as print_shifted
from trading_research.spot.state import plain, write_json

VENUE_ROWS: dict[str, tuple[str, ...]] = {
    sa.BINANCE: ("S25-PF", "S25-PFT"),
    sa.BYBIT: ("S25-BB", "S25-T", "S25-PB"),
}
PUT_BACK_ROWS: tuple[str, ...] = ("S25-BB", "S25-T")


@dataclass(frozen=True)
class ReadPlan:
    """What one read computes: its days, its access and every frozen input."""

    label: str
    block: str
    days: tuple[date, ...]
    access: Any
    models: Mapping[str, RowModels]
    max_gaps: int | None
    depth: int
    spec: Mapping[str, tuple[float, float]] | None = None
    #: Null A's hour-block sign flips per day (on D only).
    null_a: int = 0


def loader(study: SpotStudy, plan: ReadPlan) -> Callable[[date], dy.DayInputs]:
    """Load one day of both venues inside the plan's access, with every fetch log's
    terminal statuses and the fee table's exclusions."""
    logs = [study.fetch_log(s) for s in range(4) if study.records(f"fetch_stage{s}.json").exists()]
    table = study.fee_table

    def load(day: date) -> dy.DayInputs:
        return dy.load_day(day, study.spot_root, access=plan.access, logs=logs, table=table,
                           max_sequence_gaps=plan.max_gaps, depth=plan.depth, spec=plan.spec)  # fmt: skip

    return load


def _probas(
    models: RowModels, frame: pd.DataFrame, scored_from: int
) -> dict[tuple[float, float], np.ndarray]:
    """Every fit's probabilities on the day; a fit without a model never decides."""
    out = {
        fit: dy.predict(m, frame, models.layout, scored_from) for fit, m in models.models.items()
    }
    for fit in {c.fit for c in models.rules}:
        out.setdefault(fit, np.full((len(frame), 3), np.nan))
    return out


def binance_rows(
    plan: ReadPlan, inputs: dy.DayInputs, table: FeeTable
) -> tuple[dict[str, Any], dict[str, Any]]:
    """S25-PF's and S25-PFT's records on Binance's day, and their walk tables."""
    frame = dy.store_frame("binance", inputs)
    grid = stamps_ns(frame)
    tape = dy.tape(inputs, sa.BINANCE)
    forward = {OWN_CELL.hold_s: print_forward(tape, grid, OWN_CELL.hold_s)}
    records, tables = {}, {}
    for name in VENUE_ROWS[sa.BINANCE]:
        probas = _probas(plan.models[name], frame, inputs.scored_from(sa.BINANCE))
        records[name], tables[name] = print_day(
            ROWS[name], plan.models[name], tape, grid, probas, forward, inputs.binance.frame,
            table, day=inputs.day, null_a=plan.null_a,
        )  # fmt: skip
    return records, tables


def bybit_rows(
    plan: ReadPlan, inputs: dy.DayInputs, table: FeeTable, *, rows: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pd.DataFrame]]:
    """The named Bybit rows' records on the day, their tables, and the book rows'
    true trades (for FILL-S)."""
    book_day_ = inputs.book
    book = taker_book(book_day_, depth=plan.depth)
    tape = dy.tape(inputs, sa.BYBIT)
    prints = inputs.bybit.frame
    records: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    trades: dict[str, pd.DataFrame] = {}
    book_rows = [r for r in rows if ROWS[r].grid == "book"]
    if book_rows:
        frame = dy.store_frame("book", inputs)
        assert book_day_.frame is not None
        forward = {OWN_CELL.hold_s: book_forward(book_day_.frame, OWN_CELL.hold_s).to_numpy()}
        for name in book_rows:
            probas = _probas(plan.models[name], frame, inputs.scored_from(sa.BYBIT))
            records[name], tables[name], trades[name] = book_day(
                ROWS[name], plan.models[name], book, probas, forward, prints, tape, table,
                day=inputs.day, null_a=plan.null_a,
            )  # fmt: skip
    if "S25-PB" in rows:
        frame = dy.store_frame("bybit_prints", inputs)
        grid = stamps_ns(frame)
        forward = {OWN_CELL.hold_s: print_forward(tape, grid, OWN_CELL.hold_s)}
        probas = _probas(plan.models["S25-PB"], frame, inputs.scored_from(sa.BYBIT))
        records["S25-PB"], tables["S25-PB"] = print_day(
            ROWS["S25-PB"], plan.models["S25-PB"], tape, grid, probas, forward, prints, table,
            day=inputs.day, null_a=plan.null_a,
        )  # fmt: skip
        records["S25-PB"]["grid_ns_count"] = len(grid)
    return records, tables, trades


def calibration(
    plan: ReadPlan, inputs: dy.DayInputs, table: FeeTable, records: Mapping[str, Any],
    bb_trades: pd.DataFrame, folder: Any,
) -> dict[str, Any]:  # fmt: skip
    """S25-PB by the book walk, FILL-S and FILL-S thinned on one Bybit day, and S25-BB
    by the print rule; the priced orders are written beside the day's record."""
    day, start = inputs.day, dy.midnight_ns(inputs.day)
    book = taker_book(inputs.book, depth=plan.depth)
    tape = dy.tape(inputs, sa.BYBIT)
    prints = inputs.bybit.frame
    assert prints is not None
    settings = book_settings(table, sa.BYBIT, day)
    out: dict[str, Any] = {}

    def decided(name: str) -> tuple[np.ndarray, np.ndarray]:
        found = records[name].get("signal") or {"tod_ns": [], "side": []}
        return (start + np.asarray(found["tod_ns"], dtype=np.int64),
                np.asarray(found["side"], dtype=np.int64))  # fmt: skip

    ns, side = decided("S25-PB")
    walked = execute_book_day(book, ns, side, OWN_CELL.hold_ns, settings, table)
    out["pb_book"] = {"value": walked.day_bp(), "gross_bp": walked.day_bp("gross"),
                      "trades": len(walked.trades), "stale": bool(walked.excluded),
                      "tiers": tier_values(walked.trades, table, sa.BYBIT, lambda tier: (
                          table.day_fees(sa.BYBIT, day, tier).taker_bp,) * 2)}  # fmt: skip
    bb_ns, bb_side = decided("S25-BB")
    grid = np.unique(dy.grid_of(prints))
    by_rule = execute_print_day(tape, grid, bb_ns, bb_side, OWN_CELL.hold_ns,
                                print_settings(table, sa.BYBIT))  # fmt: skip
    out["bb_print_rule"] = {"value": by_rule.day_bp(), "trades": len(by_rule.trades)}
    priced = [price_orders(tape, book_orders(t)).assign(slot=name, day=day.isoformat())
              for name, t in (("S25-PB", walked.trades), ("S25-BB", bb_trades)) if len(t)]  # fmt: skip
    if priced:
        target = folder / "fill_s" / f"{day}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(priced, ignore_index=True).to_parquet(target, index=False)
    out["thinned"] = None
    if inputs.usable(sa.BINANCE) and inputs.binance.frame is not None and len(walked.trades):
        count = len(inputs.binance.frame)
        thinned = thinned_calibration(prints, count, walked.trades, day)
        target = folder / "thinned" / f"{day}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        thinned.assign(day=day.isoformat()).to_parquet(target, index=False)
        sides, sizes = trade_fills(walked.trades)
        ratios = [capacity_ratio(sides, sizes, thin_prints(prints, count, seed))
                  for seed in range(10)]  # fmt: skip
        out["thinned"] = {"binance_prints": count, "capacity": float(np.mean(ratios))}
    out["prints"] = {"bybit": len(prints),
                     "binance": None if inputs.binance.frame is None else len(inputs.binance.frame)}  # fmt: skip
    return out


def put_back(plan: ReadPlan, inputs: dy.DayInputs, table: FeeTable) -> dict[str, Any]:
    """S25-BB's and S25-T's true runs on a day excluded for its sequence gaps alone."""
    lifted = replace(inputs.book, exclusions=tuple(
        r for r in inputs.book.exclusions if r != ld.SEQUENCE_GAPS))  # fmt: skip
    view = replace(inputs, book=lifted)
    records, _, _ = bybit_rows(plan, view, table, rows=PUT_BACK_ROWS)
    keep = ("value", "trades", "gross_bp_turnover", "void")
    return {name: {k: r.get(k) for k in keep} for name, r in records.items()}


def _shift(name: str, table: Any, start: int, source: Mapping[str, Any]) -> tuple[float, bool]:
    signal = source["rows"][name].get("signal") or {"tod_ns": [], "side": []}
    tod, side = np.asarray(signal["tod_ns"], np.int64), np.asarray(signal["side"], np.int64)
    if ROWS[name].grid == "book":
        return book_shifted(table, start, tod, side)
    return print_shifted(table, start, tod, side), False


def read_day(
    plan: ReadPlan, inputs: dy.DayInputs, table: FeeTable, last: Mapping[str, Any], folder: Any
) -> dict[str, Any]:
    """One day's record: every usable venue's rows, the calibration, the put-back
    view, and each kept row's one-day shift from the venue's last kept day."""
    day, start = inputs.day, dy.midnight_ns(inputs.day)
    out: dict[str, Any] = {"day": day.isoformat(), "rows": {}, "kept": {}, "venues": {}}
    for name in VENUE_ROWS:
        out["venues"][name] = {"reasons": list(inputs.reasons[name]), "warm": inputs.warm[name]}
        frame = inputs.prints(name).frame
        out["venues"][name]["prints"] = None if frame is None else len(frame)
    tables: dict[str, Any] = {}
    stale = False
    if inputs.usable(sa.BINANCE):
        records, found = binance_rows(plan, inputs, table)
        out["rows"].update(records)
        tables.update(found)
    if inputs.usable(sa.BYBIT):
        records, found, trades = bybit_rows(plan, inputs, table, rows=VENUE_ROWS[sa.BYBIT])
        out["rows"].update(records)
        tables.update(found)
        out["calibration"] = calibration(plan, inputs, table, records,
                                         trades.get("S25-BB", pd.DataFrame()), folder)  # fmt: skip
        stale = any(bool(records[n].get("stale")) for n in PUT_BACK_ROWS)
        stale = stale or bool(out["calibration"]["pb_book"]["stale"])
    elif inputs.gap_only(sa.BYBIT):
        out["put_back"] = put_back(plan, inputs, table)
    for name in VENUE_ROWS:
        usable = inputs.usable(name) and not (name == sa.BYBIT and stale)
        if usable and name in last:
            for row in VENUE_ROWS[name]:
                if tables.get(row) is None:
                    continue
                value, flagged = _shift(row, tables[row], start, last[name])
                out["rows"][row]["shifted"], out["rows"][row]["shifted_from"] = (
                    value,
                    last[name]["day"],
                )
                stale = stale or (name == sa.BYBIT and flagged)
        out["kept"][name] = usable and not (name == sa.BYBIT and stale)
    if stale:
        out["venues"][sa.BYBIT]["reasons"].append("flatten_stale")
    return out


def own_tables(plan: ReadPlan, inputs: dy.DayInputs, table: FeeTable, venue: str) -> dict[str, Any]:
    """The venue's rows' own-cell tables on a day, for a shift computed after the day
    (they depend on the prints or the book and the settings, not on any fit)."""
    from trading_research.search_engine.walk_tables import WalkTable
    from trading_research.spot.costs import book_hold_table
    from trading_research.spot.walk_tables import print_hold_table

    out: dict[str, Any] = {}
    hold = OWN_CELL.hold_ns
    if venue == sa.BINANCE:
        tape = dy.tape(inputs, venue)
        grid = stamps_ns(dy.store_frame("binance", inputs))
        table_ = print_hold_table(tape, grid, hold, print_settings(table, venue))
        return dict.fromkeys(VENUE_ROWS[venue], table_)
    settings = book_settings(table, venue, inputs.day)
    walk = WalkTable.from_book(taker_book(inputs.book, depth=plan.depth), settings.clip)
    book_table = book_hold_table(walk, hold, settings, table)
    tape = dy.tape(inputs, venue)
    grid = stamps_ns(dy.store_frame("bybit_prints", inputs))
    out = dict.fromkeys(PUT_BACK_ROWS, book_table)
    out["S25-PB"] = print_hold_table(tape, grid, hold, print_settings(table, venue))
    return out


def read_days(study: SpotStudy, plan: ReadPlan) -> dict[str, dict[str, Any]]:
    """Every day of the plan, in order, resumed from the days already written; then the
    first kept day's shift on each venue. Returns the day records by ISO day."""
    folder = study.dir("reads", plan.label)
    load = loader(study, plan)
    table = study.fee_table
    last: dict[str, Any] = {}
    first: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    wanted = set(plan.days)
    for inputs in dy.walk_days(plan.days, load):
        if inputs.day not in wanted:
            continue
        path = folder / "days" / f"{inputs.day}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = plain(read_day(plan, inputs, table, last, folder))
            write_json(path, record)
        records[record["day"]] = record
        for name, kept in record["kept"].items():
            if kept:
                first.setdefault(name, record["day"])
                last[name] = record
    for name, day in first.items():
        record = records[day]
        if record.get(f"first_shift_{name}") or last[name]["day"] == day:
            continue
        inputs = load(date.fromisoformat(day))
        tables = own_tables(plan, inputs, table, name)
        for row in VENUE_ROWS[name]:
            if row in record["rows"] and "value" in record["rows"][row]:
                value, _ = _shift(row, tables[row], dy.midnight_ns(inputs.day), last[name])
                record["rows"][row]["shifted"], record["rows"][row]["shifted_from"] = (
                    value, last[name]["day"])  # fmt: skip
        record[f"first_shift_{name}"] = True
        write_json(folder / "days" / f"{day}.json", plain(record))
    return records
