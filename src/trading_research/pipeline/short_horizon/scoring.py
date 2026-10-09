"""Stage B: every taker cell on every test day, from walk tables, into the trial ledger.

Refused unless stage 0's walk-table parity test passed. For each test day, in
memory and never stored: the executor's book with funding, its walk table and
one hold table per hold (``search_engine.walk_tables``). Each configuration's
tape is read back and every cell of the configuration is scored from it
(``search_engine.cells.score_configuration``: the cell's own rule, gate and
guard, one position at a time by the event-jump loop, each trade's long and
short walked P&L looked up). A day keeps three files, written together:

* the trial ledger's rows (one per cell, zeros for a cell without a trade);
* the trades Null A sums by hour (stamped at the entry snapshot);
* the accepted entries Null B carries to other days (time of day and side).

A day already scored is not scored again. At the end the day's rows go into the
trial ledger as one run's rows, with the run's commit; a day an exit found the
book stale on (``flatten_stale``) is recorded and left out of every series by
stage D. The registration has the ledger committed at the end of the stage.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.taker import TakerSettings
from trading_research.pipeline.short_horizon.fits import (
    Configuration,
    configurations,
    rules_from,
    save_json,
)
from trading_research.pipeline.short_horizon.inputs import Inputs
from trading_research.pipeline.short_horizon.plan import NS_PER_S, Fold, walk_forward
from trading_research.search_engine.catalogue import CellKey, parse_key
from trading_research.search_engine.cells import ledger_frame, score_configuration, trade_rows
from trading_research.search_engine.null_b import entries_of
from trading_research.search_engine.tapes import read_tape, tape_path
from trading_research.search_engine.walk_tables import HoldTable, WalkTable, hold_table

DAY_FILES = ("ledger.parquet", "trades.parquet", "entries.parquet")


def day_tables(inputs: Inputs, day: date, holds_s: Sequence[float]) -> dict[int, HoldTable]:
    """One day's hold tables by hold in ns, from the executor's book with funding."""
    settings = TakerSettings()
    walk = WalkTable.from_book(inputs.taker_book(day), settings.clip)
    return {round(h * NS_PER_S): hold_table(walk, round(h * NS_PER_S), settings) for h in holds_s}


def test_folds(inputs: Inputs) -> dict[date, Fold]:
    """Each usable test day's fold."""
    return {day: fold for fold in walk_forward(inputs.usable("D")) for day in fold.test}


def score_day(
    run: int,
    day: date,
    tables: dict[int, HoldTable],
    configs: Sequence[Configuration],
    cells: dict[str, list[CellKey]],
    positions: dict[str, int],
    tapes: Path,
    fold_record: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """A day's ledger rows, trade rows and accepted entries, and whether a walk was stale."""
    ledgers, trades, entries = [], [], []
    stale = False
    for cfg in configs:
        tape = read_tape(tape_path(tapes, cfg.name, day), configuration=cfg.name, day=day)
        table = tables[round(cfg.hold_s * NS_PER_S)]
        rules = None
        if cfg.stratum == "C":
            rules = rules_from(fold_record["fits"][cfg.name]["thresholds"])
        scored = score_configuration(tape, table, cells[cfg.name], rules)
        ledgers.append(ledger_frame(run, day, scored))
        for key, found in scored.items():
            stale = stale or found.stale
            if found.trades:
                trades.append(pd.DataFrame(trade_rows(key, table, found)))
                taken = entries_of(table, found)
                entries.append(
                    pd.DataFrame(
                        {"cell": positions[key], "tod_ns": taken.tod_ns, "side": taken.side}
                    )
                )
    empty_trades = pd.DataFrame(
        columns=["key", "time", "side", "notional", "pnl_long_bp", "pnl_short_bp"]
    )
    empty_entries = pd.DataFrame({"cell": pd.Series(dtype=np.int64), "tod_ns": pd.Series(dtype=np.int64),
                                  "side": pd.Series(dtype=np.int64)})  # fmt: skip
    return (
        pd.concat(ledgers, ignore_index=True),
        pd.concat(trades, ignore_index=True) if trades else empty_trades,
        pd.concat(entries, ignore_index=True) if entries else empty_entries,
        stale,
    )


def score_days(
    inputs: Inputs,
    keys: Sequence[str],
    run: int,
    tapes: Path,
    folds: Path,
    out: Path,
) -> dict[str, Any]:
    """Score ``keys`` on every usable test day into ``out``; days already done are kept."""
    configs = configurations(keys)
    cells: dict[str, list[CellKey]] = {cfg.name: [] for cfg in configs}
    for key in keys:
        cells[parse_key(key).configuration].append(parse_key(key))
    positions = {key: i for i, key in enumerate(keys)}
    holds = sorted({cfg.hold_s for cfg in configs})
    stale: list[str] = []
    for day, fold in test_folds(inputs).items():
        folder = out / day.isoformat()
        marker = folder / "done.json"
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8"))["stale"]:
                stale.append(day.isoformat())
            continue
        record = json.loads((folds / f"{fold.name}.json").read_text(encoding="utf-8"))
        tables = day_tables(inputs, day, holds)
        frames = score_day(run, day, tables, configs, cells, positions, tapes, record)
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in zip(DAY_FILES, frames[:3], strict=True):
            frame.to_parquet(folder / name, index=False)
        save_json(marker, {"stale": frames[3], "rows": len(frames[0]), "trades": len(frames[1])})
        if frames[3]:
            stale.append(day.isoformat())
    return {"days": len(test_folds(inputs)), "stale_days": sorted(stale)}


def day_frames(out: Path, days: Sequence[date], name: str) -> pd.DataFrame:
    """One of the per-day files over ``days``, concatenated."""
    frames = [pd.read_parquet(out / d.isoformat() / name) for d in days]
    frames = [f for f in frames if len(f)]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.read_parquet(out / days[0].isoformat() / name)
    )


def run(study: Any) -> dict[str, Any]:
    """Stage B's record: the days scored, the stale days, the trial ledger's run."""
    from trading_research.pipeline.short_horizon.stage0 import require_parity
    from trading_research.search_engine.ledger import LEDGER_PATH, TrialLedger

    require_parity(study.record("0"))
    inputs = Inputs(study)
    keys = list(study.keys)
    out = study.dir("B", "days")
    summary = score_days(
        inputs, keys, study.run, study.work / "A" / "tapes", study.work / "A" / "folds", out
    )
    days = sorted(test_folds(inputs))
    rows = day_frames(out, days, "ledger.parquet")
    ledger = TrialLedger(study.repo / LEDGER_PATH, study.catalogue())
    written = ledger.sidecar()["runs"].get(str(study.run))
    if written is None:
        ledger.append(study.run, study.commit(), rows)
    elif written["rows"] != len(rows):
        raise RuntimeError(
            f"run {study.run} is in the trial ledger with {written['rows']} rows, not {len(rows)}"
        )
    sidecar = ledger.sidecar()
    return {
        "commit": study.commit(),
        "run": study.run,
        "cells": len(keys),
        "test_days": [d.isoformat() for d in days],
        "stale_days": summary["stale_days"],
        "ledger_rows": int(sidecar["rows"]),
        "ledger_sha256": sidecar["sha256"],
        "n_trials": int(sidecar["n_trials"]),
        "commit_the_trial_ledger": "the registration commits the ledger and its sidecar at the end of each stage",
    }
