"""Stage 0, on the reader-test days 2026-03-03 to 05: the readers' proof, the gates, the timings.

Registered (``docs/preregistration/spot_2026.md``, "The development-period
procedure", step 0). Under the reader-test access, after the disk check
(free disk above the stage's budget by 10 GB, every measurement recorded):

1. **The fetch** of the three days on every plane, timed, the book replayed at
   ten levels, the fetch log keeping each day's message counts, median interval
   (the format probe saw 200 ms between messages), sequence gaps, ten-level
   share, Bybit's ``rpi`` share and side counts, and Binance's time unit.
2. **The readers' proof** per day and instrument: Bybit's book loaded as a block
   day (no ``positive_prices`` error, no off-grid price), Bybit's prints and
   Binance's aggregated prints in the print contract with both sides present;
   each gives ``{status, stream, findings, passes}``, the D amendment's
   ``reader_test_result``.
3. **The walk depth**: the share of book rows in which the clip exceeds ten
   levels on either side, pooled over the three days; above 0.1% the walk takes
   all 200 levels and stage 1 replays at 200.
4. **The print rule's records**: the unreachable share and the median sweep
   against the clip, on each print plane.
5. **The parity gate**: the print walk tables against the print executor, entry
   by entry, at every hold, on random decisions whose busy periods depend on the
   side (and the 2025 book tables against the book executor); a mismatch stops
   the stage.
6. **The timings** of the features, labels, a fit and the scoring on one day;
   the labels, fits and scores themselves are discarded and never reported.
7. **The instruments' tick and lot**, from each venue's spot endpoint.

A failed gate writes its findings beside the stage and raises: the stage is not
complete, and a dated amendment fixes the fault before stage 1.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np

from trading_research.data import disk
from trading_research.data import fetch_log as fl
from trading_research.search_engine.walk_tables import WalkTable, entry_parity
from trading_research.spot import access as sa
from trading_research.spot import archives as ar
from trading_research.spot import days as dy
from trading_research.spot import fetch_stage as fs
from trading_research.spot import instrument as ins
from trading_research.spot import loader as ld
from trading_research.spot.bybit_book import WALK_THRESHOLD
from trading_research.spot.calibration import print_rule_checks
from trading_research.spot.context import SpotStudy
from trading_research.spot.costs import book_hold_table, book_settings, taker_book
from trading_research.spot.fills import fill_orders
from trading_research.spot.plan import HOLDS_S, LABEL_FEES_BP, NS_PER_S
from trading_research.spot.print_taker import print_settings
from trading_research.spot.state import write_json
from trading_research.spot.walk_tables import print_entry_parity, print_hold_table

PROOF = "readers' proof"
WALK_LEVELS = (10, 200)


class Stage0Failed(RuntimeError):
    """A reader or the parity gate failed: fixed by a dated amendment before stage 1."""


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    found = fn()
    return time.perf_counter() - start, found


def fetch(
    study: SpotStudy, stage: int, days: tuple[date, ...], access: Any, depth: int
) -> fl.FetchLog:
    """A stage's fetch through the study's fetch, its disk check and margin."""
    measure = study.disk or disk.free_gb
    return fs.run_stage(stage, fs.every_plane(days), study.roots, study.fetch_log(stage),
                        access=access, disk_record=study.disk_record,
                        expected_present=study.present(), opener=study.venue,
                        book_workers=min(study.workers, fs.BOOK_WORKERS_MAX),
                        book_depth=depth, measure=measure)  # fmt: skip


def stats(log: fl.FetchLog, plane: ar.Plane, day: date) -> dict[str, Any]:
    record = log.records.get(log.key(plane.name, day))
    return {} if record is None else {"status": record.status, **record.stats}


def book_proof(day: date, log: fl.FetchLog, root: Any, access: Any) -> dict[str, Any]:
    """Bybit's book replayed as a block day: no non-positive and no off-grid price."""
    loaded = ld.load_book_day(day, root, access=access, max_sequence_gaps=None)
    found = stats(log, ar.BYBIT_BOOK, day)
    bad = [
        r for r in loaded.exclusions if r in (ld.NON_POSITIVE_PRICE, ld.OFF_GRID_PRICE, ld.NO_BOOK)
    ]
    keep = ("messages", "snapshots", "deltas", "other", "median_interval_ms", "max_interval_ms",
            "sequence_gaps", "rows", "clip_beyond_levels_share", "peak_rss_bytes")  # fmt: skip
    return {"status": "usable" if not loaded.exclusions else ",".join(loaded.exclusions),
            "stream": {k: found.get(k) for k in keep},
            "findings": {"report": loaded.findings, "notes": list(loaded.notes), "tick": loaded.tick,
                         "rows_outside_day": loaded.rows_outside_day},
            "passes": bool(loaded.frame is not None and not bad)}  # fmt: skip


def prints_proof(
    plane: ar.Plane, day: date, log: fl.FetchLog, root: Any, access: Any
) -> dict[str, Any]:
    """A print archive in the print contract, both sides present."""
    from trading_research.spot.prints import PrintContractError, check

    loaded = ld.load_prints_day(plane, day, root, access=access)
    found = stats(log, plane, day)
    ok = loaded.frame is not None and not loaded.exclusions
    sides: dict[str, int] = {}
    if loaded.frame is not None:
        try:
            check(loaded.frame)
        except PrintContractError as exc:
            ok, found["contract"] = False, str(exc)
        aggressor = loaded.frame["aggressor"].to_numpy()
        sides = {"buy": int((aggressor == 1).sum()), "sell": int((aggressor == -1).sum())}
        ok = ok and sides["buy"] > 0 and sides["sell"] > 0
    return {"status": "usable" if not loaded.exclusions else ",".join(loaded.exclusions),
            "stream": {**found, "sides_loaded": sides, "rpi_rows": loaded.rpi_rows},
            "findings": {"notes": list(loaded.notes), "tick": loaded.tick,
                         "rows_outside_day": loaded.rows_outside_day},
            "passes": bool(ok)}  # fmt: skip


def readers(study: SpotStudy, log: fl.FetchLog, access: Any) -> dict[str, dict[str, Any]]:
    """``reader_test_result``: {instrument key: {day: {status, stream, findings, passes}}}."""
    root, out = study.spot_root, {}
    for name, planes in dy.VENUE_PLANES.items():
        key = sa.instrument(name).key
        out[key] = {}
        for day in study.scale.days(sa.READER_TEST):
            parts = {}
            for plane in planes:
                parts[plane.name] = (book_proof(day, log, root, access) if plane.kind == ar.BOOK
                                     else prints_proof(plane, day, log, root, access))  # fmt: skip
            if len(parts) == 1:
                out[key][day.isoformat()] = next(iter(parts.values()))
            else:
                out[key][day.isoformat()] = {
                    "status": ";".join(f"{p}: {v['status']}" for p, v in parts.items()),
                    "stream": {p: v["stream"] for p, v in parts.items()},
                    "findings": {p: v["findings"] for p, v in parts.items()},
                    "passes": all(v["passes"] for v in parts.values()),
                }
    return out


def walk_depth(log: fl.FetchLog, days: tuple[date, ...]) -> dict[str, Any]:
    """The ten-level share pooled over the days' book rows, and the walk it sets."""
    rows, beyond = 0, 0.0
    shares = {}
    for day in days:
        found = stats(log, ar.BYBIT_BOOK, day)
        if found.get("rows") and found.get("clip_beyond_levels_share") is not None:
            rows += int(found["rows"])
            beyond += float(found["clip_beyond_levels_share"]) * int(found["rows"])
            shares[day.isoformat()] = float(found["clip_beyond_levels_share"])
    if not rows:
        raise Stage0Failed("no reader-test book replayed: the ten-level share is unmeasured")
    share = beyond / rows
    return {"share": share, "per_day": shares, "threshold": WALK_THRESHOLD,
            "walk": WALK_LEVELS[1] if share > WALK_THRESHOLD else WALK_LEVELS[0]}  # fmt: skip


def print_rule(inputs: dy.DayInputs, name: str) -> dict[str, float]:
    """The unreachable share and the median sweep against the clip, from the rule's
    fills of a buy and a sell at every row of the day's print grid."""
    tape = dy.tape(inputs, name)
    prints = inputs.prints(name).frame
    assert prints is not None
    grid = np.unique(dy.grid_of(prints))
    fills = [fill_orders(tape, side, grid + 10_000_000, grid, until_ns=tape.end_ns)
             for side in (1, -1)]  # fmt: skip
    return print_rule_checks(tape, fills, 0.010)


def decisions(n: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(n, size=min(count, n), replace=False)).astype(np.int64)
    return picked, rng.choice(np.array([-1, 1]), size=len(picked))


def parity(study: SpotStudy, inputs: dy.DayInputs) -> dict[str, Any]:
    """Print walk tables against the print executor on both print planes and every hold,
    and the book tables against the book executor; with the count of decisions whose
    busy period depends on the side."""
    table, out = study.fee_table, {}
    for name in dy.VENUE_PLANES:
        tape = dy.tape(inputs, name)
        prints = inputs.prints(name).frame
        assert prints is not None
        grid = np.unique(dy.grid_of(prints))
        rows, side = decisions(len(grid), study.scale.parity_decisions, inputs.day.toordinal())
        for hold in HOLDS_S:
            found = print_hold_table(
                tape, grid, round(hold * NS_PER_S), print_settings(table, name)
            )
            check = print_entry_parity(tape, rows, side, found)
            both = found.has_entry[:, rows].all(axis=0)
            depends = int((found.free_ns[0, rows] != found.free_ns[1, rows])[both].sum())
            out[f"{name}/{inputs.day}/{hold:g}"] = {"trades": check.trades,
                                                     "mismatches": list(check.mismatches),
                                                     "side_dependent_busy": depends}  # fmt: skip
    book = taker_book(inputs.book, depth=10)
    settings = book_settings(table, sa.BYBIT, inputs.day)
    walk = WalkTable.from_book(book, settings.clip)
    rows, side = decisions(len(book.ts), study.scale.parity_decisions, inputs.day.toordinal() + 1)
    for hold in HOLDS_S:
        found = book_hold_table(walk, round(hold * NS_PER_S), settings, table)
        check = entry_parity(book, rows, side, found)
        out[f"book/{inputs.day}/{hold:g}"] = {"trades": check.trades,
                                               "mismatches": list(check.mismatches)}  # fmt: skip
    return out


def timings(study: SpotStudy, inputs: dy.DayInputs) -> dict[str, float]:
    """Seconds for the features, the labels, one fit and the scoring on one day;
    everything they compute is discarded."""
    from trading_research.spot.book_inputs import book_targets
    from trading_research.spot.fits import fit_model
    from trading_research.spot.print_label import label_column, print_targets
    from trading_research.spot.print_taker import execute_print_day

    out: dict[str, float] = {}
    for store in dy.STORES:
        out[f"features_{store}_s"], _ = timed(lambda s=store: dy.store_frame(s, inputs))
    assert inputs.binance.frame is not None
    grid = dy.grid_of(inputs.binance.frame)
    tape = dy.tape(inputs, sa.BINANCE)
    out["labels_prints_s"], labels = timed(
        lambda: print_targets(tape, grid, HOLDS_S, LABEL_FEES_BP)
    )
    assert inputs.book.frame is not None
    out["labels_book_s"], _ = timed(
        lambda: book_targets(
            inputs.book.frame, HOLDS_S, LABEL_FEES_BP, at=np.arange(0, len(inputs.book.frame), 10)
        )
    )
    features = dy.store_frame("binance", inputs)
    target = labels[label_column(50.0, 0.0)]
    x = features.drop(columns="timestamp").astype(np.float64)
    keep = target.notna().to_numpy() & x.notna().all(axis=1).to_numpy()
    out["fit_s"], _ = timed(lambda: fit_model(x[keep], target[keep], study.scale.model_params))
    rows, side = decisions(len(grid), 240, 7)
    settings = print_settings(study.fee_table, sa.BINANCE)
    out["executor_240_s"], _ = timed(
        lambda: execute_print_day(tape, grid, grid[rows], side, 50 * NS_PER_S, settings)
    )
    out["walk_table_s"], _ = timed(lambda: print_hold_table(tape, grid, 50 * NS_PER_S, settings))
    return out


def specs(study: SpotStudy) -> dict[str, dict[str, float]]:
    """``instrument_spec``: each instrument's tick and lot from its venue's spot endpoint."""
    found = [ins.fetch_spec(name, opener=study.opener) for name in (sa.BINANCE, sa.BYBIT)]
    return {sa.instrument(i.instrument).key: {"tick": i.tick, "lot": i.lot} for i in found}


def run(study: SpotStudy) -> dict[str, Any]:
    """Stage 0's record (module docstring); raises :class:`Stage0Failed` on a failed gate."""
    access = sa.reader_test_access(PROOF)
    days = study.scale.days(sa.READER_TEST)
    with study.serving():
        fetch_s, log = timed(lambda: fetch(study, 0, days, access, 10))
        spec = specs(study)
    result = readers(study, log, access)
    walk = walk_depth(log, days)
    table = study.fee_table
    rule: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    timing: dict[str, Any] = {"fetch_s": fetch_s}
    timing_access = sa.reader_test_access("stage 0 timing")
    for k, day in enumerate(days):
        inputs = dy.load_day(day, study.spot_root, access=timing_access, logs=[log], table=table,
                             max_sequence_gaps=None, depth=10)  # fmt: skip
        if not (inputs.usable(sa.BINANCE) and inputs.usable(sa.BYBIT)):
            continue
        for name in dy.VENUE_PLANES:
            rule[f"{sa.instrument(name).key}/{day}"] = print_rule(inputs, name)
        gate.update(parity(study, inputs))
        if k == 0:
            timing.update(timings(study, inputs))
    record = {
        "days": list(days), "reader_test_result": result, "walk": walk, "print_rule": rule,
        "parity": gate, "timings": timing, "instrument_spec": spec,
        "message_interval_ms": {d: r["stream"]["bybit_book"]["median_interval_ms"]
                                for d, r in result[sa.instrument(sa.BYBIT).key].items()},
        "disk": _disk(study), "access": access.stamp,
    }  # fmt: skip
    failures = _failures(record)
    if failures:
        write_json(study.dir("0") / "failed.json", {**record, "failures": failures})
        raise Stage0Failed(f"stage 0 failed: {failures[:5]} (findings in 0/failed.json)")
    return record


def _disk(study: SpotStudy) -> list[dict[str, Any]]:
    import json

    path = study.disk_record
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _failures(record: dict[str, Any]) -> list[str]:
    out = [f"reader {key} {day}" for key, days in record["reader_test_result"].items()
           for day, found in days.items() if not found["passes"]]  # fmt: skip
    out += [f"parity {run}: {found['mismatches'][:2]}" for run, found in record["parity"].items()
            if found["mismatches"]]  # fmt: skip
    if not record["parity"]:
        out.append("parity: no reader-test day usable on both venues")
    printed = [f for r, f in record["parity"].items() if "side_dependent_busy" in f]
    if printed and not any(f["side_dependent_busy"] for f in printed):
        out.append("parity: no run had a busy period that depends on the side")
    return out
