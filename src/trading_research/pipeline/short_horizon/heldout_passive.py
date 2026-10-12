"""The passive slots on a block: S-P, S-M and S25-Z through the simulator, with every variant.

Each slot's cell runs on each usable day as registered, and again under every
variant its kills read, each through the same quoter, fees, guard and
accounting:

* placebos on its decisions: the sign flipped, random directions (one seed per
  draw), the previous usable day's decisions at the same times of day
  (circularly over the block, stepping over a day any run flags stale:
  ``heldout.step_over_stale``), the decisions delayed by one hold (stale), and
  all-long and all-short (K-drift);
* its neighbourhood (S-P and S25-Z: the other cells of the signal's grid; S-M:
  the leans 0.5 and 2);
* the queue brackets (K-queue): pessimistic cancellation attribution with
  ``all_ahead`` arrival growth, and ``none`` and ``all_ahead`` alone;
* every order and cancel live 110 ms after the decision (K-latency).

Beside them, the measurements (:func:`measured_runs`): S-P's cell at the other
regime windows and S-M's guards (``guard_pull`` and ``guard_widen`` at 4, 24
and 120 h at lean 1), each with the window-0 twin when the slot has a window;
the passive-exit variant of S-P and S25-Z; on H, FILL's two other fill rules
on S-P's and S25-Z's streams; and, for a slot with a regime window, its runs
under the random flag sets of the regime attribution, each run reading its
own flags.

A day's runs share one load of the day's events and run in a worker process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import study_access as sa
from trading_research.pipeline.short_horizon.passive import (
    Decisions,
    SimCell,
    day_values,
    quoter_and_tapes,
)

LATENCY_110_MS = {"order_latency_ns": 110_000_000, "cancel_latency_ns": 110_000_000}


def brackets() -> dict[str, dict[str, Any]]:
    from trading_research.market_making.queue import ArrivalGrowth, CancelAttribution

    return {
        "pessimistic": {"cancel_attribution": CancelAttribution.PESSIMISTIC,
                        "arrival_growth": ArrivalGrowth.ALL_AHEAD},
        "none": {"arrival_growth": ArrivalGrowth.NONE},
        "all_ahead": {"arrival_growth": ArrivalGrowth.ALL_AHEAD},
    }  # fmt: skip


@dataclass(frozen=True)
class Run:
    """One simulated run of a day: the slot, the variant, the cell and its decisions."""

    slot: str
    variant: str
    cell: SimCell
    decisions: Decisions
    changes: dict[str, Any] = field(default_factory=dict)
    #: The flags this run's guard reads (the block's frozen flags when None).
    flags_ns: np.ndarray | None = field(default=None, compare=False)


@dataclass(frozen=True)
class VariantJob:
    repo: Path
    symbol: str
    day: date
    access: sa.StudyAccess
    runs: tuple[Run, ...]
    flags_ns: np.ndarray
    max_sequence_gaps: int
    sigma_ref: float
    out: Path


def fill_values(result: Any) -> dict[str, Any]:
    """What FILL reads of an entry run: orders and volume posted and filled, and each
    fill path's markouts as volume-weighted sums (summed over days, then divided)."""
    import json

    orders, fills = result.orders, result.fills
    maker = fills[fills["maker"]] if len(fills) else fills
    paths: dict[str, dict[str, float]] = {}
    for path, group in maker.groupby("path", sort=True) if len(maker) else ():
        size = group["size"].to_numpy(dtype=np.float64)
        entry = paths.setdefault(
            str(path), {"fills": float(len(group)), "volume": float(size.sum())}
        )
        for column in [c for c in group.columns if c.startswith("markout_")]:
            values = group[column].to_numpy(dtype=np.float64)
            ok = np.isfinite(values)
            entry[column] = float((values[ok] * size[ok]).sum())
            entry[column + "_weight"] = float(size[ok].sum())
    return {"orders_posted": float(len(orders)), "orders_filled": float((orders["filled"] > 0).sum()),
            "volume_posted": float(orders["size"].sum()), "volume_filled": float(orders["filled"].sum()),
            "fill_paths": json.dumps(paths, sort_keys=True)}  # fmt: skip


def day_rows(events: Any, runs: tuple[Run, ...], flags_ns: np.ndarray, max_sequence_gaps: int,
             sigma_ref: float, clip: float) -> list[dict[str, Any]]:  # fmt: skip
    """Every run of one loaded day, each under its own flags and changes.

    Each row carries the void conditions' inputs: ``accounting_failed``, the
    accounting fault a run raised (the simulator's identity, R15, or the filled
    attempts' cash against the day's; K-acct), and ``max_abs_position`` from the
    fills' positions with ``cap_exceeded`` above the cap (K-cap: one clip for the
    signal-entry quoter, the hard limit of two clips for S1)."""
    from trading_research.market_making.accounting import IdentityError
    from trading_research.market_making.simulator import simulate_day

    rows = []
    for run in runs:
        flags = flags_ns if run.flags_ns is None else run.flags_ns
        quoter, config, tapes = quoter_and_tapes(run.cell, run.decisions, flags, max_sequence_gaps,
                                                 sigma_ref)  # fmt: skip
        if run.changes:
            config = config.with_(**run.changes)
        where = {"day": events.day.isoformat(), "run_slot": run.slot, "variant": run.variant}
        try:
            result = simulate_day(events, quoter, config, tapes)
            row = day_values(run.cell, quoter, result, clip)
        except (IdentityError, RuntimeError) as exc:
            rows.append(where | {"label": run.cell.label, "excluded": False,
                                 "accounting_failed": f"{type(exc).__name__}: {exc}"})  # fmt: skip
            continue
        if run.cell.kind == "entry":
            row |= fill_values(result)
        rows.append(row | where | position_cap(result.fills, clip, run.cell.kind))
    return rows


def position_cap(fills: pd.DataFrame, clip: float, kind: str) -> dict[str, Any]:
    """K-cap's inputs from a run's fills: the largest |position| after any fill, and
    whether it exceeds the cap (one clip for the signal-entry quoter, whose order
    with a cancel in flight counts as resting; ``soft_limit_clips`` + 1 = 2 clips
    for S1's hard limit)."""
    position = fills["position_after"].to_numpy(dtype=np.float64) if len(fills) else np.zeros(0)
    largest = float(np.abs(position).max()) if len(position) else 0.0
    cap = clip * (1.0 if kind == "entry" else 2.0)
    return {"max_abs_position": largest, "cap_exceeded": bool(largest > cap * (1.0 + 1e-9)),
            "accounting_failed": ""}  # fmt: skip


def run_variants(job: VariantJob) -> Path:
    """Every run of one day, the day's events loaded once (in a worker process)."""
    from trading_research.data import study_loader as sl
    from trading_research.data.fetch_stage import StudyRoots

    target = job.out / f"{job.day}.parquet"
    if target.exists():
        return target
    roots = StudyRoots.under(job.repo / "data")
    events = sl.load_events_day(job.symbol, job.day, book_roots=[roots.book], trades_root=roots.trades,
                                funding_root=roots.funding, access=job.access)  # fmt: skip
    rows = day_rows(events, job.runs, job.flags_ns, job.max_sequence_gaps, job.sigma_ref,
                    sl.clip_for(job.symbol))  # fmt: skip
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    pd.DataFrame(rows).to_parquet(partial, index=False)
    partial.replace(target)
    return target


#: The simulator's workers: at most 6, while that many peaks of one day stay within 24 GB.
SIMULATOR_WORKERS_MAX = 6
SIMULATOR_MEMORY_BUDGET_BYTES = 24 * 10**9


def simulator_workers(requested: int, peak_rss: int) -> int:
    """Simulator workers: at most 6 and the count asked for, and fewer while that
    many times one day's peak resident memory would exceed 24 GB; at least one."""
    allowed = SIMULATOR_MEMORY_BUDGET_BYTES // peak_rss if peak_rss > 0 else SIMULATOR_WORKERS_MAX
    return int(max(1, min(SIMULATOR_WORKERS_MAX, requested, allowed)))


def run_measured(job: VariantJob) -> tuple[Path, int]:
    """:func:`run_variants` in a fresh worker, with that process's peak memory after it."""
    from trading_research.data.study_fetch import peak_rss_bytes

    return run_variants(job), peak_rss_bytes()


def variants(slot: str, cell: SimCell, own: Decisions, previous: Decisions, day_start: int,
             previous_start: int, neighbours: list[tuple[SimCell, Decisions]], draws: int,
             rngs: list[np.random.Generator]) -> list[Run]:  # fmt: skip
    """Every run a passive slot's kills read on one day."""
    magnitude = (
        np.abs(own.strength_bp) if len(own.strength_bp) == len(own.side) else np.ones(len(own.side))
    )

    def sided(side: np.ndarray, stamps: np.ndarray | None = None) -> Decisions:
        return Decisions(
            own.stamps_ns if stamps is None else stamps,
            side.astype(np.int64),
            side * magnitude,
            own.event,
        )

    runs = [Run(slot, "true", cell, own), Run(slot, "flipped", cell, sided(-own.side)),
            Run(slot, "long_only", cell, sided(np.ones_like(own.side))),
            Run(slot, "short_only", cell, sided(-np.ones_like(own.side))),
            Run(slot, "stale", cell, sided(own.side, own.stamps_ns + round(cell.hold_s * 1e9)))]  # fmt: skip
    runs.append(shifted_run(slot, cell, previous, day_start, previous_start))
    for k in range(draws):
        runs.append(
            Run(
                slot,
                f"random/{k}",
                cell,
                sided(rngs[k].choice(np.array([-1, 1]), size=len(own.side))),
            )
        )
    runs += [
        Run(slot, f"bracket/{name}", cell, own, changes) for name, changes in brackets().items()
    ]
    runs.append(Run(slot, "latency", cell, own, dict(LATENCY_110_MS)))
    runs += [Run(slot, f"neighbour/{other.label}", other, decided) for other, decided in neighbours]
    return runs


def shifted_run(slot: str, cell: SimCell, previous: Decisions, day_start: int,
                previous_start: int) -> Run:  # fmt: skip
    """The one-day shift: another day's decisions at the same times of day (in R
    with their event ids, one entry per event)."""
    shifted = previous.stamps_ns - previous_start + day_start
    return Run(slot, "shifted", cell,
               Decisions(shifted, previous.side, previous.strength_bp, previous.event))  # fmt: skip


def measured_runs(slot: str, cell: SimCell, own: Decisions, measured: list[tuple[str, SimCell, Decisions]],
                  flag_sets: list[tuple[np.ndarray, Decisions]], fill: bool) -> list[Run]:  # fmt: skip
    """The measurements of a passive slot on one day, beside its kills' runs.

    ``measured`` are the regime measurements (``regime/<window or guard>``, the
    window-0 twin among them as ``regime/none``), each on its own decisions;
    ``flag_sets`` the regime attribution's random flag sets with the decisions
    guarded by each; ``fill`` adds FILL's touch and trade-through rules (H only).
    S-P and S25-Z also run with the passive exit."""
    from dataclasses import replace

    from trading_research.market_making.queue import FillRule

    runs = [Run(slot, name, other, decided) for name, other, decided in measured]
    if cell.kind == "entry":
        runs.append(Run(slot, "exit/passive", replace(cell, passive_exit=True), own))
        if fill:
            runs += [Run(slot, f"fill/{rule.value}", cell, own, {"fill_rule": rule})
                     for rule in (FillRule.TOUCH, FillRule.THROUGH)]  # fmt: skip
    runs += [Run(slot, f"flagset/{k}", cell, decided, flags_ns=flags)
             for k, (flags, decided) in enumerate(flag_sets)]  # fmt: skip
    return runs
