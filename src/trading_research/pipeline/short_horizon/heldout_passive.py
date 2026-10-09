"""The passive slots on a block: S-P, S-M and S25-Z through the simulator, with every variant.

Each slot's cell runs on each usable day as registered, and again under every
variant its kills read, each through the same quoter, fees, guard and
accounting:

* placebos on its decisions: the sign flipped, random directions (one seed per
  draw), the previous usable day's decisions at the same times of day
  (circularly over the block), the decisions delayed by one hold (stale), and
  all-long and all-short (K-drift);
* its neighbourhood (S-P and S25-Z: the other cells of the signal's grid; S-M:
  the leans 0.5 and 2);
* the queue brackets (K-queue): pessimistic cancellation attribution with
  ``all_ahead`` arrival growth, and ``none`` and ``all_ahead`` alone;
* every order and cancel live 110 ms after the decision (K-latency).

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


def run_variants(job: VariantJob) -> Path:
    """Every run of one day, the day's events loaded once (in a worker process)."""
    from trading_research.data import study_loader as sl
    from trading_research.data.fetch_stage import StudyRoots
    from trading_research.market_making.simulator import simulate_day

    target = job.out / f"{job.day}.parquet"
    if target.exists():
        return target
    roots = StudyRoots.under(job.repo / "data")
    events = sl.load_events_day(job.symbol, job.day, book_roots=[roots.book], trades_root=roots.trades,
                                funding_root=roots.funding, access=job.access)  # fmt: skip
    mid = float(np.mean((events.bid_px[:, 0] + events.ask_px[:, 0]) / 2.0) * events.spec.tick)
    clip = sl.clip_for(job.symbol)
    rows = []
    for run in job.runs:
        quoter, config, tapes = quoter_and_tapes(run.cell, run.decisions, job.flags_ns,
                                                 job.max_sequence_gaps, job.sigma_ref)  # fmt: skip
        if run.changes:
            config = config.with_(**run.changes)
        result = simulate_day(events, quoter, config, tapes)
        rows.append(day_values(run.cell, quoter, result, clip, mid) |
                    {"day": job.day.isoformat(), "run_slot": run.slot, "variant": run.variant})  # fmt: skip
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    pd.DataFrame(rows).to_parquet(partial, index=False)
    partial.replace(target)
    return target


def variants(slot: str, cell: SimCell, own: Decisions, previous: Decisions, day_start: int,
             previous_start: int, neighbours: list[tuple[SimCell, Decisions]], draws: int,
             rngs: list[np.random.Generator]) -> list[Run]:  # fmt: skip
    """Every run a passive slot's kills read on one day."""
    magnitude = (
        np.abs(own.strength_bp) if len(own.strength_bp) == len(own.side) else np.ones(len(own.side))
    )

    def sided(side: np.ndarray, stamps: np.ndarray | None = None) -> Decisions:
        return Decisions(
            own.stamps_ns if stamps is None else stamps, side.astype(np.int64), side * magnitude
        )

    runs = [Run(slot, "true", cell, own), Run(slot, "flipped", cell, sided(-own.side)),
            Run(slot, "long_only", cell, sided(np.ones_like(own.side))),
            Run(slot, "short_only", cell, sided(-np.ones_like(own.side))),
            Run(slot, "stale", cell, sided(own.side, own.stamps_ns + round(cell.hold_s * 1e9)))]  # fmt: skip
    shifted = previous.stamps_ns - previous_start + day_start
    runs.append(Run(slot, "shifted", cell, Decisions(shifted, previous.side, previous.strength_bp)))
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
