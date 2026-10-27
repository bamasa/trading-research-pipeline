"""A block's day loaded inside an access, its exclusions, and every run of it.

:func:`load` returns a :class:`~trading_research.mm_bybit.tape.DayTape` or the
registered reason the day is excluded: ``no_prints_file``, ``no_print``,
``print_gap`` (two prints stamped between 00:10 and 23:59 more than 120 s apart),
``no_funding`` or ``fee not sourced``. :func:`run_jobs` runs every requested
(cell, variant) on one loaded day, so the day is read once for all of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.market_making.events import OffGridPrice
from trading_research.mm_bybit import archive, engine, funding, metrics
from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit.access import Access
from trading_research.mm_bybit.config import FeeNotSourced, Fees, fee_on
from trading_research.mm_bybit.tape import DayTape, build

PRINT_GAP_S = 120.0


@dataclass(frozen=True)
class Variant:
    """A named run setting: the fill rule and the latency."""

    name: str
    rule: str = "trade_through"
    latency_ms: int = 50


PRIMARY = Variant("primary")
VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        PRIMARY,
        Variant("touch", rule="touch"),
        Variant("delay", rule="delay"),
        Variant("latency150", latency_ms=150),
        Variant("latency10", latency_ms=10),
    )
}


def load(
    day: date,
    root: Path | str,
    access: Access,
    settlements: pd.DataFrame,
    fees: Fees | FeeNotSourced,
) -> DayTape | str:
    """The day's tape, or the reason it is excluded."""
    access.require(day)
    if isinstance(fees, FeeNotSourced):
        return "fee not sourced"
    path = archive.prints_path(root, day)
    if not path.exists():
        recorded = archive.read_status(root, day)
        if recorded is None:
            raise FileNotFoundError(f"{day} was never fetched: {path}")
        return recorded
    prints = pd.read_parquet(path)
    if prints.empty:
        return "no_print"
    found = funding.day_settlements(settlements, day)
    if found is None:
        return "no_funding"
    try:
        tape = build(day, prints, found[0], found[1])
    except (OffGridPrice, ValueError):
        return "unreadable"
    if tape.max_gap_s() > PRINT_GAP_S:
        return "print_gap"
    return tape


def settings_for(cfg: Mapping[str, Any], day: date, variant: str) -> engine.RunSettings:
    """A run's settings, built from the registration only: the fee is the day's sourced one."""
    fees = fee_on(cfg, day)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    v, s = VARIANTS[variant], cfg["settings"]
    return engine.RunSettings(
        maker_bp=fees.maker_bp, taker_bp=fees.taker_bp, clip=float(s["clip_btc"]),
        soft_limit_clips=float(s["soft_limit_clips"]), latency_ns=v.latency_ms * 1_000_000,
        rule=v.rule, warmup_s=int(s["warmup_s"]), stop_s=_clock_s(s["stop_quoting_at"]),
        flatten_s=_clock_s(s["flatten_at"]), pause_s=int(s["pause_s"]),
    )  # fmt: skip


def _clock_s(text: str) -> int:
    hours, minutes = (int(part) for part in str(text).split(":"))
    return hours * 3600 + minutes * 60


def run_jobs(
    tape: DayTape,
    jobs: Sequence[tuple[q.Cell, str]],
    cfg: Mapping[str, Any],
    inputs: engine.QuoterInputs,
    *,
    shifted: Mapping[str, tuple[Any, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Every (cell, variant) on the day, one summary row each.

    The settings come from :func:`settings_for` alone, so no run is given a fee
    other than the day's sourced one. ``shifted`` maps a variant name to the
    per-second (spread, sigma) state the gate reads instead of the day's own (the
    shifted-state placebo); such a variant runs the primary settings.
    """
    rows = []
    for cell, name in jobs:
        if shifted is not None and name in shifted:
            settings, state = settings_for(cfg, tape.day, PRIMARY.name), shifted[name]
        else:
            settings, state = settings_for(cfg, tape.day, name), None
        run = engine.run_day(tape, cell, settings, inputs, state)
        row = metrics.day_summary(
            tape, run, cell=cell, inputs=inputs, settings=settings,
        )  # fmt: skip
        row.update(day=tape.day.isoformat(), cell=cell.key, variant=name, tick=tape.tick,
                   lot=tape.lot)  # fmt: skip
        rows.append(row)
    return rows
