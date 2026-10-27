"""The stages: the reader's proof, the fetches, a block's runs, W's inputs.

A block's runs go day by day, each day in a worker process that loads it once
and runs every requested (cell, variant) on it; the rows come back sorted by
day, so the number of workers changes nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.mm_bybit import archive, days, engine, funding, metrics
from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit.access import Access, block_days
from trading_research.mm_bybit.config import FeeNotSourced, fee_on
from trading_research.mm_bybit.tape import DayTape

Job = tuple[str, str]  # (cell key, variant)
QUOTE_SECONDS = slice(600, 23 * 3600 + 58 * 60)


def fetch_block(
    block: str, root: Path | str, access: Access, *, opener: Any = None,
    on_day: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """Every day of the block not yet on disk, then the block's funding."""
    records = []
    kwargs = {} if opener is None else {"opener": opener}
    for day in block_days(block):
        access.require(day)
        if archive.prints_path(root, day).exists() or archive.read_status(root, day):
            continue
        try:
            record = archive.fetch_day(day, root, access, **kwargs)
            record["status"] = "on_disk"
        except archive.MissingFile:
            record = {"day": day.isoformat(), "status": "no_prints_file"}
            archive.write_status(root, day, "no_prints_file", "HTTP 404")
        except archive.UnreadableFile as exc:
            record = {"day": day.isoformat(), "status": "unreadable", "error": str(exc)}
            archive.write_status(root, day, "unreadable", str(exc))
        records.append(record)
        if on_day is not None:
            on_day(record)
    funding.fetch_block(block, root, access, **kwargs)
    return records


def load_day(
    day: date, root: Path | str, access: Access, block: str, cfg: dict[str, Any]
) -> DayTape | str:
    try:
        fees: Any = fee_on(cfg, day)
    except FeeNotSourced as exc:
        fees = exc
    return days.load(day, root, access, funding.load_block(root, block), fees)


def w_inputs(root: Path | str, access: Access, cfg: dict[str, Any]) -> dict[str, Any]:
    """sigma_ref and the quantiles from every whole second 00:10 to 23:58 of W's usable days."""
    samples, excluded, spec = [], [], {}
    for day in block_days("W"):
        tape = load_day(day, root, access, "W", cfg)
        if isinstance(tape, str):
            excluded.append([day.isoformat(), tape])
            continue
        spec[day.isoformat()] = [tape.tick, tape.lot]
        values = tape.sec_vol[QUOTE_SECONDS]
        samples.append(values[np.isfinite(values)])
    pooled = np.concatenate(samples) if samples else np.zeros(0)
    if not len(pooled):
        raise RuntimeError("W has no usable day: sigma_ref and the quantiles cannot be set")
    levels = cfg["grid"]["vol_at_most_quantile"]
    return {
        "sigma_ref_bp": float(np.median(pooled)),
        "vol_quantiles_bp": {float(k): float(np.quantile(pooled, k)) for k in levels},
        "excluded_days_W": excluded,
        "instrument_by_day_W": spec,
        "seconds_sampled": len(pooled),
    }


def _day_job(args: tuple[Any, ...]) -> tuple[str, str | list[dict[str, Any]], float]:
    day, root, access, block, cfg, jobs, inputs, source_day = args
    started = time.perf_counter()
    tape = load_day(day, root, access, block, cfg)
    if isinstance(tape, str):
        return day.isoformat(), tape, time.perf_counter() - started
    shifted = None
    if source_day is not None:
        source = load_day(source_day, root, access, block, cfg)
        if isinstance(source, str):
            raise RuntimeError(f"{day}: the shifted state's source {source_day} is {source}")
        shifted = {"shifted": (source.sec_spread, source.sec_vol)}
    cells = [(q.from_key(key), variant) for key, variant in jobs]
    rows = days.run_jobs(tape, cells, cfg, inputs, shifted=shifted)
    market = {str(h): metrics.market_markout_bp(tape, h) for h in metrics.HORIZONS_S}
    for row in rows:
        row["market_markout_bp"] = market
    return day.isoformat(), rows, time.perf_counter() - started


def run_block(
    block: str, root: Path | str, access: Access, cfg: dict[str, Any], jobs: Sequence[Job],
    inputs: engine.QuoterInputs, *, day_list: Sequence[date] | None = None,
    shift_sources: dict[date, date] | None = None, workers: int = 6,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, float]]:  # fmt: skip
    """Rows of every job on every day, the excluded days and their reasons, seconds per day."""
    chosen = list(day_list or block_days(block))
    tasks = [
        (d, root, access, block, cfg, list(jobs), inputs, (shift_sources or {}).get(d))
        for d in chosen
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_day_job, tasks))
    else:
        results = [_day_job(task) for task in tasks]
    rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    seconds: dict[str, float] = {}
    for day, out, spent in sorted(results, key=lambda r: r[0]):
        seconds[day] = spent
        if isinstance(out, str):
            excluded[day] = out
        else:
            rows.extend(out)
    return pd.DataFrame(rows), excluded, seconds
