"""The finalists page: one row per slot and S25-Z on each held-out block.

Each row carries the status, the first kill with its value against its
reference (``verdicts.KillCheck``), the daily series' mean, day-level t,
Newey-West t and median day, the trades, the MDE, the placebo percentile, the
K-nbhd median, the fee break-even per side and how many published tiers it
clears, the mean on H before 2025-08-01, the regime attribution and the
registered sentence (``headlines.slot_on_block``, which also checks that the
status and the kills agree). The
statistics are computed here from the slot's daily series, as the bar
computes them; the statuses and kills come from the entry point and are only
checked against the registered lists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier, tiers_for
from trading_research.evaluation.family import FAMILY
from trading_research.evaluation.significance import newey_west_t
from trading_research.market_making.verdicts import KillCheck, day_t
from trading_research.pipeline.regime import Attribution
from trading_research.reporting.headlines import (
    BLOCKS,
    BP_SLOTS,
    KILL_ORDER,
    HeadlineError,
    first_kill,
    slot_on_block,
)

FINALIST_COLUMNS: tuple[str, ...] = (
    "block",
    "slot",
    "status",
    "first_kill",
    "kill_value",
    "kill_reference",
    "mean",
    "t",
    "newey_west_t",
    "median_day",
    "days",
    "trades",
    "mde",
    "placebo_percentile",
    "k_nbhd_median",
    "break_even_per_side",
    "tiers_paying",
    "h_before_0801_mean",
    "regime_difference",
    "regime_random_p95",
    "unit",
    "sentence",
)
ROW_ORDER: tuple[str, ...] = (*FAMILY, "S25-Z")
#: The passive slots' break-even is a maker fee, the taker slots' a taker fee.
MAKER_SIDE = ("S-P", "S-M", "S25-Z")
PASSIVE_UNIT = "per 100 USDT of clip a day"
NAN = math.nan


@dataclass(frozen=True)
class FinalistInputs:
    """What the entry point computed for one slot on one block."""

    slot: str
    block: str
    status: str
    daily: Sequence[float] | np.ndarray
    trades: int
    kills: tuple[KillCheck, ...] = ()
    mde: float = NAN
    placebo_percentile: float = NAN
    k_nbhd_median: float = NAN
    break_even_per_side: float = NAN
    h_before_0801_mean: float = NAN
    attribution: Attribution | None = None


def tiers_paying(slot: str, break_even: float, tiers: Sequence[FeeTier] | None = None) -> int:
    """How many published tiers for BTCUSDT charge less than the break-even on
    the side it is read on (the maker fee for the passive slots)."""
    if not math.isfinite(break_even) and not math.isinf(break_even):
        return 0
    rows = tiers if tiers is not None else tiers_for("g1")
    fees = [t.maker_bp if slot in MAKER_SIDE else t.taker_bp for t in rows]
    return int(sum(fee < break_even for fee in fees))


def finalist_row(inputs: FinalistInputs) -> dict[str, object]:
    """One row of the page."""
    if inputs.slot not in KILL_ORDER or inputs.block not in BLOCKS:
        raise HeadlineError(f"{inputs.slot!r} on {inputs.block!r} is not a registered row")
    sentence = slot_on_block(inputs.slot, inputs.block, inputs.status, inputs.kills)
    values = np.asarray(inputs.daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    stats = day_t(values)
    kill = first_kill(inputs.slot, inputs.kills)
    found = inputs.attribution
    return {
        "block": inputs.block,
        "slot": inputs.slot,
        "status": inputs.status,
        "first_kill": "" if kill is None else kill.name,
        "kill_value": NAN if kill is None else kill.value,
        "kill_reference": NAN if kill is None else kill.reference,
        "mean": stats.mean,
        "t": stats.t,
        "newey_west_t": newey_west_t(values).t if len(values) >= 2 else NAN,
        "median_day": float(np.median(values)) if len(values) else NAN,
        "days": stats.days,
        "trades": int(inputs.trades),
        "mde": inputs.mde,
        "placebo_percentile": inputs.placebo_percentile,
        "k_nbhd_median": inputs.k_nbhd_median,
        "break_even_per_side": inputs.break_even_per_side,
        "tiers_paying": tiers_paying(inputs.slot, inputs.break_even_per_side),
        "h_before_0801_mean": inputs.h_before_0801_mean,
        "regime_difference": NAN if found is None else found.difference,
        "regime_random_p95": NAN if found is None else found.reference,
        "unit": "bp of traded notional" if inputs.slot in BP_SLOTS else PASSIVE_UNIT,
        "sentence": sentence,
    }


def finalists_table(rows: Sequence[FinalistInputs]) -> pd.DataFrame:
    """The page's table: H, then F, then P; the family's order, S25-Z last."""
    table = pd.DataFrame([finalist_row(r) for r in rows], columns=list(FINALIST_COLUMNS))
    if table.duplicated(["block", "slot"]).any():
        raise HeadlineError("a slot is listed twice on one block")
    order = table["block"].map(BLOCKS.index) * 10 + table["slot"].map(ROW_ORDER.index)
    return table.iloc[np.argsort(order.to_numpy(), kind="stable")].reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    """A GitHub table of ``frame``: numbers to ``digits`` places, NaN blank."""

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "" if math.isnan(value) else f"{value:.{digits}f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    lines.append("|" + "|".join("---" for _ in frame.columns) + "|")
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)
