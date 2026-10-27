"""The verdict on the held-out block: the registered statistics, the kills in order, the placebos, the fee ladder.

Every number is computed by the repository's verdict code:

* the **clustered statistic**: each attempt's net, clustered by day
  (:func:`trading_research.evaluation.significance.assess`), and the daily
  series' one-sided day-level t (:func:`trading_research.market_making.verdicts.day_t`);
* **Holm's procedure** across the execution family, m fixed by the
  registration before any read (:func:`~trading_research.market_making.verdicts.holm`);
* the **placebos**: the same decisions moved around the block by the
  shuffled-state shift (:func:`~trading_research.market_making.verdicts.state_shift_rows`,
  :func:`~trading_research.market_making.verdicts.roll_values`) and executed
  again, the result required strictly above their 95th percentile
  (:func:`~trading_research.market_making.verdicts.exceeds_placebos`);
* the **kill conditions**, in the registered order, and the status they give
  (:func:`~trading_research.market_making.verdicts.status`).

The kill conditions themselves are written into the search's registration
before any data is read (:data:`KILLS`); this module only evaluates them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from trading_research.evaluation.significance import T_BAR, SignificanceError, assess
from trading_research.market_making.verdicts import (
    INCONCLUSIVE,
    KillCheck,
    day_t,
    exceeds_placebos,
    holm,
    roll_values,
    state_shift_rows,
    status,
)

#: The kill conditions, in the order they are applied: written into the
#: registration before any data is read.
KILLS: tuple[dict[str, str], ...] = (
    {
        "name": "K1",
        "test": "the mean of the held-out daily net per attempt at or below 0 bp",
        "outcome": "killed",
    },
    {
        "name": "K-placebo",
        "test": "the held-out mean net per attempt not strictly above the placebos' 95th percentile",
        "outcome": "killed",
    },
    {
        "name": "K-days",
        "test": "fewer days with an attempt than the registration's minimum",
        "outcome": "inconclusive",
    },
)


def daily(attempts: pd.DataFrame) -> pd.Series:
    """Each day's mean net per attempt, misses included."""
    if attempts.empty:
        return pd.Series(dtype=float)
    return attempts.groupby("day")["net_bp"].mean()


def shifts(rows: int, draws: int, seed: int) -> list[int]:
    """``draws`` circular shifts of a block of ``rows`` five-second rows, by the
    registered shuffled-state rule, folded onto the block; a shift within an hour
    of no shift is drawn again from the next seed."""
    found: list[int] = []
    hour = 720
    current = seed
    while len(found) < draws and rows > 2 * hour:
        shift = state_shift_rows(current) % rows
        current += 1
        if hour <= shift <= rows - hour:
            found.append(int(shift))
        if current - seed > 100 * draws:
            break
    return found


def rolled(decision: np.ndarray, shift: int) -> np.ndarray:
    """The decisions moved ``shift`` rows later around the block."""
    return np.asarray(roll_values(np.asarray(decision), shift))


def placebo_values(
    run: Callable[[np.ndarray], pd.DataFrame], decision: np.ndarray, shift_list: Sequence[int]
) -> list[float]:
    """The mean net per attempt of the decisions under each shift."""
    values = []
    for shift in shift_list:
        attempts = run(rolled(decision, shift))
        values.append(float(attempts["net_bp"].mean()) if len(attempts) else math.nan)
    return values


def assess_mode(
    name: str,
    attempts: pd.DataFrame,
    placebos: Sequence[float],
    *,
    min_days: int,
) -> dict[str, Any]:
    """One mode's numbers and its kill checks, before Holm."""
    series = daily(attempts)
    found = day_t(series.to_numpy())
    mean = float(attempts["net_bp"].mean()) if len(attempts) else math.nan
    level, exceeds = exceeds_placebos(mean, placebos)
    clustered: dict[str, Any]
    try:
        clustered = assess(attempts, value="net_bp", cluster="day", series=None).to_dict()
    except SignificanceError as exc:
        clustered = {"note": str(exc)}
    checks = [
        KillCheck(name, "K1", KILLS[0]["test"], found.mean, 0.0, fired=not found.mean > 0),
        KillCheck(name, "K-placebo", KILLS[1]["test"], mean, level, fired=not exceeds),
        KillCheck(
            name,
            "K-days",
            KILLS[2]["test"],
            float(found.days),
            float(min_days),
            fired=found.days < min_days,
            outcome=INCONCLUSIVE,
        ),
    ]
    filled = attempts["filled"].astype(bool) if len(attempts) else pd.Series(dtype=bool)
    return {
        "mode": name,
        "attempts": len(attempts),
        "fill_rate": float(filled.mean()) if len(filled) else math.nan,
        "gross_per_attempt_bp": float(attempts["gross_bp"].mean()) if len(attempts) else math.nan,
        "cost_per_attempt_bp": float(attempts["cost_bp"].mean()) if len(attempts) else math.nan,
        "net_per_attempt_bp": mean,
        "days": found.days,
        "day_mean_bp": found.mean,
        "day_t": found.t,
        "p_one_sided": found.p_one_sided,
        "positive_days": found.positive_days,
        "clustered": clustered,
        "placebos": len([p for p in placebos if math.isfinite(p)]),
        "placebo_p95_bp": level,
        "checks": checks,
    }


def decide(modes: Mapping[str, dict[str, Any]], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm across the family, then each mode's status from its kills in order."""
    p_values = {
        name: (row["p_one_sided"] if math.isfinite(row["p_one_sided"]) else 1.0)
        for name, row in modes.items()
    }
    passed = holm(p_values, alpha)
    out: dict[str, dict[str, Any]] = {}
    for name, row in modes.items():
        bar = passed[name] and math.isfinite(row["day_t"]) and row["day_t"] >= T_BAR
        verdict, fired = status(row["checks"], bar)
        checks = [
            {
                "condition": c.name,
                "test": c.test,
                "value": c.value,
                "reference": c.reference,
                "fired": c.fired,
                "outcome": c.outcome,
            }
            for c in row["checks"]
        ]
        out[name] = {
            **row,
            "holm": passed[name],
            "status": verdict,
            "decided_by": fired,
            "checks": checks,
        }
    return out
