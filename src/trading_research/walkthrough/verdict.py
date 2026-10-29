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


#: The direction-accuracy rungs :func:`headroom` re-prices: the share of filled
#: trades whose direction a better forecast would have had right.
HIT_RUNGS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)


def headroom(attempts: pd.DataFrame, mode: str) -> dict[str, Any]:
    """What the result is short of, read from the same attempts: no re-run, no new data.

    Two levers, each an arithmetic restatement of the attempts, not a result:

    * **cost**: the gross per attempt is the most any fee and slippage could leave;
      ``cost_to_break_even_bp`` is the cost per attempt at which the net is zero,
      and is absent when the gross is not positive, because then no cost does;
    * **direction** (taker and passive only, whose trades are directional calls):
      if a better forecast had the direction right on a further share ``d`` of
      the filled trades, each such trade turns a loss of ``g`` into a gain of
      ``g``. Taking the trades it fixes to be average losers, the net per attempt
      rises by ``fill_rate * 2 * d * mean_loss``. ``hit_gain_to_break_even`` is
      the ``d`` that brings the net to zero. The assumption is stated on the
      page: a better model would also change which trades it takes, so this
      says how far the result is from zero in units a forecaster can act on,
      and does not forecast what any model would earn.

    The market maker's fills are not directional calls, so it gets the cost
    lever alone; its fee ladder is the finer statement of the same thing.
    """
    out: dict[str, Any] = {"mode": mode}
    if attempts.empty:
        return {**out, "note": "no attempts"}
    net = float(attempts["net_bp"].mean())
    gross = float(attempts["gross_bp"].mean())
    cost = float(attempts["cost_bp"].mean())
    out.update(
        net_per_attempt_bp=net,
        gross_per_attempt_bp=gross,
        cost_per_attempt_bp=cost,
        cost_to_break_even_bp=gross if gross > 0 else math.nan,
        cost_cut_to_break_even_bp=(cost - gross) if gross > 0 and net < 0 else math.nan,
    )
    if mode == "maker":
        return {**out, "direction": None,
                "note": "the market maker's fills are not directional calls: the cost lever only"}  # fmt: skip
    filled_mask = attempts["filled"].astype(bool).to_numpy()
    filled = attempts.loc[filled_mask]
    if filled.empty:
        return {**out, "direction": None, "note": "no filled trade"}
    fill_rate = float(filled_mask.mean())
    hit = float((filled["gross_bp"] > 0).mean())
    losers = filled.loc[filled["gross_bp"] < 0, "gross_bp"]
    mean_loss = float(-losers.mean()) if len(losers) else 0.0
    per_point = fill_rate * 2.0 * mean_loss
    rungs = [
        {"hit_gain": d, "hit_rate": hit + d, "net_per_attempt_bp": net + per_point * d}
        for d in HIT_RUNGS
        if hit + d <= 1.0
    ]
    if net >= 0:
        needed = 0.0
    elif per_point > 0 and -net / per_point <= 1.0 - hit:
        needed = -net / per_point
    else:
        needed = math.nan
    out["direction"] = {
        "filled_trades": len(filled),
        "fill_rate": fill_rate,
        "hit_rate": hit,
        "mean_loss_bp": mean_loss,
        "hit_gain_to_break_even": needed,
        "rungs": rungs,
        "assumption": "the trades a better forecast fixes are average losers; it would also change which trades are taken",
    }
    return out
