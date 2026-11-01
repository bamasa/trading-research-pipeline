"""The registered statuses on H: kills, the bar, Holm over {F, S0}, labels, sentences.

Precedence, one status per primary: void (K-acct, K-limit); inconclusive (too
few fills: under 100 passive fills); killed (any kill, named in the registered
order); inconclusive (capacity: the median day's capacity above 10%, after
Holm); candidate (the bar: Holm at 5% with m = 2, t >= 3, Newey-West t >= 2,
median day > 0, at least 20 days); otherwise inconclusive, "below the minimum
detectable effect" when |mean| is under the MDE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trading_research.evaluation.significance import newey_west_t
from trading_research.market_making.verdicts import day_t, exceeds_placebos, holm
from trading_research.validation.placebo_search import null_a_signs

KILL_ORDER = ("K1", "K-dir", "K-delay", "K-latency", "K-fund", "K-drift", "K-days",
              "K-nbhd", "K-placebo")  # fmt: skip
PLACEBO_DRAWS = 200


@dataclass
class Primary:
    """What one primary's verdict reads, all daily series over H's usable days."""

    name: str
    cell: str
    y: np.ndarray
    making: np.ndarray
    inventory: np.ndarray
    funding: np.ndarray
    ex_rebate: np.ndarray
    touch: np.ndarray
    delay: np.ndarray
    latency150: np.ndarray
    hourly: np.ndarray  # days x 24
    fills: int
    capacity: np.ndarray
    neighbour_means: list[float]
    shifted_mean: float | None = None
    void: list[str] = field(default_factory=list)
    #: Why the primary enters Holm with p = 1 whatever its series ("" if it does not).
    holm_one: str = ""
    mde: float = math.nan


def placebo_means(hourly: np.ndarray, draws: int = PLACEBO_DRAWS) -> np.ndarray:
    days = hourly.shape[0]
    flat = hourly.reshape(-1)
    return np.array(
        [float((flat * null_a_signs(b, len(flat))).reshape(days, 24).sum(axis=1).mean())
         for b in range(draws)]
    )  # fmt: skip


def kills(p: Primary) -> list[dict[str, Any]]:
    """Every kill condition with its value, reference and whether it fires."""
    mean = float(np.mean(p.y)) if len(p.y) else math.nan
    positive = mean > 0
    level, beats = exceeds_placebos(mean, placebo_means(p.hourly))
    out = [
        ("K1", mean, 0.0, not positive),
        ("K-dir", float(np.mean(p.making)), 0.0, not float(np.mean(p.making)) > 0),
        ("K-delay", float(np.mean(p.delay)), 0.0, positive and not float(np.mean(p.delay)) > 0),
        ("K-latency", float(np.mean(p.latency150)), 0.0,
         positive and not float(np.mean(p.latency150)) > 0),
        ("K-fund", mean - float(np.mean(p.funding)), 0.0,
         positive and not mean - float(np.mean(p.funding)) > 0),
        ("K-drift", float(np.mean(p.inventory)), 0.5 * mean,
         positive and float(np.mean(p.inventory)) > 0.5 * mean),
        ("K-days", float(np.mean(p.y > 0)), 0.5, not float(np.mean(p.y > 0)) > 0.5),
        ("K-nbhd", float(np.median(p.neighbour_means)), 0.0,
         not float(np.median(p.neighbour_means)) > 0),
        ("K-placebo", mean, level, not beats),
    ]  # fmt: skip
    rows = [{"kill": k, "value": v, "reference": r, "fires": bool(f)} for k, v, r, f in out]
    if p.shifted_mean is not None:
        rows[-1]["shifted_mean"] = p.shifted_mean
        if not mean > p.shifted_mean:
            rows[-1]["fires"] = True
    return rows


def assess(
    primaries: list[Primary], *, alpha: float = 0.05, family: tuple[str, ...] = ("F", "S0")
) -> dict[str, dict[str, Any]]:
    """Every primary's status, kills, statistics and labels; Holm across the family.

    The family is {F, S0} whatever is run: a member that is not assessed (F not
    run, for one) enters Holm with p = 1, so m stays 2.
    """
    stats: dict[str, dict[str, Any]] = {}
    for p in primaries:
        t = day_t(p.y)
        nw = newey_west_t(p.y, lag=5)
        k = kills(p)
        placebo = next(r for r in k if r["kill"] == "K-placebo")
        nbhd = next(r for r in k if r["kill"] == "K-nbhd")
        stats[p.name] = {
            "cell": p.cell, "days": t.days, "mean_y": t.mean, "t": t.t,
            "p_one_sided": t.p_one_sided, "newey_west_t": nw.t,
            "median_day": float(np.median(p.y)) if len(p.y) else math.nan,
            "fills": p.fills, "capacity_median": float(np.median(p.capacity)) if len(
                p.capacity) else 0.0, "kills": k, "mde": p.mde,
            "placebo_95th": placebo["reference"], "shifted_mean": p.shifted_mean,
            "nbhd_median": nbhd["value"], "void": list(p.void), "holm_one": p.holm_one,
            "first_kill": next((r for r in k if r["fires"]), None),
            "labels": labels(p),
        }  # fmt: skip
    p_values = dict.fromkeys(family, 1.0)
    for p in primaries:
        s = stats[p.name]
        enters = not (p.void or p.fills < 100 or s["first_kill"] or p.holm_one)
        p_values[p.name] = s["p_one_sided"] if enters else 1.0
    passed = holm(p_values, alpha)
    for p in primaries:
        s = stats[p.name]
        s["holm_m"] = len(p_values)
        s["holm_p"], s["holm_passes"] = p_values[p.name], passed[p.name]
        s["status"] = status(p, s)
        s["status_as_a_primary"] = status(p, s, labelled=False)
    return stats


def status(p: Primary, s: dict[str, Any], *, labelled: bool = True) -> str:
    """The registered precedence. A finalist not admitted on D, or the same as S0,
    carries that label as its status (it is a labelled measurement); with
    ``labelled=False`` the precedence is applied to it as if it were a primary."""
    if p.void:
        return f"void ({', '.join(p.void)})"
    if labelled and p.holm_one:
        return p.holm_one
    if p.fills < 100:
        return "inconclusive (too few fills)"
    if s["first_kill"]:
        return "killed"
    bar = (s["holm_passes"] and s["t"] >= 3.0 and s["newey_west_t"] >= 2.0
           and s["median_day"] > 0 and s["days"] >= 20)  # fmt: skip
    if bar and s["capacity_median"] > 0.10:
        return "inconclusive (capacity)"
    if bar:
        return "candidate"
    below = abs(s["mean_y"]) < p.mde if p.mde == p.mde else False
    return "inconclusive, below the minimum detectable effect" if below else "inconclusive"


def labels(p: Primary) -> dict[str, bool]:
    with_rebate = float(np.mean(p.y)) > 0
    ex_rebate = float(np.mean(p.ex_rebate)) > 0 and float(np.mean(p.making)) > 0
    return {
        "paid_with_the_rebate": with_rebate,
        "paid_ex_rebate": ex_rebate,
        "paid_at_the_touch_only": not with_rebate and float(np.mean(p.touch)) > 0,
    }
