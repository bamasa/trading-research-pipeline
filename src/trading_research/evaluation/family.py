"""The family of finalists on a held-out block, and the bar each must clear.

Each slot's verdict is made from its daily series only:

* the **one-sided day-level p-value** of the series' mean, Student t at days
  minus one (``market_making.verdicts.day_t`` and ``t_sf``); a series of
  fewer than two days enters with p = 1 (:func:`slot_p_value`);
* **Holm's procedure** across the fixed family of five, m = 5 on every block:
  a slot that is killed, void, not run, not admitted, the same as S25 or not
  taken to the block enters with p = 1, so m never depends on an outcome
  (:func:`family_holm`);
* the **bar** (:func:`bar`): Holm passes, day-level t at least 3, Newey-West
  t at least 2, median day above 0, and enough days with a trade;
* the **placebo comparison**, through ``verdicts.exceeds_placebos``
  (:func:`k_placebo`).

The precedence of statuses and the kill conditions are applied by the
caller; this module only computes the numbers they read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trading_research.evaluation.significance import NEWEY_WEST_LAG, T_BAR, newey_west_t
from trading_research.market_making.verdicts import day_t, exceeds_placebos, holm, t_sf

#: The five finalists, fixed before any data.
FAMILY = ("S25", "S-C", "S-R", "S-P", "S-M")
ALPHA = 0.05
NEWEY_WEST_BAR = 2.0
#: Days with a trade a series needs (usable days for S-P and S-M), per block.
MIN_DAYS = {"H": 20, "F": 10, "P": 10}


def slot_p_value(daily: Sequence[float] | np.ndarray) -> float:
    """The one-sided day-level p-value; 1 when the series has fewer than two
    days or no defined t."""
    found = day_t(daily)
    if found.days < 2 or not math.isfinite(found.p_one_sided):
        return 1.0
    return found.p_one_sided


def family_holm(
    p_values: Mapping[str, float | None],
    family: Sequence[str] = FAMILY,
    alpha: float = ALPHA,
) -> dict[str, bool]:
    """Holm at ``alpha`` across the whole family; a member that is absent, None
    or NaN enters with p = 1. A name outside the family is refused."""
    outside = set(p_values) - set(family)
    if outside:
        raise ValueError(f"not in the family: {sorted(outside)}")
    filled: dict[str, float] = {}
    for name in family:
        value = p_values.get(name)
        filled[name] = 1.0 if value is None or not math.isfinite(value) else float(value)
    return holm(filled, alpha)


def holm_first_step_t(df: int, m: int = len(FAMILY), alpha: float = ALPHA) -> float:
    """The day-level t at which the one-sided p equals Holm's first step,
    ``alpha / m``, at ``df`` degrees of freedom (bisection on ``t_sf``)."""
    target = alpha / m
    low, high = 0.0, 50.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if t_sf(middle, df) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


@dataclass(frozen=True)
class Bar:
    """A slot's daily series against the bar, each condition separately."""

    days: int
    mean: float
    t: float
    p_one_sided: float
    newey_west_t: float
    median: float
    holm: bool
    t_ok: bool
    newey_west_ok: bool
    median_ok: bool
    days_ok: bool

    @property
    def passes(self) -> bool:
        return self.holm and self.t_ok and self.newey_west_ok and self.median_ok and self.days_ok


def bar(
    daily: Sequence[float] | np.ndarray,
    passes_holm: bool,
    min_days: int,
    t_bar: float = T_BAR,
    newey_west_bar: float = NEWEY_WEST_BAR,
    lag: int = NEWEY_WEST_LAG,
) -> Bar:
    """The bar on a daily series. ``passes_holm`` comes from
    :func:`family_holm`; ``min_days`` from :data:`MIN_DAYS` for the block. A
    days-with-a-trade count for a passive slot counts its usable days, which is
    the length of its series."""
    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    found = day_t(values)
    nw = newey_west_t(values, lag).t
    median = float(np.median(values)) if len(values) else math.nan
    return Bar(
        days=found.days,
        mean=found.mean,
        t=found.t,
        p_one_sided=slot_p_value(values),
        newey_west_t=nw,
        median=median,
        holm=bool(passes_holm),
        t_ok=bool(found.t >= t_bar),
        newey_west_ok=bool(nw >= newey_west_bar),
        median_ok=bool(median > 0),
        days_ok=bool(found.days >= min_days),
    )


def k_placebo(true_value: float, placebos: Sequence[float] | np.ndarray) -> bool:
    """K-placebo fires when the result is not strictly above the 95th
    percentile of its placebos (or the percentile is undefined)."""
    _, exceeds = exceeds_placebos(true_value, placebos)
    return not exceeds


def passes_alone(p_claims: Sequence[float], alpha: float = ALPHA, m: int = len(FAMILY) + 1) -> bool:
    """A hypothesis outside the family, tested alone at ``alpha / m`` on the
    larger of its claims' one-sided p-values (S25-Z: alpha / 6)."""
    values = [float(p) for p in p_claims]
    if not values or not all(math.isfinite(p) for p in values):
        return False
    return max(values) <= alpha / m
