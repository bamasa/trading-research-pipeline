"""The registered statistics of the market-making study, and its status rule.

Everything here is a pure function of numbers the simulator already produced;
nothing reads market data. The pre-registration
(``docs/preregistration/market_making.md``) fixes what each verdict is made of:

* the **day-level one-sided t** of a daily series, 12 degrees of freedom on
  the 13 held-out days, and its p-value (:func:`day_t`, :func:`t_sf`);
* **Holm's procedure** at 5% across the four primaries (:func:`holm`);
* the **placebo percentile**: a true difference must exceed the 95th
  percentile of its placebo distribution (:func:`exceeds_placebos`);
* the **shuffled-state shift**: whole days in [1, 12] plus an intraday offset
  of one to twenty-three hours, from a seed (:func:`state_shift_rows`);
* the **fee break-even**: the maker fee at which a strategy's net is zero,
  interpolated over the registered grid (:func:`breakeven_on_grid`);
* the **status rule**: killed, candidate or inconclusive (:func:`status`).

The p-value of Student's t is computed in closed form for integer degrees of
freedom (Abramowitz and Stegun 26.7.3 and 26.7.4), so the verdicts depend on no
optional library.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def t_cdf_two_sided(t: float, df: int) -> float:
    """``P(|T| <= t)`` for Student's t with integer ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError(f"degrees of freedom must be at least 1, got {df}")
    if not math.isfinite(t):
        return 1.0 if not math.isnan(t) else math.nan
    theta = math.atan(abs(t) / math.sqrt(df))
    s, c = math.sin(theta), math.cos(theta)
    if df % 2 == 0:
        term, total = 1.0, 1.0
        for k in range(1, df // 2):
            term *= c * c * (2 * k - 1) / (2 * k)
            total += term
        return s * total
    if df == 1:
        return 2.0 * theta / math.pi
    term, total = 1.0, 1.0
    for k in range(1, (df - 1) // 2):
        term *= c * c * (2 * k) / (2 * k + 1)
        total += term
    return 2.0 / math.pi * (theta + s * c * total)


def t_sf(t: float, df: int) -> float:
    """One-sided p-value: ``P(T >= t)``."""
    if math.isnan(t):
        return math.nan
    two = t_cdf_two_sided(t, df)
    return (1.0 - two) / 2.0 if t >= 0 else (1.0 + two) / 2.0


@dataclass(frozen=True)
class DayT:
    """A daily series' mean, its t against zero, and the one-sided p-value."""

    days: int
    mean: float
    sd: float
    t: float
    p_one_sided: float
    positive_days: int


def day_t(values: Sequence[float] | np.ndarray) -> DayT:
    """The day-level one-sided t of a daily series (non-finite days dropped)."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    n = len(array)
    if n < 2:
        mean = float(array.mean()) if n else math.nan
        return DayT(n, mean, math.nan, math.nan, math.nan, int((array > 0).sum()))
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else math.copysign(math.inf, mean)
    if mean == 0 and sd == 0:
        t = math.nan
    return DayT(n, mean, sd, t, t_sf(t, n - 1), int((array > 0).sum()))


def holm(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm's step-down procedure: which hypotheses pass at family level ``alpha``.

    The smallest p-value is compared with ``alpha / m``, the next with
    ``alpha / (m - 1)``, and so on; the first failure stops the procedure and
    every hypothesis from it onward fails. A missing (NaN) p-value fails and
    counts in ``m``.
    """
    m = len(p_values)
    ranked = sorted(p_values.items(), key=lambda item: math.inf if item[1] != item[1] else item[1])
    passed = dict.fromkeys(p_values, False)
    for rank, (name, p) in enumerate(ranked):
        if p != p or p > alpha / (m - rank):
            break
        passed[name] = True
    return passed


#: The registered placebo percentile.
PLACEBO_PERCENTILE = 95.0


def exceeds_placebos(
    true_value: float,
    placebos: Sequence[float] | np.ndarray,
    percentile: float = PLACEBO_PERCENTILE,
) -> tuple[float, bool]:
    """The placebos' ``percentile`` (linear interpolation) and whether the true
    value is strictly above it. Non-finite placebos are dropped; with none left
    the answer is NaN and False."""
    array = np.asarray(placebos, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array) or not math.isfinite(true_value):
        return math.nan, False
    level = float(np.percentile(array, percentile))
    return level, bool(true_value > level)


#: Rows of the five-second grid in an hour and in a day.
ROWS_PER_HOUR = 720
ROWS_PER_DAY = 17_280


def state_shift_rows(
    seed: int,
    *,
    whole_days: tuple[int, int] = (1, 12),
    offset_h: tuple[float, float] = (1.0, 23.0),
) -> int:
    """The shuffled-state placebo's circular shift, in five-second rows.

    A whole number of days uniform in ``whole_days`` (inclusive) plus an
    intraday offset uniform between ``offset_h`` hours, drawn on the grid, from
    ``np.random.default_rng(seed)``. With the registered bounds the shift lies
    between 25 hours and 12 days 23 hours, so on the 13-day block it never
    returns the tape to itself and never moves it by less than an hour.
    """
    rng = np.random.default_rng(seed)
    days = int(rng.integers(whole_days[0], whole_days[1] + 1))
    low = round(offset_h[0] * ROWS_PER_HOUR)
    high = round(offset_h[1] * ROWS_PER_HOUR)
    return days * ROWS_PER_DAY + int(rng.integers(low, high + 1))


def roll_values(values: np.ndarray, shift: int) -> np.ndarray:
    """``values`` moved ``shift`` places later around the circle: the value at
    position ``i`` is the one that stood at ``i - shift``."""
    return np.roll(np.asarray(values), int(shift))


def breakeven_on_grid(fees_bp: Sequence[float], nets: Sequence[float]) -> float:
    """The maker fee at which net is zero, interpolated linearly on a grid.

    Read from the top of the grid down: the highest fee at which the net is
    still positive, and the linear crossing between it and the next fee up.
    ``+inf`` when the net is positive at the highest fee (it pays across the
    whole grid), ``-inf`` when it is positive at none, NaN without data.
    """
    fees = np.asarray(fees_bp, dtype=np.float64)
    values = np.asarray(nets, dtype=np.float64)
    order = np.argsort(fees)
    fees, values = fees[order], values[order]
    ok = np.isfinite(values)
    fees, values = fees[ok], values[ok]
    if not len(fees):
        return math.nan
    positive = np.flatnonzero(values > 0)
    if not len(positive):
        return -math.inf
    top = int(positive[-1])
    if top == len(fees) - 1:
        return math.inf
    f0, f1 = fees[top], fees[top + 1]
    v0, v1 = values[top], values[top + 1]
    return float(f0 + (f1 - f0) * v0 / (v0 - v1))


# ---------------------------------------------------------------------------
# Kill conditions and the status rule
# ---------------------------------------------------------------------------

#: What a kill condition does when it fires.
KILLED, INCONCLUSIVE, UNTESTABLE = "killed", "inconclusive", "untestable"


@dataclass(frozen=True)
class KillCheck:
    """One kill condition, evaluated mechanically.

    ``value`` is the statistic the condition reads and ``reference`` what it is
    compared with (zero, a placebo percentile, a count); ``fired`` is the
    outcome of the registered comparison. ``outcome`` is what firing means:
    ``killed`` for most, ``inconclusive`` for the minimum-count conditions
    (H2.2's K5, H3's K4), ``untestable`` for H3's K3.
    """

    hypothesis: str
    name: str
    test: str
    value: float
    reference: float
    fired: bool
    outcome: str = KILLED
    note: str = ""


def k_pess(proportional: Sequence[bool], pessimistic: Sequence[bool]) -> bool:
    """K-pess: a positive verdict under proportional attribution that is not
    positive under pessimistic attribution.

    ``proportional`` and ``pessimistic`` are the hypothesis's sign claims (each
    True when it holds) under the two rules. It fires only when every claim
    holds under the default and some claim fails under the pessimistic rule;
    when the default verdict is not positive there is nothing for it to kill.
    """
    return bool(all(proportional) and not all(pessimistic))


def k_dir(making: float, net: float) -> bool:
    """K-dir: the making part is not positive while the net is."""
    return bool(net > 0 and not making > 0)


def status(checks: Sequence[KillCheck], passes_holm: bool) -> tuple[str, list[str]]:
    """The registered status and the conditions that decided it.

    ``killed`` if any killing condition fired; else ``untestable`` or
    ``inconclusive`` if a condition with that outcome fired; else
    ``candidate`` if the day-level t passes Holm's procedure, and
    ``inconclusive`` (because of significance) if it does not. ``conditional``
    needs the boundary block and is never returned here.
    """
    for outcome in (KILLED, UNTESTABLE, INCONCLUSIVE):
        fired = [c.name for c in checks if c.fired and c.outcome == outcome]
        if fired:
            return outcome, fired
    if passes_holm:
        return "candidate", []
    return INCONCLUSIVE, ["Holm"]
