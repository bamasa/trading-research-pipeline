"""The choice on D, and the whole choice re-run under Null A and Null B.

Inputs are the D matrices over the 40 cells (rows, in :func:`quoters.grid`'s
order) and D's usable days (columns, calendar order): ``y`` (cells x days),
``hourly`` (cells x days x 24, a day's hours summing to its ``y``) and the
passive fills (cells x days).

* **S** of a cell is the median over its neighbourhood of each cell's mean ``y``;
  a cell is **admissible** when it averages at least 20 passive fills a day; F is
  the admissible cell with the highest S, ties to its own mean, then to the
  key's byte order (the earliest key wins).
* **A** is the median over the chosen cell's neighbourhood of each cell's
  day-level t of ``y``, 0 for a cell with fewer than two days or no dispersion.
* **Null A**: draw ``b`` multiplies every cell's hour by
  ``placebo_search.null_a_signs(b, hours)`` (hours in time order), sums each day.
* **Null B**: one balanced pattern, ``+1`` where ``default_rng(0).permutation(n)``
  is even, ``-1`` where odd; shift ``s`` (7 to n - 7) multiplies day ``i`` by the
  pattern at ``(i + s) mod n``.

Each replication re-runs the filter, the medians, the choice and A.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from trading_research.mm_bybit import quoters as q
from trading_research.validation.placebo_search import null_a_signs, placebo_p

MIN_FILLS_PER_DAY = 20.0


def neighbour_index(cells: Sequence[q.Cell]) -> list[np.ndarray]:
    position = {cell: i for i, cell in enumerate(cells)}
    return [np.array([position[n] for n in q.neighbourhood(cell)]) for cell in cells]


def t_rows(y: np.ndarray) -> np.ndarray:
    """Each row's day-level t; 0 with fewer than two days or no dispersion."""
    n = y.shape[1]
    if n < 2:
        return np.zeros(y.shape[0])
    sd = y.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = y.mean(axis=1) / (sd / np.sqrt(n))
    return np.where(sd > 0, t, 0.0)


@dataclass(frozen=True)
class Choice:
    index: int  # -1 when no cell is admissible
    s: float
    a: float
    peak: int  # the highest S whatever the fills


def choose(
    y: np.ndarray, admissible: np.ndarray, nbhd: list[np.ndarray], keys: Sequence[str]
) -> Choice:
    means = y.mean(axis=1)
    s = np.array([np.median(means[idx]) for idx in nbhd])
    order = sorted(range(len(keys)), key=lambda i: (-s[i], -means[i], keys[i]))
    peak = order[0]
    chosen = next((i for i in order if admissible[i]), -1)
    if chosen < 0:
        return Choice(-1, float("nan"), 0.0, peak)
    t = t_rows(y)
    return Choice(chosen, float(s[chosen]), float(np.median(t[nbhd[chosen]])), peak)


def null_b_pattern(n: int, seed: int = 0) -> np.ndarray:
    return np.where(np.random.default_rng(seed).permutation(n) % 2 == 0, 1.0, -1.0)


def null_b_shifts(n: int, low: int = 7, margin: int = 7) -> range:
    return range(low, n - margin + 1)


@dataclass(frozen=True)
class NullResult:
    a: np.ndarray  # A of every replication
    count: int  # replications with A at or above the observed
    p: float


def _result(a: np.ndarray, observed: float) -> NullResult:
    count = int((a >= observed).sum())
    return NullResult(a, count, placebo_p(count, len(a)))


def null_a(
    hourly: np.ndarray, admissible: np.ndarray, nbhd: list[np.ndarray],
    keys: Sequence[str], observed: float, draws: int = 999,
) -> NullResult:  # fmt: skip
    cells, days, hours = hourly.shape
    flat = hourly.reshape(cells, days * hours)
    a = np.empty(draws)
    for b in range(draws):
        signs = null_a_signs(b, days * hours).astype(np.float64)
        y = (flat * signs).reshape(cells, days, hours).sum(axis=2)
        a[b] = choose(y, admissible, nbhd, keys).a
    return _result(a, observed)


def null_b(
    y: np.ndarray, admissible: np.ndarray, nbhd: list[np.ndarray],
    keys: Sequence[str], observed: float, seed: int = 0,
) -> NullResult:  # fmt: skip
    n = y.shape[1]
    pattern = null_b_pattern(n, seed)
    shifts = null_b_shifts(n)
    a = np.empty(len(shifts))
    for k, shift in enumerate(shifts):
        signs = pattern[(np.arange(n) + shift) % n]
        a[k] = choose(y * signs, admissible, nbhd, keys).a
    return _result(a, observed)
