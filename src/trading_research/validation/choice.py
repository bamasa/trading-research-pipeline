"""The registered choice on the development period, as a re-runnable function.

The search scores every cell of its catalogue on every test day and then
chooses one cell per stratum. A placebo search is only honest if it repeats
that choice exactly on placebo outcomes, so the choice lives here as a pure
function of a cell-by-day matrix, with nothing read from disk and no state:

* :class:`CellDays` holds, for each cell and test day, the trades, the day's
  ``net_bp_trade`` and the day's mean G (NaN on a day without a trade);
* :func:`neighbourhood_index` lists each cell's neighbours once, as a padded
  integer matrix, from ordered axes given as numbers;
* :func:`choose` applies the rule: S is the neighbourhood median of each
  cell's mean daily net, the cell must average at least five trades a day,
  the highest S wins and ties go to the cell's own mean;
* :func:`statistic_a` is the admission statistic: the neighbourhood median of
  each cell's day-level t of G;
* :func:`carry_forward` re-runs the choice month by month and scores the
  chosen cell on the next month after an embargo.

A cell without a trade scores 0 wherever a cell value enters a median, so a
sparse corner of the grid cannot drop out of its neighbours' medians.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Trades a day a cell must average over the test days to be chosen.
MIN_TRADES_PER_DAY = 5.0


class ChoiceError(ValueError):
    """The inputs cannot describe a grid of cells over days."""


@dataclass(frozen=True)
class CellDays:
    """Per-cell daily values over the test days; every array is cells x days.

    ``net_bp`` is a day's ``net_bp_trade`` (the day's net over its entry
    notional, x 1e4) and ``g_bp`` the mean of G over the day's trades. Both are
    NaN on a day without a trade, where ``trades`` is 0.
    """

    trades: np.ndarray
    net_bp: np.ndarray
    g_bp: np.ndarray

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.trades, self.net_bp, self.g_bp)}
        if len(shapes) != 1 or self.trades.ndim != 2:
            raise ChoiceError(f"trades, net_bp and g_bp must share one 2-D shape, got {shapes}")

    @property
    def cells(self) -> int:
        return int(self.trades.shape[0])

    @property
    def days(self) -> int:
        return int(self.trades.shape[1])

    def on_days(self, columns: np.ndarray | Sequence[int]) -> CellDays:
        """The same cells on a subset of the days (positions or a mask)."""
        index = np.asarray(columns)
        return CellDays(self.trades[:, index], self.net_bp[:, index], self.g_bp[:, index])

    @classmethod
    def from_ledger(
        cls, rows: pd.DataFrame, keys: Sequence[str], days: Sequence[object]
    ) -> CellDays:
        """Pivot trial-ledger rows of one run (``key``, ``day``, ``trades``,
        ``net_bp``, ``G_bp``) onto the given cell and day order.

        A (key, day) pair without a row, or with no trade, is a day without a
        trade. A row whose key or day is not in the given order is refused.
        """
        unknown_keys = set(rows["key"]) - set(keys)
        unknown_days = set(rows["day"]) - set(days)
        if unknown_keys or unknown_days:
            raise ChoiceError(
                f"rows outside the grid: {len(unknown_keys)} keys, {len(unknown_days)} days"
            )
        if rows.duplicated(["key", "day"]).any():
            raise ChoiceError("two ledger rows share one (key, day); pass one run")
        row = pd.Index(list(keys)).get_indexer(pd.Index(rows["key"]))
        col = pd.Index(list(days), dtype=object).get_indexer(pd.Index(rows["day"], dtype=object))
        trades = np.zeros((len(keys), len(days)), dtype=np.int64)
        net = np.full(trades.shape, np.nan)
        g = np.full(trades.shape, np.nan)
        count = rows["trades"].to_numpy(dtype=np.int64)
        traded = count > 0
        trades[row, col] = count
        net[row[traded], col[traded]] = rows["net_bp"].to_numpy(dtype=np.float64)[traded]
        g[row[traded], col[traded]] = rows["G_bp"].to_numpy(dtype=np.float64)[traded]
        return cls(trades, net, g)


def cell_means(values: np.ndarray) -> np.ndarray:
    """Each row's mean over its finite entries; 0 for a row with none."""
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    total = np.where(finite, values, 0.0).sum(axis=-1)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def cell_day_t(values: np.ndarray) -> np.ndarray:
    """Each row's day-level t against zero over its finite entries.

    The same t as ``market_making.verdicts.day_t``; a row with fewer than two
    days, or with no dispersion, scores 0 (the null's statistic must be finite).
    """
    finite = np.isfinite(values)
    n = finite.sum(axis=-1)
    mean = cell_means(values)
    centred = np.where(finite, values - mean[..., None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt((centred**2).sum(axis=-1) / (n - 1))
        t = mean / (sd / np.sqrt(n))
    return np.where((n >= 2) & np.isfinite(t), t, 0.0)


def neighbourhood_index(
    table: pd.DataFrame, axes: Sequence[str], by: Sequence[str] = ()
) -> np.ndarray:
    """Each cell's neighbourhood as a padded matrix of row positions (-1 pads).

    The block of ``validation/search.neighbourhood_scores``: every cell within
    one step on every ordered axis, diagonals and the cell itself included,
    among the cells sharing the cell's values in ``by`` (its categorical values
    and stratum). A step is to the adjacent value among those the axis takes in
    the cell's group. Axes must be numbers, never strings, so that a window of
    120 hours sorts after one of 4. An axis that is missing for every cell of a
    group is not an axis of that group; one missing for some is refused.
    """
    if not axes:
        raise ChoiceError("a neighbourhood needs at least one axis")
    missing = [c for c in [*axes, *by] if c not in table.columns]
    if missing:
        raise ChoiceError(f"columns {missing} are not in the table")
    for axis in axes:
        if not pd.api.types.is_numeric_dtype(table[axis]):
            raise ChoiceError(f"axis {axis!r} must be numeric, got {table[axis].dtype}")
    if table.duplicated([*by, *axes]).any():
        raise ChoiceError("two rows share one cell of the grid")
    groups = table.groupby(list(by), sort=False, dropna=False).indices if by else {(): None}
    lists: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * len(table)
    for members in groups.values():
        rows = np.arange(len(table)) if members is None else np.asarray(members)
        steps = []
        for axis in axes:
            values = table[axis].to_numpy(dtype=np.float64)[rows]
            known = np.isfinite(values)
            if not known.any():
                continue
            if not known.all():
                raise ChoiceError(f"axis {axis!r} is missing for part of a group")
            steps.append(np.searchsorted(np.unique(values), values))
        position = np.column_stack(steps) if steps else np.zeros((len(rows), 1))
        for i, row in enumerate(rows):
            near = np.all(np.abs(position - position[i]) <= 1, axis=1)
            lists[row] = rows[near]
    width = max(len(a) for a in lists) if lists else 0
    out = np.full((len(table), width), -1, dtype=np.int64)
    for row, members_of in enumerate(lists):
        out[row, : len(members_of)] = members_of
    return out


def neighbourhood_median(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """The median of ``values`` over each cell's neighbourhood.

    ``values`` is (cells,) or (draws, cells); the result has its shape. A
    neighbour with a missing value does not contribute, and a cell whose own
    value is missing gets NaN, as in ``neighbourhood_scores``.
    """
    array = np.asarray(values, dtype=np.float64)
    padded = np.concatenate([array, np.full((*array.shape[:-1], 1), np.nan)], axis=-1)
    gathered = padded[..., np.where(index >= 0, index, array.shape[-1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(gathered, axis=-1)
    return np.where(np.isfinite(array), out, np.nan)


@dataclass(frozen=True)
class Choice:
    """The registered choice inside one stratum.

    ``cell`` is the chosen row (None when no cell averages enough trades) and
    ``peak`` the row with the highest S whatever its trade rate, reported
    beside it. ``s`` is S for every cell of the grid, not only the stratum's.
    """

    cell: int | None
    peak: int | None
    s: np.ndarray
    admissible: np.ndarray


def _best(candidates: np.ndarray, s: np.ndarray, own: np.ndarray) -> int | None:
    """Highest S, ties to the higher own mean, then to the earlier row."""
    usable = candidates[np.isfinite(s[candidates])]
    if not len(usable):
        return None
    order = np.lexsort((usable, -own[usable], -s[usable]))
    return int(usable[order[0]])


def choose(
    cells: CellDays,
    members: np.ndarray | Sequence[int],
    index: np.ndarray,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> Choice:
    """The choice among ``members`` (one stratum's rows) on these days.

    S is the neighbourhood median of each cell's mean daily ``net_bp``, a cell
    without a trade scoring 0. A cell may be chosen only if its trades over
    the days, divided by the number of days, are at least
    ``min_trades_per_day``; the filter limits the choice, never the
    neighbours. The filter is not relaxed when no cell passes.
    """
    rows = np.asarray(members, dtype=np.int64)
    own = cell_means(cells.net_bp)
    s = neighbourhood_median(own, index)
    rate = cells.trades.sum(axis=1) / max(cells.days, 1)
    admissible = rate >= min_trades_per_day
    return Choice(_best(rows[admissible[rows]], s, own), _best(rows, s, own), s, admissible)


def statistic_a(cells: CellDays, index: np.ndarray, cell: int) -> float:
    """A: the median, over ``cell``'s neighbourhood, of each cell's day-level t
    of G, a cell without trades (or with fewer than two days) scoring 0."""
    near = index[cell][index[cell] >= 0]
    return float(np.median(cell_day_t(cells.g_bp[near])))


@dataclass(frozen=True)
class StratumOutcome:
    """One stratum's chosen cell and its A (0 when nothing was chosen)."""

    cell: int | None
    a: float


def rerun(
    cells: CellDays,
    strata: dict[str, np.ndarray],
    index: np.ndarray,
    fixed: dict[str, int] | None = None,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> dict[str, StratumOutcome]:
    """The choice and A in every stratum, on one set of outcomes.

    A stratum named in ``fixed`` keeps that cell and makes no choice (a
    configuration fixed in advance, against the nulls with no selection). A
    stratum where no cell is admissible has no cell and A = 0, so that the
    maximum over strata of a null replication is still defined.
    """
    fixed = fixed or {}
    out: dict[str, StratumOutcome] = {}
    for name, members in strata.items():
        cell = fixed.get(name)
        if cell is None:
            cell = choose(cells, members, index, min_trades_per_day).cell
        out[name] = StratumOutcome(cell, 0.0 if cell is None else statistic_a(cells, index, cell))
    return out
