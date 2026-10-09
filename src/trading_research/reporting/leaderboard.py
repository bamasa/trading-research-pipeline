"""The development-period leaderboard: every configuration with its D numbers.

One row per catalogue cell, from the same cell-by-day matrix the choice reads
(``validation/choice.CellDays``), so the leaderboard cannot disagree with the
choice: S is the neighbourhood median of each cell's mean daily net (a cell
without a trade scoring 0), A the neighbourhood median of each cell's
day-level t of G, and the ranks order the cells by S, ties to the cell's own
mean and then to the earlier row, as ``choice.choose`` breaks them. The rank
ignores the five-trade filter, so the top row is the outright peak; the chosen
cell is the best-ranked admissible row of its stratum.

Every number here is a D number, labelled "selected, biased upward", and none
is quoted as a result. S25 carries a badge with its rank "r of N"; the regions
T1 and T2 carry tags, and :func:`region_rows` reports each region's best
neighbourhood cell with its S, A and rank.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.reporting.search_keys import REGIONS, S25_KEY, in_region, region_tags
from trading_research.validation.choice import (
    MIN_TRADES_PER_DAY,
    CellDays,
    ChoiceError,
    cell_day_t,
    cell_means,
    neighbourhood_median,
)

#: The registered columns, in order (``artefacts.leaderboard_columns``).
LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "key",
    "gross",
    "net",
    "G",
    "trades_per_day",
    "day_t",
    "ic",
    "S",
    "A",
    "admissible",
    "rank_in_stratum",
    "rank_overall",
    "region",
    "S25_badge",
)
SELECTED = "selected, biased upward"


def day_level_t(values: np.ndarray) -> np.ndarray:
    """Each row's day-level t of its finite days, NaN when it is undefined
    (fewer than two days, or no dispersion): a reported t, not the null's."""
    finite = np.isfinite(values)
    n = finite.sum(axis=-1)
    t = cell_day_t(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(np.where(finite, values, np.nan), axis=-1, ddof=1)
    return np.where((n >= 2) & (sd > 0), t, np.nan)


def _order(s: np.ndarray, own: np.ndarray) -> np.ndarray:
    """Rows by S descending, then own mean descending, then row (NaN S last)."""
    rows = np.arange(len(s))
    key_s = np.where(np.isfinite(s), -s, np.inf)
    return np.lexsort((rows, -own, key_s))


def leaderboard(
    table: pd.DataFrame,
    cells: CellDays,
    index: np.ndarray,
    *,
    gross_bp: np.ndarray | None = None,
    ic: np.ndarray | None = None,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> pd.DataFrame:
    """The leaderboard of ``table``'s cells (``search_keys.catalogue_table``).

    ``cells`` and the optional ``gross_bp`` (cells x days, NaN without a trade)
    are in ``table``'s row order, ``index`` is the choice's neighbourhood index
    and ``ic`` each cell's mean daily IC. Means are over days with a trade,
    0 for a cell without one; ``day_t`` is the t of the daily net.
    """
    if len(table) != cells.cells or len(index) != cells.cells:
        raise ChoiceError(f"{len(table)} keys, {len(index)} neighbourhoods, {cells.cells} cells")
    own = cell_means(cells.net_bp)
    s = neighbourhood_median(own, index)
    a = neighbourhood_median(cell_day_t(cells.g_bp), index)
    rate = cells.trades.sum(axis=1) / max(cells.days, 1)
    order = _order(s, own)
    overall = np.empty(len(table), dtype=np.int64)
    overall[order] = np.arange(1, len(table) + 1)
    in_stratum = np.empty(len(table), dtype=np.int64)
    strata = table["stratum"].to_numpy()
    for name in pd.unique(strata):
        mine = order[strata[order] == name]
        in_stratum[mine] = np.arange(1, len(mine) + 1)
    keys = table["key"].to_numpy()
    badge = [
        f"S25, rank {overall[i]} of {len(table)}" if k == S25_KEY else ""
        for i, k in enumerate(keys)
    ]
    return pd.DataFrame(
        {
            "key": keys,
            "gross": np.nan if gross_bp is None else cell_means(gross_bp),
            "net": own,
            "G": cell_means(cells.g_bp),
            "trades_per_day": rate,
            "day_t": day_level_t(cells.net_bp),
            "ic": np.nan if ic is None else np.asarray(ic, dtype=np.float64),
            "S": s,
            "A": a,
            "admissible": rate >= min_trades_per_day,
            "rank_in_stratum": in_stratum,
            "rank_overall": overall,
            "region": region_tags(table),
            "S25_badge": badge,
        },
        columns=list(LEADERBOARD_COLUMNS),
    )


REGION_COLUMNS: tuple[str, ...] = (
    "region",
    "cells",
    "key",
    "S",
    "A",
    "admissible",
    "rank_in_stratum",
    "rank_overall",
)


def region_rows(board: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Each region's best neighbourhood cell (the best-ranked, by S), with its
    S, A and ranks; measurements with no slot."""
    rows = []
    for name, values in REGIONS.items():
        members = board.loc[in_region(table, values)]
        if members.empty:
            raise ChoiceError(f"region {name} has no cell on the leaderboard")
        best = members.loc[members["rank_overall"].idxmin()]
        rows.append({"region": name, "cells": len(members), **best[list(REGION_COLUMNS[2:])]})
    return pd.DataFrame(rows, columns=list(REGION_COLUMNS))


def s25_rank(board: pd.DataFrame) -> tuple[int, int]:
    """S25's overall rank r and the N it is ranked among."""
    row = board.loc[board["key"] == S25_KEY]
    if len(row) != 1:
        raise ChoiceError("S25 is not on the leaderboard")
    return int(row["rank_overall"].iloc[0]), len(board)


def write_leaderboard(
    board: pd.DataFrame, path: Path | str, *, planned: int, scored: int, skipped: int
) -> Path:
    """The leaderboard file: a header with N planned, scored and skipped and
    the D label, then the registered columns."""
    if planned != scored + skipped:
        raise ChoiceError(f"planned {planned} is not scored {scored} plus skipped {skipped}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# N planned: {planned}\n# N scored: {scored}\n# N skipped: {skipped}\n"
        f"# every number: on development data, {SELECTED}; never quoted as a result\n"
    )
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header)
        board.loc[:, list(LEADERBOARD_COLUMNS)].to_csv(handle, index=False, float_format="%.6g")
    return target


def read_leaderboard(path: Path | str) -> tuple[dict[str, int], pd.DataFrame]:
    """The header's counts and the table, as :func:`write_leaderboard` wrote them."""
    counts: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("# N "):
                break
            name, _, value = line[4:].partition(":")
            counts[name.strip()] = int(value)
    board = pd.read_csv(path, comment="#", keep_default_na=False, na_values=[""])
    board["region"] = board["region"].fillna("")
    board["S25_badge"] = board["S25_badge"].fillna("")
    return counts, board
