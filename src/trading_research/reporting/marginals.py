"""Per-axis marginals of the development-period leaderboard.

For each value of each axis, the median over cells of each cell's mean daily
gross, net and G, in the form of the grand search's marginals: how far one
design choice moves the median configuration. Stratum C's and R's axes are
reported within their stratum, and the axes both strata share (the stratum
itself, the gate, the hold and the regime window) also over every cell. The
regime rows are one per window, in window order (none, 4, 24 and 120 h).

D numbers, "selected, biased upward"; a marginal is a description of the
catalogue and never a result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.reporting.leaderboard import SELECTED
from trading_research.validation.choice import ChoiceError

#: The axes reported in each scope ("all" is every cell of both strata).
AXES_BY_SCOPE: dict[str, tuple[str, ...]] = {
    "all": ("stratum", "gate", "hold_s", "regime"),
    "C": ("model", "inputs", "gate", "hold_s", "trades_per_day", "label_fee_bp", "regime"),
    "R": ("model", "gate", "hold_s", "k", "beta_bp", "regime"),
}
MARGINAL_COLUMNS: tuple[str, ...] = ("scope", "axis", "value", "cells", "gross", "net", "G")
_NUMBERS = ("gross", "net", "G")
#: Where the entry point writes it (``artefacts.files``).
MARGINALS_FILE = "experiments/results/short_horizon_2025_marginals.csv"


def _median(values: pd.Series) -> float:
    finite = values.to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _levels(cells: pd.DataFrame, axis: str) -> list[tuple[str, np.ndarray]]:
    """The axis's values in their order, each with its rows' mask."""
    if axis == "regime":
        order = cells.drop_duplicates("regime").sort_values("regime_h")["regime"]
        return [(str(v), (cells["regime"] == v).to_numpy()) for v in order]
    column = cells[axis]
    if pd.api.types.is_numeric_dtype(column):
        values = np.unique(column.to_numpy(dtype=np.float64))
        values = values[np.isfinite(values)]
        numbers = column.to_numpy(dtype=np.float64)
        return [(f"{v:g}", np.isclose(numbers, v)) for v in values]
    return [(str(v), (column == v).to_numpy()) for v in pd.unique(column)]


def marginals(
    board: pd.DataFrame,
    table: pd.DataFrame,
    axes_by_scope: Mapping[str, Sequence[str]] = AXES_BY_SCOPE,
) -> pd.DataFrame:
    """The marginals of ``board`` (``leaderboard.leaderboard``) over ``table``'s
    axes (``search_keys.catalogue_table``, in the same row order)."""
    if list(board["key"]) != list(table["key"]):
        raise ChoiceError("the leaderboard and the catalogue table list different keys")
    joined = pd.concat(
        [table.reset_index(drop=True), board[list(_NUMBERS)].reset_index(drop=True)], axis=1
    )
    rows: list[dict[str, object]] = []
    for scope, axes in axes_by_scope.items():
        cells = joined if scope == "all" else joined.loc[joined["stratum"] == scope]
        if cells.empty:
            continue
        for axis in axes:
            for value, mask in _levels(cells, axis):
                part = cells.loc[mask]
                rows.append(
                    {"scope": scope, "axis": axis, "value": value, "cells": len(part)}
                    | {name: _median(part[name]) for name in _NUMBERS}
                )
    return pd.DataFrame(rows, columns=list(MARGINAL_COLUMNS))


def axis_spread(table: pd.DataFrame, number: str = "net") -> pd.DataFrame:
    """How far each axis moves the median: the largest minus the smallest of
    its values' medians of ``number``, per scope and axis."""
    grouped = table.groupby(["scope", "axis"], sort=False)[number]
    return (grouped.max() - grouped.min()).rename(f"{number}_range").reset_index()


def write_marginals(table: pd.DataFrame, path: Path | str) -> Path:
    """The marginals file, under the D label."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# every number: on development data, {SELECTED}; never quoted as a result\n")
        table.loc[:, list(MARGINAL_COLUMNS)].to_csv(handle, index=False, float_format="%.6g")
    return target
