"""The 2025 search's catalogue keys as a table the presentation can group.

A key is written ``<instrument>|C|<model>|<inputs>|<gate>|hold_s=<h>|
trades_per_day=<n>|label_fee_bp=<f>|regime=<r>`` in stratum C and with ``k``
and ``beta_bp`` in place of the rate and the label fee in stratum R. The
leaderboard, the marginals and the region tags read a key's values from
:func:`catalogue_table`, where every ordered axis is a number (the regime
window in hours, ``none`` being 0) and an axis a stratum does not have is NaN.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

from trading_research.pipeline.regime import window_hours
from trading_research.validation.choice import neighbourhood_index

#: The registered named configuration.
S25_KEY = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none"

#: The categorical values that make a group, and the ordered axes of the grid.
CATEGORICAL_AXES: tuple[str, ...] = ("stratum", "model", "inputs", "gate")
ORDERED_AXES: tuple[str, ...] = ("hold_s", "trades_per_day", "label_fee_bp", "k", "beta_bp")
NUMERIC_AXES: tuple[str, ...] = (*ORDERED_AXES, "regime_h")
_FIELDS = {
    "C": ("hold_s", "trades_per_day", "label_fee_bp", "regime"),
    "R": ("hold_s", "k", "beta_bp", "regime"),
}

#: The leaderboard regions, reported as measurements with no slot.
REGIONS: dict[str, dict[str, object]] = {
    "T1": {
        "stratum": "C",
        "model": "gbm",
        "inputs": "micro",
        "label_fee_bp": 5.5,
        "gate": "open",
        "regime": "none",
    },
    "T2": {"stratum": "R", "model": "tcn", "gate": "open", "regime": "none"},
}


class CatalogueKeyError(ValueError):
    """A string that is not a catalogue key."""


def parse_key(key: str) -> dict[str, object]:
    """A key's values: strings for the categorical ones and the regime,
    floats for the ordered axes, NaN for an axis of the other stratum."""
    parts = key.split("|")
    if len(parts) != 9 or parts[1] not in _FIELDS:
        raise CatalogueKeyError(f"not a catalogue key: {key!r}")
    instrument, stratum, model, inputs, gate, *pairs = parts
    out: dict[str, object] = {
        "key": key,
        "instrument": instrument,
        "stratum": stratum,
        "model": model,
        "inputs": inputs,
        "gate": gate,
    }
    out.update(dict.fromkeys(ORDERED_AXES, np.nan))
    for name, pair in zip(_FIELDS[stratum], pairs, strict=True):
        field, sep, value = pair.partition("=")
        if field != name or not sep or not value:
            raise CatalogueKeyError(f"expected {name}=<value> in {key!r}, got {pair!r}")
        out[name] = value if name == "regime" else float(value)
    out["regime_h"] = float(window_hours(str(out["regime"])))
    return out


def catalogue_table(keys: Sequence[str]) -> pd.DataFrame:
    """One row per key, in the given order, with every axis as a column."""
    rows = [parse_key(k) for k in keys]
    if len({r["key"] for r in rows}) != len(rows):
        raise CatalogueKeyError("a key is listed twice")
    columns = ["key", "instrument", *CATEGORICAL_AXES, *ORDERED_AXES, "regime", "regime_h"]
    return pd.DataFrame(rows, columns=columns)


def neighbourhoods(table: pd.DataFrame) -> np.ndarray:
    """Every cell's registered neighbourhood (``choice.neighbourhood_index``
    over the ordered axes and the regime window, within the cell's group)."""
    return neighbourhood_index(table, list(NUMERIC_AXES), by=list(CATEGORICAL_AXES))


def in_region(table: pd.DataFrame, values: Mapping[str, object]) -> np.ndarray:
    """A mask of the rows whose values match every one of ``values``."""
    mask = np.ones(len(table), dtype=bool)
    for name, value in values.items():
        column = table[name]
        if isinstance(value, str):
            mask &= (column == value).to_numpy()
        else:
            mask &= np.isclose(column.to_numpy(dtype=np.float64), float(cast(float, value)))
    return mask


def region_tags(
    table: pd.DataFrame, regions: Mapping[str, Mapping[str, object]] = REGIONS
) -> list[str]:
    """Each row's region names, joined by ``+`` (empty outside every region)."""
    masks = {name: in_region(table, values) for name, values in regions.items()}
    return ["+".join(n for n, m in masks.items() if m[i]) for i in range(len(table))]
