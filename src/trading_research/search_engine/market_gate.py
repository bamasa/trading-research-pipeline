"""The market gate's two inputs on row stamps, in bp from the book's float64 prices.

The registered gates read two numbers per 100 ms row:

* the mid's **realised volatility over the trailing 60 s**: the square root of
  the sum of the squared log changes of the mid whose rows are stamped in
  ``(s_t - 60 s, s_t]``, in bp; NaN on the first row, which has no change;
* the **quoted spread**: ``(ask_0 - bid_0) / mid`` in bp.

Both come from the replayed book's float64 prices, never from a feature store
(float16) or a normalised column, and are in bp whatever the price unit, which
is what the K-units rule asks of a gate input. ``backtest/gating.MarketGate``'s
default columns (``log_mid_vol50`` on the 5 s grid, ``spread_bp_now``) are not
used. ``quiet_out`` keeps rows whose volatility is at or above the 30th
percentile of the fold's training rows; ``tight_only`` keeps rows whose spread
is at or below their 70th percentile; a missing input keeps the gate closed.
The levels are fitted once per fold, on its training rows, and applied
unchanged to its test days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.backtest.gating import MarketGate
from trading_research.features.normalise import normalised_columns
from trading_research.features.timed import stamps_ns, trailing_start
from trading_research.strategies.ev_gate import UnitsError, check_bp_input

BP = 1e4
WINDOW_S = 60.0
VOLATILITY_COLUMN = "mid_realised_vol_60s_bp"
SPREAD_COLUMN = "quoted_spread_bp"
QUIET_OUT_QUANTILE = 0.3
TIGHT_ONLY_QUANTILE = 0.7
GATE_NAMES: tuple[str, ...] = ("open", "quiet_out", "tight_only")


def _price(rows: pd.DataFrame, column: str) -> np.ndarray:
    """A top-of-book price column as float64, refused unless it is the book's own."""
    if column in normalised_columns(rows) or normalised_columns(rows[column]):
        raise UnitsError(f"{column} is a normalised column; gate inputs come from prices")
    if rows[column].dtype != np.float64:
        raise UnitsError(f"{column} is {rows[column].dtype}; gate inputs need float64 prices")
    values = rows[column].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise UnitsError(f"{column} has a non-positive or missing price")
    return values


def quoted_spread_bp(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """``(ask - bid) / mid`` in bp."""
    bid, ask = np.asarray(bid, dtype=np.float64), np.asarray(ask, dtype=np.float64)
    return (ask - bid) / ((bid + ask) / 2.0) * BP


def realised_volatility_bp(
    stamps: np.ndarray, mid: np.ndarray, window_s: float = WINDOW_S
) -> np.ndarray:
    """Square root of the summed squared log changes of ``mid`` ending in each
    row's trailing window ``(s_t - window_s, s_t]``, in bp."""
    stamps = np.asarray(stamps, dtype=np.int64)
    mid = np.asarray(mid, dtype=np.float64)
    squared = np.zeros(len(mid))
    squared[1:] = np.log(mid[1:] / mid[:-1]) ** 2
    total = np.concatenate([[0.0], np.cumsum(squared)])
    start = trailing_start(stamps, window_s)
    rows = np.arange(len(mid))
    summed = np.maximum(total[rows + 1] - total[start], 0.0)
    out = np.sqrt(summed) * BP
    out[rows == 0] = np.nan
    return out


def gate_inputs(rows: pd.DataFrame, window_s: float = WINDOW_S) -> pd.DataFrame:
    """Both inputs for every row of a book frame (``timestamp``, level-0 prices)."""
    bid, ask = _price(rows, "bid_price_0"), _price(rows, "ask_price_0")
    volatility = realised_volatility_bp(stamps_ns(rows), (bid + ask) / 2.0, window_s)
    spread = quoted_spread_bp(bid, ask)
    return pd.DataFrame({VOLATILITY_COLUMN: volatility, SPREAD_COLUMN: spread}, index=rows.index)


def market_gate(name: str) -> MarketGate | None:
    """The registered gate of that name on these inputs; ``open`` has none."""
    if name == "open":
        return None
    if name == "quiet_out":
        return MarketGate(
            min_volatility_quantile=QUIET_OUT_QUANTILE,
            volatility_column=VOLATILITY_COLUMN,
            spread_column=SPREAD_COLUMN,
        )
    if name == "tight_only":
        return MarketGate(
            max_spread_quantile=TIGHT_ONLY_QUANTILE,
            volatility_column=VOLATILITY_COLUMN,
            spread_column=SPREAD_COLUMN,
        )
    raise ValueError(f"unknown gate {name!r}; registered: {GATE_NAMES}")


def _checked(inputs: pd.DataFrame) -> pd.DataFrame:
    for column in (VOLATILITY_COLUMN, SPREAD_COLUMN):
        if column not in inputs.columns:
            raise KeyError(f"the gate inputs lack {column!r}")
        if column in normalised_columns(inputs):
            raise UnitsError(f"{column} is a normalised column; bp from prices only")
        check_bp_input(column, inputs[column])
    return inputs


@dataclass(frozen=True)
class GateLevels:
    """One fold's levels: ``quiet_out``'s volatility floor, ``tight_only``'s spread cap."""

    min_volatility_bp: float
    max_spread_bp: float

    def __post_init__(self) -> None:
        for name in ("min_volatility_bp", "max_spread_bp"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise UnitsError(f"{name} must be a finite level in bp, got {value!r}")

    def to_dict(self) -> dict[str, float]:
        return {"min_volatility_bp": self.min_volatility_bp, "max_spread_bp": self.max_spread_bp}

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> GateLevels:
        return cls(float(values["min_volatility_bp"]), float(values["max_spread_bp"]))


def fit_gate_levels(training: pd.DataFrame) -> GateLevels:
    """The 30th percentile of the volatility and the 70th of the spread over
    the fold's training rows (``MarketGate.thresholds``, missing values dropped)."""
    inputs = _checked(training)
    quiet, tight = market_gate("quiet_out"), market_gate("tight_only")
    assert quiet is not None and tight is not None
    floor = quiet.thresholds(inputs).get("min_volatility")
    cap = tight.thresholds(inputs).get("max_spread")
    if floor is None or cap is None:
        raise ValueError("the training rows have no gate input to fit a level on")
    return GateLevels(floor, cap)


def gate_states(inputs: pd.DataFrame, levels: GateLevels) -> dict[str, np.ndarray]:
    """Whether each registered gate lets each row through, under fitted levels."""
    inputs = _checked(inputs)
    quiet, tight = market_gate("quiet_out"), market_gate("tight_only")
    assert quiet is not None and tight is not None
    floor = {"min_volatility": levels.min_volatility_bp}
    cap = {"max_spread": levels.max_spread_bp}
    return {
        "open": np.ones(len(inputs), dtype=bool),
        "quiet_out": quiet.mask(inputs, floor).to_numpy(dtype=bool),
        "tight_only": tight.mask(inputs, cap).to_numpy(dtype=bool),
    }
