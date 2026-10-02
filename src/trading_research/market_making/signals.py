"""External signals, joined into event time at the end of their bin.

A signal computed on the five-second grid — the reversion index, a regime
statistic — is a value per bin. :func:`trading_research.data.grid.to_grid`
labels a bin by its start by default and carries its last value, so joining it
into event time at that label hands a decision a value observed up to five
seconds later. A :class:`SignalTape` therefore always holds **bin-end** labels,
and a decision at time ``t`` reads the last value whose label is strictly
before ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_S, to_ns


@dataclass(frozen=True)
class SignalTape:
    """A causal signal: ``values[i]`` is known from ``ts[i]`` (ns) onward."""

    name: str
    ts: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.ts) != len(self.values):
            raise ValueError("a tape needs one value per timestamp")
        if len(self.ts) > 1 and np.any(np.diff(self.ts) <= 0):
            raise ValueError("tape timestamps must be strictly increasing")

    @classmethod
    def from_grid(
        cls,
        name: str,
        grid: pd.DataFrame,
        column: str,
        *,
        seconds: int,
        label: Literal["left", "right"],
    ) -> SignalTape:
        """A tape from a gridded frame, relabelled to bin ends if need be.

        ``label`` must say how ``grid`` was labelled; there is no default,
        because guessing is how the look-ahead gets in. A left-labelled grid
        has every label moved one bin later.
        """
        if label not in ("left", "right"):
            raise ValueError(f"label must be 'left' or 'right', got {label!r}")
        stamps = to_ns(grid["timestamp"])
        if label == "left":
            stamps = stamps + seconds * NS_PER_S
        values = grid[column].to_numpy(dtype=np.float64)
        return cls(name, stamps, values)

    def strictly_before(self, at_ns: np.ndarray) -> np.ndarray:
        """The value known strictly before each time in ``at_ns``; NaN before the first."""
        at = np.asarray(at_ns, dtype=np.int64)
        index = np.searchsorted(self.ts, at, side="left") - 1
        out = np.full(len(at), np.nan)
        known = index >= 0
        out[known] = self.values[index[known]]
        return out
