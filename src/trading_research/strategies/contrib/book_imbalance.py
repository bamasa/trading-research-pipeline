"""Book imbalance at the touch, a threshold set from a trade rate, and a clock exit.

The worked example of ``docs/extending.md``: one file, one decorator, and the
strategy is listed, checked and backtested by the plugin commands with no
other edit.

The rule: the touch imbalance ``(bid size - ask size) / (bid size + ask size)``
says which side of the book is heavier. Where it is far enough from zero, go
with the heavier side (a heavy bid is buying interest) and hold for ``hold``
rows, whatever happens: the clock exit, which every execution applies from the
strategy's ``horizon``.

"Far enough" is not tuned on outcomes. ``fit`` reads the history only and sets
the threshold at the ``1 - share`` quantile of the absolute imbalance there, so
about ``share`` of the rows would be decision rows: a statement about how often
to trade, made before the trading rows exist.

On the synthetic demo market the generator plants exactly this effect (its
latent signal skews the touch sizes and drifts the next mid), weakly: the
example is a test that the pipeline finds a planted edge and charges for it,
not a claim about any market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.plugins import register_strategy


@register_strategy("book_imbalance")
class BookImbalance:
    """Go with the heavier side of the touch once its imbalance is in the top share; exit on the clock."""

    #: The decision reads the touch sizes of its own row and nothing else.
    inputs = ("bid_size_0", "ask_size_0")
    lookback = 0

    def __init__(self, *, hold: int = 30, share: float = 0.05) -> None:
        if not 0.0 < share < 1.0:
            raise ValueError(f"share must be in (0, 1), got {share}")
        #: Rows a position is held: the clock exit.
        self.horizon = hold
        self.share = share
        self.threshold = float("nan")

    @staticmethod
    def imbalance(frame: pd.DataFrame) -> np.ndarray:
        bid = frame["bid_size_0"].to_numpy(dtype="float64")
        ask = frame["ask_size_0"].to_numpy(dtype="float64")
        total = bid + ask
        out = np.full(len(frame), np.nan)
        np.divide(bid - ask, total, out=out, where=total > 0)
        return out

    def fit(self, history: pd.DataFrame) -> BookImbalance:
        values = np.abs(self.imbalance(history))
        self.threshold = float(np.nanquantile(values, 1.0 - self.share))
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        if not np.isfinite(self.threshold):
            raise RuntimeError("book_imbalance has not been fitted")
        value = self.imbalance(frame)
        out = np.zeros(len(frame), dtype=np.int8)
        strong = np.isfinite(value) & (np.abs(value) >= self.threshold)
        out[strong] = np.sign(value[strong]).astype(np.int8)
        return out
