"""Expressing everything relative to a rolling centre.

Two problems solved by one change.

**Stationarity.** A feature whose level drifts asks a model to learn the drift
as well as the relationship. Spread in basis points is stable on BTCUSDT and
not on an altcoin whose liquidity changed over the sample; queue imbalance is
bounded but its dispersion is not. Fitting on a fortnight and trading the next
day means the model meets a distribution slightly different from the one it
learned, and the drift is a large part of that difference.

**Pooling, which is the bigger prize.** Features on a common scale can be
*stacked across instruments*. Thirty-four days of one instrument is a few
hundred thousand usable rows and one realisation of one market; the same days
across twenty instruments is twenty times the sample and twenty partly
independent realisations. Nothing else available here multiplies the data by an
order of magnitude, and §17 said the sample is what limits every conclusion in
this project.

Causality is the whole difficulty
---------------------------------
A z-score is trivial and a *causal* z-score is not. Statistics computed over the
whole sample leak the future into every row — including rows in the training
block, which is why it is not caught by a naive train/test split — and the leak
is invisible because the values still look like features. This uses trailing
windows only, and the leakage checker asserts it: mutate the data after a
cutoff, and nothing before the cutoff may move.

Robust by default
-----------------
Median and interquartile range rather than mean and standard deviation. A book
feature's tail is not Gaussian: one repriced quote produces a value ten standard
deviations out, and a mean-based scaler lets that single observation set the
scale for the entire window. The median is unmoved by it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingNormaliser:
    """Centre and scale every column against its own trailing window.

    ``window``
        Rows of history the statistics are computed over. Long enough to be
        stable, short enough to track a regime — a fortnight of five-second
        rows is about 240,000, and the default is far shorter than that on
        purpose: the point is to follow the regime, not to average it away.
    ``method``
        ``"robust"`` uses the median and the interquartile range; ``"zscore"``
        the mean and standard deviation. Robust is the default because a single
        repriced quote otherwise sets the scale for a whole window.
    ``clip``
        Values beyond this many normalised units are clipped. Extreme
        observations survive as extreme rather than as outliers that dominate a
        fit, and the number is stated rather than left to whatever the data
        happened to contain.
    """

    window: int = 2000
    method: str = "robust"
    clip: float = 8.0
    min_periods: int | None = None
    #: Columns left alone. Anything already bounded and centred — an imbalance
    #: in [-1, 1], a share in [0, 1], a calendar term — gains nothing from being
    #: rescaled and loses its interpretation. Re-normalising a bounded feature
    #: also amplifies it exactly where it is least informative: in a quiet
    #: window its own small variation is divided by its own small spread.
    exclude: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.window < 10:
            raise ValueError(f"window must be at least 10, got {self.window}")
        if self.method not in ("robust", "zscore"):
            raise ValueError(f"method must be 'robust' or 'zscore', got {self.method!r}")
        if self.clip <= 0:
            raise ValueError(f"clip must be positive, got {self.clip}")

    @property
    def warmup(self) -> int:
        """Rows that cannot be normalised, and must be treated as missing."""
        return self.min_periods if self.min_periods is not None else self.window // 4

    def transform(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str] | None = None,
        *,
        floor: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Normalise the named columns, leaving everything else untouched.

        Every statistic is trailing and *shifted by one row*: the value at *t*
        is centred on a window ending at *t-1*. Including *t* would let a row
        contribute to its own scale, which is a small leak and an unnecessary
        one.

        ``floor`` sets a minimum scale, per row. For price-derived columns the
        natural floor is the spread: a scale smaller than one tick says the
        price was pinned, and dividing by it turns a single tick into a large
        normalised value. Without it the fallback below applies, which is
        weaker — it recovers a usable number but does not know what "too small"
        means for this instrument.
        """
        chosen = (
            list(columns)
            if columns is not None
            else [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        )
        chosen = [c for c in chosen if c not in self.exclude]
        out = frame.copy()
        minimum = self.warmup

        for column in chosen:
            values = frame[column]
            rolling = values.shift(1).rolling(self.window, min_periods=minimum)
            if self.method == "robust":
                centre = rolling.median()
                spread = rolling.quantile(0.75) - rolling.quantile(0.25)
                # The interquartile range of a normal is about 1.349 standard deviations; dividing
                # by it puts robust and z-score output on the same scale, so the
                # two methods are interchangeable without retuning `clip`.
                scale = spread / 1.349
            else:
                centre = rolling.mean()
                scale = rolling.std()

            # A zero scale is not missing data. It happens whenever a feature
            # is constant over the window — an instrument quoted one tick wide,
            # or a return that is exactly zero because the price did not move —
            # and dividing by it silently discarded 81% of XRPUSDT's rows the
            # first time this ran, which then selected the strategy's moments
            # for it. Fall back to a wider measure, and to zero when the column
            # genuinely does not vary: a constant feature carries no
            # information, which is a value, not an absence of one.
            fallback = (values - centre).abs().shift(1).rolling(
                self.window, min_periods=minimum
            ).mean() * 1.4826
            usable = scale.where(scale > 0, fallback)
            if floor is not None:
                usable = usable.combine(floor.reindex(values.index), max)
            normalised = (values - centre) / usable.where(usable > 0)
            normalised = normalised.where(centre.notna(), np.nan).fillna(
                pd.Series(0.0, index=values.index).where(centre.notna())
            )
            out[column] = normalised.clip(-self.clip, self.clip)
        return out

    def describe(self) -> dict[str, object]:
        return {
            "window": self.window,
            "method": self.method,
            "clip": self.clip,
            "warmup": self.warmup,
        }


def normalise_pooled(
    frames: dict[str, pd.DataFrame],
    columns: Sequence[str],
    normaliser: RollingNormaliser | None = None,
    *,
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Normalise each instrument separately, then stack them into one frame.

    Separately is the point. Normalising the pool as a whole would centre every
    instrument on the pool's average, which is a quantity none of them
    experiences — and it would make each row's features depend on which other
    instruments happened to be included.

    The result carries a ``symbol`` column so a fit can be checked for having
    learned one instrument rather than the common structure, and so folds can be
    cut by time across all of them at once.
    """
    normaliser = normaliser or RollingNormaliser()
    parts: list[pd.DataFrame] = []

    for symbol, frame in frames.items():
        present = [c for c in columns if c in frame.columns]
        if not present:
            continue
        scaled = normaliser.transform(frame, present)
        scaled[symbol_column] = symbol
        parts.append(scaled)

    if not parts:
        raise ValueError("no instrument carried any of the requested columns")

    pooled = pd.concat(parts, ignore_index=True)
    if "timestamp" in pooled.columns:
        # Sorted by time across instruments, so a chronological split cuts all
        # of them at the same instant rather than at the same row number.
        pooled = pooled.sort_values("timestamp").reset_index(drop=True)
    return pooled


def coverage(frame: pd.DataFrame, columns: Sequence[str]) -> float:
    """Share of rows where every named column is present.

    Worth checking after normalising: a long window discards its warm-up on
    every instrument, and pooling twenty instruments discards it twenty times.
    """
    present = [c for c in columns if c in frame.columns]
    if not present:
        return 0.0
    return float(frame[present].notna().all(axis=1).mean())
