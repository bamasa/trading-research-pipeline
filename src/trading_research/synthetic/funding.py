"""A funding-rate series with a chosen persistence, for the carry strategies.

What is planted
---------------
One settlement every ``interval_h`` hours from the first midnight, the rate an
AR(1) around ``mean_bp``: ``r_t - mu = persistence * (r_{t-1} - mu) + e_t``,
with the innovations scaled so the stationary standard deviation is ``sd_bp``
(bp per settlement; positive means longs pay shorts, as the venue publishes it).
The persistence is what a gated carry bets on: a high rate now says a high rate
at the next settlements. ``truth.json`` records the mean, the standard
deviation, the persistence and its half-life in settlements.

It is written in the layout of :mod:`trading_research.data.bybit_funding` (a
file per instrument-day under ``funding/<SYMBOL>/<day>.parquet`` with
``timestamp`` and ``rate`` as a fraction), so the market-making loader's
``funding_root`` and a carry study read it as they read the venue's history.

The null
--------
``null=True`` keeps the mean and the standard deviation and sets the
persistence to zero: the rate now says nothing about the next one.

What it can prove, and what it cannot
-------------------------------------
It can show whether a carry rule earns the persistence it bets on, and nothing
where there is none, before it is run on a venue's history. It cannot show that
real funding is an AR(1), what its persistence is, or that a basis, a margin
call or a fee would leave the carry anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.synthetic.common import (
    DEFAULT_START,
    NS_PER_S,
    Truth,
    day_list,
    day_start_ns,
    rng_for,
    spec_dict,
    stamps,
    write_plane,
)

GENERATOR = "trading_research.synthetic.funding.v1"

PROVES = (
    "Whether a carry rule earns the planted persistence of the funding rate, and nothing "
    "on a null without it."
)
DOES_NOT_PROVE = (
    "That real funding is an AR(1), what its persistence is, or that basis, margin and "
    "fees would leave a carry anything."
)


@dataclass(frozen=True)
class FundingSpec:
    """Every parameter of a funding series. The defaults generate in milliseconds."""

    symbol: str = "SYNTHUSDT"
    start: str = DEFAULT_START
    days: int = 365
    interval_h: int = 8
    mean_bp: float = 1.0
    sd_bp: float = 1.5
    #: AR(1) coefficient per settlement, in [0, 1).
    persistence: float = 0.9
    null: bool = False
    seed: int = 20200106

    def __post_init__(self) -> None:
        if self.interval_h <= 0 or 24 % self.interval_h:
            raise ValueError("the settlement interval must divide a day")
        if not 0.0 <= self.persistence < 1.0:
            raise ValueError("persistence must lie in [0, 1)")
        if self.sd_bp < 0:
            raise ValueError("the standard deviation cannot be negative")

    @property
    def phi(self) -> float:
        return 0.0 if self.null else self.persistence

    @property
    def half_life_settlements(self) -> float:
        return math.inf if self.phi <= 0 else -math.log(2.0) / math.log(self.phi)


@dataclass
class FundingDataset:
    spec: FundingSpec
    #: ``timestamp`` (UTC) and ``rate`` (a fraction per settlement), oldest first.
    series: pd.DataFrame
    truth: Truth

    def by_day(self) -> dict[date, pd.DataFrame]:
        days = self.series["timestamp"].dt.date
        return {
            day: rows.reset_index(drop=True)
            for day, rows in self.series.groupby(days, sort=True)
            if isinstance(day, date)
        }

    def write(self, directory: Path | str) -> Path:
        root = Path(directory)
        write_plane(
            root, "funding", self.spec.symbol, self.by_day(), generator=GENERATOR,
            seed=self.spec.seed, extra={"interval_h": self.spec.interval_h},
        )  # fmt: skip
        self.truth.write(root)
        return root


def generate(spec: FundingSpec | None = None) -> FundingDataset:
    spec = spec or FundingSpec()
    first = day_list(spec.start, 1)[0]
    per_day = 24 // spec.interval_h
    n = spec.days * per_day
    rng = rng_for(spec.seed, 7)
    phi = spec.phi
    shocks = rng.normal(0.0, spec.sd_bp * math.sqrt(1.0 - phi**2), n)
    deviation = np.empty(n)
    deviation[0] = rng.normal(0.0, spec.sd_bp)
    for k in range(1, n):
        deviation[k] = phi * deviation[k - 1] + shocks[k]
    rate = (spec.mean_bp + deviation) * 1e-4
    ns = day_start_ns(first) + np.arange(n, dtype=np.int64) * spec.interval_h * 3600 * NS_PER_S
    series = pd.DataFrame({"timestamp": stamps(ns), "rate": rate})
    truth = Truth(
        generator=GENERATOR,
        seed=spec.seed,
        null=spec.null,
        planted={
            "mean_bp": spec.mean_bp,
            "sd_bp": spec.sd_bp,
            "persistence": phi,
            "half_life_settlements": spec.half_life_settlements,
            "interval_h": spec.interval_h,
        },
        spec=spec_dict(spec),
        proves=PROVES,
        does_not_prove=DOES_NOT_PROVE,
    )
    return FundingDataset(spec, series, truth)


def estimate_persistence(rate: np.ndarray) -> float:
    """OLS slope of the rate on its previous value: what the series says its persistence is."""
    x = np.asarray(rate, dtype=np.float64)
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    return float((a @ b) / (a @ a))
