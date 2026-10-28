"""Two instruments on bars with a planted cointegration, and its two nulls.

What is planted
---------------
The second leg's log close is a Gaussian random walk (``vol_b_bp`` a bar). The
first leg's is

    log a(t) = alpha + beta * log b(t) + s(t),

with ``s`` an AR(1) whose coefficient gives a half-life of ``half_life_bars``
and whose stationary standard deviation is ``spread_sd_bp``. So the hedge ratio
is ``beta``, the residual reverts at a known speed, and a band rule on its
z-score is owed a known gross.

The nulls
---------
* ``variant="independent"``: the first leg is a random walk of its own, of
  the same bar volatility, independent of the second. Nothing cointegrates, and
  a pairs rule is owed nothing before costs.
* ``variant="break"``: planted as above until ``break_fraction`` of the bars,
  then the residual becomes a random walk with the same innovations: a pair
  that cointegrates on a formation window before the break and does not after.

Bars
----
Every bar's close is the path; its open is the previous close moved by a small
noise (``open_noise_bp``), its high and low wrap both, and the stamps follow the
public klines: ``open_time`` on the interval, ``timestamp`` (the close) one
millisecond before the next open. Written in the layout ``trp data download
--kind klines`` writes, ``<dir>/<SYMBOL>/klines-<interval>/<day>.parquet`` with
a ``manifest.json``, and ``truth.json`` at the top, so ``trp pairs`` reads it as
it reads downloaded bars.

What it can prove, and what it cannot
-------------------------------------
It can show whether the screen finds a planted cointegration and rejects
independent walks after correction, whether the hedge ratio and half-life come
back, and whether the walk-forward backtest earns a planted reversion before
costs and nothing on the nulls. It cannot show that any two real instruments
cointegrate, at what half-life, or that the relation would hold after the
window it was found on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from trading_research import __version__
from trading_research.data.schema import BARS_SCHEMA, SCHEMA_VERSION
from trading_research.synthetic.common import (
    DEFAULT_START,
    NOTE,
    SYNTHETIC_SOURCE,
    Truth,
    day_list,
    day_start_ns,
    rng_for,
    spec_dict,
    stamps,
)

GENERATOR = "trading_research.synthetic.pairs.v1"

Variant = Literal["planted", "independent", "break"]

PROVES = (
    "Whether the pairs screen finds a planted cointegration and rejects independent walks "
    "after correction, and whether the walk-forward backtest earns a planted reversion "
    "before costs and nothing on the nulls."
)
DOES_NOT_PROVE = (
    "That any two real instruments cointegrate, at what half-life, or that a relation "
    "found on one window holds on the next."
)


@dataclass(frozen=True)
class PairSpec:
    """Every parameter of a generated pair. The defaults generate in milliseconds."""

    symbol_a: str = "SYNTHAUSDT"
    symbol_b: str = "SYNTHBUSDT"
    start: str = DEFAULT_START
    days: int = 120
    interval_minutes: int = 60
    beta: float = 0.8
    alpha: float = 0.5
    price_b: float = 100.0
    vol_b_bp: float = 60.0
    half_life_bars: float = 6.0
    spread_sd_bp: float = 60.0
    open_noise_bp: float = 2.0
    variant: Variant = "planted"
    break_fraction: float = 0.5
    seed: int = 20200106

    def __post_init__(self) -> None:
        if self.days < 1 or self.interval_minutes < 1 or 1440 % self.interval_minutes:
            raise ValueError("days must be positive and the interval must divide a day")
        if self.half_life_bars <= 0 or self.spread_sd_bp < 0 or self.vol_b_bp <= 0:
            raise ValueError("half-life and volatilities must be positive")
        if self.variant not in ("planted", "independent", "break"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if not 0.0 < self.break_fraction < 1.0:
            raise ValueError("break_fraction must lie in (0, 1)")
        if self.symbol_a == self.symbol_b:
            raise ValueError("the two legs need different symbols")

    @property
    def phi(self) -> float:
        """AR(1) coefficient of the residual per bar."""
        return 0.5 ** (1.0 / self.half_life_bars)

    @property
    def bars(self) -> int:
        return self.days * 1440 // self.interval_minutes

    @property
    def interval(self) -> str:
        m = self.interval_minutes
        return f"{m // 60}h" if m % 60 == 0 else f"{m}m"

    @property
    def break_bar(self) -> int | None:
        return int(self.bars * self.break_fraction) if self.variant == "break" else None


@dataclass
class PairDataset:
    spec: PairSpec
    #: One contract bars frame per symbol.
    bars: dict[str, pd.DataFrame]
    #: The planted residual ``s(t)`` per bar (log units); NaN for the independent null.
    residual: np.ndarray
    truth: Truth

    def write(self, directory: Path | str) -> Path:
        root = Path(directory)
        for symbol, frame in self.bars.items():
            folder = root / symbol / f"klines-{self.spec.interval}"
            folder.mkdir(parents=True, exist_ok=True)
            days = frame["open_time"].dt.strftime("%Y-%m-%d")
            for day, rows in frame.groupby(days, sort=True):
                rows.reset_index(drop=True).to_parquet(folder / f"{day}.parquet", index=False)
            manifest = {
                "plane": "bars",
                "symbol": symbol,
                "interval": self.spec.interval,
                "source": SYNTHETIC_SOURCE,
                "schema_version": SCHEMA_VERSION,
                "rows": len(frame),
                "seed": self.spec.seed,
                "generator": GENERATOR,
                "trading_research_version": __version__,
                "note": NOTE,
            }
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
        self.truth.write(root)
        return root


def _bars(symbol: str, log_close: np.ndarray, spec: PairSpec, key: int) -> pd.DataFrame:
    rng = rng_for(spec.seed, 91, key)
    n = len(log_close)
    close = np.exp(log_close)
    previous = np.concatenate([[log_close[0]], log_close[:-1]])
    open_ = np.exp(previous + rng.normal(0.0, spec.open_noise_bp * 1e-4, n))
    wick = np.abs(rng.normal(0.0, spec.vol_b_bp * 0.5e-4, (2, n)))
    first = day_start_ns(day_list(spec.start, 1)[0])
    step = spec.interval_minutes * 60 * 1_000_000_000
    open_ns = first + np.arange(n, dtype=np.int64) * step
    frame = pd.DataFrame(
        {
            "timestamp": stamps(open_ns + step - 1_000_000),
            "open_time": stamps(open_ns),
            "symbol": pd.array([symbol] * n, dtype="string"),
            "open": open_,
            "high": np.maximum(open_, close) * np.exp(wick[0]),
            "low": np.minimum(open_, close) * np.exp(-wick[1]),
            "close": close,
            "volume": rng.uniform(100.0, 200.0, n),
            "source": pd.array([SYNTHETIC_SOURCE] * n, dtype="string"),
        }
    )
    BARS_SCHEMA.validate(frame)
    return frame


def generate(spec: PairSpec | None = None) -> PairDataset:
    spec = spec or PairSpec()
    n = spec.bars
    rng = rng_for(spec.seed, 90)
    log_b = math.log(spec.price_b) + np.cumsum(rng.normal(0.0, spec.vol_b_bp * 1e-4, n))
    phi = spec.phi
    sd = spec.spread_sd_bp * 1e-4
    shocks = rng.normal(0.0, sd * math.sqrt(1.0 - phi**2), n)
    residual = np.empty(n)
    residual[0] = rng.normal(0.0, sd)
    cut = spec.break_bar if spec.break_bar is not None else n
    for k in range(1, n):
        pull = phi if k < cut else 1.0
        residual[k] = pull * residual[k - 1] + shocks[k]
    if spec.variant == "independent":
        # The same bar volatility as the planted first leg: beta's share of the
        # second leg's plus the residual's change.
        vol = math.sqrt((spec.beta * spec.vol_b_bp) ** 2 + 2.0 * (1.0 - phi) * spec.spread_sd_bp**2)
        log_a = spec.alpha + spec.beta * log_b[0] + np.cumsum(rng.normal(0.0, vol * 1e-4, n))
        residual = np.full(n, np.nan)
    else:
        log_a = spec.alpha + spec.beta * log_b + residual
    bars = {
        spec.symbol_a: _bars(spec.symbol_a, log_a, spec, 0),
        spec.symbol_b: _bars(spec.symbol_b, log_b, spec, 1),
    }
    cointegrated = spec.variant == "planted"
    planted = {
        "variant": spec.variant,
        "cointegrated": cointegrated,
        "beta": spec.beta if spec.variant != "independent" else None,
        "alpha": spec.alpha if spec.variant != "independent" else None,
        "half_life_bars": spec.half_life_bars if spec.variant != "independent" else None,
        "spread_sd_bp": spec.spread_sd_bp if spec.variant != "independent" else None,
        "bars": n,
        "interval": spec.interval,
    }
    if spec.break_bar is not None:
        planted["break_bar"] = spec.break_bar
        planted["break_at_open"] = str(bars[spec.symbol_a]["open_time"].iloc[spec.break_bar])
    truth = Truth(
        generator=GENERATOR,
        seed=spec.seed,
        null=not cointegrated,
        planted=planted,
        spec=spec_dict(spec),
        proves=PROVES,
        does_not_prove=DOES_NOT_PROVE,
    )
    return PairDataset(spec, bars, residual, truth)
