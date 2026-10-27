"""A universe of instruments on bars, with a planted reversion (or momentum), and its null.

What is planted
---------------
``n_instruments`` instruments on ``bar_s``-second bars over ``days`` whole days.
Each log price is a market part shared by all and an idiosyncratic part, and
each part is a random walk plus a **transient** component ``z``, an AR(1) with
half-life ``half_life_bars`` (``phi = 2 ** (-1 / half_life)``):

* ``variant="reversal"``: the price carries ``z`` itself. A shock moves the
  price at once and then decays back, so after the price has moved by ``z``
  above its permanent level it is expected to move ``(phi**H - 1) * z`` over
  the next ``H`` bars: a cross-sectional reversion for the idiosyncratic part
  (stationary sd ``size_bp``), and a market-wide one for the market part
  (``market_size_bp``) that an index of the other instruments reveals;
* ``variant="momentum"``: the price carries ``S - z``, ``S`` the running sum of
  the transient's shocks, so a shock enters the price gradually: after a move
  the price is expected to continue by ``(1 - phi**H) * z``.

``truth.json`` records the sizes, the half-life and ``phi``; the generator also
keeps the latent ``z`` of every instrument and of the market, from which a
check computes what any position was owed by the planted effect alone.

The null
--------
``null=True`` removes both transients and adds, in their place, a random walk
with the variance of the transient's own increments, so each instrument's
returns keep their variance and lose every autocorrelation.

What it can prove, and what it cannot
-------------------------------------
It can show that the repository's cross-sectional rules recover a reversion of
known size and half-life, lose on a momentum they bet against, and find nothing
on a null with the same return variance. It cannot show that any universe of
real instruments reverts, at what half-life, or that costs at that horizon
would leave anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from trading_research.synthetic.common import (
    DEFAULT_START,
    NS_PER_S,
    SYNTHETIC_SOURCE,
    Truth,
    check_frame,
    day_list,
    day_start_ns,
    rng_for,
    spec_dict,
    stamps,
    strings,
    write_plane,
)

GENERATOR = "trading_research.synthetic.cross_section.v1"

Variant = Literal["reversal", "momentum"]

PROVES = (
    "The repository's cross-sectional rules recover a planted reversion of known size and "
    "half-life, lose on a planted momentum they bet against, and find nothing on the null."
)
DOES_NOT_PROVE = (
    "That real instruments revert, at what half-life, or that the effect would survive "
    "costs at that horizon."
)


@dataclass(frozen=True)
class CrossSectionSpec:
    """Every parameter of a cross-section dataset. The defaults generate in seconds."""

    n_instruments: int = 20
    start: str = DEFAULT_START
    days: int = 16
    bar_s: int = 5
    #: Random-walk volatilities per bar, bp.
    market_vol_bp: float = 2.0
    idio_vol_bp: float = 3.0
    #: Stationary sd of the idiosyncratic transient, bp.
    size_bp: float = 8.0
    #: Stationary sd of the market transient, bp.
    market_size_bp: float = 20.0
    half_life_bars: float = 60.0
    variant: Variant = "reversal"
    null: bool = False
    seed: int = 20200106

    def __post_init__(self) -> None:
        if self.n_instruments < 6:
            raise ValueError("a cross-section needs at least six instruments")
        if self.variant not in ("reversal", "momentum"):
            raise ValueError(f"variant is 'reversal' or 'momentum', got {self.variant!r}")
        if self.half_life_bars <= 0 or self.bar_s < 1 or 86_400 % self.bar_s:
            raise ValueError("the half-life must be positive and a bar must divide a day")

    @property
    def phi(self) -> float:
        return float(2.0 ** (-1.0 / self.half_life_bars))

    @property
    def rows_per_day(self) -> int:
        return 86_400 // self.bar_s

    @property
    def symbols(self) -> list[str]:
        return [f"SYN{k:02d}USDT" for k in range(self.n_instruments)]

    def expected_move(self, z: np.ndarray, hold: int) -> np.ndarray:
        """Expected log move over ``hold`` bars owed to the transient ``z`` (log units)."""
        factor = self.phi**hold - 1.0
        return (factor if self.variant == "reversal" else -factor) * z


@dataclass
class CrossSectionDataset:
    """The panel, the bars, the latent transients and the truth."""

    spec: CrossSectionSpec
    days: list[date]
    #: Log closes, one column per instrument, indexed by bar close (UTC).
    log_prices: pd.DataFrame
    #: Latent transient per instrument (idiosyncratic plus market), log units.
    transient: pd.DataFrame
    #: The market's own transient, log units.
    market_transient: pd.Series
    bars: dict[str, dict[date, pd.DataFrame]]
    truth: Truth

    def day_of_row(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(rows, dtype=np.int64) // self.spec.rows_per_day

    def write(self, directory: Path | str, *, validate: bool = True) -> Path:
        root = Path(directory)
        for symbol, frames in self.bars.items():
            if validate:
                for frame in frames.values():
                    check_frame(frame, "bars")
            write_plane(
                root, "bars", symbol, frames, generator=GENERATOR, seed=self.spec.seed,
                extra={"interval_s": self.spec.bar_s},
            )  # fmt: skip
        latent = self.transient.copy()
        latent["__market__"] = self.market_transient
        latent.reset_index().to_parquet(root / "latent_transient.parquet", index=False)
        self.truth.write(root)
        return root


def _ar1(innovations: np.ndarray, phi: float) -> np.ndarray:
    """``z_t = phi * z_{t-1} + e_t`` down the first axis, ``z_0 = e_0``."""
    out = np.array(innovations, dtype=np.float64, copy=True)
    for t in range(1, len(out)):
        out[t] += phi * out[t - 1]
    return out


def generate(spec: CrossSectionSpec | None = None) -> CrossSectionDataset:
    """Generate the whole span at once; the same seed gives the same panel."""
    spec = spec or CrossSectionSpec()
    days = day_list(spec.start, spec.days)
    rng = rng_for(spec.seed, 0)
    n = spec.rows_per_day * spec.days
    k = spec.n_instruments
    bp = 1e-4
    phi = spec.phi
    common = rng.normal(0.0, spec.market_vol_bp * bp, n)
    idio = rng.normal(0.0, spec.idio_vol_bp * bp, (n, k))
    eta_m = rng.normal(0.0, spec.market_size_bp * bp * np.sqrt(1.0 - phi**2), n)
    eta_i = rng.normal(0.0, spec.size_bp * bp * np.sqrt(1.0 - phi**2), (n, k))
    # Started from the stationary distribution, so the head of the span has no
    # transient a model could learn as a time effect.
    eta_m[0] = rng.normal(0.0, spec.market_size_bp * bp)
    eta_i[0] = rng.normal(0.0, spec.size_bp * bp, k)
    z_m = _ar1(eta_m, phi)
    z_i = _ar1(eta_i, phi)
    if spec.variant == "reversal":
        part_m, part_i = z_m, z_i
        step_var = 2.0 * (1.0 - phi)
    else:
        part_m = np.cumsum(eta_m) - z_m
        part_i = np.cumsum(eta_i, axis=0) - z_i
        step_var = (1.0 - phi) ** 2
    if spec.null:
        # The transient's increments keep their variance and lose their memory.
        part_m = np.cumsum(rng.normal(0.0, spec.market_size_bp * bp * np.sqrt(step_var), n))
        part_i = np.cumsum(rng.normal(0.0, spec.size_bp * bp * np.sqrt(step_var), (n, k)), axis=0)
        z_m = np.zeros(n)
        z_i = np.zeros((n, k))
    start = np.log(rng.uniform(20.0, 200.0, k))
    log_p = start[None, :] + np.cumsum(common)[:, None] + part_m[:, None]
    log_p = log_p + np.cumsum(idio, axis=0) + part_i

    first = day_start_ns(days[0])
    bar_ns = spec.bar_s * NS_PER_S
    close_ns = first + (np.arange(n, dtype=np.int64) + 1) * bar_ns
    index = stamps(close_ns)
    symbols = spec.symbols
    log_prices = pd.DataFrame(log_p, index=index, columns=symbols)
    transient = pd.DataFrame(z_i + z_m[:, None], index=index, columns=symbols)
    market = pd.Series(z_m, index=index, name="market_transient")
    bars = {s: _bars(spec, rng, s, log_p[:, j], close_ns, days) for j, s in enumerate(symbols)}
    truth = Truth(
        generator=GENERATOR,
        seed=spec.seed,
        null=spec.null,
        planted={
            "variant": spec.variant,
            "size_bp": 0.0 if spec.null else spec.size_bp,
            "market_size_bp": 0.0 if spec.null else spec.market_size_bp,
            "half_life_bars": spec.half_life_bars,
            "half_life_s": spec.half_life_bars * spec.bar_s,
            "phi": phi,
        },
        spec=spec_dict(spec),
        proves=PROVES,
        does_not_prove=DOES_NOT_PROVE,
    )
    return CrossSectionDataset(spec, days, log_prices, transient, market, bars, truth)


def _bars(
    spec: CrossSectionSpec,
    rng: np.random.Generator,
    symbol: str,
    log_close: np.ndarray,
    close_ns: np.ndarray,
    days: list[date],
) -> dict[date, pd.DataFrame]:
    """OHLCV bars for one instrument, a frame per day; the open is the last close."""
    close = np.exp(log_close)
    opened = np.concatenate([[close[0] * np.exp(rng.normal(0.0, 1e-4))], close[:-1]])
    wiggle = np.abs(rng.normal(0.0, spec.idio_vol_bp * 1e-4, (2, len(close))))
    high = np.maximum(opened, close) * np.exp(wiggle[0])
    low = np.minimum(opened, close) * np.exp(-wiggle[1])
    volume = np.round(rng.lognormal(3.0, 0.8, len(close)), 3)
    bar_ns = spec.bar_s * NS_PER_S
    out: dict[date, pd.DataFrame] = {}
    rows = spec.rows_per_day
    for d, day in enumerate(days):
        part = slice(d * rows, (d + 1) * rows)
        count = rows
        out[day] = pd.DataFrame(
            {
                "timestamp": stamps(close_ns[part]),
                "open_time": stamps(close_ns[part] - bar_ns),
                "symbol": strings(symbol, count),
                "open": opened[part],
                "high": high[part],
                "low": low[part],
                "close": close[part],
                "volume": volume[part],
                "source": strings(SYNTHETIC_SOURCE, count),
            }
        )
    return out
