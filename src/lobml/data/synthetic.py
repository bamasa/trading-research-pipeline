"""A synthetic market that is honest about what it is.

Purpose and non-purpose
-----------------------
This generator exists so that the pipeline can be tested, demonstrated and run
in CI without downloading anything. It does **not** exist to demonstrate alpha.
A profitable backtest on synthetic data proves only that the generator and the
model agree, which is a statement about this file rather than about markets.
Every artefact produced here is stamped ``source="synthetic"`` so that a result
can never be mistaken for a real-data result further down the line.

What it does provide is a market with a *known* answer. The mid-price contains
a small, deliberately weak predictable component driven by order flow, with a
tunable strength. That gives the test suite something it otherwise could not
have: a dataset where the correct conclusion is known in advance. Set the
signal to zero and any pipeline that still reports an edge has a leak — which
is exactly the check :mod:`lobml.validation.leakage` performs.

Model
-----
Discrete time, one snapshot per step, deliberately simple and readable rather
than a serious microstructure simulation:

- **Mid-price.** A random walk in log space whose per-step volatility switches
  between a calm and an excited state via a two-state Markov chain. Regimes are
  what make a strategy's stability testable, so they are in the data from the
  start rather than bolted on later.
- **Signal.** An AR(1) latent process drives both the order flow imbalance and
  the *next* step's drift. The predictability is therefore genuinely causal:
  flow observable at *t* carries information about the return from *t* to
  *t+1*, and nothing observable at *t* depends on anything after it. This is
  the one property the generator must get right, because a generator with
  accidental look-ahead would make the leakage tests pass for the wrong reason.
- **Book.** Spread widens with volatility and narrows with liquidity. Depth
  decays geometrically away from the touch, is scaled by a slowly varying
  liquidity state, and is skewed by the latent signal so that queue imbalance
  is informative.
- **Trades.** Poisson arrivals with an intensity that rises with volatility.
  Aggressor side is drawn with a probability tilted by the signal, and size is
  lognormal. Trades execute against the book, so signed flow and the book agree
  by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from lobml import __version__
from lobml.data.schema import (
    SCHEMA_VERSION,
    DatasetManifest,
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
)

GENERATOR_NAME: Final = "lobml.synthetic.v1"

#: Marker written into the ``source`` column of every generated row.
SYNTHETIC_SOURCE: Final = "synthetic"


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic market.

    Defaults describe a liquid crypto perpetual sampled a few times a second:
    tight spread relative to price, frequent trades, occasional volatility
    bursts. They are chosen to make the demo quick and the tests fast, not to
    match any particular instrument.
    """

    symbol: str = "DEMOUSDT"
    n_steps: int = 20_000
    step_ms: int = 200
    start: str = "2026-01-01T00:00:00Z"
    depth: int = 10
    seed: int = 20260101

    initial_price: float = 30_000.0
    tick_size: float = 0.1
    lot_size: float = 0.001

    #: Per-step log-return volatility in the calm and excited states.
    vol_calm: float = 2.0e-5
    vol_excited: float = 8.0e-5
    #: Per-step transition probabilities of the volatility chain.
    p_calm_to_excited: float = 0.0008
    p_excited_to_calm: float = 0.0100

    #: Persistence and scale of the latent signal driving flow and drift.
    signal_ar: float = 0.97
    signal_scale: float = 1.0
    #: How much of the next return the signal explains, as a multiple of the
    #: current volatility. Small on purpose: a realistic edge is thin, and a
    #: strong signal would let a broken pipeline still look successful.
    signal_strength: float = 0.15

    #: Spread in ticks at the calm volatility, and how much it widens with vol.
    base_spread_ticks: float = 2.0
    spread_vol_sensitivity: float = 1.5
    max_spread_ticks: float = 40.0

    #: Resting size at the touch, and geometric decay per level away from it.
    base_depth_size: float = 5.0
    depth_decay: float = 0.82
    #: Persistence and scale of the slowly varying liquidity state.
    liquidity_ar: float = 0.995
    liquidity_scale: float = 0.25
    #: How strongly the latent signal skews resting size between the two sides.
    imbalance_strength: float = 0.45

    #: Trade arrivals per step at calm volatility, and sensitivity to vol.
    trade_intensity: float = 0.35
    trade_intensity_vol_sensitivity: float = 2.0
    trade_size_log_mean: float = -1.6
    trade_size_log_sd: float = 0.9
    #: How strongly the signal tilts aggressor side away from a coin flip.
    aggressor_tilt: float = 0.35

    def __post_init__(self) -> None:
        if self.n_steps < 2:
            raise ValueError(f"n_steps must be at least 2, got {self.n_steps}")
        if self.depth < 1:
            raise ValueError(f"depth must be at least 1, got {self.depth}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if not 0.0 <= self.signal_ar < 1.0:
            raise ValueError(f"signal_ar must be in [0, 1), got {self.signal_ar}")
        if not 0.0 <= self.liquidity_ar < 1.0:
            raise ValueError(f"liquidity_ar must be in [0, 1), got {self.liquidity_ar}")
        if self.base_spread_ticks < 1.0:
            raise ValueError(f"base_spread_ticks must be at least 1, got {self.base_spread_ticks}")


@dataclass
class SyntheticDataset:
    """Both planes of a generated market, plus the latent state that made it.

    ``latent`` is returned deliberately. It holds the true volatility regime and
    the true signal, which no model may see but which tests and diagnostics
    need: it is what lets a test assert "the pipeline recovers the regime" or
    "the measured edge matches the edge that was injected".
    """

    trades: pd.DataFrame
    book: pd.DataFrame
    latent: pd.DataFrame
    config: SyntheticConfig
    manifests: dict[str, DatasetManifest] = field(default_factory=dict)


def generate(config: SyntheticConfig | None = None, **overrides: object) -> SyntheticDataset:
    """Generate a synthetic market.

    The same ``seed`` always produces the same dataset, on any platform: all
    randomness comes from one seeded :class:`numpy.random.Generator` consumed in
    a fixed order. Reproducibility here is not a nicety — a demo that changes
    between runs cannot be used to check that a refactor was behaviour
    preserving.
    """
    cfg = config or SyntheticConfig()
    if overrides:
        cfg = replace_config(cfg, **overrides)

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps

    volatility, excited = _volatility_path(cfg, rng, n)
    signal = _ar1(rng, n, phi=cfg.signal_ar, scale=cfg.signal_scale)
    liquidity = _liquidity_path(cfg, rng, n)

    mid = _mid_path(cfg, rng, n, volatility, signal)
    timestamps = _timestamps(cfg, n)

    book = _build_book(cfg, rng, mid, volatility, liquidity, signal, timestamps)
    trades = _build_trades(cfg, rng, book, volatility, signal, timestamps)

    latent = pd.DataFrame(
        {
            "timestamp": timestamps,
            "mid_true": mid,
            "volatility": volatility,
            "regime_excited": excited,
            "signal": signal,
            "liquidity": liquidity,
        }
    )

    manifests = {
        "trades": DatasetManifest(
            plane="trades",
            symbol=cfg.symbol,
            source=SYNTHETIC_SOURCE,
            schema_version=SCHEMA_VERSION,
            rows=len(trades),
            start=str(timestamps[0]),
            end=str(timestamps[-1]),
            seed=cfg.seed,
            generator=GENERATOR_NAME,
            lobml_version=__version__,
            extra={"note": "Synthetic data. Not a market. Results are not evidence of edge."},
        ),
        "book": DatasetManifest(
            plane="book",
            symbol=cfg.symbol,
            source=SYNTHETIC_SOURCE,
            schema_version=SCHEMA_VERSION,
            rows=len(book),
            start=str(timestamps[0]),
            end=str(timestamps[-1]),
            depth=cfg.depth,
            seed=cfg.seed,
            generator=GENERATOR_NAME,
            lobml_version=__version__,
            extra={"note": "Synthetic data. Not a market. Results are not evidence of edge."},
        ),
    }

    return SyntheticDataset(
        trades=trades, book=book, latent=latent, config=cfg, manifests=manifests
    )


def replace_config(cfg: SyntheticConfig, **overrides: object) -> SyntheticConfig:
    """Return a copy of ``cfg`` with the given fields replaced, validating names."""
    known = set(SyntheticConfig.__dataclass_fields__)
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"unknown config field(s): {', '.join(sorted(unknown))}")
    merged = {f: getattr(cfg, f) for f in known}
    merged.update(overrides)
    return SyntheticConfig(**merged)


# ---------------------------------------------------------------------------
# Latent processes
# ---------------------------------------------------------------------------


def _ar1(rng: np.random.Generator, n: int, *, phi: float, scale: float) -> np.ndarray:
    """A stationary AR(1) path with unit-variance innovations scaled to ``scale``.

    Started from its stationary distribution rather than from zero, so there is
    no burn-in transient at the head of the sample that a model could learn as a
    spurious time effect.
    """
    innovation_sd = scale * np.sqrt(1.0 - phi**2)
    out = np.empty(n, dtype=np.float64)
    out[0] = rng.normal(0.0, scale)
    noise = rng.normal(0.0, innovation_sd, size=n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def _volatility_path(
    cfg: SyntheticConfig, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Markov volatility. Returns per-step vol and the regime flag."""
    excited = np.zeros(n, dtype=bool)
    draws = rng.random(n)
    state = False
    for i in range(n):
        threshold = cfg.p_excited_to_calm if state else cfg.p_calm_to_excited
        if draws[i] < threshold:
            state = not state
        excited[i] = state
    volatility = np.where(excited, cfg.vol_excited, cfg.vol_calm)
    return volatility, excited


def _liquidity_path(cfg: SyntheticConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """A slowly varying positive multiplier on resting size."""
    raw = _ar1(rng, n, phi=cfg.liquidity_ar, scale=cfg.liquidity_scale)
    return np.exp(raw)


def _mid_path(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    n: int,
    volatility: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    """Log random walk with a signal-driven drift.

    The causal structure is the important part. The return realised between
    step ``i-1`` and step ``i`` uses ``signal[i - 1]``: the value of the latent
    process *before* the move. Since the book and the flow at step ``i-1`` are
    also built from ``signal[i - 1]``, everything an observer can see at
    ``i-1`` is informative about the next return and nothing more. Shift this
    by one index and the generator would leak the future into the present,
    quietly invalidating every leakage test that relies on it.
    """
    drift = np.zeros(n, dtype=np.float64)
    drift[1:] = cfg.signal_strength * volatility[1:] * signal[:-1]

    shock = rng.normal(0.0, 1.0, size=n) * volatility
    shock[0] = 0.0

    log_mid = np.log(cfg.initial_price) + np.cumsum(drift + shock)
    return np.exp(log_mid)


def _timestamps(cfg: SyntheticConfig, n: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(cfg.start)
    if start.tz is None:
        start = start.tz_localize("UTC")
    return pd.date_range(start=start, periods=n, freq=f"{cfg.step_ms}ms", tz="UTC")


# ---------------------------------------------------------------------------
# Book construction
# ---------------------------------------------------------------------------


def _build_book(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    mid: np.ndarray,
    volatility: np.ndarray,
    liquidity: np.ndarray,
    signal: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble snapshots around the mid path.

    Prices are snapped to the tick grid, which is what makes the spread an
    integer number of ticks and keeps level ordering exact. Building the two
    sides by stepping outward from a rounded mid, rather than rounding each
    level independently, guarantees the book is never crossed and that levels
    never collide — invariants the validator enforces and that real
    reconstruction bugs violate.
    """
    n = len(mid)
    tick = cfg.tick_size

    vol_ratio = volatility / cfg.vol_calm
    spread_ticks = cfg.base_spread_ticks * (1.0 + cfg.spread_vol_sensitivity * (vol_ratio - 1.0))
    spread_ticks = spread_ticks / np.maximum(liquidity, 0.2)
    spread_ticks = np.clip(np.rint(spread_ticks), 1.0, cfg.max_spread_ticks)

    # Half-spreads that differ by at most one tick when the total is odd; the
    # mid then sits on or beside the tick grid exactly as it does on a real
    # venue, instead of always landing between two levels.
    half_lower = np.floor(spread_ticks / 2.0)
    half_upper = spread_ticks - half_lower

    mid_ticks = np.rint(mid / tick)
    best_bid = (mid_ticks - half_lower) * tick
    best_ask = (mid_ticks + half_upper) * tick

    # Level spacing widens with volatility, so a stressed book is both wider at
    # the touch and thinner further out.
    gap_ticks = np.maximum(1.0, np.rint(spread_ticks / 2.0))

    skew = np.tanh(cfg.imbalance_strength * signal)

    columns: dict[str, np.ndarray] = {}
    decay = cfg.depth_decay ** np.arange(cfg.depth)

    for level in range(cfg.depth):
        offset = level * gap_ticks * tick
        columns[bid_price_col(level)] = best_bid - offset
        columns[ask_price_col(level)] = best_ask + offset

        scale = cfg.base_depth_size * decay[level] * liquidity
        noise_bid = rng.gamma(shape=4.0, scale=0.25, size=n)
        noise_ask = rng.gamma(shape=4.0, scale=0.25, size=n)

        # A positive signal means buying pressure: the bid side is deeper.
        columns[bid_size_col(level)] = _round_lots(scale * (1.0 + skew) * noise_bid, cfg.lot_size)
        columns[ask_size_col(level)] = _round_lots(scale * (1.0 - skew) * noise_ask, cfg.lot_size)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": pd.array([cfg.symbol] * n, dtype="string"),
            "sequence_id": np.arange(1, n + 1, dtype=np.int64),
            "source": pd.array([SYNTHETIC_SOURCE] * n, dtype="string"),
            **columns,
        }
    )


def _round_lots(size: np.ndarray, lot_size: float) -> np.ndarray:
    """Snap sizes to the lot grid, keeping at least one lot at every level."""
    return np.maximum(np.rint(size / lot_size), 1.0) * lot_size


# ---------------------------------------------------------------------------
# Trade construction
# ---------------------------------------------------------------------------


def _build_trades(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    book: pd.DataFrame,
    volatility: np.ndarray,
    signal: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Draw trades that execute against the book snapshots.

    Trades are priced at the touch of the side they hit, so an aggressive buy
    pays the ask and an aggressive sell hits the bid. Order flow imbalance
    computed from these trades and queue imbalance read off the book therefore
    point the same way by construction, which is what makes the two planes
    usable together.
    """
    n = len(book)
    vol_ratio = volatility / cfg.vol_calm
    intensity = cfg.trade_intensity * (
        1.0 + cfg.trade_intensity_vol_sensitivity * (vol_ratio - 1.0)
    )
    counts = rng.poisson(np.maximum(intensity, 0.0))

    total = int(counts.sum())
    if total == 0:
        return _empty_trades()

    step_index = np.repeat(np.arange(n), counts)

    # A positive signal tilts the aggressor towards buying.
    p_buy = 0.5 + 0.5 * np.tanh(cfg.aggressor_tilt * signal[step_index])
    is_aggressive_buy = rng.random(total) < p_buy

    best_bid = book[bid_price_col(0)].to_numpy()[step_index]
    best_ask = book[ask_price_col(0)].to_numpy()[step_index]
    price = np.where(is_aggressive_buy, best_ask, best_bid)

    quantity = _round_lots(
        rng.lognormal(cfg.trade_size_log_mean, cfg.trade_size_log_sd, size=total),
        cfg.lot_size,
    )

    # Trades within one step are spread across its duration so that timestamps
    # stay non-decreasing and inter-arrival features are not degenerate.
    offsets = pd.to_timedelta(_intra_step_offsets(step_index, cfg.step_ms, rng), unit="ms")
    trade_time = timestamps[step_index] + offsets

    return pd.DataFrame(
        {
            "timestamp": trade_time,
            "symbol": pd.array([cfg.symbol] * total, dtype="string"),
            "price": price,
            "quantity": quantity,
            # The exchange convention: True means the resting order was the buy.
            # An aggressive buy therefore lifts a resting ask, so is_buyer_maker
            # is False.
            "is_buyer_maker": ~is_aggressive_buy,
            "trade_id": np.arange(1, total + 1, dtype=np.int64),
            "source": pd.array([SYNTHETIC_SOURCE] * total, dtype="string"),
        }
    )


def _intra_step_offsets(
    step_index: np.ndarray, step_ms: int, rng: np.random.Generator
) -> np.ndarray:
    """Millisecond offsets within each step, ascending inside every step.

    ``step_index`` arrives already sorted, so ordering the offsets by
    ``(step_index, offset)`` and reading them back in that order leaves each
    group internally ascending while every group stays in place. Timestamps are
    then non-decreasing overall, which the validator requires.
    """
    offsets = rng.random(len(step_index)) * step_ms
    order = np.lexsort((offsets, step_index))
    return np.floor(offsets[order])


def _empty_trades() -> pd.DataFrame:
    """An empty trade frame that still satisfies the contract.

    A quiet market is a legitimate outcome, not an error, so it must produce
    something the rest of the pipeline can read rather than a special case.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex([], tz="UTC"),
            "symbol": pd.array([], dtype="string"),
            "price": np.array([], dtype=np.float64),
            "quantity": np.array([], dtype=np.float64),
            "is_buyer_maker": np.array([], dtype=bool),
            "trade_id": np.array([], dtype=np.int64),
            "source": pd.array([], dtype="string"),
        }
    )
