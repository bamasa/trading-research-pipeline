"""Order-book days with a planted short-horizon edge, and their null.

What is planted
---------------
A ten-level book photographed every ``step_ms`` (100 ms) and the prints with
their aggressor, over ``days`` sessions of ``session_s`` seconds. The log mid is
a random walk (``noise_bp`` per square-root second, the unpredictable part) plus
a drift that arrives in **episodes**. An episode starts at a snapshot ``k0``
with a direction ``d``; from that snapshot on, for ``signal_s`` seconds, the
chosen feature shows ``d``:

* ``feature="imbalance"``: the touch on side ``d`` is ``(1 + q) / (1 - q)``
  times its usual size, so the queue imbalance reads about ``q * d``;
* ``feature="flow"``: prints arrive ``flow_burst`` times faster and their
  aggressor is ``d`` with probability ``flow_tilt`` instead of one half, so
  signed flow reads ``d``.

From the same snapshot the expected log mid moves ``d * size_bp`` over the next
``horizon_s`` seconds and then stops, at a rate that decays with half-life
``decay_s``: the share of the move done ``u`` seconds after the onset is
``D(u) = (1 - 2**(-u / decay)) / (1 - 2**(-horizon / decay))``. A taker who sees
the feature at ``k0`` and crosses after a latency captures ``size_bp * (1 -
D(delay))`` in expectation, before costs.

Causality is the one property this generator must get right. The feature is
visible at ``k0`` (the snapshot, and the prints in the interval ending at it,
which the backward rule of :mod:`trading_research.features.prints` assigns to
``k0``); the drift moves the mid only from ``k0`` to ``k0 + 1`` onward. Nothing
observable at a snapshot depends on anything after it.

The null
--------
``null=True`` keeps the episodes, the feature's skew or tilt and the size of
each episode's move, draws the move's direction independently of the
feature's, and makes the whole move in the one step after the onset. The book,
the prints and the mid's moves over the horizon have the same distribution as
in the planted data; nothing observable predicts what is left of a move,
because nothing is left. (A drift spread over the horizon, as planted, would be
predictable from its own start even with a random direction: the price's
momentum inside an episode is part of the planted edge, not of the null.)

What it can prove, and what it cannot
-------------------------------------
It can show that a pipeline (features, labels, a model or a rule, the taker
executor) finds a predictable component of a known size where one exists, that
it measures its size without bias (realised gross against the planted drift of
the very trades taken), and that it finds nothing on the null or after the
decisions are moved to other days. It cannot show that any such component
exists in a market, that the feature's real-world strength resembles this one,
or that costs, queues and latency behave as modelled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from trading_research.data.schema import (
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
)
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

GENERATOR = "trading_research.synthetic.book_days.v1"

Feature = Literal["imbalance", "flow"]

PROVES = (
    "A pipeline run on this data finds a planted short-horizon drift of known size, "
    "measures it without bias, and finds nothing on the null or on decisions moved "
    "to other days."
)
DOES_NOT_PROVE = (
    "That such a drift exists in any market, that real features carry it with this "
    "strength, or that real costs, queues and latency behave as modelled here."
)


@dataclass(frozen=True)
class BookDaysSpec:
    """Every parameter of a book-days dataset. The defaults generate in a few seconds."""

    symbol: str = "SYNTHUSDT"
    start: str = DEFAULT_START
    days: int = 16
    #: Seconds of data per day, from midnight UTC.
    session_s: int = 3600
    step_ms: int = 100
    depth: int = 10
    price0: float = 1000.0
    tick: float = 0.1
    lot: float = 0.001
    feature: Feature = "imbalance"
    #: Expected move of the log mid over the horizon after an episode's onset.
    size_bp: float = 16.0
    horizon_s: float = 30.0
    #: Half-life of the drift's rate; long makes the move nearly linear.
    decay_s: float = 10.0
    #: Volatility of the unpredictable part, bp per square-root second.
    noise_bp: float = 1.0
    episodes_per_hour: float = 40.0
    #: How long an episode's feature shows its direction.
    signal_s: float = 2.0
    #: Queue imbalance the touch shows during an imbalance episode.
    imbalance_skew: float = 0.6
    #: Probability that a print's aggressor is the episode's direction (flow).
    flow_tilt: float = 0.85
    trades_per_s: float = 2.0
    #: Print rate during a flow episode's signal, as a multiple of the usual
    #: rate: informed flow arrives in a burst.
    flow_burst: float = 5.0
    spread_ticks: int = 2
    #: Share of time the spread is ``wide_ticks`` wider, in spells of
    #: ``wide_mean_s`` seconds on average: a spread that moves on its own.
    wide_share: float = 0.15
    wide_ticks: int = 1
    wide_mean_s: float = 3.0
    base_depth: float = 5.0
    #: Gamma shape of the touch sizes' noise; larger makes the background
    #: imbalance quieter and the planted one easier to see.
    touch_noise_shape: float = 16.0
    depth_decay: float = 0.85
    #: Seconds at the head of each session without an onset, so features warm up.
    warmup_s: float = 60.0
    null: bool = False
    seed: int = 20200106

    def __post_init__(self) -> None:
        if self.feature not in ("imbalance", "flow"):
            raise ValueError(f"feature is 'imbalance' or 'flow', got {self.feature!r}")
        if self.step_ms < 1 or (1000 % self.step_ms and self.step_ms % 1000):
            raise ValueError(
                f"step_ms must divide a second or be whole seconds, got {self.step_ms}"
            )
        if self.horizon_s <= 0 or self.decay_s <= 0 or self.signal_s <= 0:
            raise ValueError("horizon, decay and signal duration must be positive")
        if self.session_s < self.warmup_s + 2 * self.horizon_s:
            raise ValueError("the session is too short for a warm-up and an episode")
        if not 0.0 <= self.imbalance_skew < 1.0:
            raise ValueError("imbalance_skew must lie in [0, 1)")
        if not 0.5 <= self.flow_tilt <= 1.0:
            raise ValueError("flow_tilt must lie in [0.5, 1]")
        if self.spread_ticks < 1 or self.depth < 1 or self.tick <= 0 or self.lot <= 0:
            raise ValueError("spread, depth, tick and lot must be positive")

    @property
    def step_s(self) -> float:
        return self.step_ms / 1000.0

    @property
    def rows_per_day(self) -> int:
        return round(self.session_s / self.step_s)

    def share_done(self, seconds: np.ndarray | float) -> np.ndarray:
        """``D(u)``: the share of an episode's move done ``u`` seconds after its onset."""
        u = np.clip(np.asarray(seconds, dtype=np.float64), 0.0, self.horizon_s)
        return (1.0 - 2.0 ** (-u / self.decay_s)) / (1.0 - 2.0 ** (-self.horizon_s / self.decay_s))


@dataclass(frozen=True)
class Regime:
    """How a stretch of a day differs from the spec: the break generator's lever."""

    vol_multiplier: float = 1.0
    spread_ticks: int | None = None
    #: Multiplies each drift (and its sign) on top of the volatility: -1 flips
    #: the planted edge.
    edge_multiplier: float = 1.0


@dataclass
class BookDataset:
    """Generated days: the planes, the latent state that made them, and the truth."""

    spec: BookDaysSpec
    days: list[date]
    book: dict[date, pd.DataFrame]
    trades: dict[date, pd.DataFrame]
    #: Per day: ``timestamp``, ``drift_log`` (the planted cumulative drift, log
    #: units), ``signal`` (the direction the feature shows, 0 outside episodes),
    #: ``regime`` (0 before a break, 1 after). No model may read it.
    latent: dict[date, pd.DataFrame]
    truth: Truth
    episodes: pd.DataFrame = field(default_factory=pd.DataFrame)

    def prints(self, day: date) -> pd.DataFrame:
        """The day's prints in the Bybit contract (``timestamp``, ``price``, ``size``,
        ``aggressor``) that the print features read."""
        trades = self.trades[day]
        return pd.DataFrame(
            {
                "timestamp": trades["timestamp"],
                "price": trades["price"].to_numpy(dtype=np.float64),
                "size": trades["quantity"].to_numpy(dtype=np.float64),
                "aggressor": np.where(trades["is_buyer_maker"].to_numpy(), -1, 1).astype(np.int64),
            }
        )

    def planted_move_bp(self, day: date, entry_ns: np.ndarray, exit_ns: np.ndarray) -> np.ndarray:
        """The planted drift between two snapshot stamps, in bp of log mid, per pair."""
        latent = self.latent[day]
        ts = latent["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        drift = latent["drift_log"].to_numpy(dtype=np.float64)
        e = np.clip(np.searchsorted(ts, np.asarray(entry_ns, dtype=np.int64), "right") - 1, 0, None)
        x = np.clip(np.searchsorted(ts, np.asarray(exit_ns, dtype=np.int64), "right") - 1, 0, None)
        return (drift[x] - drift[e]) * 1e4

    def write(self, directory: Path | str, *, validate: bool = True) -> Path:
        """Write the planes, the latent state and ``truth.json``; validate first."""
        root = Path(directory)
        if validate:
            for day in self.days:
                check_frame(self.book[day], "book", depth=self.spec.depth)
                check_frame(self.trades[day], "trades")
        write_plane(
            root, "book", self.spec.symbol, self.book, generator=GENERATOR,
            seed=self.spec.seed, depth=self.spec.depth,
            extra={"step_ms": self.spec.step_ms},
        )  # fmt: skip
        write_plane(
            root, "trades", self.spec.symbol, self.trades, generator=GENERATOR,
            seed=self.spec.seed,
        )  # fmt: skip
        latent_dir = root / "latent" / self.spec.symbol
        latent_dir.mkdir(parents=True, exist_ok=True)
        for day, frame in self.latent.items():
            frame.to_parquet(latent_dir / f"{day.isoformat()}.parquet", index=False)
        self.episodes.to_csv(root / "episodes.csv", index=False)
        self.truth.write(root)
        return root


def planted_truth(spec: BookDaysSpec, generator: str = GENERATOR, **extra: object) -> Truth:
    """The truth record of a spec: the planted edge, or none for the null."""
    planted = {
        "feature": spec.feature,
        "size_bp": 0.0 if spec.null else spec.size_bp,
        "drift_size_bp": spec.size_bp,
        "horizon_s": spec.horizon_s,
        "decay_s": spec.decay_s,
        "noise_bp_per_sqrt_s": spec.noise_bp,
        "episodes_per_hour": spec.episodes_per_hour,
        "signal_s": spec.signal_s,
        "spread_ticks": spec.spread_ticks,
        "tick_bp_at_start": spec.tick / spec.price0 * 1e4,
        "capture_after_one_step": spec.size_bp * float(1.0 - spec.share_done(spec.step_s)),
    }
    return Truth(
        generator=generator,
        seed=spec.seed,
        null=spec.null,
        planted=planted,
        spec=spec_dict(spec),
        proves=PROVES,
        does_not_prove=DOES_NOT_PROVE,
        extra=dict(extra),
    )


def generate(
    spec: BookDaysSpec | None = None,
    *,
    break_ns: int | None = None,
    regime: Regime | None = None,
    truth: Truth | None = None,
) -> BookDataset:
    """Generate every day of ``spec``.

    With ``break_ns`` and ``regime``, every snapshot stamped at or after
    ``break_ns`` follows ``regime``: the lever :mod:`.break_days` uses to plant
    a break. Plain book days pass neither.
    """
    spec = spec or BookDaysSpec()
    days = day_list(spec.start, spec.days)
    log_price = float(np.log(spec.price0))
    books: dict[date, pd.DataFrame] = {}
    trades: dict[date, pd.DataFrame] = {}
    latents: dict[date, pd.DataFrame] = {}
    episodes: list[pd.DataFrame] = []
    sequence = 1
    trade_id = 1
    for index, day in enumerate(days):
        out = _one_day(spec, day, index, log_price, sequence, trade_id, break_ns, regime)
        books[day], trades[day], latents[day], found, log_price = out
        sequence += len(books[day])
        trade_id += len(trades[day])
        episodes.append(found)
    return BookDataset(
        spec=spec,
        days=days,
        book=books,
        trades=trades,
        latent=latents,
        truth=truth or planted_truth(spec),
        episodes=pd.concat(episodes, ignore_index=True),
    )


def _onsets(spec: BookDaysSpec, rng: np.random.Generator) -> np.ndarray:
    """Episode onset rows: a Poisson process thinned so episodes never overlap."""
    n = spec.rows_per_day
    warm = int(spec.warmup_s / spec.step_s)
    horizon_rows = int(np.ceil(spec.horizon_s / spec.step_s))
    last = n - horizon_rows - 1
    expected = spec.episodes_per_hour * spec.session_s / 3600.0
    count = int(rng.poisson(expected * 1.6)) + 1
    candidates = np.sort(rng.integers(warm, max(last, warm + 1), size=count))
    gap = horizon_rows + int(np.ceil(spec.signal_s / spec.step_s))
    kept: list[int] = []
    for row in candidates.tolist():
        if not kept or row - kept[-1] >= gap:
            kept.append(row)
    keep = rng.random(len(kept)) < min(1.0, expected / max(len(kept), 1))
    return np.asarray(kept, dtype=np.int64)[keep]


def _one_day(
    spec: BookDaysSpec,
    day: date,
    index: int,
    log_price: float,
    sequence: int,
    trade_id: int,
    break_ns: int | None,
    regime: Regime | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    rng = rng_for(spec.seed, index)
    n = spec.rows_per_day
    step = spec.step_s
    start_ns = day_start_ns(day)
    step_ns = spec.step_ms * 1_000_000
    row_ns = start_ns + np.arange(n, dtype=np.int64) * step_ns
    if break_ns is None or regime is None:
        after = np.zeros(n, dtype=bool)
        regime = Regime()
    else:
        after = row_ns >= break_ns

    onsets = _onsets(spec, rng)
    direction = rng.choice(np.array([-1, 1]), size=len(onsets))
    drift_sign = rng.choice(np.array([-1, 1]), size=len(onsets)) if spec.null else direction
    # A regime scales every part of the return, the drift included, so the
    # volatility ratio is the ratio of the whole return's scale.
    multiplier = np.where(after[onsets], regime.edge_multiplier * regime.vol_multiplier, 1.0)
    drift = np.zeros(n)
    signal = np.zeros(n, dtype=np.int8)
    signal_rows = max(1, round(spec.signal_s / step))
    horizon_rows = int(np.ceil(spec.horizon_s / step))
    for k0, d, e, m in zip(onsets, direction, drift_sign, multiplier, strict=True):
        u = np.arange(0, horizon_rows + 1) * step
        # The null's move is all in the step after the onset: no part of it is
        # left for anyone who saw the onset, from the feature or from the price.
        shape = (u > 0).astype(np.float64) if spec.null else spec.share_done(u)
        piece = float(e) * float(m) * spec.size_bp * 1e-4 * shape
        stop = k0 + len(piece)
        drift[k0:stop] += piece
        drift[stop:] += piece[-1]
        signal[k0 : min(n, k0 + signal_rows)] = d

    vol = spec.noise_bp * 1e-4 * np.sqrt(step) * np.where(after, regime.vol_multiplier, 1.0)
    shocks = rng.normal(0.0, 1.0, n) * vol
    shocks[0] = 0.0
    log_mid = log_price + np.cumsum(shocks) + drift
    mid = np.exp(log_mid)

    spread = np.full(n, float(spec.spread_ticks))
    if regime.spread_ticks is not None:
        spread = np.where(after, float(regime.spread_ticks), spread)
    spread = spread + spec.wide_ticks * _spells(rng, n, step, spec.wide_share, spec.wide_mean_s)
    half_lower = np.floor(spread / 2.0)
    half_upper = spread - half_lower
    mid_ticks = np.rint(mid / spec.tick)
    best_bid = mid_ticks - half_lower
    best_ask = mid_ticks + half_upper

    columns: dict[str, np.ndarray] = {}
    decay = spec.depth_decay ** np.arange(spec.depth)
    skew = (1.0 + spec.imbalance_skew) / (1.0 - spec.imbalance_skew)
    for level in range(spec.depth):
        columns[bid_price_col(level)] = np.round((best_bid - level) * spec.tick, 10)
        columns[ask_price_col(level)] = np.round((best_ask + level) * spec.tick, 10)
        scale = spec.base_depth * decay[level]
        gamma_shape = spec.touch_noise_shape if level == 0 else 4.0
        bid_size = scale * rng.gamma(gamma_shape, 1.0 / gamma_shape, n)
        ask_size = scale * rng.gamma(gamma_shape, 1.0 / gamma_shape, n)
        if level == 0 and spec.feature == "imbalance":
            bid_size = np.where(signal > 0, bid_size * skew, bid_size)
            ask_size = np.where(signal < 0, ask_size * skew, ask_size)
        columns[bid_size_col(level)] = _lots(bid_size, spec.lot)
        columns[ask_size_col(level)] = _lots(ask_size, spec.lot)

    book = pd.DataFrame(
        {
            "timestamp": stamps(row_ns),
            "symbol": strings(spec.symbol, n),
            "sequence_id": np.arange(sequence, sequence + n, dtype=np.int64),
            "source": strings(SYNTHETIC_SOURCE, n),
            **columns,
        }
    )
    trades = _prints(spec, rng, row_ns, signal, best_bid, best_ask, trade_id)
    latent = pd.DataFrame(
        {
            "timestamp": stamps(row_ns),
            "drift_log": drift,
            "signal": signal,
            "regime": after.astype(np.int8),
            "mid_latent": mid,
        }
    )
    found = pd.DataFrame(
        {
            "day": day.isoformat(),
            "onset": stamps(row_ns[onsets]) if len(onsets) else pd.DatetimeIndex([], tz="UTC"),
            "row": onsets,
            "feature_direction": direction,
            "drift_direction": np.asarray(drift_sign) * np.asarray(multiplier, dtype=float),
        }
    )
    return book, trades, latent, found, float(log_mid[-1])


def _spells(
    rng: np.random.Generator, n: int, step: float, share: float, mean_s: float
) -> np.ndarray:
    """1 inside spells covering about ``share`` of the rows, 0 outside them."""
    out = np.zeros(n)
    if share <= 0 or mean_s <= 0:
        return out
    calm_s = mean_s * (1.0 - share) / share
    count = int(n * step / (mean_s + calm_s) * 2) + 4
    lengths = np.empty(2 * count)
    lengths[0::2] = rng.exponential(calm_s, count)
    lengths[1::2] = rng.exponential(mean_s, count)
    edges = np.minimum(np.rint(np.cumsum(lengths) / step).astype(np.int64), n)
    for begin, end in zip(edges[0::2], edges[1::2], strict=True):
        out[begin:end] = 1.0
    return out


def _lots(size: np.ndarray, lot: float) -> np.ndarray:
    return np.round(np.maximum(np.rint(size / lot), 1.0) * lot, 10)


def _prints(
    spec: BookDaysSpec,
    rng: np.random.Generator,
    row_ns: np.ndarray,
    signal: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    first_id: int,
) -> pd.DataFrame:
    """Prints in the interval ending at each row, priced at the previous row's touch.

    A print stamped in ``(t_{j-1}, t_j]`` belongs to row ``j`` under the backward
    rule, so the flow a row shows is the flow of the interval it closes; it hits
    the book that stood at ``t_{j-1}``. Its aggressor leans to row ``j``'s
    signal when the feature is flow.
    """
    n = len(row_ns)
    rate = np.full(n, spec.trades_per_s * spec.step_s)
    if spec.feature == "flow":
        rate = np.where(signal != 0, rate * spec.flow_burst, rate)
    counts = rng.poisson(rate)
    counts[0] = 0
    total = int(counts.sum())
    rows = np.repeat(np.arange(n), counts)
    p_buy = np.full(total, 0.5)
    if spec.feature == "flow":
        lean = signal[rows].astype(np.float64)
        p_buy = np.where(lean > 0, spec.flow_tilt, np.where(lean < 0, 1.0 - spec.flow_tilt, 0.5))
    buy = rng.random(total) < p_buy
    price_ticks = np.where(buy, best_ask[rows - 1], best_bid[rows - 1])
    offsets_ms = rng.integers(1, spec.step_ms + 1, total)
    order = np.lexsort((offsets_ms, rows))
    rows, buy, price_ticks = rows[order], buy[order], price_ticks[order]
    offsets_ms = offsets_ms[order]
    step_ns = spec.step_ms * 1_000_000
    ts = row_ns[rows] - step_ns + offsets_ms * 1_000_000
    quantity = _lots(rng.lognormal(np.log(0.05), 0.9, total), spec.lot)
    return pd.DataFrame(
        {
            "timestamp": stamps(ts),
            "symbol": strings(spec.symbol, total),
            "price": np.round(price_ticks * spec.tick, 10),
            "quantity": quantity,
            "is_buyer_maker": ~buy,
            "trade_id": np.arange(first_id, first_id + total, dtype=np.int64),
            "source": strings(SYNTHETIC_SOURCE, total),
        }
    )


def drift_rows(spec: BookDaysSpec) -> int:
    """Rows of the planted horizon: the label horizon a pipeline should use."""
    return round(spec.horizon_s / spec.step_s)


def hold_ns(spec: BookDaysSpec) -> int:
    return int(spec.horizon_s * NS_PER_S)
