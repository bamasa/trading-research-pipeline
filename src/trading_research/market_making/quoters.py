"""What a quoting rule may see, what it returns, and the baseline rule.

A quoter is called at book events only, with a :class:`MarketView` of that
snapshot and the position, and returns one :class:`Quote` per side. It sees no
fill, no markout and nothing after the snapshot; the arrays it is handed are
read-only views. Whether a quote becomes an order — the clip, the inventory
limits, the depth of the visible book, post-only on arrival — is the
simulator's decision, not the quoter's, so every quoter is held to the same
rules.

The rules the pre-registration compares:

* :class:`TouchQuoter` (S0) — bid at the best bid, ask at the best ask;
* :class:`SkewQuoter` (S1) — an Avellaneda-Stoikov reservation price skewed by
  inventory and a half-spread scaled by volatility, gated so that a quote must
  clear the maker fee plus a minimum edge; never better than the touch;
* :class:`InsideQuoter` (S2, H1) — S1, plus one tick inside the touch when the
  spread is wide enough and the improved price still clears the gate; with
  ``inside=False`` it is H1's twin, S1 at S2's parameters;
* :class:`RegimeGuard` (S3, H3) — S1 or S2, pulling its quotes or doubling its
  half-spread for a window after a regime flag;
* :class:`ReversionLean` (S4, H2.1) — S1 with the reservation price shifted
  against the index's ten-minute move while a trigger is fresh;
* :class:`SignalExecutor` (X1, H2.2) — one-sided passive execution of the
  frozen reversion rule's triggers, with a taker exit on timeout.

Fair value is the mid; economic quantities are in basis points and prices in
integer ticks. Signals arrive in :attr:`MarketView.signals` by name, as of
strictly before the snapshot (see :mod:`.signals` and :mod:`.flags`).

The value types are named tuples rather than dataclasses because one of each is
built at every snapshot — several hundred thousand a day — and a tuple is the
cheapest immutable record Python has.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd


class MarketView(NamedTuple):
    """Everything a quoter may read at a decision. Prices in ticks."""

    #: The snapshot's timestamp, ns.
    ts: int
    tick: float
    #: The snapshot's levels, read-only, best first.
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    best_bid: int
    best_ask: int
    #: Mid of the snapshot, in price units.
    mid: float
    #: EWMA of one-second mid returns, half-life 60 s, scaled to one minute, bp.
    vol_bp_1m: float
    #: Median touch size over the trailing hour, base units.
    touch_median_1h: float
    #: The simulator's clip for this decision, base units, already on the lot.
    clip: float
    #: The soft inventory limit for this decision, base units.
    soft_limit: float
    #: External signals, as of strictly before ``ts``.
    signals: Mapping[str, float]


class Quote(NamedTuple):
    """One side's wish. ``price`` None means do not quote this side."""

    price: int | None
    size: float


class Quotes(NamedTuple):
    bid: Quote
    ask: Quote
    #: A reduce-only taker order to send with this decision: signed size, +
    #: buys and - sells. Zero for every market maker; X1 uses it to cross for
    #: an exit that timed out. The simulator never lets it grow the position.
    cross: float = 0.0


NO_QUOTE = Quote(None, 0.0)


class Quoter(Protocol):
    """A quoting rule."""

    #: Short label carried into results.
    name: str
    #: True only for rules that read the future (the ladder's forecast rungs).
    uses_future: bool

    def quotes(self, view: MarketView, position: float) -> Quotes: ...


@dataclass(frozen=True)
class TouchQuoter:
    """S0: join both touches with the clip, and nothing else.

    The baseline every other rule is measured against, and rung 0 of the
    advantage ladder. Its decisions read neither the fee nor the volatility, so
    its fee break-even is exact without a re-run.
    """

    name: str = "S0"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:  # noqa: ARG002
        return Quotes(Quote(view.best_bid, view.clip), Quote(view.best_ask, view.clip))


# ---------------------------------------------------------------------------
# S1-S4 and X1
# ---------------------------------------------------------------------------

#: Below this, a floating-point product that should be a whole number of ticks
#: is treated as one.
_TICK_EPS = 1e-9

NO_QUOTES = Quotes(NO_QUOTE, NO_QUOTE)


def _floor_tick(price: float, tick: float) -> int:
    return math.floor(price / tick + _TICK_EPS)


def _ceil_tick(price: float, tick: float) -> int:
    return math.ceil(price / tick - _TICK_EPS)


@dataclass(frozen=True)
class SkewQuoter:
    """S1: inventory-skewed reservation price, volatility-scaled half-spread.

    With ``q`` the position, ``Q`` the soft limit and ``sigma`` the one-minute
    volatility in bp::

        r     = mid * (1 - skew_bp * 1e-4 * clamp((sigma / sigma_ref)**2, 0.25, 4) * q / Q)
        delta = maker_bp + min_edge_bp + k * sigma                       (bp)
        bid   = min(floor_tick(r * (1 - delta * 1e-4)), best bid)
        ask   = max(ceil_tick(r * (1 + delta * 1e-4)), best ask)

    Long inventory lowers both quotes, short raises them, and more so when the
    market is more volatile than usual. A quote never improves the touch, and
    the gate keeps it off a touch too close to fair value to clear the fee. The
    simulator then applies the clip, the soft and hard limits and the visible
    depth, as for every quoter.
    """

    skew_bp: float
    k: float
    min_edge_bp: float
    sigma_ref: float
    maker_bp: float = 2.0
    name: str = "S1"
    uses_future: bool = False

    def __post_init__(self) -> None:
        if not self.sigma_ref > 0:
            raise ValueError(f"sigma_ref must be positive, got {self.sigma_ref}")
        if self.k < 0 or self.min_edge_bp < 0 or self.skew_bp < 0:
            raise ValueError("skew, k and the minimum edge cannot be negative")

    def reservation(self, view: MarketView, position: float) -> float:
        """The reservation price r, in price units."""
        ratio = (view.vol_bp_1m / self.sigma_ref) ** 2
        ratio = min(max(ratio, 0.25), 4.0)
        return view.mid * (1.0 - self.skew_bp * 1e-4 * ratio * position / view.soft_limit)

    def half_spread_bp(self, view: MarketView) -> float:
        """The gate delta, in bp: fee, minimum edge, and k times volatility."""
        return self.maker_bp + self.min_edge_bp + self.k * view.vol_bp_1m

    def usable(self, view: MarketView) -> bool:
        return view.soft_limit > 0 and view.vol_bp_1m == view.vol_bp_1m and view.mid > 0

    def prices(self, view: MarketView, r: float, delta_bp: float) -> tuple[int, int]:
        """Bid and ask in ticks from a reservation price and a half-spread."""
        tick = view.tick
        bid = min(_floor_tick(r * (1.0 - delta_bp * 1e-4), tick), view.best_bid)
        ask = max(_ceil_tick(r * (1.0 + delta_bp * 1e-4), tick), view.best_ask)
        return bid, ask

    def quote_prices(
        self, view: MarketView, position: float, *, delta_scale: float = 1.0
    ) -> tuple[int, int] | None:
        """The rule's bid and ask, or None when it cannot quote at all."""
        if not self.usable(view):
            return None
        r = self.reservation(view, position)
        return self.prices(view, r, self.half_spread_bp(view) * delta_scale)

    def quotes(self, view: MarketView, position: float) -> Quotes:
        prices = self.quote_prices(view, position)
        if prices is None:
            return NO_QUOTES
        return Quotes(Quote(prices[0], view.clip), Quote(prices[1], view.clip))


@dataclass(frozen=True)
class InsideQuoter:
    """S2: S1, plus one tick inside the touch when the spread allows it.

    When the spread is at least ``m_ticks``, a side improves to best bid + 1
    tick (best ask - 1 tick) if that price still clears S1's gate,
    ``r - bid >= delta`` (``ask - r >= delta``), and is post-only safe.
    When the spread is exactly two ticks the two improved prices would meet,
    so only one side may improve: the side that reduces ``|q|``, and the bid
    when flat. A side that may not improve keeps S1's price.

    ``inside=False`` is H1's twin: the same parameters with the inside rule
    off, which is S1 itself.
    """

    skew_bp: float
    k: float
    min_edge_bp: float
    m_ticks: int
    sigma_ref: float
    maker_bp: float = 2.0
    inside: bool = True
    name: str = "S2"
    uses_future: bool = False

    def __post_init__(self) -> None:
        if self.m_ticks < 2:
            raise ValueError("improving by one tick needs a spread of at least two ticks")

    @property
    def base(self) -> SkewQuoter:
        return SkewQuoter(self.skew_bp, self.k, self.min_edge_bp, self.sigma_ref, self.maker_bp)

    def quote_prices(
        self, view: MarketView, position: float, *, delta_scale: float = 1.0
    ) -> tuple[int, int] | None:
        base = self.base
        if not base.usable(view):
            return None
        r = base.reservation(view, position)
        delta_bp = base.half_spread_bp(view) * delta_scale
        bid, ask = base.prices(view, r, delta_bp)
        if not self.inside:
            return bid, ask
        best_bid, best_ask = view.best_bid, view.best_ask
        spread = best_ask - best_bid
        if spread < self.m_ticks:
            return bid, ask
        tick = view.tick
        bid_ok = (best_bid + 1) * tick <= r * (1.0 - delta_bp * 1e-4) * (1.0 + _TICK_EPS)
        ask_ok = (best_ask - 1) * tick >= r * (1.0 + delta_bp * 1e-4) * (1.0 - _TICK_EPS)
        if spread == 2 and bid_ok and ask_ok:
            # The two improved prices would meet: one side only.
            if position > 0:
                bid_ok = False
            else:
                ask_ok = False
        if bid_ok:
            bid = best_bid + 1
        if ask_ok:
            ask = best_ask - 1
        return bid, ask

    def quotes(self, view: MarketView, position: float) -> Quotes:
        prices = self.quote_prices(view, position)
        if prices is None:
            return NO_QUOTES
        return Quotes(Quote(prices[0], view.clip), Quote(prices[1], view.clip))


#: The name of the tape :class:`RegimeGuard` reads: the effective time, in
#: epoch seconds, of the latest regime flag.
FLAG_SIGNAL = "flag_s"


@dataclass(frozen=True)
class RegimeGuard:
    """S3: the guarded quoter, pulled or widened for a window after each flag.

    A flag effective at ``e`` opens a window ``(e, e + window_min]``; windows
    that overlap merge, because the latest flag always extends it. Inside it
    the action is ``pull`` (no quotes) or ``widen`` (the half-spread doubled).
    The flags arrive as the ``flag_s`` signal (:mod:`.flags`), as of strictly
    before the snapshot.
    """

    base: SkewQuoter | InsideQuoter
    action: str
    window_min: float
    widen_factor: float = 2.0
    name: str = "S3"
    uses_future: bool = False

    def __post_init__(self) -> None:
        if self.action not in ("pull", "widen"):
            raise ValueError(f"action must be 'pull' or 'widen', got {self.action!r}")
        if not self.window_min > 0:
            raise ValueError("the guard window must be positive")

    def guarded(self, view: MarketView) -> bool:
        flag = view.signals.get(FLAG_SIGNAL, math.nan)
        if flag != flag:
            return False
        return view.ts / 1e9 - flag <= self.window_min * 60.0

    def quotes(self, view: MarketView, position: float) -> Quotes:
        scale = 1.0
        if self.guarded(view):
            if self.action == "pull":
                return NO_QUOTES
            scale = self.widen_factor
        prices = self.base.quote_prices(view, position, delta_scale=scale)
        if prices is None:
            return NO_QUOTES
        return Quotes(Quote(prices[0], view.clip), Quote(prices[1], view.clip))


#: Tapes :class:`ReversionLean` reads: the index's ten-minute return excluding
#: the instrument (bp), and the time (epoch seconds) of the latest label at
#: which its size cleared theta.
INDEX_SIGNAL = "index_return_bp"
LEAN_TRIGGER_SIGNAL = "lean_trigger_s"


@dataclass(frozen=True)
class ReversionLean:
    """S4: S1 with the reservation price leaning against the index's move.

    A trigger fires when ``|s| >= theta`` (``lean_trigger_s``); for the next
    ``window_s`` the reservation price becomes
    ``r * (1 + lam * beta * s * 1e-4)`` with ``s`` read live. With ``beta``
    negative, as reversion implies, a rise in the index lowers both quotes:
    the quoter leans to sell into it. With ``one_sided``, the side that would
    follow the move — the bid after a rise, the ask after a fall — is not
    quoted inside the window. ``lam`` negative is the flipped placebo.
    """

    base: SkewQuoter
    beta: float
    lam: float
    one_sided: bool
    window_s: float = 600.0
    name: str = "S4"
    uses_future: bool = False

    def window_signal(self, view: MarketView) -> float | None:
        """The live signal when a trigger is fresh, else None."""
        trigger = view.signals.get(LEAN_TRIGGER_SIGNAL, math.nan)
        s = view.signals.get(INDEX_SIGNAL, math.nan)
        if trigger != trigger or s != s:
            return None
        if view.ts / 1e9 - trigger > self.window_s:
            return None
        return s

    def quotes(self, view: MarketView, position: float) -> Quotes:
        base = self.base
        if not base.usable(view):
            return NO_QUOTES
        r = base.reservation(view, position)
        s = self.window_signal(view)
        if s is not None:
            r *= 1.0 + self.lam * self.beta * s * 1e-4
        bid, ask = base.prices(view, r, base.half_spread_bp(view))
        bid_quote = Quote(bid, view.clip)
        ask_quote = Quote(ask, view.clip)
        if s is not None and self.one_sided:
            if s > 0:
                bid_quote = NO_QUOTE
            elif s < 0:
                ask_quote = NO_QUOTE
        return Quotes(bid_quote, ask_quote)


#: Tapes :class:`SignalExecutor` reads: the latest trigger's time (epoch
#: seconds) and its direction (+1 buy, -1 sell).
TRIGGER_SIGNAL = "x1_trigger_s"
DIRECTION_SIGNAL = "x1_direction"


@dataclass
class SignalExecutor:
    """X1: passive execution of the frozen reversion rule's triggers.

    At each new trigger the fading side is posted at the touch with the clip
    (a buy at the best bid after the index fell, a sell at the best ask after
    it rose), following the touch while it waits. The entry waits at most
    ``entry_timeout_s`` from the trigger; unfilled, the attempt is a miss.
    Once the position shows a fill, the remaining entry is pulled and the exit
    is posted at the opposite touch for the position held; whatever is still
    open ``exit_timeout_s`` after the fill is crossed with a reduce-only taker
    order. The fill is seen at the first decision after it, so the exit clock
    starts within one snapshot of the fill.

    Which triggers are attempted — not in the warm-up, not in the last 25
    minutes of a day — is decided when the trigger tape is built, so the taker
    twin can be scored on exactly the same ones. A trigger that arrives while an
    attempt is still open is skipped and counted.

    Stateful, unlike the market makers: a fresh instance per day.
    """

    entry_timeout_s: float = 600.0
    exit_timeout_s: float = 600.0
    name: str = "X1"
    uses_future: bool = False
    state: str = field(default="idle", repr=False, compare=False)
    side: int = field(default=0, repr=False, compare=False)
    acted: float = field(default=math.nan, repr=False, compare=False)
    deadline: float = field(default=math.inf, repr=False, compare=False)
    crossed: bool = field(default=False, repr=False, compare=False)
    counts: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    #: One entry per attempt: (trigger time s, side, first post ns, posted
    #: price in price units, posted size).
    log: list[tuple[float, int, int, float, float]] = field(
        default_factory=list, repr=False, compare=False
    )

    def _count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def attempts(self, fills: pd.DataFrame) -> pd.DataFrame:
        """One row per attempt with its result, from the day's fills.

        An attempt owns every fill from its first post to the next attempt's
        (or the day's end, the day-end flatten included): X1 is flat between
        attempts, so that is its round trip. ``net_bp`` is the cash it made,
        fees included, over the notional it posted; a miss is zero.
        """
        columns = ["trigger_s", "side", "posted_ns", "price", "size", "filled", "net", "net_bp"]
        if not self.log:
            return pd.DataFrame(columns=columns)
        ts = fills["ts"].to_numpy(dtype=np.int64)
        cash = -fills["side"].to_numpy(dtype=np.float64) * fills["price"].to_numpy(
            dtype=np.float64
        ) * fills["size"].to_numpy(dtype=np.float64) - fills["fee"].to_numpy(dtype=np.float64)
        starts = [entry[2] for entry in self.log]
        rows = []
        for k, (trigger, side, posted, price, size) in enumerate(self.log):
            end = starts[k + 1] if k + 1 < len(starts) else np.iinfo(np.int64).max
            mine = (ts >= posted) & (ts < end)
            net = float(cash[mine].sum())
            notional = price * size
            rows.append(
                (
                    trigger,
                    side,
                    posted,
                    price,
                    size,
                    bool(mine.any()),
                    net,
                    net / notional * 1e4 if notional > 0 else 0.0,
                )
            )
        return pd.DataFrame(rows, columns=columns)

    def summarise(self, fills: pd.DataFrame) -> dict[str, float]:
        """Totals over the day's attempts, for a results row."""
        table = self.attempts(fills)
        return {
            "attempts_logged": float(len(table)),
            "attempts_filled": float(table["filled"].sum()) if len(table) else 0.0,
            "attempt_net_bp_sum": float(table["net_bp"].sum()) if len(table) else 0.0,
            "attempt_net_sum": float(table["net"].sum()) if len(table) else 0.0,
        }

    def quotes(self, view: MarketView, position: float) -> Quotes:
        now = view.ts / 1e9
        trigger = view.signals.get(TRIGGER_SIGNAL, math.nan)
        fresh = trigger == trigger and trigger != self.acted
        if self.state == "idle" and position == 0.0:
            if fresh:
                self.acted = trigger
                if now - trigger > self.entry_timeout_s:
                    self._count("expired_unseen")
                else:
                    direction = view.signals.get(DIRECTION_SIGNAL, math.nan)
                    self.side = 1 if direction > 0 else -1
                    self.deadline = trigger + self.entry_timeout_s
                    self.state = "entering"
                    self._count("attempts")
                    price = (view.best_bid if self.side > 0 else view.best_ask) * view.tick
                    self.log.append((trigger, self.side, view.ts, price, view.clip))
        elif fresh:
            self.acted = trigger
            self._count("skipped_busy")

        if self.state == "entering":
            if position != 0.0:
                self.state = "exiting"
                self.deadline = now + self.exit_timeout_s
                self.crossed = False
                self._count("entries_filled")
            elif now > self.deadline:
                self.state = "idle"
                self._count("misses")
                return NO_QUOTES
            elif self.side > 0:
                return Quotes(Quote(view.best_bid, view.clip), NO_QUOTE)
            else:
                return Quotes(NO_QUOTE, Quote(view.best_ask, view.clip))

        if self.state == "exiting":
            if position == 0.0:
                self.state = "idle"
                self._count("exits_done")
                return NO_QUOTES
            if now >= self.deadline:
                if self.crossed:
                    return NO_QUOTES
                self.crossed = True
                self._count("exits_crossed")
                return Quotes(NO_QUOTE, NO_QUOTE, -position)
            size = abs(position)
            if position > 0:
                return Quotes(NO_QUOTE, Quote(view.best_ask, size))
            return Quotes(Quote(view.best_bid, size), NO_QUOTE)
        return NO_QUOTES
