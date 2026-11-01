"""The three slots' rules as quoters of the market-making simulator, for the calibration.

The simulator (``market_making/simulator.simulate_day``) asks its quoter at
every book snapshot, prices in ticks, and keeps a resting order whose price is
unchanged (its place in the queue with it). Each quoter here reproduces its
slot's rule on the book: the touch is the book's best bid and ask, a quote is
held off the other side by a tick (post-only), and the position is the one the
simulator reports at the snapshot, so the skew reads the position known then.

* :class:`BookQuoter` (MK-Q, MK-SQ): both sides at the registered requote
  instants, skewed, the signal's side doubled for MK-SQ, each side cut by the
  limit; between instants an order is not replenished (a fill seen in the
  position shrinks that side's order until the next instant).
* :class:`BookSignalMaker` (MK-S): one clip at its own side's touch at a
  decision taken flat and not busy; cancelled at the earliest of the hold, the
  next opposite decision and 23:59; when filled, the exit at the other side's
  touch for the position from the earliest of the first fill (seen at the
  snapshot after it) + hold, the opposite decision and 23:59, requoted every
  ``requote_s``. The simulator flattens at 23:59.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trading_research.market_making.quoters import NO_QUOTE, MarketView, Quote, Quotes
from trading_research.spot_maker import quoter as qt
from trading_research.spot_maker.executor import MakerSettings
from trading_research.spot_maker.registration import REMAINDER_BTC
from trading_research.spot_maker.signal_maker import next_opposite

NO_QUOTES = Quotes(NO_QUOTE, NO_QUOTE)


def _price(view: MarketView, side: int, position: float, settings: MakerSettings) -> int:
    """A quote of ``side`` at its own touch, skewed by the position, post-only, in ticks."""
    tick, shift = view.tick, 1.0 - settings.skew_bp * 1e-4 * position / settings.clip
    if side > 0:
        return math.floor(
            min(view.best_bid * tick * shift, (view.best_ask - 1) * tick) / tick + 1e-6
        )
    return math.ceil(max(view.best_ask * tick * shift, (view.best_bid + 1) * tick) / tick - 1e-6)


@dataclass
class BookQuoter:
    """MK-Q or MK-SQ in the simulator's interface."""

    settings: MakerSettings
    instants: np.ndarray
    signal_ns: np.ndarray
    signal_side: np.ndarray
    name: str = "MK-Q"
    uses_future: bool = False
    _k: int = field(default=-1, init=False)
    _bid: tuple[int | None, float] = field(default=(None, 0.0), init=False)
    _ask: tuple[int | None, float] = field(default=(None, 0.0), init=False)
    _position: float = field(default=0.0, init=False)

    def _requote(self, view: MarketView, position: float, instant: int) -> None:
        s = self.settings
        sigma = int(qt.signal_sides(np.array([instant]), self.signal_ns, self.signal_side,
                                    s.hold_ns)[0])  # fmt: skip
        want_bid, want_ask = s.clip * (2 if sigma > 0 else 1), s.clip * (2 if sigma < 0 else 1)
        self._bid = (_price(view, 1, position, s), min(want_bid, s.limit - position))
        self._ask = (_price(view, -1, position, s), min(want_ask, s.limit + position))
        self._position = position

    def quotes(self, view: MarketView, position: float) -> Quotes:
        k = int(np.searchsorted(self.instants, view.ts, side="right")) - 1
        if k < 0:
            return NO_QUOTES
        if k != self._k:
            self._k = k
            self._requote(view, position, int(self.instants[k]))
        moved = position - self._position
        self._position = position
        if moved > 0:
            self._bid = (self._bid[0], self._bid[1] - moved)
        elif moved < 0:
            self._ask = (self._ask[0], self._ask[1] + moved)
        bid, ask = (Quote(p, q) if p is not None and q > REMAINDER_BTC else NO_QUOTE
                    for p, q in (self._bid, self._ask))  # fmt: skip
        return Quotes(bid, ask)


@dataclass
class BookSignalMaker:
    """MK-S in the simulator's interface: idle, entry, held, exit."""

    settings: MakerSettings
    decision_ns: np.ndarray
    decision_side: np.ndarray
    first_ns: int
    last_ns: int
    flatten_ns: int
    name: str = "MK-S"
    uses_future: bool = False
    state: str = field(default="idle", init=False)
    _k: int = field(default=0, init=False)
    _side: int = field(default=0, init=False)
    _price: int | None = field(default=None, init=False)
    _cancel: int = field(default=0, init=False)
    _flip: float = field(default=math.inf, init=False)
    _fill: int = field(default=-1, init=False)
    _exit: float = field(default=math.inf, init=False)
    _requote: float = field(default=math.inf, init=False)
    _busy: float = field(default=-math.inf, init=False)

    def __post_init__(self) -> None:
        self._opposite = next_opposite(np.asarray(self.decision_side, dtype=np.int64))

    def _advance(self, t: int, position: float) -> None:
        flat = abs(position) <= REMAINDER_BTC
        if self.state == "entry":
            if not flat and self._fill < 0:
                self._fill = t
            if t >= self._cancel:
                if flat:
                    self.state, self._busy = "idle", self._cancel
                else:
                    self.state = "held"
                    hold = self._fill + self.settings.hold_ns
                    self._exit = self._cancel if self._cancel == self._flip else min(
                        hold, self._flip, self.flatten_ns)  # fmt: skip
        if self.state == "held" and t >= self._exit:
            self.state, self._requote = "exit", self._exit
        if self.state == "exit" and flat:
            self.state, self._busy = "idle", t

    def _take(self, view: MarketView) -> None:
        while self._k < len(self.decision_ns) and self.decision_ns[self._k] <= view.ts:
            k, self._k = self._k, self._k + 1
            t, side = int(self.decision_ns[k]), int(self.decision_side[k])
            if self.state != "idle" or t < self._busy or not self.first_ns <= t <= self.last_ns:
                continue
            j = int(self._opposite[k])
            self._flip = float(self.decision_ns[j]) if j >= 0 else math.inf
            self._cancel = int(min(t + self.settings.hold_ns, self._flip, self.flatten_ns))
            self.state, self._side, self._fill = "entry", side, -1
            self._price = _price(view, side, 0.0, self.settings)

    def quotes(self, view: MarketView, position: float) -> Quotes:
        self._advance(view.ts, position)
        self._take(view)
        if self.state == "entry":
            size = self.settings.clip - abs(position)
            quote = Quote(self._price, size) if size > REMAINDER_BTC else NO_QUOTE
            return Quotes(quote, NO_QUOTE) if self._side > 0 else Quotes(NO_QUOTE, quote)
        if self.state == "exit":
            if view.ts >= self._requote:
                self._price = _price(view, -self._side, position, self.settings)
                self._requote += self.settings.requote_ns
                while self._requote <= view.ts:
                    self._requote += self.settings.requote_ns
            quote = Quote(self._price, abs(position))
            return Quotes(quote, NO_QUOTE) if self._side < 0 else Quotes(NO_QUOTE, quote)
        return NO_QUOTES
