"""The signal-entry quoter: one clip posted at the touch on a signal, kept at its first price.

A decision at a row stamped ``t`` reaches the simulator as a tape row stamped
``t`` (:func:`entry_tapes`). Tapes are read strictly before each snapshot, so
the decision is first seen at the first snapshot after ``t``, and the order the
quoter sends then is live after the order latency: never earlier than ``t``
plus that latency. A decision first seen more than ``hold_s`` after ``t`` is
not posted (``expired_unseen``).

One attempt, from the post to the return to a flat book:

* **Entry.** One clip on the signal's side at that snapshot's touch, kept at
  that price and never re-pegged. After a partial fill the rest stays at the
  same price. It is no longer quoted, and so cancelled, at the first snapshot
  ``hold_s`` after the post, and as soon as it is done for any other reason
  (filled, rejected on arrival, pulled by a feed pause, or cancelled by the
  simulator at the first event after a regime flag when the configuration's
  ``cancel_entries_on_flag`` is set): an entry is never posted twice.
* **Exit.** ``hold_s`` after the position opened (the time of the first fill,
  which the simulator reports), a reduce-only taker order for the position
  held, which the simulator prices by the shared book walk. A fill that lands
  after it (the entry's cancel was still in flight) is crossed at the next
  snapshot. With ``passive_exit`` the exit is first posted at the touch on the
  other side, kept at its first price, and whatever is left ``hold_s`` later is
  crossed once that order is done.
* **One position at a time.** A new decision is taken only when the position
  is flat and no own order is out; **an order whose cancel is in flight counts
  as resting** until the cancel lands or the order fills. A decision that
  arrives meanwhile is skipped (``skipped_busy``), not queued.

The quoter is told its own orders and the opening time through the simulator's
``observe`` hook, and keeps a log of attempts for :meth:`attempts`. Stateful:
a fresh instance per simulated day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_research.market_making.orders import OwnOrder
from trading_research.market_making.quoters import (
    NO_QUOTE,
    NO_QUOTES,
    MarketView,
    Quote,
    Quotes,
)
from trading_research.market_making.signals import SignalTape

#: The decision's stamp in epoch seconds, and its side (+1 buy, -1 sell).
ENTRY_SIGNAL = "entry_decision_s"
ENTRY_SIDE_SIGNAL = "entry_side"


def entry_tapes(decision_ns: np.ndarray, side: np.ndarray) -> dict[str, SignalTape]:
    """The two tapes the quoter reads, each row stamped at its decision row's stamp."""
    ts = np.asarray(decision_ns, dtype=np.int64)
    sides = np.asarray(side, dtype=np.float64)
    if len(ts) != len(sides):
        raise ValueError("one side per decision")
    if not np.isin(sides, (-1.0, 1.0)).all():
        raise ValueError("a side is +1 (buy) or -1 (sell)")
    return {
        ENTRY_SIGNAL: SignalTape(ENTRY_SIGNAL, ts, ts / 1e9),
        ENTRY_SIDE_SIGNAL: SignalTape(ENTRY_SIDE_SIGNAL, ts.copy(), sides),
    }


@dataclass
class _Attempt:
    decision_s: float
    side: int
    price: int
    posted_s: float
    entry_open: bool = True
    exit_at: float = math.nan
    exit_price: int | None = None
    cross_at: float = math.nan


@dataclass
class SignalEntryQuoter:
    """One clip at the touch per accepted decision; a taker exit ``hold_s`` after the fill."""

    hold_s: float
    passive_exit: bool = False
    name: str = "signal_entry"
    uses_future: bool = False
    counts: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    #: One entry per attempt: (decision s, side, post ns, price, size).
    log: list[tuple[float, int, int, float, float]] = field(
        default_factory=list, repr=False, compare=False
    )
    _own: tuple[OwnOrder, ...] = field(default=(), repr=False, compare=False)
    _opened_ns: int = field(default=-1, repr=False, compare=False)
    _acted: float = field(default=math.nan, repr=False, compare=False)
    _attempt: _Attempt | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.hold_s > 0:
            raise ValueError(f"the hold must be positive, got {self.hold_s}")

    def observe(self, own: tuple[OwnOrder, ...], opened_ns: int) -> None:
        self._own = own
        self._opened_ns = opened_ns

    def _count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def quotes(self, view: MarketView, position: float) -> Quotes:
        now = view.ts / 1e9
        decision = view.signals.get(ENTRY_SIGNAL, math.nan)
        fresh = decision == decision and decision != self._acted
        if fresh:
            self._acted = decision
        attempt = self._attempt
        if attempt is not None and self._update(attempt, position, now):
            self._attempt = attempt = None
        if attempt is not None:
            if fresh:
                self._count("skipped_busy")
            return self._quotes(attempt, view, position, now)
        if not fresh:
            return NO_QUOTES
        if position != 0.0 or self._own:
            self._count("skipped_busy")
            return NO_QUOTES
        if now - decision > self.hold_s:
            self._count("expired_unseen")
            return NO_QUOTES
        side = 1 if view.signals.get(ENTRY_SIDE_SIGNAL, math.nan) > 0 else -1
        price = view.best_bid if side > 0 else view.best_ask
        attempt = _Attempt(decision, side, int(price), now)
        self._attempt = attempt
        self._count("attempts")
        self.log.append((decision, side, view.ts, price * view.tick, view.clip))
        return self._quotes(attempt, view, position, now)

    def _update(self, a: _Attempt, position: float, now: float) -> bool:
        """Close the entry when its time is up or it is done; True when the attempt is over."""
        if a.entry_open:
            alive = any(
                o.side == a.side and o.price == a.price and not o.cancelling for o in self._own
            )
            if not alive:
                a.entry_open = False
            elif now >= a.posted_s + self.hold_s:
                a.entry_open = False
                self._count("entries_expired")
        if position != 0.0 and a.exit_at != a.exit_at:
            a.exit_at = self._opened_ns / 1e9 + self.hold_s
            self._count("entries_filled")
        if a.entry_open or position != 0.0 or self._own:
            return False
        self._count("round_trips" if a.exit_at == a.exit_at else "entries_unfilled")
        return True

    def _quotes(self, a: _Attempt, view: MarketView, position: float, now: float) -> Quotes:
        bid = ask = NO_QUOTE
        if a.entry_open:
            entry = Quote(a.price, view.clip - abs(position))
            bid, ask = (entry, NO_QUOTE) if a.side > 0 else (NO_QUOTE, entry)
        cross = 0.0
        if position != 0.0 and now >= a.exit_at:
            if not self.passive_exit:
                cross = -position
            else:
                if a.exit_price is None:
                    a.exit_price = view.best_ask if position > 0 else view.best_bid
                    a.cross_at = now + self.hold_s
                    self._count("passive_exits_posted")
                exit_side = -1 if position > 0 else 1
                if now < a.cross_at:
                    out = Quote(a.exit_price, abs(position))
                    bid, ask = (out, ask) if exit_side > 0 else (bid, out)
                elif not any(o.side == exit_side for o in self._own):
                    cross = -position
        if cross:
            self._count("exit_crosses")
        return Quotes(bid, ask, cross)

    def attempts(self, fills: pd.DataFrame) -> pd.DataFrame:
        """One row per attempt, from the day's fills; funding is in the day's totals only.

        An attempt owns every fill from its post to the next attempt's post (or
        the day's end, the day-end flatten included): the quoter is flat between
        attempts. ``filled`` is True for an entry with any fill; ``net`` is
        the cash of its fills, fees included; ``notional`` is the clip at the
        posted price.
        """
        columns = ["decision_s", "side", "posted_ns", "price", "size", "filled"]
        columns += ["filled_size", "gross", "fees", "net", "notional"]
        if not self.log:
            return pd.DataFrame(columns=columns)
        ts = fills["ts"].to_numpy(dtype=np.int64)
        flow = (
            -fills["side"].to_numpy(dtype=np.float64)
            * fills["price"].to_numpy(dtype=np.float64)
            * fills["size"].to_numpy(dtype=np.float64)
        )
        fee = fills["fee"].to_numpy(dtype=np.float64)
        maker = fills["maker"].to_numpy(dtype=bool)
        size = fills["size"].to_numpy(dtype=np.float64)
        fill_side = fills["side"].to_numpy(dtype=np.int64)
        starts = [entry[2] for entry in self.log]
        rows = []
        for k, (decision, side, posted, price, clip) in enumerate(self.log):
            end = starts[k + 1] if k + 1 < len(starts) else np.iinfo(np.int64).max
            mine = (ts >= posted) & (ts < end)
            entry_size = float(size[mine & maker & (fill_side == side)].sum())
            gross = float(flow[mine].sum())
            fees = float(fee[mine].sum())
            rows.append(
                (
                    decision,
                    side,
                    posted,
                    price,
                    clip,
                    entry_size > 0,
                    entry_size,
                    gross,
                    fees,
                    gross - fees,
                    price * clip,
                )
            )
        return pd.DataFrame(rows, columns=columns)

    def summarise(self, fills: pd.DataFrame) -> dict[str, float]:
        """Totals over the day's attempts, for a results row."""
        table = self.attempts(fills)
        return {
            "attempts_logged": float(len(table)),
            "entries_with_fill": float(table["filled"].sum()) if len(table) else 0.0,
            "attempt_net_sum": float(table["net"].sum()) if len(table) else 0.0,
        }
