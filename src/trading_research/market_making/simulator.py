"""The event loop: one instrument-day, one quoter, every rule applied in order.

The simulator walks the merged stream of :mod:`.events` once. Between events it
applies what latency has made due — orders arriving at the venue, cancels
landing, the timed stop and flatten — strictly before the event at hand, so an
action that becomes due at a print's exact timestamp loses the tie. Then:

* a **funding** settlement charges or pays the position held at that instant;
* a **print** is offered to own orders on the side it hits, best price first,
  and fills them only as :func:`.queue.on_print` allows;
* a **book snapshot** updates the mark, corrects every resting order's queue
  estimate (:func:`.queue.on_snapshot`), and, if quoting is allowed, asks the
  quoter for its quotes and reconciles them with the orders already out.

Reconciling keeps a resting order whose price is unchanged — priority is an
asset — and turns a price change into a cancel and a new order, which joins the
queue afresh. A size change alone never touches a resting order. New orders and
cancels take effect after their latency; until a cancel lands the order can
still fill.

What the simulator decides, not the quoter
------------------------------------------
The clip, the inventory limits, the depth of the visible book and post-only on
arrival are enforced here, identically for every quoter:

* clip = ``min(clip_notional / mid, clip_touch_share * trailing one-hour median
  touch)``, rounded down to the lot; below one lot the side is not quoted;
* soft limit = ``soft_limit_clips * clip``: a side whose fill would take the
  position past it is not quoted;
* hard limit = soft + one clip: a new order is sent only if the position plus
  every order still out on that side (cancels in flight included) stays within
  it, so the hard limit holds however long the latency;
* a fill that takes the position past the soft limit is flattened back to it
  by a taker order walking the last snapshot, with slippage on top;
* a quote beyond the tenth visible level is not placed, and an order at or
  through the opposite touch when it arrives is rejected.

The day
-------
Every day starts flat, quotes nothing before ``warmup_s`` or from
``stop_quoting_at``, and is flattened by a costed taker order at
``flatten_at``. If no snapshot arrives for ``suspend_after_pause_s``, every
order is pulled and nothing is quoted until the next snapshot. The accounting
identity is checked after every change to the account and once more at the
end. The full set of rules, each with the direction it biases a result and the
test that pins it, is in ``docs/market_making_simulator.md``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import date, time
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import BYBIT_BASE, FeeTier
from trading_research.market_making import analysis
from trading_research.market_making.accounting import Account, IdentityError
from trading_research.market_making.events import (
    BOOK,
    NS_PER_S,
    TRADE,
    DayEvents,
    DayUnavailable,
    load_day,
)
from trading_research.market_making.orders import (
    CANCELLED,
    FILLED,
    REJECTED,
    Order,
    OrderState,
)
from trading_research.market_making.queue import (
    CROSSED,
    FLATTEN,
    ArrivalGrowth,
    CancelAttribution,
    CrossedPolicy,
    QueuePriority,
    on_arrival,
    on_print,
    on_snapshot,
)
from trading_research.market_making.quoters import MarketView, Quote, Quoter
from trading_research.market_making.signals import SignalTape

_EMPTY_SIGNALS: Mapping[str, float] = MappingProxyType({})

#: Pending actions, ordered by effective time and then by the order they were
#: scheduled in.
_ACTIVATE, _CANCEL, _STOP, _FLATTEN = 0, 1, 2, 3

_NS_PER_MINUTE = 60 * NS_PER_S


class OracleRefused(RuntimeError):
    """A quoter that reads the future was run outside the advantage ladder."""


@dataclass(frozen=True, kw_only=True)
class SimConfig:
    """Every setting a simulated day depends on. Defaults are the pre-registered ones."""

    #: Notional cap on the clip, quote currency. Fixed per instrument from the
    #: development block; there is no default because there is no neutral value.
    clip_notional: float
    fees: FeeTier = BYBIT_BASE
    order_latency_ns: int = 10_000_000
    cancel_latency_ns: int = 10_000_000
    feed_latency_ns: int = 0
    cancel_attribution: CancelAttribution = CancelAttribution.PROPORTIONAL
    arrival_growth: ArrivalGrowth = ArrivalGrowth.PRO_RATA_TIME
    queue_priority: QueuePriority = QueuePriority.TAIL
    crossed_policy: CrossedPolicy = CrossedPolicy.TRADE_TAPE
    clip_touch_share: float = 0.10
    soft_limit_clips: float = 6.0
    flatten_slippage_bp: float = 0.5
    warmup_s: float = 600.0
    stop_quoting_at: time = time(23, 58)
    flatten_at: time = time(23, 59)
    suspend_after_pause_s: float = 5.0
    touch_window_s: int = 3600
    vol_half_life_s: float = 60.0
    markout_horizons_s: tuple[float, ...] = (1.0, 5.0, 30.0)
    decomposition_horizon_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.clip_notional > 0:
            raise ValueError(f"clip_notional must be positive, got {self.clip_notional}")
        for name in ("order_latency_ns", "cancel_latency_ns", "feed_latency_ns"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not self.clip_touch_share > 0:
            raise ValueError("clip_touch_share must be positive")
        if not self.soft_limit_clips >= 1:
            raise ValueError("the soft limit must allow at least one clip")
        if self.flatten_at < self.stop_quoting_at:
            raise ValueError("the flatten cannot come before quoting stops")
        if self.suspend_after_pause_s <= 0 or self.touch_window_s < 1:
            raise ValueError("the pause and the touch window must be positive")

    @property
    def hard_limit_clips(self) -> float:
        """Soft plus one clip: one fill from the soft limit cannot breach it."""
        return self.soft_limit_clips + 1.0

    def with_(self, **changes: Any) -> SimConfig:
        return replace(self, **changes)


@dataclass(frozen=True)
class DayResult:
    """One simulated day. Times are int64 nanoseconds since the epoch."""

    symbol: str
    day: date
    #: One row per fill: ts, oid, side, price, size, maker, fee, path,
    #: queue_wait_ns, position_after, mid_ref, soft_limit, hard_limit and a
    #: markout column per horizon. Flattens have oid 0 and maker False.
    fills: pd.DataFrame
    #: One row per own order: what was decided, when, and how it ended.
    orders: pd.DataFrame
    #: One row a minute: ts, equity, position, mid, realised, fees, funding.
    equity: pd.DataFrame
    #: One row per episode of a snapshot showing the opposite touch at or
    #: through a resting order: ts, oid, side, price, mid.
    crossed: pd.DataFrame
    counters: dict[str, float]
    #: spread, adverse, inventory, fees, funding, gross, making, net.
    decomposition: dict[str, float]
    #: True when the quoter read the future; such a day is an upper bound only.
    oracle: bool = False


@dataclass
class _Clock:
    """The latest decision's sizing, which fills and flattens are judged against."""

    clip: float = 0.0
    soft: float = 0.0
    hard: float = 0.0


@dataclass
class _Counters:
    values: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, amount: float = 1.0) -> None:
        self.values[name] = self.values.get(name, 0.0) + amount


def _per_second(events: DayEvents, config: SimConfig) -> tuple[list[float], list[float]]:
    """Volatility and trailing median touch at every whole second, causally.

    The value at second ``s`` uses only snapshots at or before that second, so
    a decision at time ``t`` reads the value of ``floor(t)``.
    """
    start = events.day_start_ns
    last = int(events.ts[-1])
    count = max(2, (last - start) // NS_PER_S + 2)
    boundaries = start + np.arange(count, dtype=np.int64) * NS_PER_S
    index = np.searchsorted(events.book_ts, boundaries, side="right") - 1
    known = index >= 0
    mid = np.full(count, np.nan)
    touch = np.full(count, np.nan)
    rows = index[known]
    mid[known] = (events.bid_px[rows, 0] + events.ask_px[rows, 0]) * (0.5 * events.spec.tick)
    touch[known] = 0.5 * (events.bid_sz[rows, 0] + events.ask_sz[rows, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(mid), prepend=np.nan)
    variance = (
        pd.Series(returns**2)
        .ewm(halflife=config.vol_half_life_s, adjust=False, ignore_na=True)
        .mean()
        .to_numpy()
    )
    vol = np.sqrt(variance * 60.0) * 1e4
    median = pd.Series(touch).rolling(config.touch_window_s, min_periods=1).median().to_numpy()
    return vol.tolist(), median.tolist()


class _Day:
    """The state of one simulated day. Built, run once, then read."""

    def __init__(
        self,
        events: DayEvents,
        quoter: Quoter,
        config: SimConfig,
        tapes: Mapping[str, SignalTape] | None,
    ) -> None:
        self.events = events
        self.quoter = quoter
        self.config = config
        spec = events.spec
        self.tick = spec.tick
        self.lot = spec.lot
        self.depth = events.depth
        self.start = events.day_start_ns

        self.bid0: list[int] = events.bid_px[:, 0].tolist()
        self.ask0: list[int] = events.ask_px[:, 0].tolist()
        self.bid_last: list[int] = events.bid_px[:, -1].tolist()
        self.ask_last: list[int] = events.ask_px[:, -1].tolist()
        self.book_ts: list[int] = events.book_ts.tolist()
        self.mid_array = analysis.book_mid(events)
        self.mids: list[float] = self.mid_array.tolist()
        self.trade_px: list[int] = events.trade_px.tolist()
        self.trade_sz: list[float] = events.trade_sz.tolist()
        self.trade_aggressor: list[int] = events.trade_aggressor.tolist()
        self.funding_rate: list[float] = events.funding_rate.tolist()
        self.vol, self.touch_median = _per_second(events, config)
        self.signals: dict[str, list[float]] = {
            name: tape.strictly_before(events.book_ts).tolist()
            for name, tape in (tapes or {}).items()
        }

        self.account = Account()
        self.clock = _Clock()
        self.counters = _Counters()
        self.heap: list[tuple[int, int, int, Order | None]] = []
        self.sequence = 0
        self.next_oid = 0
        self.orders: list[Order] = []
        self.working: dict[int, Order | None] = {1: None, -1: None}
        self.resting: dict[int, list[Order]] = {1: [], -1: []}
        self.open: dict[int, list[Order]] = {1: [], -1: []}
        self.fill_rows: list[tuple[Any, ...]] = []
        self.equity_rows: list[tuple[Any, ...]] = []
        self.crossed_rows: list[tuple[Any, ...]] = []

        self.now = self.start
        self.last_book = -1
        self.last_book_ts: int | None = None
        self.mid = math.nan
        self.stopped = False
        self.suspended = False
        self.stale_deadline: float = math.inf
        self.pause_ns = round(config.suspend_after_pause_s * NS_PER_S)
        self.quote_start = self.start + round(config.warmup_s * NS_PER_S)
        self.next_minute = self.start + _NS_PER_MINUTE
        self.max_abs_position = 0.0
        self.clip_over_touch_sum = 0.0
        self.clip_over_touch_n = 0

        stop = self.start + _time_ns(config.stop_quoting_at)
        flatten = self.start + _time_ns(config.flatten_at)
        self.stop_ns = stop
        self._schedule(stop, _STOP, None)
        self._schedule(flatten, _FLATTEN, None)

    # -- scheduling ---------------------------------------------------------

    def _schedule(self, effective: int, action: int, order: Order | None) -> None:
        self.sequence += 1
        heapq.heappush(self.heap, (effective, self.sequence, action, order))

    def _pop(self) -> None:
        effective, _, action, order = heapq.heappop(self.heap)
        self.now = max(self.now, effective)
        if effective >= self.next_minute:
            self._minutes(effective)
        if action == _ACTIVATE and order is not None:
            self._activate(order, effective)
        elif action == _CANCEL and order is not None:
            self._cancel_lands(order, effective)
        elif action == _STOP:
            self.stopped = True
            self._cancel_all(effective)
        elif action == _FLATTEN:
            self.stopped = True
            self._cancel_all(effective)
            position = self.account.position
            if position != 0.0:
                self._taker(-1 if position > 0 else 1, abs(position), effective, "day_end")

    # -- the loop -----------------------------------------------------------

    def run(self) -> None:
        events = self.events
        ts: list[int] = events.ts.tolist()
        kinds: list[int] = events.kind.tolist()
        refs: list[int] = events.ref.tolist()
        heap = self.heap
        aggressor = self.trade_aggressor
        resting = self.resting
        self.counters.add("events", len(ts))
        for k in range(len(ts)):
            t = ts[k]
            self.now = t
            if t > self.stale_deadline:
                self._suspend()
            while heap and heap[0][0] < t:
                self._pop()
            if t >= self.next_minute:
                self._minutes(t)
            kind = kinds[k]
            if kind == BOOK:
                self._book(refs[k], t)
            elif kind == TRADE:
                row = refs[k]
                side = -aggressor[row]
                if resting[side]:
                    self._trade(row, t, side)
            else:
                self._funding(refs[k])
        while heap:
            self._pop()
        self._close(self.now)

    # -- events -------------------------------------------------------------

    def _book(self, row: int, t: int) -> None:
        previous = self.last_book_ts
        self.last_book = row
        self.last_book_ts = t
        self.mid = self.mids[row]
        self.stale_deadline = t + self.pause_ns
        self.suspended = False
        config = self.config
        best_bid, best_ask = self.bid0[row], self.ask0[row]
        crossed_now: list[Order] = []
        for side in (1, -1):
            for order in self.resting[side]:
                level = self._level(side, order.price, row)
                if level is None and previous is not None:
                    self.counters.add("unobserved_level_s", (t - previous) / NS_PER_S)
                crossed = best_ask <= order.price if side == 1 else best_bid >= order.price
                if crossed and not order.crossed:
                    self.counters.add("crossed_episodes")
                    self.crossed_rows.append((t, order.oid, side, order.price, self.mid))
                    crossed_now.append(order)
                order.crossed = crossed
                on_snapshot(
                    order,
                    level,
                    rule=config.cancel_attribution,
                    growth=config.arrival_growth,
                    snapshot_ns=t,
                )
        if config.crossed_policy is CrossedPolicy.ASSUME_FILLED:
            for order in crossed_now:
                if order.can_fill and order.remaining > 0:
                    filled = order.remaining
                    order.remaining = 0.0
                    self._fill(order, filled, CROSSED, t)
        if self.quote_start <= t < self.stop_ns and not self.stopped:
            self._decide(row, t)

    def _trade(self, row: int, t: int, side: int) -> None:
        price = self.trade_px[row]
        available = self.trade_sz[row]
        aggressor = self.trade_aggressor[row]
        for order in list(self.resting[side]):
            filled, path = on_print(order, price, available, aggressor)
            if filled > 0.0:
                self._fill(order, filled, path, t)
                available -= filled
                if available <= 0.0:
                    break

    def _funding(self, row: int) -> None:
        rate = self.funding_rate[row]
        position = self.account.position
        if position != 0.0 and not self.mid == self.mid:
            raise RuntimeError("a funding settlement with a position and no mark")
        self.account.settle_funding(rate, self.mid)
        self.counters.add("funding_settlements")
        if position != 0.0:
            self.counters.add("funding_settlements_with_position")
            self.account.check_identity(self.mid)

    # -- decisions ----------------------------------------------------------

    def _level(self, side: int, price: int, row: int) -> float | None:
        """Visible size at ``price`` on ``side``: 0.0 if empty or inside the
        touch, None if beyond the visible levels."""
        events = self.events
        if side == 1:
            best = self.bid0[row]
            if price > best:
                return 0.0
            if price < self.bid_last[row]:
                return None
            prices = events.bid_px[row]
            sizes = events.bid_sz[row]
            guess = best - price
        else:
            best = self.ask0[row]
            if price < best:
                return 0.0
            if price > self.ask_last[row]:
                return None
            prices = events.ask_px[row]
            sizes = events.ask_sz[row]
            guess = price - best
        if guess < self.depth and prices[guess] == price:
            return float(sizes[guess])
        for level in range(self.depth):
            here = int(prices[level])
            if here == price:
                return float(sizes[level])
            if (here - price) * side < 0:
                return 0.0
        return 0.0

    def _decide(self, row: int, t: int) -> None:
        config = self.config
        events = self.events
        mid = self.mids[row]
        second = min(max((t - self.start) // NS_PER_S, 0), len(self.vol) - 1)
        touch = self.touch_median[second]
        cap = config.clip_notional / mid
        if touch == touch:
            cap = min(cap, config.clip_touch_share * touch)
        lot = self.lot
        clip = math.floor(cap / lot + 1e-9) * lot
        if touch > 0:
            self.clip_over_touch_sum += clip / touch
            self.clip_over_touch_n += 1
        clock = self.clock
        clock.clip = clip
        clock.soft = config.soft_limit_clips * clip
        clock.hard = clock.soft + clip
        signals = (
            {name: values[row] for name, values in self.signals.items()}
            if self.signals
            else _EMPTY_SIGNALS
        )
        view = MarketView(
            t,
            self.tick,
            events.bid_px[row],
            events.bid_sz[row],
            events.ask_px[row],
            events.ask_sz[row],
            self.bid0[row],
            self.ask0[row],
            mid,
            self.vol[second],
            touch,
            clip,
            clock.soft,
            signals,
        )
        wanted = self.quoter.quotes(view, self.account.position)
        self.counters.add("decisions")
        decided = t + config.feed_latency_ns
        self._reconcile(1, wanted.bid, row, decided)
        self._reconcile(-1, wanted.ask, row, decided)

    def _reconcile(self, side: int, quote: Quote, row: int, decided: int) -> None:
        current = self.working[side]
        price = quote.price
        position = self.account.position
        clock = self.clock
        lot = self.lot
        size = 0.0
        tolerance = 1e-9 * max(1.0, clock.hard)
        if price is not None:
            size = math.floor(min(quote.size, clock.clip) / lot + 1e-9) * lot
            if size < lot * (1 - 1e-9):
                price = None
                self.counters.add("not_quoted_below_lot")
            elif side * position + size > clock.soft + tolerance:
                price = None
                self.counters.add("not_quoted_soft_limit")
        if price is None:
            if current is not None:
                self._send_cancel(current, decided)
            return
        if current is not None:
            if current.price == price:
                return
            self._send_cancel(current, decided)
            self.counters.add("replaced")
        beyond = price < self.bid_last[row] if side == 1 else price > self.ask_last[row]
        if beyond:
            self.counters.add("not_placed_beyond_book")
            return
        outstanding = sum(order.remaining for order in self.open[side])
        if side * position + outstanding + size > clock.hard + tolerance:
            self.counters.add("not_placed_hard_limit")
            return
        self._place(side, int(price), size, decided, position)

    def _place(self, side: int, price: int, size: float, decided: int, position: float) -> None:
        self.next_oid += 1
        order = Order(
            oid=self.next_oid,
            side=side,
            price=price,
            size=size,
            remaining=size,
            decided_ns=decided,
            live_ns=decided + self.config.order_latency_ns,
            position_at_decision=position,
        )
        self.orders.append(order)
        self.open[side].append(order)
        self.working[side] = order
        self.counters.add("placed")
        self._schedule(order.live_ns, _ACTIVATE, order)

    def _send_cancel(self, order: Order, decided: int) -> None:
        if order.cancel_ns is not None or order.state is OrderState.DONE:
            return
        order.cancel_ns = decided + self.config.cancel_latency_ns
        if order.state is OrderState.LIVE:
            order.state = OrderState.PENDING_CANCEL
        if self.working[order.side] is order:
            self.working[order.side] = None
        self.counters.add("cancels_sent")
        self._schedule(order.cancel_ns, _CANCEL, order)

    def _cancel_all(self, decided: int) -> None:
        for side in (1, -1):
            for order in list(self.open[side]):
                self._send_cancel(order, decided)

    def _suspend(self) -> None:
        """No snapshot for too long: pull everything until the next one."""
        deadline = int(self.stale_deadline)
        self.stale_deadline = math.inf
        self.suspended = True
        self.counters.add("feed_pauses")
        self._cancel_all(deadline)

    # -- the venue ----------------------------------------------------------

    def _activate(self, order: Order, effective: int) -> None:
        if order.state is not OrderState.PENDING_NEW:
            return
        row = self.last_book
        side = order.side
        if row < 0 or (
            order.price >= self.ask0[row] if side == 1 else order.price <= self.bid0[row]
        ):
            self._finish(order, effective, REJECTED)
            self.counters.add("rejected")
            return
        inside = order.price > self.bid0[row] if side == 1 else order.price < self.ask0[row]
        on_arrival(
            order,
            self._level(side, order.price, row),
            inside_spread=inside,
            priority=self.config.queue_priority,
        )
        order.arrival_snapshot_ns = self.book_ts[row]
        order.state = OrderState.PENDING_CANCEL if order.cancel_ns is not None else OrderState.LIVE
        book = self.resting[side]
        book.append(order)
        if len(book) > 1:
            book.sort(key=lambda o: (-o.side * o.price, o.oid))

    def _cancel_lands(self, order: Order, effective: int) -> None:
        if order.state is OrderState.DONE:
            return
        if order.state is OrderState.PENDING_NEW:
            # The cancel overtook its order: the venue applies it on arrival.
            self._schedule(order.live_ns, _CANCEL, order)
            return
        self._finish(order, effective, CANCELLED)
        self.counters.add("cancelled")

    def _finish(self, order: Order, when: int, outcome: str) -> None:
        order.state = OrderState.DONE
        order.done_ns = when
        order.outcome = outcome
        side = order.side
        if order in self.resting[side]:
            self.resting[side].remove(order)
        if order in self.open[side]:
            self.open[side].remove(order)
        if self.working[side] is order:
            self.working[side] = None

    # -- money --------------------------------------------------------------

    def _fill(self, order: Order, size: float, path: str, t: int) -> None:
        account = self.account
        side = order.side
        before = account.position
        price = order.price * self.tick
        fee = account.fill(side, price, size, self.config.fees.maker_bp, maker=True)
        account.check_identity(self.mid)
        if order.remaining <= self.lot * 1e-6:
            order.remaining = 0.0
            self._finish(order, t, FILLED)
        position = account.position
        self.max_abs_position = max(self.max_abs_position, abs(position))
        self.fill_rows.append(
            (
                t,
                order.oid,
                side,
                price,
                size,
                True,
                fee,
                path,
                t - order.live_ns,
                position,
                self.mid,
                self.clock.soft,
                self.clock.hard,
            )
        )
        self.counters.add(f"fills_{path}")
        soft = self.clock.soft
        if (
            side * position > 0
            and abs(position) > abs(before)
            and abs(position) > soft * (1 + 1e-9)
        ):
            self._taker(-side, abs(position) - soft, t, "soft_limit")

    def _taker(self, side: int, size: float, t: int, reason: str) -> None:
        """Cross the spread for ``size``, walking the last snapshot."""
        row = self.last_book
        if row < 0:
            raise RuntimeError("a taker order with no book to walk")
        events = self.events
        prices = events.ask_px[row] if side == 1 else events.bid_px[row]
        sizes = events.ask_sz[row] if side == 1 else events.bid_sz[row]
        left = size
        cost = 0.0
        for level in range(self.depth):
            take = min(left, float(sizes[level]))
            cost += take * float(prices[level])
            left -= take
            if left <= 0.0:
                break
        if left > 0.0:
            cost += left * float(prices[self.depth - 1])
            self.counters.add("flatten_beyond_visible_book")
        slip = self.config.flatten_slippage_bp * 1e-4
        price = cost / size * self.tick * (1.0 + side * slip)
        account = self.account
        fee = account.fill(side, price, size, self.config.fees.taker_bp, maker=False)
        account.check_identity(self.mid)
        self.fill_rows.append(
            (
                t,
                0,
                side,
                price,
                size,
                False,
                fee,
                FLATTEN,
                0,
                account.position,
                self.mid,
                self.clock.soft,
                self.clock.hard,
            )
        )
        self.counters.add("flattens")
        self.counters.add(f"flattens_{reason}")

    # -- bookkeeping ----------------------------------------------------------

    def _minutes(self, upto: int) -> None:
        account = self.account
        while self.next_minute <= upto:
            mid = self.mid
            equity = account.equity(mid) if mid == mid else account.cash
            self.equity_rows.append(
                (
                    self.next_minute,
                    equity,
                    account.position,
                    mid,
                    account.realised,
                    account.fees,
                    account.funding,
                )
            )
            self.next_minute += _NS_PER_MINUTE

    def _close(self, last: int) -> None:
        for side in (1, -1):
            for order in list(self.open[side]):
                self._finish(order, last, CANCELLED)
        if self.account.position != 0.0:
            position = self.account.position
            self._taker(-1 if position > 0 else 1, abs(position), last, "close")
        self.account.check_identity(self.mid)
        account = self.account
        self.equity_rows.append(
            (
                last,
                account.equity(self.mid),
                account.position,
                self.mid,
                account.realised,
                account.fees,
                account.funding,
            )
        )

    # -- the result -----------------------------------------------------------

    def result(self, *, oracle: bool) -> DayResult:
        events = self.events
        config = self.config
        fills = pd.DataFrame(
            self.fill_rows,
            columns=[
                "ts",
                "oid",
                "side",
                "price",
                "size",
                "maker",
                "fee",
                "path",
                "queue_wait_ns",
                "position_after",
                "mid_ref",
                "soft_limit",
                "hard_limit",
            ],
        ).astype(
            {
                "ts": "int64",
                "oid": "int64",
                "side": "int64",
                "price": "float64",
                "size": "float64",
                "maker": "bool",
                "fee": "float64",
                "path": "object",
                "queue_wait_ns": "int64",
                "position_after": "float64",
                "mid_ref": "float64",
                "soft_limit": "float64",
                "hard_limit": "float64",
            }
        )
        fills = analysis.markouts(fills, events.book_ts, self.mid_array, config.markout_horizons_s)
        account = self.account
        decomposition = analysis.decompose(
            fills,
            book_ts=events.book_ts,
            mid=self.mid_array,
            horizon_s=config.decomposition_horizon_s,
            final_position=account.position,
            final_mark=self.mid,
            fees=account.fees,
            funding=account.funding,
        )
        equity = account.equity(self.mid)
        scale = max(1.0, account.maker_turnover + account.taker_turnover)
        if abs(decomposition["net"] - equity) > 1e-9 * scale:
            raise IdentityError(
                f"decomposition net {decomposition['net']!r} differs from cash {equity!r}"
            )

        orders = pd.DataFrame(
            [
                (
                    o.oid,
                    o.side,
                    o.price,
                    o.size,
                    o.size - o.remaining,
                    o.decided_ns,
                    o.live_ns,
                    -1 if o.cancel_ns is None else o.cancel_ns,
                    -1 if o.done_ns is None else o.done_ns,
                    o.outcome,
                    o.position_at_decision,
                )
                for o in self.orders
            ],
            columns=[
                "oid",
                "side",
                "price",
                "size",
                "filled",
                "decided_ns",
                "live_ns",
                "cancel_ns",
                "done_ns",
                "outcome",
                "position_at_decision",
            ],
        )
        equity_frame = pd.DataFrame(
            self.equity_rows,
            columns=["ts", "equity", "position", "mid", "realised", "fees", "funding"],
        )
        crossed = pd.DataFrame(self.crossed_rows, columns=["ts", "oid", "side", "price", "mid"])

        counters = dict(self.counters.values)
        for side, name in ((1, "bid"), (-1, "ask")):
            counters[f"{name}_quoted_s"] = _time_quoted(self.orders, side) / NS_PER_S
        counters["book_events"] = float(len(events.book_ts))
        counters["trade_events"] = float(len(events.trade_ts))
        counters["sequence_gaps"] = float(events.sequence_gaps)
        counters["has_funding"] = float(events.has_funding)
        counters["max_abs_position"] = self.max_abs_position
        counters["maker_turnover"] = account.maker_turnover
        counters["taker_turnover"] = account.taker_turnover
        counters["fills"] = float(len(fills))
        counters["clip_over_touch"] = (
            self.clip_over_touch_sum / self.clip_over_touch_n
            if self.clip_over_touch_n
            else math.nan
        )
        return DayResult(
            symbol=events.spec.symbol,
            day=events.day,
            fills=fills,
            orders=orders,
            equity=equity_frame,
            crossed=crossed,
            counters=counters,
            decomposition=decomposition,
            oracle=oracle,
        )


def _time_ns(moment: time) -> int:
    return (moment.hour * 3600 + moment.minute * 60 + moment.second) * NS_PER_S + (
        moment.microsecond * 1000
    )


def _time_quoted(orders: Sequence[Order], side: int) -> int:
    """Nanoseconds during which at least one order on ``side`` was in the book."""
    spans = sorted(
        (o.live_ns, o.done_ns)
        for o in orders
        if o.side == side
        and o.outcome != REJECTED
        and o.done_ns is not None
        and o.done_ns > o.live_ns
    )
    total = 0
    end = None
    begin = 0
    for start, stop in spans:
        if end is None or start > end:
            if end is not None:
                total += end - begin
            begin, end = start, stop
        else:
            end = max(end, stop)
    if end is not None:
        total += end - begin
    return total


def simulate_day(
    events: DayEvents,
    quoter: Quoter,
    config: SimConfig,
    tapes: Mapping[str, SignalTape] | None = None,
    *,
    allow_oracle: bool = False,
) -> DayResult:
    """Simulate one instrument-day with one quoter.

    ``tapes`` are external signals; a quoter reads them through
    :attr:`MarketView.signals`, as of strictly before each snapshot.
    ``allow_oracle`` admits a quoter that reads the future and is for the
    advantage ladder alone; its result is stamped ``oracle``.
    """
    if quoter.uses_future and not allow_oracle:
        raise OracleRefused(
            f"quoter {quoter.name!r} reads the future and may run only inside the ladder"
        )
    if len(events.book_ts) == 0 or len(events.trade_ts) == 0:
        raise DayUnavailable(f"{events.spec.symbol} {events.day}: a plane is empty")
    day = _Day(events, quoter, config, tapes)
    day.run()
    return day.result(oracle=bool(quoter.uses_future))


# ---------------------------------------------------------------------------
# Many days
# ---------------------------------------------------------------------------

#: Columns of the per-day summary, in order.
SUMMARY_COLUMNS = (
    "symbol",
    "day",
    "quoter",
    "status",
    "net",
    "gross",
    "spread",
    "adverse",
    "inventory",
    "fees",
    "funding",
    "making",
    "fills",
    "maker_turnover",
    "taker_turnover",
    "flattens",
    "max_abs_position",
    "placed",
    "cancelled",
    "rejected",
    "crossed_episodes",
    "bid_quoted_s",
    "ask_quoted_s",
    "sequence_gaps",
)


def _summary(result: DayResult, quoter: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": result.symbol,
        "day": result.day.isoformat(),
        "quoter": quoter,
        "status": "ok",
    }
    for name in ("net", "gross", "spread", "adverse", "inventory", "fees", "funding", "making"):
        row[name] = float(result.decomposition[name])
    for name in SUMMARY_COLUMNS:
        if name not in row:
            row[name] = float(result.counters.get(name, 0.0))
    return row


def _job_key(symbol: str, quoter: Quoter, config: SimConfig) -> str:
    description = json.dumps(
        {"symbol": symbol, "quoter": repr(quoter), "config": asdict(config)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(description.encode()).hexdigest()[:16]


def _run_one(
    job: tuple[
        str,
        date,
        Callable[[], Quoter],
        SimConfig,
        tuple[Path, ...],
        Path,
        Path | None,
        Path | None,
    ],
) -> dict[str, Any]:
    """One instrument-day. Module level so a worker process can receive it."""
    symbol, day, make_quoter, config, book_roots, trades_root, funding_root, cache = job
    quoter = make_quoter()
    target: Path | None = None
    if cache is not None:
        target = cache / _job_key(symbol, quoter, config) / symbol / day.isoformat()
        summary = target / "summary.json"
        if summary.exists():
            loaded: dict[str, Any] = json.loads(summary.read_text(encoding="utf-8"))
            return loaded
    try:
        events = load_day(
            symbol,
            day,
            book_roots=book_roots,
            trades_root=trades_root,
            funding_root=funding_root,
        )
    except DayUnavailable as exc:
        row: dict[str, Any] = dict.fromkeys(SUMMARY_COLUMNS, np.nan)
        row.update(
            {"symbol": symbol, "day": day.isoformat(), "quoter": quoter.name, "status": str(exc)}
        )
        return row
    result = simulate_day(events, quoter, config)
    row = _summary(result, quoter.name)
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        result.fills.to_parquet(target / "fills.parquet", index=False)
        result.equity.to_parquet(target / "equity.parquet", index=False)
        (target / "summary.json").write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    return row


def run_days(
    symbol: str,
    days: Sequence[date],
    make_quoter: Callable[[], Quoter],
    config: SimConfig,
    *,
    book_roots: Sequence[Path],
    trades_root: Path,
    funding_root: Path | None,
    workers: int = 1,
    cache: Path | None = None,
) -> pd.DataFrame:
    """Simulate independent days, one row each, sorted by day.

    Each day starts flat and ends flat, so days are independent jobs: with
    ``workers`` above one they run in separate processes, and the result does
    not depend on how many. A day that lacks a plane is a row whose
    ``status`` says why, with no numbers. With ``cache``, each day's fills,
    equity and summary are written under a key of the symbol, quoter and
    configuration, and a day already there is read back instead of re-run.
    ``make_quoter`` must be picklable when ``workers`` is above one.
    """
    probe = make_quoter()
    if probe.uses_future:
        raise OracleRefused(
            f"quoter {probe.name!r} reads the future; its days cannot enter a results table"
        )
    jobs = [
        (
            symbol,
            day,
            make_quoter,
            config,
            tuple(Path(r) for r in book_roots),
            Path(trades_root),
            None if funding_root is None else Path(funding_root),
            None if cache is None else Path(cache),
        )
        for day in sorted(set(days))
    ]
    if workers <= 1:
        rows = [_run_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, jobs))
    frame = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    return frame.sort_values("day", kind="stable").reset_index(drop=True)
