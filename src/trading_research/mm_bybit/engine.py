"""The event loop: one day, one cell, every registered rule, from prints only.

Market events are the day's prints, the funding settlements and, for the
record, the hour marks; own actions (an order arriving, a cancel landing, the
stop, the flatten and its fill) fire only strictly before the next market event,
so at an equal stamp the market goes first. At equal times: an hour mark, then a
settlement, then an own action. A decision is taken at each row of the print
grid, after every print stamped at the row.

**Fill rules** (``rule``): ``trade_through`` (primary) fills a live bid at ``p``
only on a seller-initiated print strictly below ``p``, at most the print's size,
at ``p``, floored to the lot; ``touch`` fills the whole remaining order at the
first print of the hitting side at or through ``p``; ``delay`` is the primary
rule counting only prints stamped strictly after the stamp of the first print
that traded through ``p`` after the order's arrival.

The limits are enforced before quoting (a side is quoted only if its fill would
leave |q| within the soft limit, every order of that side not yet done
counted), so the only taker order is the 23:59 flatten, priced by the spot
study's side-correct print rule (``spot/fills.fill_orders``, fallback on).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from trading_research.market_making.accounting import Account
from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit.tape import SECONDS, DayTape
from trading_research.spot.fills import PrintFillSettings, fill_orders

NS_PER_S = 1_000_000_000
RULES = ("trade_through", "touch", "delay")
_ACTIVATE, _CANCEL, _STOP, _FLATTEN, _TAKER = 0, 1, 2, 3, 4
_LOT_EPS = 1e-9
_NEVER = 2**62


@dataclass(frozen=True)
class RunSettings:
    maker_bp: float
    taker_bp: float
    clip: float = 0.010
    soft_limit_clips: float = 6.0
    latency_ns: int = 50_000_000
    rule: str = "trade_through"
    warmup_s: int = 600
    stop_s: int = 23 * 3600 + 58 * 60
    flatten_s: int = 23 * 3600 + 59 * 60
    pause_s: int = 5

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unknown fill rule {self.rule!r}")
        if self.latency_ns < 0 or self.clip <= 0 or self.soft_limit_clips < 1:
            raise ValueError("latency, clip and soft limit must be valid")

    @property
    def soft(self) -> float:
        return self.soft_limit_clips * self.clip


@dataclass(frozen=True)
class QuoterInputs:
    """Set on W: the volatility reference and the gate's quantiles."""

    sigma_ref: float
    quantiles: dict[float, float]


@dataclass(eq=False)
class _Order:
    side: int
    price: int
    remaining: float
    decided: int
    live_at: int
    live: bool = False
    done: bool = False
    cancel_at: int | None = None
    first_through: int | None = None
    #: A taker fill's price (USDT); NaN for a resting order.
    fill_price: float = math.nan


@dataclass
class DayRun:
    """One cell on one day. Fill arrays share their order; equity is per UTC hour end."""

    fill_ts: np.ndarray
    fill_side: np.ndarray
    fill_price: np.ndarray
    fill_size: np.ndarray
    fill_maker: np.ndarray
    fill_fee: np.ndarray
    hourly_equity: np.ndarray
    net: float
    fees: float
    funding: float
    maker_turnover: float
    taker_turnover: float
    max_abs_position: float
    counters: dict[str, float] = field(default_factory=dict)
    identity_ok: bool = True


class _Day:
    def __init__(
        self,
        tape: DayTape,
        cell: q.Cell,
        settings: RunSettings,
        inputs: QuoterInputs,
        shifted: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        self.tape, self.cell, self.s, self.inputs, self.shifted = (
            tape, cell, settings, inputs, shifted)  # fmt: skip
        self.acct = Account()
        self.heap: list[tuple[int, int, int, _Order | None]] = []
        self.seq = 0
        self.open: dict[int, list[_Order]] = {1: [], -1: []}
        self.working: dict[int, _Order | None] = {1: None, -1: None}
        self.bid = math.nan  # bid proxy, ticks (last seller-initiated print)
        self.ask = math.nan  # ask proxy, ticks (last buyer-initiated print)
        self.stopped = False
        self.fills: list[tuple[int, int, float, float, bool, float]] = []
        self.counters: dict[str, float] = {}
        self.hourly = np.zeros(24)
        self.next_hour = 1
        self.next_funding = 0
        self.hour_at = self._hour_time()
        self.fund_at = self._fund_time()
        self.max_abs = 0.0
        self.identity_ok = True
        start = tape.start_ns
        self.quote_from = start + settings.warmup_s * NS_PER_S
        self.quote_until = start + settings.stop_s * NS_PER_S
        self._push(start + settings.stop_s * NS_PER_S, _STOP, None)
        self._push(start + settings.flatten_s * NS_PER_S, _FLATTEN, None)

    def count(self, name: str, amount: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + amount

    def _push(self, when: int, action: int, order: _Order | None) -> None:
        self.seq += 1
        heapq.heappush(self.heap, (when, self.seq, action, order))

    def mid(self) -> float:
        return (self.bid + self.ask) * 0.5 * self.tape.tick

    # -- the clock ---------------------------------------------------------

    def advance(self, t: int) -> None:
        """Hour marks and settlements at or before ``t``, own actions strictly before it."""
        heap = self.heap
        while True:
            hour, fund = self.hour_at, self.fund_at
            act = heap[0][0] if heap else _NEVER
            if hour <= t and hour <= fund and hour <= act:
                self.hourly[self.next_hour - 1] = self.acct.equity(self._mark())
                self.next_hour += 1
                self.hour_at = self._hour_time()
            elif fund <= t and fund <= act:
                self._settle(float(self.tape.funding_rate[self.next_funding]))
                self.next_funding += 1
                self.fund_at = self._fund_time()
            elif act < t:
                when, _, action, order = heapq.heappop(heap)
                self._act(when, action, order)
            else:
                return

    def _hour_time(self) -> int:
        if self.next_hour > 24:
            return _NEVER
        return self.tape.start_ns + self.next_hour * 3600 * NS_PER_S

    def _fund_time(self) -> int:
        if self.next_funding >= len(self.tape.funding_ts):
            return _NEVER
        return int(self.tape.funding_ts[self.next_funding])

    def _mark(self) -> float:
        mid = self.mid()
        return mid if mid == mid else 0.0

    def _settle(self, rate: float) -> None:
        if self.acct.position != 0.0:
            self.acct.settle_funding(rate, self.mid())
            self.count("funding_settlements_with_position")
            self._check()

    def _check(self) -> None:
        try:
            self.acct.check_identity(self._mark())
        except AssertionError:
            self.identity_ok = False

    # -- own actions ---------------------------------------------------------

    def _act(self, when: int, action: int, order: _Order | None) -> None:
        if action == _ACTIVATE and order is not None:
            if order.done:
                return
            if order.cancel_at is not None and order.cancel_at <= when:
                self._finish(order)
                return
            crossing = (
                order.price >= self.ask if order.side > 0 else order.price <= self.bid
            )  # a NaN proxy never crosses
            if crossing:
                self.count("rejected_post_only")
                self._finish(order)
                return
            order.live = True
        elif action == _CANCEL and order is not None:
            self._finish(order)
        elif action == _STOP:
            self.stopped = True
            for side in (1, -1):
                self._cancel_side(side, when)
        elif action == _FLATTEN:
            self.stopped = True
            self._flatten(when)
        elif action == _TAKER and order is not None:
            self._book(order.side, order.fill_price, order.remaining, when, maker=False)

    def _finish(self, order: _Order) -> None:
        order.done = True
        order.live = False
        if order in self.open[order.side]:
            self.open[order.side].remove(order)
        if self.working[order.side] is order:
            self.working[order.side] = None

    def _cancel(self, order: _Order, decided: int) -> None:
        if order.cancel_at is None and not order.done:
            order.cancel_at = decided + self.s.latency_ns
            self._push(order.cancel_at, _CANCEL, order)
            self.count("cancels")
        if self.working[order.side] is order:
            self.working[order.side] = None

    def _cancel_side(self, side: int, decided: int) -> None:
        for order in list(self.open[side]):
            self._cancel(order, decided)

    def _flatten(self, decided: int) -> None:
        position = self.acct.position
        if position == 0.0:
            return
        side = -1 if position > 0 else 1
        arrival = decided + self.s.latency_ns
        settings = PrintFillSettings(clip=abs(position), latency_ns=self.s.latency_ns)
        fill = fill_orders(
            self.tape.prints, side, np.array([arrival]), np.array([decided]),
            until_ns=self.tape.start_ns + SECONDS * NS_PER_S, fallback=True, settings=settings,
        )  # fmt: skip
        # The tape caches its groups per clip; a flatten's clip is its own position, so
        # the entry is dropped rather than kept for a size no other order will have.
        self.tape.prints._groups.pop((side, settings), None)
        price = float(fill.rule[0])
        if not math.isfinite(price):
            raise RuntimeError(f"{self.tape.day}: the flatten found no print to price it")
        when = int(fill.print_ns[0]) if bool(fill.found[0]) else arrival
        if bool(fill.fallback[0]):
            self.count("flatten_fallback")
        taker = _Order(side, 0, abs(position), decided, when, fill_price=price)
        self._push(when, _TAKER, taker)

    def _book(self, side: int, price: float, size: float, when: int, *, maker: bool) -> None:
        fee_bp = self.s.maker_bp if maker else self.s.taker_bp
        fee = self.acct.fill(side, price, size, fee_bp, maker=maker)
        self.fills.append((when, side, price, size, maker, fee))
        self.max_abs = max(self.max_abs, abs(self.acct.position))
        if abs(self.acct.position) > self.s.soft + self.s.clip + 1e-12:
            self.count("limit_breach")
        self._check()

    # -- market events -------------------------------------------------------

    def on_print(self, t: int, px: int, size: float, aggressor: int) -> None:
        side = -aggressor  # a seller-initiated print hits bids (+1)
        resting = self.open[side]
        if resting:
            rule, lot, tick = self.s.rule, self.tape.lot, self.tape.tick
            available = size
            for order in sorted(resting, key=lambda o: -side * o.price):
                if not order.live:
                    continue
                through = px < order.price if side > 0 else px > order.price
                if rule == "touch":
                    if not (through or px == order.price):
                        continue
                    filled = order.remaining
                else:
                    if not through:
                        continue
                    if rule == "delay":
                        if order.first_through is None:
                            order.first_through = t
                        if t <= order.first_through:
                            continue
                    filled = math.floor(min(available, order.remaining) / lot + _LOT_EPS) * lot
                    if filled <= 0.0:
                        continue
                self._book(side, order.price * tick, filled, t, maker=True)
                self.count("passive_fills")
                order.remaining -= filled
                if order.remaining < lot * (1 - _LOT_EPS):
                    self._finish(order)
                if rule != "touch":
                    available -= filled
                    if available <= 0.0:
                        break
        if aggressor > 0:
            self.ask = float(px)
        else:
            self.bid = float(px)

    def decide(self, t: int) -> None:
        if t < self.quote_from or t >= self.quote_until or self.stopped:
            return
        self.count("decisions")
        bid, ask = self.bid, self.ask
        if bid != bid or ask != ask:
            for side in (1, -1):
                self._cancel_side(side, t)
            return
        tape, second = self.tape, self.tape.second_of(t)
        vol = float(tape.sec_vol[second])
        if self.shifted is None:
            state = q.State(ask - bid, vol)
        else:
            state = q.State(float(self.shifted[0][second]), float(self.shifted[1][second]))
        want_bid, want_ask = q.wanted(
            self.cell, self.acct.position, self.s.soft, vol,
            self.inputs.sigma_ref, state, self.inputs.quantiles,
        )  # fmt: skip
        bid_px, ask_px = int(bid), int(ask)
        if want_bid and want_ask and bid_px >= ask_px:
            # Own quotes never lock or cross: pull both apart around their middle (R4).
            total = bid_px + ask_px
            bid_px, ask_px = (total + 1) // 2 - 1, total // 2 + 1
            self.count("self_cross_clamped")
        self._reconcile(1, want_bid, bid_px, t)
        self._reconcile(-1, want_ask, ask_px, t)

    def _reconcile(self, side: int, want: bool, price: int, t: int) -> None:
        current = self.working[side]
        if not want:
            if current is not None:
                self._cancel(current, t)
            return
        tol = 1e-9 * self.s.clip
        exposure = side * self.acct.position + sum(o.remaining for o in self.open[side])
        if current is not None and current.price == price:
            if exposure <= self.s.soft + tol:
                return
            self._cancel(current, t)
            self.count("cancelled_for_limits")
            return
        if current is not None:
            self._cancel(current, t)
            self.count("replaced")
        if exposure + self.s.clip > self.s.soft + tol:
            self.count("not_quoted_limit")
            return
        order = _Order(side, price, self.s.clip, t, t + self.s.latency_ns)
        self.open[side].append(order)
        self.working[side] = order
        self._push(order.live_at, _ACTIVATE, order)
        self.count("placed")

    # -- the day -------------------------------------------------------------

    def run(self) -> DayRun:
        tape = self.tape
        ts, px = tape.ts.tolist(), tape.px.tolist()
        size, agg = tape.size.tolist(), tape.agg.tolist()
        rows, row_ts = tape.row_end.tolist(), tape.row_ts.tolist()
        next_row, n_rows = 0, len(rows)
        pause = self.s.pause_s * NS_PER_S
        last: int | None = None
        for i, t in enumerate(ts):
            if last is not None and t - last > pause and (self.open[1] or self.open[-1]):
                self.advance(last + pause)
                self.count("pauses")
                for side in (1, -1):
                    self._cancel_side(side, last + pause)
            self.advance(t)
            self.on_print(t, px[i], size[i], agg[i])
            last = t
            if next_row < n_rows and rows[next_row] == i:
                self.decide(row_ts[next_row])
                next_row += 1
        self.advance(_NEVER - 1)
        mark = self._mark()
        self._check()
        if self.acct.position != 0.0:
            self.count("not_flat_at_end")
        f = self.fills
        return DayRun(
            fill_ts=np.array([r[0] for r in f], dtype=np.int64),
            fill_side=np.array([r[1] for r in f], dtype=np.int64),
            fill_price=np.array([r[2] for r in f], dtype=np.float64),
            fill_size=np.array([r[3] for r in f], dtype=np.float64),
            fill_maker=np.array([r[4] for r in f], dtype=bool),
            fill_fee=np.array([r[5] for r in f], dtype=np.float64),
            hourly_equity=self.hourly.copy(),
            net=self.acct.equity(mark),
            fees=self.acct.fees,
            funding=self.acct.funding,
            maker_turnover=self.acct.maker_turnover,
            taker_turnover=self.acct.taker_turnover,
            max_abs_position=self.max_abs,
            counters=dict(self.counters),
            identity_ok=self.identity_ok,
        )


def run_day(
    tape: DayTape,
    cell: q.Cell,
    settings: RunSettings,
    inputs: QuoterInputs,
    shifted: tuple[np.ndarray, np.ndarray] | None = None,
) -> DayRun:
    """One cell on one day. ``shifted`` replaces the gate's state (spread, sigma) per second."""
    return _Day(tape, cell, settings, inputs, shifted).run()
