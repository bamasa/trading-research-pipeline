"""Order flow for the market-making simulator, from an order-level book with a known truth.

What is planted
---------------
An order-level limit order book, simulated order by order and photographed every
``snapshot_ms`` (100 ms) at ten levels a side, with a constant spread of
``spread_ticks`` ticks and these flows, each a Poisson process:

* **arrivals**: limit orders join the tail of a uniformly chosen visible level
  at ``arrival_rate`` per level per second (sizes geometric, mean
  ``order_size_mean`` lots);
* **cancels**: every resting order leaves at ``cancel_rate`` per second, chosen
  uniformly among the orders of the book (``cancel_position="uniform"``), or
  favouring the later arrivals of its level (``"back"``) or the earlier
  (``"front"``);
* **market orders**: ``market_rate`` per side per second, mean
  ``market_size_mean`` lots, consuming the touch first in, first out and never
  more than it holds (a touch emptied this way is refilled by one new order),
  each recorded as a print with its aggressor;
* **adverse selection**: after a print, with probability ``pi`` the whole book
  moves ``move_ticks`` in the aggressor's direction within ``move_delay_ms``
  (the touch in the way is pulled, a new touch forms on the other side). A
  resting order filled by that print is marked against by ``pi * move_ticks``
  ticks on average, so the planted post-fill adverse selection is
  ``adverse_bp = pi * move_ticks * tick / price * 1e4``;
* **noise moves**: ``noise_rate`` one-tick moves a second in a random
  direction, independent of the prints.

Because every order is simulated, the generator knows what the snapshots
cannot say: the exact queue in front of an order of our own at the moment it
arrives, and which orders cancelled from in front of it. :func:`true_fills`
replays the order log with zero-impact own orders (the simulator's own
assumption, R11) and returns when each would really have filled: the truth the
simulator's queue estimate is checked against.

The null
--------
``null=True`` plants no adverse selection (``pi = 0``) and adds the same
expected number of moves as noise, random in direction, so the mid's moves keep
their rate and size and lose their link to the prints.

What it can prove, and what it cannot
-------------------------------------
It can show that the simulator's markouts recover a planted adverse selection
and find none where none is planted, and how far its queue estimate, built from
100 ms snapshots, is from the fills an order-level replay of the same orders
gives. It cannot show that real queues, cancellations or informed flow behave
like these Poisson flows, or that a market maker would earn anything.
"""

from __future__ import annotations

import math
import random
from bisect import insort
from dataclasses import dataclass
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
    spec_dict,
    stamps,
    strings,
    write_plane,
)

GENERATOR = "trading_research.synthetic.order_flow.v1"

CancelPosition = Literal["uniform", "back", "front"]

#: Kinds of the order log.
ADD, CANCEL, TRADE = 0, 1, 2

PROVES = (
    "The simulator's markouts recover a planted adverse selection and find none on the "
    "null, and its queue estimate is measured against an order-level replay of the "
    "same orders."
)
DOES_NOT_PROVE = (
    "That real queues, cancellations or informed flow behave like these Poisson flows, "
    "or that a market maker would earn anything."
)


@dataclass(frozen=True)
class OrderFlowSpec:
    """Every parameter of an order-flow dataset. The defaults generate in seconds."""

    symbol: str = "FLOWUSDT"
    start: str = DEFAULT_START
    days: int = 4
    session_s: int = 3600
    snapshot_ms: int = 100
    depth: int = 10
    tick: float = 0.1
    start_ticks: int = 10_000
    lot: float = 1.0
    spread_ticks: int = 2
    arrival_rate: float = 0.8
    cancel_rate: float = 0.08
    order_size_mean: float = 2.0
    market_rate: float = 0.5
    market_size_mean: float = 4.0
    #: Planted post-fill adverse selection, bp of price at the start.
    adverse_bp: float = 0.6
    move_delay_ms: float = 300.0
    noise_rate: float = 0.2
    cancel_position: CancelPosition = "uniform"
    null: bool = False
    seed: int = 20200106

    def __post_init__(self) -> None:
        if self.spread_ticks < 2:
            raise ValueError("the spread must be at least two ticks, so a move never locks it")
        if self.cancel_position not in ("uniform", "back", "front"):
            raise ValueError(f"unknown cancel position {self.cancel_position!r}")
        if min(self.arrival_rate, self.cancel_rate, self.market_rate) <= 0:
            raise ValueError("arrival, cancel and market rates must be positive")
        if self.adverse_bp < 0 or self.noise_rate < 0 or self.move_delay_ms <= 0:
            raise ValueError("adverse selection, noise and delay cannot be negative")
        if self.order_size_mean < 1 or self.market_size_mean < 1:
            raise ValueError("mean sizes are at least one lot")

    @property
    def tick_bp(self) -> float:
        return 1e4 / self.start_ticks

    @property
    def move_ticks(self) -> int:
        return max(1, math.ceil(self.adverse_bp / self.tick_bp - 1e-9))

    @property
    def informed_share(self) -> float:
        """``pi``: the probability that a print is followed by a move against its maker."""
        return 0.0 if self.null else self.adverse_bp / (self.move_ticks * self.tick_bp)

    @property
    def planted_adverse_bp(self) -> float:
        return 0.0 if self.null else self.adverse_bp


@dataclass
class OrderFlowDay:
    """One day: the snapshots and prints as arrays (prices in ticks) and the order log."""

    day: date
    book_ts: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    trade_ts: np.ndarray
    trade_px: np.ndarray
    trade_sz: np.ndarray
    trade_aggressor: np.ndarray
    #: The order log: ``ts``, ``kind`` (0 add, 1 cancel, 2 trade), ``side`` (+1
    #: bid, -1 ask; for a trade the side hit), ``price`` (ticks), ``size``
    #: (lots; a cancel's remaining size), ``added_ns`` (a cancel's order's
    #: arrival; -1 otherwise).
    log: pd.DataFrame
    #: Every move of the book: ``ts``, ``direction``, ``why`` (informed or
    #: noise) and the best bid after it, in ticks.
    moves: pd.DataFrame
    #: The best bid before the first move, in ticks.
    first_bid: int = 0

    def true_mid(self, at_ns: np.ndarray, spread_ticks: int, tick: float) -> np.ndarray:
        """The book's mid at each time, from the move log: no snapshot staleness."""
        stamps_ = self.moves["ts"].to_numpy(dtype=np.int64)
        after = self.moves["best_bid_after"].to_numpy(dtype=np.int64)
        row = np.searchsorted(stamps_, np.asarray(at_ns, dtype=np.int64), side="right") - 1
        bid = np.where(row >= 0, after[np.maximum(row, 0)], self.first_bid)
        return (bid + spread_ticks / 2.0) * tick


@dataclass
class OrderFlowDataset:
    spec: OrderFlowSpec
    days: list[OrderFlowDay]
    truth: Truth

    def events(self, day: OrderFlowDay) -> object:
        """The day as the simulator's :class:`DayEvents`, assembled in memory."""
        from trading_research.market_making.events import InstrumentSpec, assemble

        spec = self.spec
        return assemble(
            InstrumentSpec(spec.symbol, spec.tick, spec.lot),
            day.day,
            book_ts=day.book_ts,
            bid_px=day.bid_px,
            bid_sz=day.bid_sz * spec.lot,
            ask_px=day.ask_px,
            ask_sz=day.ask_sz * spec.lot,
            trade_ts=day.trade_ts,
            trade_px=day.trade_px,
            trade_sz=day.trade_sz * spec.lot,
            trade_aggressor=day.trade_aggressor,
        )

    def book_frame(self, day: OrderFlowDay) -> pd.DataFrame:
        spec = self.spec
        n = len(day.book_ts)
        columns: dict[str, np.ndarray] = {}
        for level in range(spec.depth):
            columns[bid_price_col(level)] = np.round(day.bid_px[:, level] * spec.tick, 10)
            columns[bid_size_col(level)] = day.bid_sz[:, level] * spec.lot
            columns[ask_price_col(level)] = np.round(day.ask_px[:, level] * spec.tick, 10)
            columns[ask_size_col(level)] = day.ask_sz[:, level] * spec.lot
        first = 1 + sum(len(d.book_ts) for d in self.days if d.day < day.day)
        return pd.DataFrame(
            {
                "timestamp": stamps(day.book_ts),
                "symbol": strings(spec.symbol, n),
                "sequence_id": np.arange(first, first + n, dtype=np.int64),
                "source": strings(SYNTHETIC_SOURCE, n),
                **columns,
            }
        )

    def trades_frame(self, day: OrderFlowDay) -> pd.DataFrame:
        spec = self.spec
        n = len(day.trade_ts)
        first = 1 + sum(len(d.trade_ts) for d in self.days if d.day < day.day)
        return pd.DataFrame(
            {
                "timestamp": stamps(day.trade_ts),
                "symbol": strings(spec.symbol, n),
                "price": np.round(day.trade_px * spec.tick, 10),
                "quantity": day.trade_sz * spec.lot,
                "is_buyer_maker": day.trade_aggressor < 0,
                "trade_id": np.arange(first, first + n, dtype=np.int64),
                "source": strings(SYNTHETIC_SOURCE, n),
            }
        )

    def write(self, directory: Path | str, *, validate: bool = True) -> Path:
        root = Path(directory)
        books = {d.day: self.book_frame(d) for d in self.days}
        trades = {d.day: self.trades_frame(d) for d in self.days}
        if validate:
            for d in self.days:
                check_frame(books[d.day], "book", depth=self.spec.depth)
                check_frame(trades[d.day], "trades")
        write_plane(
            root, "book", self.spec.symbol, books, generator=GENERATOR, seed=self.spec.seed,
            depth=self.spec.depth, extra={"step_ms": self.spec.snapshot_ms},
        )  # fmt: skip
        write_plane(
            root, "trades", self.spec.symbol, trades, generator=GENERATOR, seed=self.spec.seed
        )
        logs = root / "order_log" / self.spec.symbol
        logs.mkdir(parents=True, exist_ok=True)
        for d in self.days:
            d.log.to_parquet(logs / f"{d.day.isoformat()}.parquet", index=False)
        self.truth.write(root)
        return root


def generate(spec: OrderFlowSpec | None = None) -> OrderFlowDataset:
    """Generate every day; each day starts from a fresh book at the same price."""
    spec = spec or OrderFlowSpec()
    days = [_Day(spec, day, k).run() for k, day in enumerate(day_list(spec.start, spec.days))]
    truth = Truth(
        generator=GENERATOR,
        seed=spec.seed,
        null=spec.null,
        planted={
            "adverse_bp": spec.planted_adverse_bp,
            "informed_share": spec.informed_share,
            "move_ticks": spec.move_ticks,
            "spread_ticks": spec.spread_ticks,
            "half_spread_bp": spec.spread_ticks / 2 * spec.tick_bp,
            "arrival_rate_per_level_s": spec.arrival_rate,
            "cancel_rate_per_order_s": spec.cancel_rate,
            "market_rate_per_side_s": spec.market_rate,
            "cancel_position": spec.cancel_position,
            "fills": "per order, from true_fills() on the order log",
        },
        spec=spec_dict(spec),
        proves=PROVES,
        does_not_prove=DOES_NOT_PROVE,
    )
    return OrderFlowDataset(spec, days, truth)


class _Day:
    """The order-level simulation of one day."""

    def __init__(self, spec: OrderFlowSpec, day: date, index: int) -> None:
        self.spec = spec
        self.day = day
        self.rng = random.Random(spec.seed * 1009 + index)
        self.start = day_start_ns(day)
        self.end = self.start + spec.session_s * NS_PER_S
        self.best_bid = spec.start_ticks - spec.spread_ticks // 2
        #: price -> list of [added_ns, size] in time order, one map per side.
        self.queues: dict[int, dict[int, list[list[int]]]] = {1: {}, -1: {}}
        self.count = 0
        self.log: list[tuple[int, int, int, int, int, int]] = []
        self.prints: list[tuple[int, int, int, int]] = []
        self.moves: list[tuple[int, int, str, int]] = []
        self.pending: list[tuple[int, int]] = []
        mean_orders = spec.arrival_rate / spec.cancel_rate
        self.level_orders = mean_orders
        for side in (1, -1):
            for price in self.visible(side):
                self.populate(side, price, self.start)

    @property
    def best_ask(self) -> int:
        return self.best_bid + self.spec.spread_ticks

    def visible(self, side: int) -> list[int]:
        touch = self.best_bid if side == 1 else self.best_ask
        return [touch - side * k for k in range(self.spec.depth)]

    def size(self, mean: float) -> int:
        """Geometric on 1, 2, ... with the given mean."""
        p = 1.0 / mean
        return 1 + int(math.log(1.0 - self.rng.random()) / math.log(1.0 - p)) if p < 1 else 1

    def add(self, side: int, price: int, t: int, size: int) -> None:
        self.queues[side].setdefault(price, []).append([t, size])
        self.count += 1
        self.log.append((t, ADD, side, price, size, -1))

    def populate(self, side: int, price: int, t: int) -> None:
        orders = max(1, self.poisson(self.level_orders))
        for _ in range(orders):
            self.add(side, price, t, self.size(self.spec.order_size_mean))

    def poisson(self, mean: float) -> int:
        # Knuth's method is fine for the small means used here.
        limit, k, product = math.exp(-mean), 0, self.rng.random()
        while product > limit:
            k += 1
            product *= self.rng.random()
        return k

    def remove_level(self, side: int, price: int, t: int) -> None:
        for added, size in self.queues[side].pop(price, []):
            self.count -= 1
            self.log.append((t, CANCEL, side, price, size, added))

    def cancel(self, t: int) -> None:
        if self.count == 0:
            return
        pick = self.rng.randrange(self.count)
        for side in (1, -1):
            for price, queue in self.queues[side].items():
                if pick < len(queue):
                    position = self.position(len(queue), pick)
                    added, size = queue.pop(position)
                    self.count -= 1
                    self.log.append((t, CANCEL, side, price, size, added))
                    if not queue:
                        del self.queues[side][price]
                    return
                pick -= len(queue)

    def position(self, length: int, uniform: int) -> int:
        """Which order of a level of ``length`` cancels, by the planted rule."""
        rule = self.spec.cancel_position
        if rule == "uniform" or length == 1:
            return uniform
        # A triangular choice: weight k + 1 for the k-th arrival (back), or the reverse.
        u = self.rng.random()
        k = min(length - 1, int(math.sqrt(u) * length))
        return k if rule == "back" else length - 1 - k

    def market(self, t: int) -> None:
        aggressor = 1 if self.rng.random() < 0.5 else -1
        side = -aggressor
        price = self.best_bid if side == 1 else self.best_ask
        queue = self.queues[side].get(price)
        if not queue:
            return
        wanted = self.size(self.spec.market_size_mean)
        traded = 0
        while queue and traded < wanted:
            take = min(queue[0][1], wanted - traded)
            queue[0][1] -= take
            traded += take
            if queue[0][1] == 0:
                queue.pop(0)
                self.count -= 1
        self.log.append((t, TRADE, side, price, traded, -1))
        self.prints.append((t, price, traded, aggressor))
        if not queue:
            self.queues[side].pop(price, None)
            self.add(side, price, t, self.size(self.spec.order_size_mean))
        if self.rng.random() < self.spec.informed_share:
            delay = 1 + int(self.rng.random() * self.spec.move_delay_ms * 1_000_000)
            insort(self.pending, (t + delay, aggressor))

    def move(self, t: int, direction: int, why: str) -> None:
        """The whole book moves one tick: the side moved away from is pulled."""
        depth = self.spec.depth
        for _ in range(self.spec.move_ticks if why == "informed" else 1):
            if direction > 0:
                self.remove_level(-1, self.best_ask, t)
                self.best_bid += 1
                self.populate(1, self.best_bid, t)
                self.remove_level(1, self.best_bid - depth, t)
                self.populate(-1, self.best_ask + depth - 1, t)
            else:
                self.remove_level(1, self.best_bid, t)
                self.best_bid -= 1
                self.populate(-1, self.best_ask, t)
                self.remove_level(-1, self.best_ask + depth, t)
                self.populate(1, self.best_bid - depth + 1, t)
        self.moves.append((t, direction, why, self.best_bid))

    def snapshot(self) -> tuple[list[int], list[int], list[int], list[int]]:
        out = []
        for side in (1, -1):
            prices = self.visible(side)
            queues = self.queues[side]
            out.append(prices)
            out.append([sum(o[1] for o in queues.get(p, ())) for p in prices])
        return out[0], out[1], out[2], out[3]

    def run(self) -> OrderFlowDay:
        spec = self.spec
        step = spec.snapshot_ms * 1_000_000
        next_snap = self.start + step
        extra_noise = 0.0
        if spec.null:
            # The moves an informed share would have made, as noise instead.
            extra_noise = 2 * spec.market_rate * (spec.adverse_bp / spec.tick_bp)
        noise_rate = spec.noise_rate + extra_noise
        arrivals = 2 * spec.depth * spec.arrival_rate
        markets = 2 * spec.market_rate
        books: list[tuple[int, list[int], list[int], list[int], list[int]]] = []
        t = self.start
        while True:
            total = arrivals + markets + noise_rate + spec.cancel_rate * self.count
            t_next = t + int(self.rng.expovariate(total) * NS_PER_S) + 1
            scheduled = min(next_snap, self.pending[0][0] if self.pending else self.end)
            if scheduled <= t_next or t_next >= self.end:
                t = min(scheduled, self.end)
                if t >= self.end:
                    break
                if self.pending and self.pending[0][0] == t and t < next_snap:
                    _, direction = self.pending.pop(0)
                    self.move(t, direction, "informed")
                else:
                    books.append((t, *self.snapshot()))
                    next_snap += step
                continue
            t = t_next
            u = self.rng.random() * total
            if u < arrivals:
                side = 1 if self.rng.random() < 0.5 else -1
                level = self.rng.randrange(spec.depth)
                touch = self.best_bid if side == 1 else self.best_ask
                self.add(side, touch - side * level, t, self.size(spec.order_size_mean))
            elif u < arrivals + markets:
                self.market(t)
            elif u < arrivals + markets + noise_rate:
                self.move(t, 1 if self.rng.random() < 0.5 else -1, "noise")
            else:
                self.cancel(t)
        return self.finish(books)

    def finish(
        self, books: list[tuple[int, list[int], list[int], list[int], list[int]]]
    ) -> OrderFlowDay:
        book_ts = np.array([b[0] for b in books], dtype=np.int64)
        bid_px = np.array([b[1] for b in books], dtype=np.int64)
        bid_sz = np.array([b[2] for b in books], dtype=np.float64)
        ask_px = np.array([b[3] for b in books], dtype=np.int64)
        ask_sz = np.array([b[4] for b in books], dtype=np.float64)
        prints = np.array(self.prints, dtype=np.int64).reshape(-1, 4)
        log = pd.DataFrame(
            self.log, columns=["ts", "kind", "side", "price", "size", "added_ns"]
        ).astype("int64")
        moves = pd.DataFrame(self.moves, columns=["ts", "direction", "why", "best_bid_after"])
        return OrderFlowDay(
            day=self.day,
            book_ts=book_ts,
            bid_px=bid_px,
            bid_sz=bid_sz,
            ask_px=ask_px,
            ask_sz=ask_sz,
            trade_ts=prints[:, 0],
            trade_px=prints[:, 1],
            trade_sz=prints[:, 2].astype(np.float64),
            trade_aggressor=prints[:, 3].astype(np.int8),
            log=log,
            moves=moves,
            first_bid=self.spec.start_ticks - self.spec.spread_ticks // 2,
        )


def true_fills(log: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Replay the order log with zero-impact own orders; return each one's true fill.

    ``orders`` has ``side`` (+1 bid, -1 ask), ``price`` (ticks), ``size``
    (lots), ``live_ns`` (arrival at the venue) and ``end_ns`` (when it stops
    resting). An own order joins the tail of the true queue at its price after
    every logged event stamped at or before its arrival (ties go to the market,
    as in the simulator) and rests through every event stamped at or before its
    end. A cancel of an order that arrived no later than it shortens the queue
    in front by that order's remaining size; a print at its price consumes the
    queue in front first and fills it with the excess; a print strictly through
    its price fills it up to the print's size. The book is not changed by an own
    order (the simulator's own assumption). Returns ``filled`` and
    ``first_fill_ns`` (-1 when unfilled), aligned with ``orders``.
    """
    ts = log["ts"].to_numpy(dtype=np.int64)
    kind = log["kind"].to_numpy(dtype=np.int64)
    side = log["side"].to_numpy(dtype=np.int64)
    price = log["price"].to_numpy(dtype=np.int64)
    size = log["size"].to_numpy(dtype=np.int64)
    added = log["added_ns"].to_numpy(dtype=np.int64)
    order = np.argsort(orders["live_ns"].to_numpy(dtype=np.int64), kind="stable")
    o_side = orders["side"].to_numpy(dtype=np.int64)
    o_price = orders["price"].to_numpy(dtype=np.int64)
    o_size = orders["size"].to_numpy(dtype=np.float64)
    o_live = orders["live_ns"].to_numpy(dtype=np.int64)
    o_end = orders["end_ns"].to_numpy(dtype=np.int64)
    filled = np.zeros(len(orders))
    first = np.full(len(orders), -1, dtype=np.int64)
    ahead = np.zeros(len(orders))
    volume: dict[tuple[int, int], float] = {}
    active: list[int] = []
    pointer = 0
    for k in range(len(ts)):
        t = int(ts[k])
        while pointer < len(order) and o_live[order[pointer]] < t:
            g = int(order[pointer])
            ahead[g] = volume.get((int(o_side[g]), int(o_price[g])), 0.0)
            active.append(g)
            pointer += 1
        if active:
            active = [g for g in active if o_end[g] >= t and filled[g] < o_size[g]]
        s, p, q = int(side[k]), int(price[k]), float(size[k])
        key = (s, p)
        if kind[k] == ADD:
            volume[key] = volume.get(key, 0.0) + q
            continue
        volume[key] = volume.get(key, 0.0) - q
        for g in active:
            if o_side[g] != s:
                continue
            if kind[k] == CANCEL:
                if o_price[g] == p and added[k] <= o_live[g]:
                    ahead[g] = max(0.0, ahead[g] - q)
                continue
            beyond = (int(o_price[g]) - p) * s
            take = 0.0
            if beyond > 0:
                ahead[g] = 0.0
                take = q
            elif beyond == 0:
                if q <= ahead[g]:
                    ahead[g] -= q
                else:
                    take = q - ahead[g]
                    ahead[g] = 0.0
            if take > 0:
                got = min(take, o_size[g] - filled[g])
                if got > 0 and first[g] < 0:
                    first[g] = t
                filled[g] += got
    return pd.DataFrame({"filled": filled, "first_fill_ns": first}, index=orders.index)
