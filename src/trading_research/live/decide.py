"""Decisions at each boundary, and the paper orders that carry them out.

**The decision.** A model is loaded by path (the ``model.pkl`` that ``train``
writes, ``{"model", "features"}``) or by name (``always_hold``). At a
boundary whose features are all present, ``predict_proba`` gives (sell, hold,
buy) and :func:`trading_research.backtest.evaluate.decide` turns it into
-1 / 0 / +1 with the confidence threshold chosen elsewhere, exactly as
``backtest`` does. A boundary with a missing feature is not scored, as
``predict`` does not score it.

**The execution policies.** No order leaves the process: an order here is a
row in a table, filled against public prints by the repository's rules.

``taker``
    The order arrives at the decision plus the latency and fills at the
    **first print of its side** stamped at or after its arrival (a buy at a
    buyer-initiated print), sweeping that stamp's same-side prints at their
    volume-weighted price, the part beyond the sweep at the sweep's worst
    price: the side-correct print rule of
    :mod:`trading_research.spot.fills`. RPI prints never fill. The book walk
    of the first row at or after the arrival
    (:func:`trading_research.backtest.walk.walk_book`) is recorded beside it.
    The exit is decided at the first row stamped at or after the entry fill +
    ``hold`` and filled by the same rule; an entry that finds no print within
    ``hold`` is dropped.
``passive``
    One clip posted at the touch on the decision's side, kept at its first
    price, filled **only by prints strictly through it**, up to their size
    (:func:`trading_research.market_making.queue.on_print_rule` with
    ``FillRule.THROUGH``), as the signal-entry quoter is filled. It is
    cancelled ``hold`` after the post (it can still fill until the cancel
    lands); ``hold`` after its first fill the position is closed by a taker
    order, and a fill that lands after that is closed the same way.
``maker``
    A quoter of :mod:`trading_research.market_making.quoters` (S0 at the touch,
    or S1 skewed by inventory) is asked at every boundary; every quote is
    logged; a working order whose price or size differs is cancelled and the
    new one posted; resting orders fill by the same trade-through rule; a side
    whose fill could pass the hard limit (soft limit plus one clip) is not
    quoted. A model, if given, is scored and logged but does not quote.

One position at a time for ``taker`` and ``passive``: a decision while an order
is out or a position is held is counted as busy and dropped. Once the kill
switch latches, every resting order is cancelled and the position is closed by
a taker order. An exit with no print after it (the run ended) is priced at the
last print of its side at or before its arrival, else of either side, and
flagged ``no_print``.
"""

from __future__ import annotations

import hashlib
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.evaluate import decide as decide_rule
from trading_research.backtest.walk import walk_book
from trading_research.live.events import BookRow, Print
from trading_research.live.ledger import PaperLedger
from trading_research.market_making.orders import Order
from trading_research.market_making.queue import ArrivalGrowth, FillRule, on_print_rule
from trading_research.market_making.quoters import MarketView, SkewQuoter, TouchQuoter
from trading_research.models.base import CLASSES, AlwaysHold

MODES = ("taker", "passive", "maker")
NAMED_MODELS = ("always_hold",)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class LoadedModel:
    """A fitted model, the features it reads, and the hash of the file it came from."""

    name: str
    estimator: Any
    features: list[str]
    sha256: str
    path: str | None

    def proba(self, values: pd.Series) -> np.ndarray | None:
        """Class probabilities for one boundary, or None when a feature is missing."""
        if not self.features:
            return np.asarray(self.estimator.predict_proba(pd.DataFrame(index=[0])), dtype=float)[0]
        row = values.reindex(self.features)
        if row.isna().any():
            return None
        frame = pd.DataFrame([row.to_numpy(dtype="float64")], columns=self.features)
        return np.asarray(self.estimator.predict_proba(frame), dtype=float)[0]


def load_model(spec: str | Path) -> LoadedModel:
    """``spec`` is a directory holding ``model.pkl``, a ``.pkl`` file, or a named rule.

    A pickle runs code when it is read; load only a model you trained.
    """
    text = str(spec)
    if text in NAMED_MODELS:
        model = AlwaysHold().fit(pd.DataFrame(), pd.Series(dtype=float))
        return LoadedModel(text, model, [], hashlib.sha256(text.encode()).hexdigest(), None)
    path = Path(text)
    if path.is_dir():
        path = path / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"no model at {spec}; give train's output directory or {NAMED_MODELS}"
        )
    payload_bytes = path.read_bytes()
    payload = pickle.loads(payload_bytes)
    if not isinstance(payload, dict) or "model" not in payload or "features" not in payload:
        raise ValueError(f"{path} is not a model written by train ({{'model', 'features'}})")
    estimator = payload["model"]
    name = getattr(estimator, "name", type(estimator).__name__)
    return LoadedModel(
        str(name),
        estimator,
        list(payload["features"]),
        hashlib.sha256(payload_bytes).hexdigest(),
        str(path),
    )


def decide(proba: np.ndarray, min_confidence: float) -> int:
    """The backtest's rule on one boundary's probabilities."""
    return int(decide_rule(proba.reshape(1, len(CLASSES)), min_confidence=min_confidence)[0])


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionSettings:
    """What the paper orders depend on besides the data. Defaults are the backtests'."""

    mode: str = "taker"
    clip: float = 0.001
    latency_ns: int = 10_000_000
    cancel_latency_ns: int = 10_000_000
    hold_ns: int = 120_000_000_000
    quoter: str = "touch"
    skew_bp: float = 5.0
    k: float = 0.5
    min_edge_bp: float = 0.0
    sigma_ref: float = 5.0
    soft_limit_clips: float = 6.0
    #: The maker fee S1's gate must clear: the run's own, set from its schedule.
    maker_bp: float = 0.0
    vol_half_life_s: float = 60.0
    touch_window_s: int = 3600

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"execution must be one of {MODES}, got {self.mode!r}")
        if not self.clip > 0 or self.hold_ns <= 0 or self.latency_ns < 0:
            raise ValueError("the clip and the hold must be positive and the latency non-negative")
        if self.quoter not in ("touch", "skew"):
            raise ValueError("the quoter is 'touch' (S0) or 'skew' (S1)")


@dataclass
class PaperOrder:
    oid: int
    kind: str  # "taker" or "limit"
    side: int
    size: float
    decided_ns: int
    arrival_ns: int
    reason: str
    price: float = float("nan")
    remaining: float = 0.0
    cancel_ns: int | None = None
    done_ns: int | None = None
    outcome: str = "open"
    filled: float = 0.0
    fill_price: float = float("nan")
    path: str = ""
    book_walk_price: float = float("nan")
    sweep_ts: int | None = None
    sweep: list[tuple[float, float]] = field(default_factory=list)
    queue: Order | None = None

    @property
    def done(self) -> bool:
        return self.done_ns is not None

    def row(self) -> tuple[Any, ...]:
        return (
            self.oid,
            self.kind,
            self.side,
            self.size,
            self.price,
            self.decided_ns,
            self.arrival_ns,
            self.cancel_ns,
            self.done_ns,
            self.outcome,
            self.filled,
            self.fill_price,
            self.path,
            self.book_walk_price,
            self.reason,
        )


ORDER_COLUMNS = (
    "oid",
    "kind",
    "side",
    "size",
    "price",
    "decided_ns",
    "arrival_ns",
    "cancel_ns",
    "done_ns",
    "outcome",
    "filled",
    "fill_price",
    "path",
    "book_walk_price",
    "reason",
)


@dataclass
class _Seconds:
    """``vol_bp_1m`` and the trailing median touch the simulator gives a quoter, per second."""

    half_life_s: float
    window_s: int
    mids: deque[float] = field(default_factory=deque)
    touch: deque[float] = field(default_factory=deque)
    last_second: int | None = None
    previous: BookRow | None = None
    vol: float = float("nan")
    median: float = float("nan")

    def update(self, row: BookRow) -> None:
        second = row.ts_ns // 1_000_000_000
        if self.last_second is None:
            self.last_second = second - (0 if row.ts_ns % 1_000_000_000 == 0 else 1)
        while self.last_second < second:
            self.last_second += 1
            exact = self.last_second == second and row.ts_ns % 1_000_000_000 == 0
            sample = row if exact else self.previous
            self._push(sample)
        self.previous = row

    def _push(self, sample: BookRow | None) -> None:
        if sample is None:
            self.mids.append(float("nan"))
            self.touch.append(float("nan"))
        else:
            self.mids.append(sample.mid)
            self.touch.append(0.5 * (sample.bid_sz[0] + sample.ask_sz[0]))
        while len(self.mids) > self.window_s + 1:
            self.mids.popleft()
            self.touch.popleft()
        mid = np.asarray(self.mids, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(np.log(mid), prepend=np.nan)
        variance = (
            pd.Series(returns**2)
            .ewm(halflife=self.half_life_s, adjust=False, ignore_na=True)
            .mean()
            .to_numpy()
        )
        self.vol = float(np.sqrt(variance[-1] * 60.0) * 1e4)
        touch = np.asarray(self.touch, dtype=float)
        touch = touch[np.isfinite(touch)]
        self.median = float(np.median(touch[-self.window_s :])) if len(touch) else float("nan")


@dataclass
class Execution:
    """One paper account's orders under one policy."""

    settings: ExecutionSettings
    ledger: PaperLedger
    tick: float
    orders: list[PaperOrder] = field(default_factory=list)
    decisions: list[tuple[Any, ...]] = field(default_factory=list)
    quotes: list[tuple[Any, ...]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    phase: str = "flat"
    entry_oid: int | None = None
    entry_fill_ns: int | None = None
    post_ns: int | None = None
    last_print: dict[int, tuple[int, float]] = field(default_factory=dict)
    last_row: BookRow | None = None
    flattening: bool = False

    def __post_init__(self) -> None:
        self._next = 1
        self._open: list[PaperOrder] = []
        self._entry: PaperOrder | None = None
        self._need_walk: list[PaperOrder] = []
        self.quoter: TouchQuoter | SkewQuoter | None = None
        if self.settings.mode == "maker":
            self.quoter = (
                TouchQuoter()
                if self.settings.quoter == "touch"
                else SkewQuoter(
                    self.settings.skew_bp,
                    self.settings.k,
                    self.settings.min_edge_bp,
                    self.settings.sigma_ref,
                    self.settings.maker_bp,
                )
            )
        self.seconds = _Seconds(self.settings.vol_half_life_s, self.settings.touch_window_s)

    # -- bookkeeping ---------------------------------------------------------

    def _count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def _ticks(self, price: float) -> int:
        return round(price / self.tick)

    def working(self) -> list[PaperOrder]:
        self._open = [o for o in self._open if not o.done]
        return list(self._open)

    def _new(self, kind: str, side: int, size: float, decided: int, arrival: int, reason: str,
             price: float = float("nan")) -> PaperOrder:  # fmt: skip
        order = PaperOrder(self._next, kind, side, size, decided, arrival, reason, price, size)
        self._next += 1
        if kind == "limit":
            order.queue = Order(order.oid, side, self._ticks(price), size, size, decided, arrival)
        self.orders.append(order)
        self._open.append(order)
        if kind == "taker":
            self._need_walk.append(order)
        self._count(f"orders_{kind}")
        return order

    # -- events --------------------------------------------------------------

    def on_print(self, p: Print) -> None:
        self._close_sweeps(p.ts_ns)
        if not p.rpi:
            for order in self.working():
                if order.kind == "taker":
                    if order.arrival_ns <= p.ts_ns and p.aggressor == order.side:
                        if order.sweep_ts is None:
                            order.sweep_ts = p.ts_ns
                        if order.sweep_ts == p.ts_ns:
                            order.sweep.append((p.price, p.size))
                elif self._resting(order, p.ts_ns):
                    self._through(order, p)
            self.last_print[p.aggressor] = (p.ts_ns, p.price)

    def _resting(self, order: PaperOrder, ts: int) -> bool:
        if order.arrival_ns > ts:
            return False
        return order.cancel_ns is None or ts < order.cancel_ns

    def _through(self, order: PaperOrder, p: Print) -> None:
        assert order.queue is not None
        filled, path = on_print_rule(
            order.queue,
            self._ticks(p.price),
            p.size,
            p.aggressor,
            rule=FillRule.THROUGH,
            ts=p.ts_ns,
            growth=ArrivalGrowth.PRO_RATA_TIME,
        )
        if filled <= 0:
            return
        self.ledger.fill(
            p.ts_ns, order.side, order.price, filled, maker=True, oid=order.oid,
            reason=order.reason, path=path,
        )  # fmt: skip
        order.filled += filled
        order.remaining = order.queue.remaining
        order.fill_price = order.price
        order.path = path
        if order.remaining <= 1e-12:
            order.done_ns, order.outcome = p.ts_ns, "filled"
        self._after_fill(order, p.ts_ns)

    def _close_sweeps(self, now: int) -> None:
        """Fill every taker order whose sweep stamp is behind ``now``."""
        for order in self.working():
            if order.kind == "taker" and order.sweep_ts is not None and order.sweep_ts < now:
                self._fill_sweep(order)

    def _fill_sweep(self, order: PaperOrder) -> None:
        remaining, cost, worst = order.size, 0.0, None
        for price, size in order.sweep:
            take = min(remaining, size)
            cost += take * price
            remaining -= take
            worst = (
                price
                if worst is None
                else (max(worst, price) if order.side > 0 else min(worst, price))
            )
            if remaining <= 1e-12:
                break
        assert worst is not None
        path = "print"
        if remaining > 1e-9 * order.size:
            cost += remaining * worst
            path = "print_beyond_sweep"
        price = cost / order.size
        self._taker_fill(order, order.sweep_ts or order.arrival_ns, price, path)

    def _taker_fill(self, order: PaperOrder, ts: int, price: float, path: str) -> None:
        self.ledger.fill(
            ts, order.side, price, order.size, maker=False, oid=order.oid,
            reason=order.reason, path=path,
        )  # fmt: skip
        order.filled, order.remaining, order.fill_price, order.path = order.size, 0.0, price, path
        order.done_ns, order.outcome = ts, "filled"
        self._after_fill(order, ts)

    def _after_fill(self, order: PaperOrder, ts: int) -> None:
        if order.reason == "entry" and self.entry_fill_ns is None:
            self.entry_fill_ns = ts
            self.phase = "holding"

    def on_row(self, row: BookRow, delay_ns: int = 0) -> None:
        """Advance to a new row: close sweeps, land cancels, run the timers, quote."""
        self._close_sweeps(row.ts_ns)
        waiting = []
        for order in self._need_walk:
            if order.arrival_ns > row.ts_ns:
                waiting.append(order)
                continue
            px, sz = (row.ask_px, row.ask_sz) if order.side > 0 else (row.bid_px, row.bid_sz)
            walk = walk_book(np.asarray(px), np.asarray(sz), order.size)
            order.book_walk_price = float(walk.vwap[0])
        self._need_walk = waiting
        for order in self.working():
            if (
                order.kind == "limit"
                and order.cancel_ns is not None
                and order.cancel_ns <= row.ts_ns
            ):
                order.done_ns = order.cancel_ns
                order.outcome = "partial" if order.filled > 0 else "cancelled"
        self.last_row = row
        self.seconds.update(row)
        if self.ledger.killed is not None or self.flattening:
            self._flatten(row, delay_ns, self.ledger.killed or "flatten")
            return
        if self.settings.mode == "maker":
            self._quote(row, delay_ns)
        else:
            self._timers(row, delay_ns)

    def _send_exit(self, row: BookRow, delay_ns: int, reason: str) -> None:
        position = self.ledger.account.position
        if position == 0 or any(o.kind == "taker" for o in self.working()):
            return
        side = -1 if position > 0 else 1
        self._new(
            "taker",
            side,
            abs(position),
            row.ts_ns,
            row.ts_ns + delay_ns + self.settings.latency_ns,
            reason,
        )

    def _cancel(self, order: PaperOrder, row: BookRow) -> None:
        if order.kind == "limit" and order.cancel_ns is None and not order.done:
            order.cancel_ns = row.ts_ns + self.settings.cancel_latency_ns

    def _reset_entry(self) -> None:
        self.phase, self.entry_oid, self.entry_fill_ns, self.post_ns = "flat", None, None, None
        self._entry = None

    def _timers(self, row: BookRow, delay_ns: int) -> None:
        hold = self.settings.hold_ns
        entry = self._entry
        if self.phase == "entering" and entry is not None:
            if entry.kind == "taker" and not entry.done and row.ts_ns >= entry.arrival_ns + hold:
                entry.done_ns, entry.outcome = row.ts_ns, "no_print"
                self._count("entries_no_print")
                self._reset_entry()
            elif (
                entry.kind == "limit"
                and self.post_ns is not None
                and row.ts_ns >= self.post_ns + hold
            ):
                self._cancel(entry, row)
                if entry.done and entry.filled == 0:
                    self._reset_entry()
        if self.phase in ("holding", "exiting") and self.entry_fill_ns is not None:
            if row.ts_ns >= self.entry_fill_ns + hold:
                if entry is not None:
                    self._cancel(entry, row)
                self.phase = "exiting"
                self._send_exit(row, delay_ns, "exit")
            entry_done = entry is None or entry.done
            if (
                self.phase == "exiting"
                and entry_done
                and self.ledger.account.position == 0
                and not self.working()
            ):
                self._reset_entry()

    def on_decision(self, side: int, row: BookRow, delay_ns: int = 0) -> str:
        """Act on a decision at ``row``; returns what happened to it."""
        if side == 0:
            return "no_signal"
        if self.settings.mode == "maker":
            return "logged"
        if self.ledger.killed is not None:
            return "killed"
        if self.phase != "flat" or self.working():
            self._count("decisions_busy")
            return "busy"
        clip = self.settings.clip
        if not self.ledger.allows(side, clip):
            return "refused"
        arrival = row.ts_ns + delay_ns + self.settings.latency_ns
        if self.settings.mode == "taker":
            order = self._new("taker", side, clip, row.ts_ns, arrival, "entry")
        else:
            price = row.bid_px[0] if side > 0 else row.ask_px[0]
            order = self._new("limit", side, clip, row.ts_ns, arrival, "entry", price)
            self.post_ns = row.ts_ns
        self.entry_oid, self.entry_fill_ns, self.phase = order.oid, None, "entering"
        self._entry = order
        return "entered"

    # -- the market maker ------------------------------------------------------

    def _view(self, row: BookRow) -> MarketView:
        tick = self.tick
        bid_px = np.asarray([self._ticks(p) for p in row.bid_px], dtype=np.int64)
        ask_px = np.asarray([self._ticks(p) for p in row.ask_px], dtype=np.int64)
        for array in (bid_px, ask_px):
            array.setflags(write=False)
        clip = self.settings.clip
        return MarketView(
            row.ts_ns, tick, bid_px, np.asarray(row.bid_sz), ask_px, np.asarray(row.ask_sz),
            int(bid_px[0]), int(ask_px[0]), row.mid, self.seconds.vol, self.seconds.median,
            clip, clip * self.settings.soft_limit_clips, {},
        )  # fmt: skip

    def _quote(self, row: BookRow, delay_ns: int) -> None:
        assert self.quoter is not None
        position = self.ledger.account.position
        quotes = self.quoter.quotes(self._view(row), position)
        hard = self.settings.clip * (self.settings.soft_limit_clips + 1.0)
        logged: list[Any] = [row.ts_ns, position]
        for side, quote in ((1, quotes.bid), (-1, quotes.ask)):
            price = None if quote.price is None else round(quote.price * self.tick, 10)
            size = quote.size
            if price is not None and abs(position + side * size) > hard + 1e-12:
                price = None
            if price is not None and not self.ledger.allows(side, size):
                price = None
            logged += [price if price is not None else float("nan"), size if price else 0.0]
            current = [
                o
                for o in self.working()
                if o.kind == "limit" and o.side == side and o.cancel_ns is None
            ]
            keep = [
                o
                for o in current
                if price is not None
                and abs(o.price - price) < self.tick / 2
                and abs(o.remaining - size) < 1e-12
            ]
            for order in current:
                if order not in keep:
                    self._cancel(order, row)
            if price is not None and not keep:
                arrival = row.ts_ns + delay_ns + self.settings.latency_ns
                self._new("limit", side, size, row.ts_ns, arrival, "quote", price)
        self.quotes.append(tuple(logged))

    # -- flattening ------------------------------------------------------------

    def _flatten(self, row: BookRow, delay_ns: int, reason: str) -> None:
        for order in self.working():
            if order.kind == "limit":
                self._cancel(order, row)
            elif order.reason == "entry":
                order.done_ns, order.outcome = row.ts_ns, "cancelled"
        self._send_exit(row, delay_ns, f"flatten:{reason}")
        self.phase = "flattening"

    def flatten_now(self) -> None:
        """Ask for a flatten at the next row (the run is ending)."""
        self.flattening = True

    def finish(self, ts_ns: int) -> None:
        """End of the run: cancel what rests, price what is still open without a print."""
        for order in self.working():
            if order.kind == "limit":
                order.done_ns = max(ts_ns, order.cancel_ns or ts_ns)
                order.outcome = "partial" if order.filled > 0 else "cancelled"
            elif order.sweep_ts is not None:
                self._fill_sweep(order)
        if self.last_row is not None:
            self._send_exit(self.last_row, 0, "flatten:session_end")
        for order in self.working():
            if order.kind == "taker":
                if order.reason == "entry":
                    order.done_ns, order.outcome = ts_ns, "unfilled"
                    continue
                self._fallback(order, ts_ns)

    def _fallback(self, order: PaperOrder, ts_ns: int) -> None:
        own = self.last_print.get(order.side)
        either = max(self.last_print.values(), default=None)
        if own is not None:
            price, path = own[1], "no_print_last_same_side"
        elif either is not None:
            price, path = either[1], "no_print_last_either_side"
        elif self.last_row is not None:
            price, path = self.last_row.mid, "no_print_last_mid"
        else:
            return
        self._taker_fill(order, max(ts_ns, order.arrival_ns), price, path)
        self._count("exits_no_print")

    # -- tables ----------------------------------------------------------------

    def orders_frame(self) -> pd.DataFrame:
        return pd.DataFrame([o.row() for o in self.orders], columns=list(ORDER_COLUMNS))

    def quotes_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.quotes, columns=["ts_ns", "position", "bid", "bid_size", "ask", "ask_size"]
        )

    def fill_rate(self) -> dict[str, float]:
        frame = self.orders_frame()
        out: dict[str, float] = {}
        for kind in ("taker", "limit"):
            mine = frame[frame["kind"] == kind]
            if len(mine):
                out[f"{kind}_orders"] = float(len(mine))
                out[f"{kind}_with_fill"] = float((mine["filled"] > 0).sum())
                out[f"{kind}_fill_rate"] = float((mine["filled"] > 0).mean())
        return out
