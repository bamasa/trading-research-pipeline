"""The market-making simulator, run on planted order flow and read against the truth.

Two measurements, each on every day of an :class:`OrderFlowDataset`:

* **Adverse selection.** The repository's touch quoter (S0) quotes both sides
  through the simulator. Each passive fill's five-second markout (the
  simulator's ``markout_5s``) is the half-spread it captured against the true
  mid at the fill, less what the mid then moved against it; that second part,
  ``side * (true_mid(t) - price) / price * 1e4 - markout_5s``, is the recovered
  adverse selection, averaged by volume and compared with the planted
  ``pi * move_ticks * tick / price * 1e4`` at the same fills. The simulator's
  own split (R18) measures the captured spread from the last snapshot's mid,
  up to 100 ms stale, so its adverse part also carries any move between that
  snapshot and the fill; it is reported beside, as the simulator's documents
  say it should be read: indicative.
* **Fill rate.** :class:`ProbeQuoter` posts one lot at each touch, holds it at
  its price for ``hold_s`` and then cancels it, whatever happens to the market,
  so that every order's life is fixed in advance. The share of these orders the
  simulator fills, under each cancellation bracket, is compared with the share
  :func:`~trading_research.synthetic.order_flow.true_fills` gives for the very
  same orders on the order log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making.orders import OwnOrder
from trading_research.market_making.quoters import NO_QUOTE, MarketView, Quote, Quotes
from trading_research.synthetic.common import NS_PER_S
from trading_research.synthetic.order_flow import OrderFlowDataset, OrderFlowDay, true_fills


@dataclass
class ProbeQuoter:
    """One lot at each touch, held at its price for ``hold_s``, then cancelled.

    It reads its own resting orders through the simulator's ``observe`` hook, so
    a filled order is replaced only after ``gap_s``. It never chases the touch:
    an order's whole life is decided when it is posted.
    """

    hold_s: float = 3.0
    gap_s: float = 0.5
    size: float = 1.0
    name: str = "probe"
    uses_future: bool = False
    _posted: dict[int, tuple[int, int] | None] = field(default_factory=lambda: {1: None, -1: None})
    _free_at: dict[int, int] = field(default_factory=lambda: {1: 0, -1: 0})
    _resting: set[tuple[int, int]] = field(default_factory=set)

    def observe(self, own: tuple[OwnOrder, ...], opened_ns: int) -> None:  # noqa: ARG002
        self._resting = {(o.side, o.price) for o in own if not o.cancelling}

    def quotes(self, view: MarketView, position: float) -> Quotes:  # noqa: ARG002
        hold = int(self.hold_s * NS_PER_S)
        gap = int(self.gap_s * NS_PER_S)
        wanted: dict[int, Quote] = {}
        for side, touch in ((1, view.best_bid), (-1, view.best_ask)):
            posted = self._posted[side]
            if posted is not None:
                price, at = posted
                gone = (side, price) not in self._resting and view.ts > at
                if gone or view.ts - at >= hold:
                    self._posted[side] = None
                    self._free_at[side] = view.ts + gap
                    wanted[side] = NO_QUOTE
                    continue
                wanted[side] = Quote(price, self.size)
                continue
            if view.ts >= self._free_at[side]:
                self._posted[side] = (int(touch), view.ts)
                wanted[side] = Quote(int(touch), self.size)
            else:
                wanted[side] = NO_QUOTE
        return Quotes(wanted[1], wanted[-1])


def sim_config(dataset: OrderFlowDataset, **changes: Any) -> Any:
    """The simulator's settings for a session: warm-up, stop and flatten inside it."""
    from trading_research.market_making.simulator import SimConfig

    end = dataset.spec.session_s

    def clock(seconds: float) -> time:
        return time(int(seconds // 3600), int(seconds % 3600 // 60), int(seconds % 60))

    base: dict[str, Any] = {
        "clip_btc": 1.0,
        "warmup_s": 120.0,
        "stop_quoting_at": clock(end - 60),
        "flatten_at": clock(end - 30),
        "flatten_slippage_bp": 0.0,
    }
    base.update(changes)
    return SimConfig(**base)


def adverse_fills(dataset: OrderFlowDataset, day: OrderFlowDay) -> pd.DataFrame:
    """S0's passive fills on one day with their adverse markout and the planted one."""
    from trading_research.market_making.quoters import TouchQuoter
    from trading_research.market_making.simulator import simulate_day

    events = dataset.events(day)
    result = simulate_day(events, TouchQuoter(), sim_config(dataset))  # type: ignore[arg-type]
    fills = result.fills[result.fills["maker"]].copy()
    if fills.empty:
        return fills
    mid = (events.bid_px[:, 0] + events.ask_px[:, 0]) * 0.5 * events.spec.tick  # type: ignore[attr-defined]
    from trading_research.market_making.analysis import mid_at

    ts = fills["ts"].to_numpy(dtype=np.int64)
    later = mid_at(events.book_ts, mid, ts + 5 * NS_PER_S)  # type: ignore[attr-defined]
    side = fills["side"].to_numpy(dtype=np.float64)
    price = fills["price"].to_numpy(dtype=np.float64)
    spec = dataset.spec
    true_mid = day.true_mid(ts, spec.spread_ticks, spec.tick)
    fills["captured_true_bp"] = side * (true_mid - price) / price * 1e4
    fills["adverse_bp"] = fills["captured_true_bp"] - fills["markout_5s"]
    fills["adverse_split_bp"] = -side * (later - fills["mid_ref"].to_numpy()) / price * 1e4
    fills["planted_bp"] = spec.informed_share * spec.move_ticks * spec.tick / price * 1e4
    fills["day"] = day.day.isoformat()
    return fills.dropna(subset=["adverse_bp"])


def probe_orders(
    dataset: OrderFlowDataset,
    day: OrderFlowDay,
    *,
    attribution: str = "proportional",
    hold_s: float = 3.0,
) -> pd.DataFrame:
    """The probe's orders on one day: the simulator's fills beside the true ones."""
    from trading_research.market_making.queue import CancelAttribution
    from trading_research.market_making.simulator import simulate_day

    events = dataset.events(day)
    config = sim_config(
        dataset, soft_limit_clips=1e9, cancel_attribution=CancelAttribution(attribution)
    )
    quoter = ProbeQuoter(hold_s=hold_s)
    result = simulate_day(events, quoter, config)  # type: ignore[arg-type]
    orders = result.orders
    orders = orders[orders["outcome"] != "rejected"].copy()
    if orders.empty:
        return orders
    book_ts = events.book_ts  # type: ignore[attr-defined]
    hold = int(hold_s * NS_PER_S)
    decided = orders["decided_ns"].to_numpy(dtype=np.int64)
    cancel_row = np.searchsorted(book_ts, decided + hold, side="left")
    intended = np.where(
        cancel_row < len(book_ts),
        book_ts[np.minimum(cancel_row, len(book_ts) - 1)] + config.cancel_latency_ns,
        np.iinfo(np.int64).max,
    )
    done = orders["done_ns"].to_numpy(dtype=np.int64)
    cancelled = orders["outcome"].to_numpy() == "cancelled"
    orders["end_ns"] = np.where(cancelled, np.minimum(done, intended), intended)
    lots = orders.assign(size=orders["size"] / dataset.spec.lot)
    truth = true_fills(day.log, lots[["side", "price", "size", "live_ns", "end_ns"]])
    orders["true_filled"] = truth["filled"].to_numpy() * dataset.spec.lot
    orders["sim_hit"] = orders["filled"] > 0
    orders["true_hit"] = orders["true_filled"] > 0
    orders["day"] = day.day.isoformat()
    orders["attribution"] = attribution
    return orders


def share(values: Sequence[bool] | np.ndarray | pd.Series) -> tuple[float, float]:
    """A proportion and its binomial standard error."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    p = float(array.mean())
    return p, float(np.sqrt(max(p * (1.0 - p), 1e-12) / len(array)))
