"""Latency: an order exists at the venue only after it arrives, and until a
cancel lands it can still be filled."""

from __future__ import annotations

from dataclasses import dataclass

from trading_research.market_making.events import day_start_ns
from trading_research.market_making.quoters import MarketView, Quote, Quotes
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import SYNTHETIC_DAY, hand_built_market

CONFIG = SimConfig(clip_notional=1e12, warmup_s=0.0)
START = day_start_ns(SYNTHETIC_DAY)


@dataclass(frozen=True)
class Scripted:
    """Bids the price of the last script entry at or before the snapshot.

    ``script`` is ((from milliseconds, bid price or None), ...).
    """

    script: tuple[tuple[float, int | None], ...]
    name: str = "scripted"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:
        elapsed = (view.ts - START) / 1e6
        price = None
        for since, bid in self.script:
            if since <= elapsed:
                price = bid
        return Quotes(Quote(price, view.clip), Quote(None, 0.0))


ALWAYS_100 = Scripted(((0.0, 100),))


def _book(best_bid: int, best_ask: int, size: float = 10.0):
    return (
        [(best_bid - i, size) for i in range(3)],
        [(best_ask + i, size) for i in range(3)],
    )


BOOK = _book(100, 102)


def _maker_fill_times(result) -> list[float]:
    return [(t - START) / 1e6 for t in result.fills.query("maker")["ts"]]


def test_an_order_cannot_fill_before_it_arrives() -> None:
    """Decided at 0, live at 10 ms: a sale through the bid at 5 ms misses it."""
    market = hand_built_market(
        [(0.0, *BOOK)],
        [(5.0, 99, 1.0, -1), (15.0, 99, 1.0, -1)],
    )
    result = simulate_day(market, ALWAYS_100, CONFIG)
    assert _maker_fill_times(result) == [15.0]
    assert result.orders["live_ns"].iloc[0] - START == 10_000_000


def test_an_order_arriving_at_a_prints_timestamp_misses_it() -> None:
    """Activations lose ties to prints."""
    market = hand_built_market(
        [(0.0, *BOOK)],
        [(10.0, 99, 1.0, -1), (10.001, 99, 1.0, -1)],
    )
    assert _maker_fill_times(simulate_day(market, ALWAYS_100, CONFIG)) == [10.001]


def test_a_cancel_in_flight_does_not_protect_the_order() -> None:
    """The quote is withdrawn at 100 ms; the cancel lands at 110 ms; a sale
    through the bid at 105 ms still fills it, one at 115 ms does not."""
    market = hand_built_market(
        [(0.0, *BOOK), (100.0, *BOOK)],
        [(105.0, 99, 0.5, -1), (115.0, 99, 0.5, -1)],
    )
    quoter = Scripted(((0.0, 100), (100.0, None)))
    result = simulate_day(market, quoter, CONFIG)
    assert _maker_fill_times(result) == [105.0]
    order = result.orders.iloc[0]
    assert order["cancel_ns"] - START == 110_000_000
    assert order["outcome"] == "cancelled"


def test_a_replace_loses_priority_and_a_kept_quote_keeps_it() -> None:
    """Six of the ten ahead trade at 50 ms. A kept quote is then under five from
    the front and fills on the next five; a quote moved away and back is a new
    order at the back of the queue, and does not."""
    after = _book(100, 102, size=10.0)
    prints = [(50.0, 100, 6.0, -1), (250.0, 100, 5.0, -1)]
    snapshots = [(0.0, *BOOK), (100.0, *after), (200.0, *after)]
    market = hand_built_market(snapshots, prints)

    kept = simulate_day(market, ALWAYS_100, CONFIG)
    assert _maker_fill_times(kept) == [250.0]
    assert len(kept.orders) == 1

    moved = simulate_day(market, Scripted(((0.0, 100), (100.0, 99), (200.0, 100))), CONFIG)
    assert moved.fills.query("maker").empty
    assert len(moved.orders) == 3
    assert moved.counters["replaced"] == 2


def test_post_only_rejects_an_order_that_would_cross_on_arrival() -> None:
    """Decided inside the spread at 0; by the time it arrives at 10 ms the offer
    has come down to its price."""
    lower_ask = _book(100, 101)
    market = hand_built_market(
        [(0.0, *BOOK), (5.0, *lower_ask)],
        [(50.0, 101, 1.0, 1)],
    )
    result = simulate_day(market, Scripted(((0.0, 101),)), CONFIG)
    assert result.counters["rejected"] == 1
    assert result.orders["outcome"].iloc[0] == "rejected"
    assert result.fills.query("maker").empty


def test_quotes_are_pulled_across_a_book_gap() -> None:
    """No snapshot for 5 s: everything is cancelled, so a sale at 6 s that would
    have filled the bid finds nothing."""
    market = hand_built_market(
        [(0.0, *BOOK), (100.0, *BOOK), (20_000.0, *BOOK)],
        [(6_000.0, 100, 20.0, -1)],
    )
    pulled = simulate_day(market, ALWAYS_100, CONFIG)
    assert pulled.fills.query("maker").empty
    assert pulled.counters["feed_pauses"] == 1
    assert pulled.orders["cancel_ns"].iloc[0] - START == 5_110_000_000

    patient = simulate_day(market, ALWAYS_100, CONFIG.with_(suspend_after_pause_s=30.0))
    assert _maker_fill_times(patient) == [6_000.0]


def test_feed_latency_delays_every_decision() -> None:
    market = hand_built_market(
        [(0.0, *BOOK)],
        [(55.0, 99, 1.0, -1), (65.0, 99, 1.0, -1)],
    )
    result = simulate_day(market, ALWAYS_100, CONFIG.with_(feed_latency_ns=50_000_000))
    assert result.orders["decided_ns"].iloc[0] - START == 50_000_000
    assert _maker_fill_times(result) == [65.0]


def test_a_cancel_faster_than_its_order_waits_for_it() -> None:
    """With a cancel latency shorter than the order latency, a cancel can
    overtake its order; the venue applies it when the order arrives."""
    market = hand_built_market(
        [(0.0, *BOOK), (1.0, *BOOK)],
        [(20.0, 99, 1.0, -1)],
    )
    config = CONFIG.with_(order_latency_ns=10_000_000, cancel_latency_ns=1_000_000)
    result = simulate_day(market, Scripted(((0.0, 100), (1.0, None))), config)
    order = result.orders.iloc[0]
    assert order["outcome"] == "cancelled"
    assert order["done_ns"] == order["live_ns"]
    assert result.fills.query("maker").empty


def test_an_order_arriving_at_a_snapshots_timestamp_joins_that_snapshots_queue() -> None:
    """Ties go to the market: the snapshot stamped with the arrival time is
    processed first. The 10 ms snapshot shows 30 at 100, so the order decided
    at 0 joins behind 30, not the 10 of the snapshot it was decided on, and a
    sale of 15 at 20 ms does not reach it."""
    market = hand_built_market(
        [
            (0.0, *BOOK),
            (10.0, [(100, 30.0), (99, 10.0), (98, 10.0)], BOOK[1]),
            (100.0, [(100, 15.0), (99, 10.0), (98, 10.0)], BOOK[1]),
        ],
        [(20.0, 100, 15.0, -1)],
    )
    result = simulate_day(market, ALWAYS_100, CONFIG.with_(clip_notional=10.0))
    assert result.orders["live_ns"].iloc[0] - START == 10_000_000
    assert result.fills.query("maker").empty
