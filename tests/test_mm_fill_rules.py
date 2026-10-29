"""The three fill rules: on touch, trade-through, and the queue."""

from __future__ import annotations

from datetime import timedelta
from functools import partial

import numpy as np

from trading_research.market_making.fill_rules import compare_fill_rules
from trading_research.market_making.orders import Order
from trading_research.market_making.queue import (
    THROUGH,
    TOUCH,
    ArrivalGrowth,
    FillRule,
    on_print,
    on_print_rule,
)
from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
from trading_research.market_making.simulator import SimConfig
from trading_research.market_making.synthetic import SYNTHETIC_DAY, random_market


def _bid(queue_ahead: float = 5.0, size: float = 2.0) -> Order:
    order = Order(oid=1, side=1, price=100, size=size, remaining=size, decided_ns=0, live_ns=0)
    order.queue_ahead = queue_ahead
    order.level_at_last_snapshot = queue_ahead
    return order


def _print(order: Order, rule: FillRule, price: int, size: float, aggressor: int = -1):
    return on_print_rule(
        order, price, size, aggressor, rule=rule, ts=1, growth=ArrivalGrowth.PRO_RATA_TIME
    )


def test_on_touch_fills_in_full_at_the_first_print_at_its_price_whatever_is_ahead() -> None:
    order = _bid()
    assert _print(order, FillRule.TOUCH, 100, 0.1) == (2.0, TOUCH)
    assert order.remaining == 0.0
    assert _print(_bid(), FillRule.TOUCH, 99, 0.1) == (2.0, TOUCH)


def test_no_rule_fills_from_the_wrong_side_or_a_price_short_of_the_order() -> None:
    for rule in FillRule:
        assert _print(_bid(), rule, 100, 9.0, aggressor=1) == (0.0, "")
        assert _print(_bid(), rule, 101, 9.0) == (0.0, "")


def test_trade_through_fills_only_through_the_price_up_to_the_print() -> None:
    order = _bid(queue_ahead=0.0)
    assert _print(order, FillRule.THROUGH, 100, 9.0) == (0.0, "")
    assert _print(order, FillRule.THROUGH, 99, 0.5) == (0.5, THROUGH)
    assert order.remaining == 1.5


def test_the_queue_rule_is_on_print_itself() -> None:
    for price, size in ((100, 3.0), (100, 6.0), (99, 1.0)):
        ruled, plain = _bid(), _bid()
        got = _print(ruled, FillRule.QUEUE, price, size)
        expected = on_print(plain, price, size, -1, ts=1, growth=ArrivalGrowth.PRO_RATA_TIME)
        assert got == expected
        assert ruled.queue_ahead == plain.queue_ahead


def _signal_days(count: int):
    """Random days, each with a decision every 25 snapshots, and the quoter factory."""
    days, stamps, sides = [], [], []
    for seed in range(count):
        market = random_market(
            seed,
            n_snapshots=900,
            prints_per_snapshot=1.0,
            funding=True,
            day=SYNTHETIC_DAY + timedelta(seed),
        )
        days.append(market)
        chosen = market.book_ts[10:-100:25]
        stamps.append(chosen)
        sides.append(np.where(np.arange(len(chosen)) % 3 == 0, -1, 1))
    tapes = entry_tapes(np.concatenate(stamps), np.concatenate(sides))
    return days, tapes, partial(SignalEntryQuoter, 2.0)


CONFIG = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0)


def test_the_comparison_orders_the_rules_as_their_biases_say() -> None:
    days, tapes, make = _signal_days(3)
    comparison = compare_fill_rules(days, make, CONFIG, tapes)
    summary = comparison.summary.set_index("rule")
    ratio = summary["fill_ratio"]
    assert ratio["touch"] >= ratio["queue"] >= ratio["trade_through"]
    assert (summary["days"] == 3).all()
    assert summary.loc["touch", "bias"] == "optimistic bound"
    days_table = comparison.days[~comparison.days["excluded"].astype(bool)]
    filled = days_table.groupby("rule")["notional_filled"].sum()
    assert np.allclose(summary["notional_filled"], filled.reindex(summary.index))
    assert ((summary["notional_filled"] > 0) == (summary["volume_filled"] > 0)).all()
    paths = comparison.markouts.groupby("rule")["path"].apply(set).to_dict()
    assert paths == {
        "touch": {"touch"},
        "trade_through": {"through"},
        "queue": {"queue", "through"},
    }
    assert len(comparison.days) == 9


def test_prints_at_the_price_that_never_clear_the_queue_fill_only_on_touch() -> None:
    """One bid at the touch behind 10 lots; prints of 3 at its price only."""
    from trading_research.market_making.synthetic import hand_built_market

    book = ([(100 - i, 10.0) for i in range(3)], [(102 + i, 10.0) for i in range(3)])
    snapshots = [(100.0 * i, *book) for i in range(30)]
    market = hand_built_market(snapshots, [(450.0, 100, 3.0, -1)], funding=[(0.0, 1e-4)])
    start = market.day_start_ns
    tapes = entry_tapes(np.array([start]), np.array([1]))
    comparison = compare_fill_rules([market], partial(SignalEntryQuoter, 1.0), CONFIG, tapes)
    ratio = comparison.summary.set_index("rule")["fill_ratio"]
    assert ratio.to_dict() == {"touch": 1.0, "trade_through": 0.0, "queue": 0.0}
    assert comparison.prediction() == {
        "touch_fill_ratio": 1.0,
        "queue_fill_ratio": 0.0,
        "holds": True,
    }
