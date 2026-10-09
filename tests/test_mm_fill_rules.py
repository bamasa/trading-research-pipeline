"""The three fill rules: on touch, trade-through, and the queue."""

from __future__ import annotations

from trading_research.market_making.orders import Order
from trading_research.market_making.queue import (
    THROUGH,
    TOUCH,
    ArrivalGrowth,
    FillRule,
    on_print,
    on_print_rule,
)


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
