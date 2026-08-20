"""The passive fill model, whose whole value is in what it refuses to fill."""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.backtest.maker import (
    BUY,
    SELL,
    MakerCosts,
    PostingRules,
    score,
    simulate,
)


def _book(n: int, bid: float = 99.0, ask: float = 101.0, size: float = 10.0):
    return (
        np.full(n, bid),
        np.full(n, ask),
        np.full(n, size),
        np.full(n, size),
    )


def _no_flow(n: int):
    return np.zeros(n), np.zeros(n)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_a_fully_passive_round_trip_pays_only_fees() -> None:
    costs = MakerCosts(fee_bp_per_side=2.0)
    assert costs.round_trip_bp(passive_exit=True, spread_bp=20.0) == 4.0


def test_a_crossing_exit_pays_the_spread_it_crosses() -> None:
    costs = MakerCosts(fee_bp_per_side=2.0, taker_fee_bp_per_side=5.5, slippage_bp=0.5)
    assert costs.round_trip_bp(passive_exit=False, spread_bp=3.0) == pytest.approx(11.0)


def test_passive_is_cheaper_than_taking_on_both_legs() -> None:
    """The premise of the whole module, asserted rather than assumed."""
    costs = MakerCosts()
    passive = costs.round_trip_bp(passive_exit=True)
    crossing = costs.round_trip_bp(passive_exit=False, spread_bp=4.0)
    assert passive < crossing


def test_negative_fees_are_refused() -> None:
    with pytest.raises(ValueError, match="fee_bp_per_side"):
        MakerCosts(fee_bp_per_side=-1.0)


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------


def test_an_order_does_not_fill_while_nobody_trades() -> None:
    n = 20
    bid, ask, bid_size, ask_size = _book(n)
    sells, buys = _no_flow(n)
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY

    trades = simulate(decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules())
    assert len(trades) == 1
    assert not trades[0].filled
    assert trades[0].exit_reason == "unfilled"


def test_the_queue_ahead_must_be_consumed_before_the_order_fills() -> None:
    """Ten lots rest ahead; two lots a row means five rows of waiting."""
    n = 20
    bid, ask, bid_size, ask_size = _book(n, size=10.0)
    sells, buys = _no_flow(n)
    sells[:] = 2.0
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY

    trades = simulate(decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules())
    assert trades[0].filled
    assert trades[0].wait_rows == 5
    assert trades[0].queue_ahead == 10.0


def test_posting_behind_the_touch_starts_with_an_empty_queue() -> None:
    n = 10
    bid, ask, bid_size, ask_size = _book(n, size=1000.0)
    sells, buys = _no_flow(n)
    sells[:] = 1.0
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY

    behind = PostingRules(join_touch=False)
    # Posting one tick behind means no visible size ahead in this model — but
    # the price is worse, and the touch is no longer where prints land.
    trades = simulate(decision, bid, ask, bid_size, ask_size, sells, buys, behind, tick=0.01)
    assert trades[0].queue_ahead == 0.0


def test_an_order_is_left_behind_when_the_market_moves_its_way() -> None:
    """The adverse-selection mechanism, in isolation.

    A buy is posted at 99. The bid immediately rises to 99.5 — the move the
    signal wanted — and every subsequent print happens up there. The order sits
    at 99 making no progress and times out unfilled, which is precisely why
    passive fills are drawn from the losing half.
    """
    n = 30
    bid, ask, bid_size, ask_size = _book(n)
    sells, buys = _no_flow(n)
    sells[:] = 100.0
    bid[1:] = 99.5
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY

    trades = simulate(decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules())
    assert not trades[0].filled


def test_a_touch_that_moves_through_the_price_fills_the_order() -> None:
    """The other half: price falls past 99, so everything resting there is taken."""
    n = 30
    bid, ask, bid_size, ask_size = _book(n)
    sells, buys = _no_flow(n)
    bid[3:] = 98.0
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY

    trades = simulate(decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules())
    assert trades[0].filled
    assert trades[0].fill_index == 3
    assert trades[0].fill_price == 99.0


def test_a_sell_is_the_mirror_of_a_buy() -> None:
    n = 20
    bid, ask, bid_size, ask_size = _book(n, size=6.0)
    sells, buys = _no_flow(n)
    buys[:] = 3.0
    decision = np.zeros(n, dtype=int)
    decision[0] = SELL

    trades = simulate(decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules())
    assert trades[0].filled
    assert trades[0].fill_price == 101.0
    assert trades[0].wait_rows == 2


def test_the_timeout_bounds_the_wait() -> None:
    n = 60
    bid, ask, bid_size, ask_size = _book(n, size=1000.0)
    sells, buys = _no_flow(n)
    sells[:] = 1.0
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY

    trades = simulate(
        decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules(timeout_rows=10)
    )
    assert not trades[0].filled


def test_only_one_order_is_live_at_a_time() -> None:
    """Signals during a live attempt are ignored, as everywhere else."""
    n = 40
    bid, ask, bid_size, ask_size = _book(n, size=2.0)
    sells, buys = _no_flow(n)
    sells[:] = 1.0
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY
    decision[1] = BUY
    decision[2] = BUY

    trades = simulate(
        decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules(hold_rows=20)
    )
    assert len(trades) == 1


def test_mismatched_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="bid_size"):
        simulate(
            np.zeros(10, dtype=int),
            np.ones(10),
            np.ones(10),
            np.ones(5),
            np.ones(10),
            np.zeros(10),
            np.zeros(10),
            PostingRules(),
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_misses_are_counted_in_the_per_attempt_figure() -> None:
    """A strategy that rarely fills is not the strategy that was signalled."""
    n = 200
    bid, ask, bid_size, ask_size = _book(n, size=4.0)
    sells, buys = _no_flow(n)
    sells[::10] = 5.0
    decision = np.zeros(n, dtype=int)
    decision[::20] = BUY

    trades = simulate(
        decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules(hold_rows=5)
    )
    result = score(trades, MakerCosts(), np.full(n, 2.0))
    assert result["attempts"] >= result["fills"]
    assert 0.0 <= result["fill_rate"] <= 1.0
    if result["fills"]:
        assert abs(result["net_per_attempt_bp"]) <= abs(result["net_per_fill_bp"]) + 1e-9


def test_a_short_that_falls_is_profitable_before_costs() -> None:
    """The sign convention, which a previous bug in this project inverted."""
    n = 30
    bid, ask, bid_size, ask_size = _book(n)
    sells, buys = _no_flow(n)
    buys[:] = 100.0
    decision = np.zeros(n, dtype=int)
    decision[0] = SELL
    # Filled short at 101, then the whole book drops ten per cent.
    bid[2:] = 90.0
    ask[2:] = 91.0

    trades = simulate(
        decision, bid, ask, bid_size, ask_size, sells, buys, PostingRules(hold_rows=5)
    )
    result = score(trades, MakerCosts(), np.full(n, 2.0))
    assert result["gross_per_fill_bp"] > 0


def test_no_trades_scores_without_raising() -> None:
    assert score([], MakerCosts(), np.zeros(5))["attempts"] == 0.0
