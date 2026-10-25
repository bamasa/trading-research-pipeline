"""The print rule's stand-in: the side-correct fill, its sweep and its four brackets."""

from __future__ import annotations

import numpy as np
import pytest

from test_spot_prints_fixture import CLIP, DAY, make_prints
from trading_research.backtest.walk import walk_book
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.spot.fills_standin import RULES, DayPrints, fill, sweeps

START = day_start_ns(DAY)
FLATTEN = START + (23 * 3600 + 59 * 60) * NS_PER_S
END = START + 86_400 * NS_PER_S


def arrivals(prints: DayPrints, count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = rng.choice(prints.ts, count)
    return np.sort(picks + rng.integers(-300, 300, count) * 1_000_000)


@pytest.mark.parametrize("side", [1, -1])
def test_an_order_fills_at_the_first_reachable_print_of_its_side(side: int) -> None:
    prints = make_prints(1)
    sw = sweeps(prints, side, CLIP, window_end_ns=FLATTEN)
    arrival = arrivals(prints, 400)
    got = fill(sw, arrival, until_ns=END, decision_ns=arrival - 10_000_000)
    for a, found, pos, stamp in zip(arrival, got.found, got.position, got.print_ns, strict=True):
        ok = (prints.ts >= a) & (prints.side == side) & prints.reachable
        if not ok.any():
            assert not found and pos == -1
            continue
        want = int(np.flatnonzero(ok)[0])
        assert found and pos == want and stamp == prints.ts[want]
    assert got.found.sum() > 300


@pytest.mark.parametrize("side", [1, -1])
def test_the_sweep_is_walk_book_over_the_stamps_prints(side: int) -> None:
    prints = make_prints(2)
    sw = sweeps(prints, side, CLIP, window_end_ns=FLATTEN)
    got = fill(sw, arrivals(prints, 300, 2), until_ns=END, decision_ns=START)
    for pos, price, beyond in zip(got.position, got.price, got.beyond, strict=True):
        if pos < 0:
            continue
        same = (prints.ts == prints.ts[pos]) & (prints.side == side) & prints.reachable
        walked = walk_book(prints.price[same], prints.size[same], CLIP)
        assert (price, beyond) == (walked.vwap[0], walked.beyond[0])
    assert (got.beyond > 0).any() and (got.beyond == 0).any()


def test_a_fill_must_come_before_its_bound() -> None:
    prints = make_prints(3)
    sw = sweeps(prints, 1, CLIP, window_end_ns=FLATTEN)
    late = np.array([FLATTEN - 30 * NS_PER_S])
    assert fill(sw, late, until_ns=END, decision_ns=late).found[0]
    first = fill(sw, late, until_ns=END, decision_ns=late).print_ns[0]
    assert not fill(sw, late, until_ns=int(first), decision_ns=late).found[0]
    assert not fill(sw, np.array([END]), until_ns=END + NS_PER_S, decision_ns=late).found[0]


@pytest.mark.parametrize("side", [1, -1])
def test_the_brackets_price_the_same_print(side: int) -> None:
    prints = make_prints(4)
    sw = sweeps(prints, side, CLIP, window_end_ns=FLATTEN)
    arrival = arrivals(prints, 300, 4)
    decision = arrival - 10_000_000
    by = {r: fill(sw, arrival, until_ns=END, decision_ns=decision, rule=r) for r in RULES}
    for r in RULES:
        np.testing.assert_array_equal(by[r].print_ns, by["rule"].print_ns)
    found = by["rule"].found
    for k in np.flatnonzero(found)[:120]:
        pos, stamp = int(by["rule"].position[k]), int(by["rule"].print_ns[k])
        assert by["optimistic"].price[k] == prints.price[pos]
        mine = (prints.side == side) & prints.reachable
        window = mine & (prints.ts > stamp) & (prints.ts <= stamp + 60 * NS_PER_S)
        window &= prints.ts < FLATTEN
        same = mine & (prints.ts == stamp)
        walked = walk_book(prints.price[same], prints.size[same], CLIP)
        left = walked.beyond[0]
        worst = (prints.price[window].max() if side > 0 else prints.price[window].min()) if (
            window.any()) else prints.price[same][-1]  # fmt: skip
        taken = walked.vwap[0] * CLIP - left * prints.price[same][-1]
        np.testing.assert_allclose(by["pessimistic_depth"].price[k], (taken + left * worst) / CLIP,
                                   rtol=1e-12)  # fmt: skip
        before = mine & (prints.ts <= decision[k])
        stale = prints.price[np.flatnonzero(before)[-1]] if before.any() else by["rule"].price[k]
        assert by["stale"].price[k] == stale
        pick = max if side > 0 else min
        assert by["worse_of_two"].price[k] == pick(by["pessimistic_depth"].price[k], stale)
    with pytest.raises(ValueError, match="unknown fill rule"):
        fill(sw, arrival, until_ns=END, decision_ns=decision, rule="mid")


def test_one_order_alone_fills_as_it_does_among_many() -> None:
    prints = make_prints(5)
    for side in (1, -1):
        sw = sweeps(prints, side, CLIP, window_end_ns=FLATTEN)
        arrival = arrivals(prints, 200, 5)
        for rule in RULES:
            many = fill(sw, arrival, until_ns=END, decision_ns=arrival, rule=rule)
            for k in range(0, 200, 17):
                one = fill(sw, arrival[k : k + 1], until_ns=END, decision_ns=arrival[k], rule=rule)
                for a, b in zip(one, many, strict=True):
                    np.testing.assert_array_equal(a, b[k : k + 1])


def test_a_side_without_prints_never_fills() -> None:
    prints = make_prints(6)
    only_buys = DayPrints(DAY, prints.ts, prints.price, prints.size,
                          np.ones_like(prints.side), prints.reachable)  # fmt: skip
    sw = sweeps(only_buys, -1, CLIP, window_end_ns=FLATTEN)
    got = fill(sw, arrivals(prints, 10), until_ns=END, decision_ns=START)
    assert not got.found.any() and (got.position == -1).all() and np.isnan(got.price).all()
    with pytest.raises(ValueError):
        sweeps(prints, 0, CLIP, window_end_ns=FLATTEN)
