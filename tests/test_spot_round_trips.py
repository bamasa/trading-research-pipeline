"""Print fills and round trips against a plain loop that reads the registration's words."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test_spot_synthetic_prints import DAY, make_prints, no_network
from trading_research.spot.fills import (
    BUY,
    SELL,
    PrintFillSettings,
    PrintTape,
    fill_orders,
    round_trips,
)
from trading_research.spot.grid import grid_stamps_ns

S = 1_000_000_000
SETTINGS = PrintFillSettings()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


def loop_fill(prints: pd.DataFrame, side: int, arrival: int, decision: int, until: int,
              not_before: int, fallback: bool, flatten: int) -> dict[str, float]:  # fmt: skip
    """One order, priced by reading the prints one at a time."""
    ts = prints["timestamp"].astype("int64").to_numpy()
    px, sz = prints["price"].to_numpy(), prints["size"].to_numpy()
    mine = (prints["aggressor"].to_numpy() == side) & ~prints["rpi"].to_numpy()
    clip, effective = SETTINGS.clip, max(arrival, not_before)
    earlier = [i for i in range(len(ts)) if mine[i] and ts[i] <= decision]
    stale = px[earlier[-1]] if earlier else np.nan  # none: the rule's price, set below
    later = [i for i in range(len(ts)) if mine[i] and effective <= ts[i] < until]
    if not later:
        if not fallback:
            return {"rule": np.nan, "pessimistic": np.nan, "stale": stale, "index": -1}
        before = [i for i in range(len(ts)) if mine[i] and ts[i] <= effective]
        if not before:
            before = [i for i in range(len(ts)) if not prints["rpi"].iloc[i] and ts[i] <= effective]
        j = before[-1]
        return {"rule": px[j], "pessimistic": px[j], "stale": stale if earlier else px[j],
                "index": j}  # fmt: skip
    stamp = ts[later[0]]
    left, cost = clip, 0.0
    sweep = [i for i in later if ts[i] == stamp]
    for i in sweep:
        take = min(left, sz[i])
        cost, left = cost + take * px[i], left - take
    left = left if left > 1e-9 * clip else 0.0
    last = px[sweep[-1]]
    window = [px[i] for i in range(len(ts)) if mine[i] and stamp < ts[i] <= stamp + 60 * S
              and ts[i] < flatten]  # fmt: skip
    worst = (max(window) if side > 0 else min(window)) if window else last
    rule = (cost + left * last) / clip
    return {"rule": rule, "pessimistic": (cost + left * worst) / clip,
            "stale": stale if earlier else rule, "index": later[0]}  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fill_orders_equals_the_loop(seed: int) -> None:
    prints = make_prints(600, seed=seed, quiet_at=300)
    tape = PrintTape.from_frame(prints, DAY)
    rng = np.random.default_rng(seed)
    ts = tape.ts
    arrival = np.sort(rng.integers(ts[0] - S, ts[-1] + S, 200))
    decision = arrival - SETTINGS.latency_ns
    side = np.where(rng.random(200) < 0.5, BUY, SELL)
    not_before = arrival + rng.integers(0, 3 * S, 200) * (rng.random(200) < 0.3)
    until = int(ts[-1]) - 5 * S
    for fallback in (False, True):
        fills = fill_orders(tape, side, arrival, decision, until_ns=until,
                            not_before_ns=not_before, fallback=fallback)  # fmt: skip
        for k in range(len(arrival)):
            ref = loop_fill(prints, int(side[k]), int(arrival[k]), int(decision[k]), until,
                            int(not_before[k]), fallback, tape.flatten_ns(SETTINGS))  # fmt: skip
            assert fills.print_index[k] == ref["index"]
            for name in ("rule", "pessimistic", "stale"):
                np.testing.assert_allclose(getattr(fills, name)[k], ref[name], rtol=1e-12)


def loop_round_trip(prints: pd.DataFrame, grid: np.ndarray, decision: int, side: int,
                    hold: int, flatten_on: bool, end: int, flatten: int) -> tuple[int, int, int]:  # fmt: skip
    """(entry print, exit print, busy-until) by the words of the registration; -1 if none."""
    latency = SETTINGS.latency_ns
    ref = loop_fill(prints, side, decision + latency, decision, flatten if flatten_on else end,
                    0, False, flatten)  # fmt: skip
    if ref["index"] < 0:
        return -1, -1, -1
    fill_ns = int(prints["timestamp"].astype("int64").iloc[ref["index"]])
    after = grid[grid >= fill_ns + hold]
    exit_decision = int(after[0]) if len(after) else -1
    if flatten_on and (exit_decision < 0 or exit_decision > flatten):
        exit_decision = flatten
    if exit_decision < 0:
        return ref["index"], -1, -1
    out = loop_fill(prints, -side, exit_decision + latency, exit_decision, end, fill_ns,
                    flatten_on, flatten)  # fmt: skip
    if out["index"] < 0:
        return ref["index"], -1, -1
    exit_ns = int(prints["timestamp"].astype("int64").iloc[out["index"]])
    return ref["index"], out["index"], max(exit_ns, exit_decision + latency)


@pytest.mark.parametrize("flatten_on", [True, False])
def test_round_trips_equal_the_loop_both_sides(flatten_on: bool) -> None:
    prints = make_prints(800, seed=4, quiet_at=500, tail_at_s=23 * 3600 + 58 * 60 + 40)
    # The day's last prints are all buyer-initiated: late long exits fall back.
    prints.loc[prints.index[-25:], "aggressor"] = BUY
    tape = PrintTape.from_frame(prints, DAY)
    grid = grid_stamps_ns(prints)
    decisions = grid[::3]
    for side in (BUY, SELL):
        trips = round_trips(tape, grid, decisions, side, 20 * S, flatten=flatten_on)
        for k, t in enumerate(decisions):
            entry, exit_, busy = loop_round_trip(prints, grid, int(t), side, 20 * S, flatten_on,
                                                 tape.end_ns, tape.flatten_ns(SETTINGS))  # fmt: skip
            assert trips.entry.print_index[k] == entry
            if exit_ >= 0:
                assert trips.exit.print_index[k] == exit_ and trips.complete[k]
                if flatten_on:
                    assert trips.busy_until_ns[k] == busy
            else:
                assert not trips.complete[k]
    long = round_trips(tape, grid, decisions, BUY, 20 * S)
    short = round_trips(tape, grid, decisions, SELL, 20 * S)
    # Busy periods depend on the side: the fixture must exercise that.
    assert (long.busy_until_ns != short.busy_until_ns).any()
    assert (long.exit_reason == "day_end").any() and long.exit.fallback.any()
    assert not short.exit.fallback.any() and (long.entry.beyond > 0).any()
