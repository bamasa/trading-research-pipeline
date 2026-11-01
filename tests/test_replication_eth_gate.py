"""The replication's gate and event rule against the original's definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.replication_eth.gate import (
    FLAT,
    GateConfig,
    gate,
    one_per_event,
    strategy_inputs,
)

CFG = GateConfig(
    "t",
    tau_move=0.9,
    min_margin=0.45,
    buffer_bp=4,
    t_min=1.6,
    event_max_gap=15,
    cooldown_seconds=10,
)


def _original_events(
    labels: np.ndarray, seconds: np.ndarray, gap: float, cooldown: float, kind: str = "stable"
) -> np.ndarray:
    """The original's grouping and cooldown, its loops kept; ``kind`` is the sort's.

    As in the original, the kept positions index the sorted frame and are applied
    to the labels in their given order.
    """
    df = (
        pd.DataFrame({"t": seconds, "label": labels})
        .sort_values("t", kind=kind)
        .reset_index(drop=True)
    )
    events, cur, start, end, prev = [], None, 0, 0, None
    for i, (t, lab) in enumerate(zip(df["t"], df["label"], strict=True)):
        if cur is None:
            cur, start, end = lab, i, i
        elif lab == cur and t - prev <= gap:
            end = i
        else:
            events.append((df["t"][start], df["t"][end], cur, start))
            cur, start, end = lab, i, i
        prev = t
    events.append((df["t"][start], df["t"][end], cur, start))
    out, last = np.full(len(labels), FLAT), -np.inf
    for t0, t1, lab, idx in events:
        if lab == FLAT or t0 < last + cooldown:
            continue
        out[idx], last = labels[idx], t1
    return out


def test_one_per_event_matches_the_original_loop() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        seconds = np.sort(rng.integers(0, 400, 300))
        labels = rng.choice([0, 1, 2], size=300, p=[0.1, 0.8, 0.1])
        cfg = GateConfig(
            "t", 0.9, 0.4, 3, 2.5, float(rng.choice([1, 8, 15])), float(rng.choice([0, 1, 10, 60]))
        )
        expected = _original_events(labels, seconds, cfg.event_max_gap, cfg.cooldown_seconds)
        np.testing.assert_array_equal(one_per_event(labels, seconds, cfg), expected)


def test_unstable_order_re_enacts_the_original_sort() -> None:
    rng = np.random.default_rng(4)
    seconds = np.repeat(np.arange(2000), 10)
    labels = rng.choice([0, 1, 2], size=len(seconds), p=[0.02, 0.96, 0.02])
    cfg = GateConfig("t", 0.9, 0.45, 4, 1.6, 15, 10, event_order="unstable_sort")
    expected = _original_events(labels, seconds, 15, 10, kind="quicksort")
    got = one_per_event(labels, seconds, cfg)
    np.testing.assert_array_equal(got, expected)
    stable = one_per_event(labels, seconds, GateConfig("t", 0.9, 0.45, 4, 1.6, 15, 10))
    assert (got != FLAT).sum() <= (stable != FLAT).sum()


def test_gate_holds_on_each_condition() -> None:
    probs = np.array(
        [
            [0.02, 0.05, 0.93],  # trades up
            [0.93, 0.05, 0.02],  # trades down
            [0.02, 0.15, 0.83],  # 1 - p_flat below tau
            [0.30, 0.05, 0.65],  # margin below 0.45
            [0.02, 0.05, 0.93],  # not tradable
            [0.02, 0.05, 0.93],  # edge below cost plus buffer
        ]
    )
    rv = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1.0, 17.0])
    spread = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(gate(probs, rv, spread, CFG), [2, 0, 1, 1, 1, 1])


def test_strategy_inputs_fill_and_clip_as_the_original() -> None:
    rv = np.array([-0.1, 0.2, 0.3])
    mid_mean = np.array([0.0, 2.0, 0.0])
    spread_10 = np.array([1.0, 1.0, -1.0])
    rv_bp, spread_bp = strategy_inputs(rv, mid_mean, spread_10)
    np.testing.assert_allclose(rv_bp, [0.0, 2000.0, 3000.0])
    np.testing.assert_allclose(spread_bp, [5000.0, 5000.0, 5000.0])
