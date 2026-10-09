"""Stage C's money for the signal-entry quoter: per attempt, over its own entry's notional."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market
from trading_research.pipeline.short_horizon.passive import SimCell, day_values, per_100_clip

MS = 1_000_000
CONFIG = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0, flatten_slippage_bp=0.0)
CELL = SimCell("S-P", "key", "entry", 0.5)


def _market(bids: list[int], prints: list[tuple[float, int, float, int]]):
    """Snapshots every 100 ms with the ask two ticks above the bid."""
    snapshots = [
        (100.0 * i, [(b - k, 10.0) for k in range(3)], [(b + 2 + k, 10.0) for k in range(3)])
        for i, b in enumerate(bids)
    ]
    return hand_built_market(snapshots, prints, lot=1.0)


def _run(market, decisions_ms: list[float]):
    stamps = np.array([market.day_start_ns + round(t * MS) for t in decisions_ms])
    quoter = SignalEntryQuoter(0.5)
    tapes = entry_tapes(stamps, np.ones(len(stamps)))
    return quoter, simulate_day(market, quoter, CONFIG, tapes)


def test_an_exit_at_the_next_post_stays_in_its_attempt_and_the_day_reads_per_attempt() -> None:
    """Bought at 1.00 at 150 ms, out at 1.03 at 800 ms where the second attempt
    posts at 1.03 and never fills: the first attempt carries the whole day."""
    market = _market([100] * 8 + [103] * 12, [(150.0, 99, 5.0, -1)])
    quoter, result = _run(market, [0, 750])
    row = day_values(CELL, quoter, result, clip=1.0, mid=1.0)
    fees = float(result.fills["fee"].sum())
    assert row["attempts"] == 2 and row["trades"] == 1
    assert row["net_per_100"] == pytest.approx((0.03 - fees) * 100.0 / 1.00)
    assert row["gross_bp_trade"] == pytest.approx(300.0)


def test_each_attempt_is_taken_over_its_own_entry_notional() -> None:
    """Two round trips at different entry prices: each net over its own notional."""
    bids = [100] * 8 + [103] * 4 + [110] * 8
    prints = [(150.0, 99, 5.0, -1), (1_250.0, 109, 5.0, -1)]
    market = _market(bids, prints)
    quoter, result = _run(market, [0, 1_150])
    table = quoter.attempts(result.fills)
    assert table["filled"].tolist() == [True, True]
    assert table["notional"].tolist() == pytest.approx([1.00, 1.10])
    row = day_values(CELL, quoter, result, clip=1.0, mid=1.0)
    expected = (table["net"] / table["notional"] * 100.0).sum()
    assert row["net_per_100"] == pytest.approx(expected)
    assert row["net_per_100"] != pytest.approx(row["net_usdt"] * 100.0 / 1.05)


def test_funding_is_taken_over_the_mean_entry_notional_and_no_fill_is_zero() -> None:
    filled = pd.DataFrame({"net": [0.02, -0.01], "notional": [1.0, 3.0]})
    assert per_100_clip(filled, 0.0) == pytest.approx(2.0 - 1.0 / 3.0)
    assert per_100_clip(filled, -0.004) == pytest.approx(2.0 - 1.0 / 3.0 - 0.2)
    assert per_100_clip(filled.iloc[:0], 0.5) == 0.0
