"""Capacity: an infinite day ratio survives the day record and lifts no cap."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from trading_research.spot import assemble as asm
from trading_research.spot import statuses as st
from trading_research.spot.calibration import capacity_ratio
from trading_research.spot.state import plain


def _records(capacities: list[float], trades: list[int]) -> dict[str, dict]:
    days = [date(2026, 8, 8) + timedelta(days=k) for k in range(len(capacities))]
    records = {}
    for d, capacity, n in zip(days, capacities, trades, strict=True):
        row = {"value": 1.0 if n else None, "trades": n, "gross_bp_turnover": 12.0, "ic": 0.1,
               "random": [0.0] * st.RANDOM_DRAWS, "flipped": -1.0, "shifted": 0.0,
               "all_long": 0.0, "all_short": 0.0, "latency": 1.0,
               "neighbourhood": [1.0] * st.NEIGHBOURHOOD_CELLS,
               "worse": {"net_quote": 1.0, "entry_notional": 600.0, "buy_notional": 600.0,
                         "sell_notional": 600.0},
               "fills": 2 * n, "beyond_sweep": 0, "capacity": capacity}  # fmt: skip
        records[d.isoformat()] = plain(
            {"day": d.isoformat(), "kept": {"binance-spot:BTCFDUSD": True},
             "rows": {"S25-PF": row}}
        )  # fmt: skip
    return records


def test_an_infinite_ratio_is_read_back_as_infinite_and_caps_the_slot() -> None:
    prints = pd.DataFrame({"aggressor": [1, 1, 1], "size": [0.5, 0.5, 0.5]})
    ratio = capacity_ratio(np.array([1, -1]), np.array([0.01, 0.01]), prints)
    assert ratio == math.inf
    records = _records([math.inf, math.inf, 0.001], [40, 40, 40])
    assert records["2026-08-08"]["rows"]["S25-PF"]["capacity"] is None  # JSON has no inf
    s = asm.slot_days(records, "S25-PF")
    assert list(s.capacity) == [math.inf, math.inf, 0.001]
    assert st.caps(s)["capacity_median"] == math.inf > st.CAPACITY_CAP
    found = asm.measurements(records, s)
    assert found["capacity_median_infinite"]


def test_a_day_without_a_trade_counts_zero() -> None:
    s = asm.slot_days(_records([0.0, 0.03, 0.04], [0, 40, 40]), "S25-PF")
    assert list(s.capacity) == [0.0, 0.03, 0.04]
    assert st.caps(s)["capacity_median"] == 0.03
