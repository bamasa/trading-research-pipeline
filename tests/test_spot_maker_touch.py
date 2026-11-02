"""Touch proxies, carried prices, post-only and skewed quote prices."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import touch as tc

DAY = date(2026, 5, 4)
T0 = int(np.datetime64("2026-05-04T00:00:00", "ns").astype(np.int64))


def _tape() -> PrintTape:
    ts = T0 + np.array([10, 20, 20, 30], dtype=np.int64) * 1_000_000_000
    price = np.array([100.02, 100.00, 99.99, 100.03])
    side = np.array([1, -1, -1, 1])
    return PrintTape(DAY, ts, price, np.full(4, 0.01), side)


def test_proxies_after_prints_and_carried() -> None:
    proxies = tc.Proxies.from_tape(_tape(), carried=(101.0, 99.0))
    ask, bid = proxies.at(np.array([T0, T0 + 10 * 10**9, T0 + 25 * 10**9]))
    assert list(ask) == [101.0, 100.02, 100.02]
    assert list(bid) == [99.0, 99.0, 99.99]
    ask, bid = proxies.at(T0 + 20 * 10**9, strict=True)
    assert bid[0] == 99.0  # strictly before the stamp: the carried bid
    cold = tc.Proxies.from_tape(_tape())
    assert math.isnan(cold.one(T0)[0]) and cold.mid(T0 + 35 * 10**9)[0] == (100.03 + 99.99) / 2
    assert tc.last_prices(_tape()) == (100.03, 99.99)


def test_quote_price_post_only_skew_and_tick() -> None:
    q = tc.quote_price
    assert q(1, 100.02, 100.00, 0.0, skew_bp=1, clip=0.01, tick=0.01) == 100.00
    assert q(-1, 100.02, 100.00, 0.0, skew_bp=1, clip=0.01, tick=0.01) == 100.02
    # a long position lowers both quotes; the sell is held a tick above the bid
    assert q(-1, 100.02, 100.00, 0.01, skew_bp=1, clip=0.01, tick=0.01) == 100.01
    assert q(1, 100.02, 100.00, 0.01, skew_bp=1, clip=0.01, tick=0.01) == 99.99
    # crossed proxies: the buy is held below the ask proxy
    assert q(1, 99.98, 100.00, 0.0, skew_bp=0, clip=0.01, tick=0.01) == 99.97
    assert math.isnan(q(1, 100.02, math.nan, 0.0, skew_bp=0, clip=0.01, tick=0.01))
    assert q(1, math.nan, 100.0, 0.0, skew_bp=0, clip=0.01, tick=0.01) == 100.0
    assert tc.on_tick(100.004999, 0.01, 1) == 100.0 and tc.on_tick(100.004999, 0.01, -1) == 100.01
