"""A run's money, markouts, capacity and void conditions."""

from __future__ import annotations

import dataclasses
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml

from trading_research.spot.costs import FeeTable
from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import money as mn
from trading_research.spot_maker.touch import Proxies

REPO = Path(__file__).resolve().parents[1]
DAY = date(2026, 5, 4)
T0 = int(np.datetime64("2026-05-04T00:00:00", "ns").astype(np.int64))
S = 1_000_000_000


def _table() -> FeeTable:
    config = yaml.safe_load((REPO / "configs/spot_maker_2026.yaml").read_text(encoding="utf-8"))
    return FeeTable.from_config(config["fees"])


def _tape() -> PrintTape:
    ts = T0 + np.array([1, 2, 7, 8], dtype=np.int64) * S
    return PrintTape(DAY, ts, np.array([100.0, 99.0, 101.0, 100.5]), np.full(4, 0.01),
                     np.array([1, -1, 1, -1]))  # fmt: skip


def _fills(flatten_side: int = -1) -> mn.DayFills:
    log = mn.FillLog()
    log.add(T0 + 3 * S, 1, 99.5, 0.01, True, "entry")
    log.add(T0 + 9 * S, flatten_side, 100.0, 0.01, False, "flatten")
    return log.day(mn.Pricing(_table(), "regular"))


def test_money_charges_maker_zero_and_taker_ten() -> None:
    fills = _fills()
    assert list(fills.fee_bp) == [0.0, 10.0]
    found = mn.money(fills, 100.0, 0.010)
    assert found["net_quote"] == pytest.approx(0.005 - 0.001)
    assert found["net_bp_fill"] == pytest.approx(0.004 / 1.995 * 1e4)
    assert found["taker_fills"] == 1 and found["maker_buy_btc"] == 0.01
    tiers = mn.tier_net_bp(fills, _table(), ["vip9"])
    assert tiers["zero_fees"] == pytest.approx(0.005 / 1.995 * 1e4)
    assert tiers["vip9"] == pytest.approx((0.005 - 0.00023) / 1.995 * 1e4)
    empty = mn.FillLog().day(mn.Pricing(_table(), "regular"))
    assert math.isnan(mn.money(empty, 100.0, 0.01)["net_bp_fill"])
    assert mn.voids(empty, first_price=100.0, last_price=100.5, unit=0.01, table=_table(),
                    tier="regular") == []  # fmt: skip


def test_markouts_and_capacity() -> None:
    fills = _fills()
    found = mn.markouts(fills, Proxies.from_tape(_tape()), (5.0,))
    # at 3 s + 5 s the proxies are ask 101.0 and bid 100.5 (a print at 8 s counts)
    assert found["markout_5s"][0] == pytest.approx((100.75 - 99.5) / 99.5 * 1e4)
    assert found["spread"][0] == pytest.approx((99.5 - 99.5) / 99.5 * 1e4)
    sums = mn.markout_sums(fills, Proxies.from_tape(_tape()), (5.0,))
    assert sums["markout_5s_n"] == 1.0
    assert mn.capacity(fills, _tape()) == (0.01, 0.0)


def test_voids_fire_on_wrong_fee_open_position_and_beyond_unit() -> None:
    kw = {"first_price": 100.0, "last_price": 100.5, "table": _table(), "tier": "regular"}
    assert mn.voids(_fills(), unit=0.01, **kw) == []
    wrong = dataclasses.replace(_fills(), fee_bp=np.array([0.0, 9.0]))
    assert "K-units" in mn.voids(wrong, unit=0.01, **kw)
    assert "K-acct" in mn.voids(_fills(flatten_side=1), unit=0.01, **kw)
    assert "K-pos" in mn.voids(_fills(flatten_side=1), unit=0.01, **kw)
