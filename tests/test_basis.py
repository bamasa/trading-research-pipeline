"""The basis trade's engine: it earns a planted premium's fall, pays its costs, and never reads ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.search_engine.basis import GRID
from trading_research.strategies.basis import BasisRule, deviation, panel, replay, shifted
from trading_research.strategies.carry import Market

HOURS = pd.date_range("2022-01-01", periods=24 * 60, freq="1h", tz="UTC")


def market(jumps: dict[int, float], funding: float = 0.0) -> Market:
    spot = pd.Series(100.0, index=HOURS)
    premium = np.full(len(HOURS), -3.0)
    for at, bp in jumps.items():
        premium[at : at + 6] = bp  # rich for six hours, then back
    perp = spot * (1 + premium / 1e4)
    rates = pd.Series(funding, index=HOURS[::8])
    return Market(spot, perp, rates)


def everything(_: str, __: pd.Period) -> bool:
    return True


def test_a_planted_jump_is_traded_and_pays_after_cost() -> None:
    p = panel({"AAA": market({24 * 20: 150.0})}, HOURS, everything)
    frame = replay(p, BasisRule(entry_bp=60, window_days=7, slots=1), HOURS[0], HOURS[-1])
    assert frame.attrs["trades"] == 1
    # 153 bp of premium's fall less the 38 bp round trip, on a leg of half the capital
    assert np.isclose((frame.attrs["final_equity"] - 1) * 1e4, (153 - 38) / 2)


def test_a_premium_below_the_entry_is_never_traded() -> None:
    p = panel({"AAA": market({24 * 20: 40.0})}, HOURS, everything)
    frame = replay(p, BasisRule(entry_bp=60, slots=1), HOURS[0], HOURS[-1])
    assert frame.attrs["trades"] == 0
    assert frame.attrs["final_equity"] == 1.0


def test_the_deviation_reads_only_earlier_hours() -> None:
    p = panel({"AAA": market({24 * 20: 150.0})}, HOURS, everything)
    before = deviation(p, 7)
    p.perp[24 * 30 :] *= 1.01  # change the future
    after = deviation(p, 7)
    assert np.allclose(before[: 24 * 30], after[: 24 * 30], equal_nan=True)


def test_an_unreadable_month_is_never_held() -> None:
    def no_february(_: str, m: pd.Period) -> bool:
        return m != pd.Period("2022-02", "M")

    p = panel({"AAA": market({24 * 30 + 12: 150.0})}, HOURS, no_february)  # 31 Jan, 12:00
    assert not p.readable[HOURS.searchsorted(pd.Timestamp("2022-02-10", tz="UTC"))].any()
    frame = replay(p, BasisRule(entry_bp=60, slots=1), HOURS[0], HOURS[-1])
    assert frame.attrs["trades"] <= 1


def test_placebo_keeps_each_coins_signals_and_moves_them() -> None:
    p = panel({"AAA": market({24 * 20: 150.0})}, HOURS, everything)
    dev = deviation(p, 7)
    moved = shifted(dev, 0, len(HOURS), 3)
    assert np.isclose(np.nansum(moved > 60), np.nansum(dev > 60))
    assert not np.allclose(np.nan_to_num(moved), np.nan_to_num(dev))


def test_the_grid_is_the_registrations() -> None:
    assert len(GRID) == 12
    assert {r.entry_bp for r in GRID} == {30.0, 45.0, 60.0}
    assert {r.exit_bp for r in GRID} == {0.0, 15.0}
    assert {r.window_days for r in GRID} == {7, 30}
