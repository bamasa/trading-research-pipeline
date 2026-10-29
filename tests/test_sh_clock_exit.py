"""Close by time on row stamps, the day's limits, and thinning with the exit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.strategies.clock_exit import ClockExit, DayLimits
from trading_research.strategies.ev_gate import thin_entries

MS = 1_000_000
DAY = pd.Timestamp("2025-03-04", tz="UTC").value


def test_entry_and_exit_snapshots_follow_the_stated_steps() -> None:
    stamps = DAY + np.array([3_600_000, 3_600_005, 3_600_020, 3_602_000, 3_602_040]) * MS
    schedule = ClockExit(hold_s=2.0).schedule(stamps)
    # Row 0: live at +10 ms -> fills on row 2 (+20 ms); exit decided at +2020 ms,
    # live at +2030 ms -> first snapshot at or after is row 4 (+2040 ms).
    assert schedule["entry_index"].tolist() == [2, 2, 3, 4, 5]
    assert schedule["exit_index"].tolist() == [4, 4, 5, 5, 5]
    assert schedule["allowed"].all()


def test_the_day_limits_bound_entries_and_cap_the_hold() -> None:
    limits = DayLimits()
    clock = np.array([0, 59, 60, 23 * 3600 + 58 * 60, 23 * 3600 + 58 * 60 + 1]) * 1000 * MS
    assert limits.entry_allowed(DAY + clock).tolist() == [False, False, True, True, False]
    exit_rule = ClockExit(hold_s=120.0)
    late_fill = DAY + (23 * 3600 + 58 * 60) * 1000 * MS
    flatten = DAY + (23 * 3600 + 59 * 60) * 1000 * MS
    assert exit_rule.exit_decisions(np.array([late_fill]))[0] == flatten
    after_flatten = flatten + 5 * 1000 * MS
    assert exit_rule.exit_decisions(np.array([after_flatten]))[0] == after_flatten
    with pytest.raises(ValueError, match="positive"):
        ClockExit(hold_s=0.0)


def test_thinning_with_the_clock_exit_holds_one_position() -> None:
    rng = np.random.default_rng(0)
    stamps = DAY + 3_600_000 * MS + np.cumsum(rng.integers(50, 300, 2_000)) * MS
    schedule = ClockExit(hold_s=5.0).schedule(stamps)
    sides = rng.choice([-1, 0, 0, 0, 1], size=len(stamps))
    entries = thin_entries(sides, schedule["exit_index"].to_numpy())
    exits = schedule["exit_index"].to_numpy()[entries]
    assert (entries[1:] > exits[:-1]).all()
    fills = stamps[schedule["entry_index"].to_numpy()[entries]]
    assert (stamps[exits] >= fills + 5_000 * MS + 10 * MS).all()
