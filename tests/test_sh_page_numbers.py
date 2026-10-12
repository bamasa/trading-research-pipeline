"""What the pages show beside the statuses, from a block's per-day series."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from trading_research.pipeline.short_horizon import page_numbers as pn
from trading_research.pipeline.short_horizon import statuses as st
from trading_research.pipeline.short_horizon.reads import regime_segments

F_DAYS = [date(2025, 8, 21) + timedelta(days=i) for i in range(40)]


def _slot(slot: str, kind: str = "taker", value: float = 1.0) -> st.SlotDays:
    n = len(F_DAYS)
    zeros = np.zeros(n)
    return st.SlotDays(slot, kind, list(F_DAYS), np.full(n, value), np.full(n, 5), np.full(n, 1.0),
                       np.full(n, 100.0), zeros, np.zeros((2, n)), zeros, zeros, zeros, zeros,
                       np.zeros((1, n)), zeros)  # fmt: skip


def test_only_the_slots_taken_to_f_are_numbered_on_f_minus(monkeypatch: pytest.MonkeyPatch) -> None:
    """A split at the 25th day: S-C, taken to F, is numbered on its 24 days before b;
    S25, never taken to F, on all of F."""
    monkeypatch.setattr(pn, "save_close", lambda _: None)
    monkeypatch.setattr(pn, "fill_rule_summary", lambda _: {})
    block = SimpleNamespace(name="F", detector={"alarms": []})
    found = {"split": {"b": F_DAYS[24].isoformat(), "on_all_of_F": False}, "after_F": {"S-C": "x"}}
    series = {"S-C": _slot("S-C"), "S25": _slot("S25")}
    numbers = pn.block_numbers(block, series, found)["slots"]
    assert numbers["S-C"]["window_days"] == 24 and numbers["S25"]["window_days"] == len(F_DAYS)


def test_s25zs_regime_map_cuts_its_net_at_zero_fees() -> None:
    s = _slot("S25-Z", "zero", value=3.0)
    s.second = np.full(len(F_DAYS), -1.0)
    rows = regime_segments({"S25-Z": s}, [], "F")
    assert rows and all(r["mean"] == pytest.approx(-1.0) for r in rows)


def test_delta_b_is_reported_with_its_percentile_among_the_admissible_days() -> None:
    """A series that is 2 before b, -1 after it and 0.5 on b's own day, which is in
    neither side of Delta(b) but in one side of every other split's: Delta(b) is
    the largest of the admissible days'."""
    s = _slot("S-C")
    s.value = np.where(np.arange(len(F_DAYS)) < 25, 2.0, -1.0)
    s.value[25] = 0.5
    found = pn.delta_b(s, F_DAYS[25])
    assert found["delta_b"] == pytest.approx(3.0)
    assert found["admissible_days"] == len(F_DAYS) - 20 - 10 - 1
    assert found["percentile_among_admissible"] == 100


def test_f_numbers_carry_delta_b_for_a_slot_tested_for_r(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pn, "save_close", lambda _: None)
    monkeypatch.setattr(pn, "fill_rule_summary", lambda _: {})
    block = SimpleNamespace(name="F", detector={"alarms": []})
    found = {"split": {"b": F_DAYS[25].isoformat(), "on_all_of_F": False}, "after_F": {"S-C": "x"},
             "R": {"S-C": {"outcome": "not supported"}}, "assessments": {"S-C": {"status": "killed"}}}  # fmt: skip
    numbers = pn.block_numbers(block, {"S-C": _slot("S-C"), "S25": _slot("S25")}, found)["slots"]
    assert numbers["S-C"]["R_delta_b"]["passes_on_F_minus"] is False
    assert "R_delta_b" not in numbers["S25"]
