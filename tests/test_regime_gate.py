"""The regime gate: pauses only after an alarm is known, placebos keep the time paused, the decision."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.bench.regime_gate import GateResult, pause_mask, shifted, verdict


def _labels(n: int = 12) -> pd.DatetimeIndex:
    return pd.date_range("2024-03-01 00:00:05", periods=n, freq="5s", tz="UTC")


def test_a_rise_pauses_from_the_next_row_and_a_fall_resumes() -> None:
    labels = _labels()
    found = [(labels[2], 1), (labels[6], -1), (labels[9], 1)]
    mask = pause_mask(labels, found)
    assert mask.tolist() == [False] * 3 + [True] * 4 + [False] * 3 + [True] * 2


def test_an_alarm_before_the_scored_rows_sets_the_state_they_start_in() -> None:
    labels = _labels()
    early, later = pd.Timestamp("2024-02-28", tz="UTC"), pd.Timestamp("2024-02-28 01:00", tz="UTC")
    assert pause_mask(labels, [(early, 1)]).all()
    assert not pause_mask(labels, [(early, 1), (later, -1)]).any()


def test_a_placebo_pauses_as_long_as_the_gate_at_other_times() -> None:
    mask = np.array([False] * 4 + [True, True, False, False, False, False, True, False])
    for j in range(1, 20):
        moved = shifted(mask, 4, j)
        assert moved[:4].tolist() == mask[:4].tolist() and moved.sum() == mask.sum()


def test_the_decision_needs_a_gain_and_every_placebo_beaten() -> None:
    def panel(gain: float, placebo: float, found: bool = True) -> GateResult:
        return GateResult("x", alarms=[("2024-03-06T05:30", 1)] if found else [],
                          ungated_pct=1.0, gated_pct=1.0 + gain,
                          placebo_pct=[1.0 + placebo] * 19)  # fmt: skip

    assert verdict([panel(2.0, 1.0), panel(1.0, 0.0)])["status"] == "passes"
    assert verdict([panel(-1.0, -2.0)])["first_kill"] == "the gate's gain is not positive"
    assert verdict([panel(1.0, 1.0)])["first_kill"] == "a placebo gained as much"
    assert verdict([panel(0.0, 0.0, found=False)])["status"] == "not tested"
