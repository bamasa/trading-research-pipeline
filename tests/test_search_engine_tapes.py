"""Candidate-row tapes: the loosest cell's rows, and every stricter cell inside them."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.pipeline.execution import guarded_rows
from trading_research.search_engine.tapes import (
    GUARD_WINDOWS_H,
    CandidateTape,
    TapeError,
    ev_gate_tape,
    hold_first_tape,
    read_tape,
    tape_path,
    write_tape,
)
from trading_research.strategies.ev_gate import EVGate
from trading_research.strategies.hold_first import fit_hold_first

DAY = date(2025, 3, 4)
C_CONFIG = "BTCUSDT|C|gbm|micro+tensor|hold_s=20|label_fee_bp=5.5"
R_CONFIG = "BTCUSDT|R|tcn|tensor+micro|hold_s=5"
ROWS = 5000


def _stamps() -> np.ndarray:
    start = pd.Timestamp(DAY, tz="UTC").value
    return start + np.arange(ROWS, dtype=np.int64) * 17_280_000_000  # the whole day


def _proba(seed: int = 0) -> np.ndarray:
    # columns in the models' class order: SELL, HOLD, BUY
    return np.random.default_rng(seed).dirichlet([1.0, 6.0, 1.0], ROWS)


def _gates(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"quiet_out": rng.random(ROWS) < 0.7, "tight_only": rng.random(ROWS) < 0.6}


def _flags() -> np.ndarray:
    start = pd.Timestamp(DAY, tz="UTC").value
    return np.array([start - 30 * 3_600_000_000_000, start + 6 * 3_600_000_000_000])


def test_hold_first_tape_holds_the_loosest_rows_and_stricter_ones_inside() -> None:
    proba = _proba()
    loose = fit_hold_first(_proba(1), days=ROWS / 1000, trades_per_day=240)
    strict = fit_hold_first(_proba(1), days=ROWS / 1000, trades_per_day=25)
    tape = hold_first_tape(DAY, C_CONFIG, _stamps(), proba, loose, _gates(), _flags())
    np.testing.assert_array_equal(tape.rows, np.flatnonzero(loose.decide_proba(proba)))
    stricter = strict.decide_proba(proba)
    inside = np.flatnonzero(stricter)
    assert 0 < len(inside) < len(tape) and np.isin(inside, tape.rows).all()
    position = np.searchsorted(tape.rows, inside)
    np.testing.assert_array_equal(tape.side[position], stricter[inside])
    np.testing.assert_array_equal(tape.scores["p_long"], proba[tape.rows, 2])
    assert tape.loosest == {"theta_long": loose.theta_long, "theta_short": loose.theta_short}


def test_ev_gate_tape_holds_k_zero_beta_zero_and_stricter_cells_inside() -> None:
    rng = np.random.default_rng(2)
    mu, sigma = rng.normal(0, 12, ROWS), rng.uniform(0.5, 6, ROWS)
    walk = rng.uniform(0.01, 0.5, ROWS)
    tape = ev_gate_tape(DAY, R_CONFIG, _stamps(), mu, sigma, walk, _gates(), _flags())
    np.testing.assert_array_equal(tape.rows, np.flatnonzero(EVGate(0, 0).sides(mu, sigma, walk)))
    strict = EVGate(1.0, 2.0).sides(mu, sigma, walk)
    assert np.isin(np.flatnonzero(strict), tape.rows).all()
    assert tape.loosest == {"cost_bp": 11.0}


def test_gate_and_guard_states_are_carried_per_row() -> None:
    gates = _gates(3)
    tape = hold_first_tape(DAY, C_CONFIG, _stamps(), _proba(3),
                           fit_hold_first(_proba(4), days=5, trades_per_day=240), gates, _flags())  # fmt: skip
    for name, state in gates.items():
        np.testing.assert_array_equal(tape.gates[name], state[tape.rows])
    for hours in GUARD_WINDOWS_H:
        expected = guarded_rows(_stamps()[tape.rows], _flags(), hours * 3_600_000_000_000)
        np.testing.assert_array_equal(tape.guards[hours], expected)
    assert tape.guards[4].any() and not tape.guards[4].all()
    assert tape.guards[120].sum() >= tape.guards[24].sum() >= tape.guards[4].sum()


def test_a_tape_round_trips_through_its_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    tape = ev_gate_tape(DAY, R_CONFIG, _stamps(), rng.normal(0, 15, ROWS),
                        rng.uniform(1, 3, ROWS), rng.uniform(0, 1, ROWS), _gates(), _flags())  # fmt: skip
    path = write_tape(tape, tape_path(tmp_path, R_CONFIG, DAY))
    assert "|" not in str(path.relative_to(tmp_path)) and "+" not in str(path.relative_to(tmp_path))
    back = read_tape(path, configuration=R_CONFIG, day=DAY)
    pd.testing.assert_frame_equal(back.to_frame(), tape.to_frame())
    assert back.metadata() == tape.metadata()
    with pytest.raises(TapeError, match="not"):
        read_tape(path, configuration=C_CONFIG)
    with pytest.raises(TapeError, match="not"):
        read_tape(path, day=date(2025, 3, 5))
    tape.to_frame().to_parquet(tmp_path / "plain.parquet")
    with pytest.raises(TapeError, match="not a candidate tape"):
        read_tape(tmp_path / "plain.parquet")


def test_malformed_tapes_are_refused() -> None:
    tape = hold_first_tape(DAY, C_CONFIG, _stamps(), _proba(),
                           fit_hold_first(_proba(1), days=5, trades_per_day=240), _gates(), _flags())  # fmt: skip
    frame = tape.to_frame()
    with pytest.raises(TapeError, match="increasing"):
        CandidateTape.from_frame(frame.iloc[::-1].reset_index(drop=True), tape.metadata())
    with pytest.raises(TapeError, match="carry"):
        CandidateTape.from_frame(frame, {**tape.metadata(), "stratum": "R"})
    with pytest.raises(TapeError, match="one value per row"):
        hold_first_tape(DAY, C_CONFIG, _stamps()[:-1], _proba(),
                        fit_hold_first(_proba(1), days=5, trades_per_day=240), _gates(), _flags())  # fmt: skip
