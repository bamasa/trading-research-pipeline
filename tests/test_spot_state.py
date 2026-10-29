"""The spot study's stage machine: once, in order, refused out of order, resumable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading_research.spot.state import ORDER, StageOrderError, StageState, plain


def test_a_stage_runs_once_and_returns_its_record(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    calls = []
    first = state.run("0", lambda: calls.append(1) or {"value": 1})
    again = state.run("0", lambda: calls.append(2) or {"value": 2})
    assert first == again == {"value": 1} and calls == [1]
    assert state.entries()["0"]["attempts"] == 1


def test_stages_are_refused_out_of_order(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    with pytest.raises(StageOrderError, match="needs stage 0"):
        state.run("1", lambda: {})
    state.run("0", lambda: {})
    state.run("1", lambda: {})
    with pytest.raises(StageOrderError, match="cannot run again"):
        state.require_ready("0")
    (tmp_path / "0" / "record.json").write_text('{"changed": 1}\n')
    with pytest.raises(StageOrderError, match="changed since it finished"):
        state.require_ready("D")


def test_a_stage_that_stopped_is_started_again(tmp_path: Path) -> None:
    state = StageState(tmp_path)

    def fails() -> dict[str, int]:
        raise RuntimeError("stopped")

    with pytest.raises(RuntimeError):
        state.run("0", fails)
    assert not state.complete("0")
    assert state.run("0", lambda: {"ok": 1}) == {"ok": 1}
    assert state.entries()["0"]["attempts"] == 2
    assert set(state.status()) == set(ORDER)


def test_records_are_plain_values() -> None:
    found = plain({"a": np.float64(1.5), "b": [np.int64(2), float("nan")], "c": np.bool_(True)})
    assert found == {"a": 1.5, "b": [2, None], "c": True}
