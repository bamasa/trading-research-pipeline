"""The stage machine: the registered order, idempotent stages, no stage out of order."""

from __future__ import annotations

from pathlib import Path

import pytest

from trading_research.pipeline.short_horizon.state import ORDER, StageOrderError, StageState


def test_the_registered_order() -> None:
    assert ORDER == ("0", "1", "A", "B", "D", "C", "E", "H", "F", "P")


def test_a_stage_refuses_to_run_before_the_stages_it_reads(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    with pytest.raises(StageOrderError, match="needs stage 0, which is not run"):
        state.run("1", lambda: {"x": 1})
    assert state.run("0", lambda: {"x": 0}) == {"x": 0}
    assert state.run("1", lambda: {"x": 1}) == {"x": 1}


def test_a_complete_stage_returns_its_record_and_computes_nothing(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    calls = []
    state.run("0", lambda: calls.append(1) or {"n": len(calls)})
    again = state.run("0", lambda: calls.append(1) or {"n": len(calls)})
    assert again == {"n": 1} and len(calls) == 1


def test_a_stage_that_failed_is_started_again(tmp_path: Path) -> None:
    state = StageState(tmp_path)

    def fails() -> dict[str, int]:
        raise RuntimeError("stopped part way")

    with pytest.raises(RuntimeError):
        state.run("0", fails)
    assert not state.complete("0")
    assert state.run("0", lambda: {"ok": 1}) == {"ok": 1}
    assert state.entries()["0"]["attempts"] == 2


def test_a_changed_record_blocks_the_next_stage(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.run("0", lambda: {"x": 0})
    state.record_path("0").write_text('{"x": 9}\n', encoding="utf-8")
    with pytest.raises(StageOrderError, match="changed since it finished"):
        state.run("1", lambda: {})


def test_an_earlier_stage_cannot_run_again_once_a_later_one_started(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    state.run("0", lambda: {})
    state.run("1", lambda: {})
    state.record_path("0").unlink()
    with pytest.raises(StageOrderError, match="stage 1 has started"):
        state.run("0", lambda: {})


def test_a_block_is_read_only_after_the_one_before_it(tmp_path: Path) -> None:
    state = StageState(tmp_path)
    for stage in ("0", "1", "A", "B", "D", "C", "E"):
        state.run(stage, lambda: {})
    with pytest.raises(StageOrderError, match="needs stage H"):
        state.run("F", lambda: {})
    state.run("H", lambda: {})
    state.run("F", lambda: {})
