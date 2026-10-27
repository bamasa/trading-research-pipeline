"""``trp agent``: the harness's own entry point, its demonstration with the scripted client, no key read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.walkthrough import agent

runner = CliRunner()
SENTINEL = "not-a-key-0123456789"


def test_a_model_acts_only_with_a_key_a_model_name_and_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert agent.live_missing() == "ANTHROPIC_API_KEY is not set"
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    monkeypatch.delenv("TRP_AGENT_MODEL", raising=False)
    found = agent.live_missing()
    assert found is not None and "TRP_AGENT_MODEL" in found and SENTINEL not in found


def test_the_demonstration_runs_the_loop_with_the_scripted_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    monkeypatch.delenv("TRP_AGENT_MODEL", raising=False)
    result = runner.invoke(app, ["agent", "demo", "--into", str(tmp_path / "scratch")])
    assert result.exit_code == 0, result.output
    assert "scripted client" in result.output and "status: complete" in result.output
    assert SENTINEL not in result.output
    written = [
        p for p in (tmp_path / "scratch").rglob("*") if p.is_file() and ".git" not in p.parts
    ]
    assert written and not [p for p in written if SENTINEL in p.read_text(errors="ignore")]
    (run,) = list((tmp_path / "scratch" / "agents" / "runs").iterdir())
    state = agent.status(run)
    assert state["study"] == "synth_carry" and state["verdict"]["ledger_match"] is True
    assert [s["step"] for s in state["steps"]][-1] == "record"
    shown = runner.invoke(app, ["agent", "status", str(run)])
    assert shown.exit_code == 0 and json.loads(shown.output)["study"] == "synth_carry"


def test_the_harness_refuses_a_run_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRP_AGENT_MODEL", raising=False)
    result = runner.invoke(app, ["agent", "run", "--study", "funding_carry_2026", "--brief", "x"])
    assert result.exit_code == 2


def test_the_calendar_is_the_harnesss() -> None:
    done = agent.loop(["calendar"])
    assert done.returncode == 0 and "spot_2026:H" in done.stdout
