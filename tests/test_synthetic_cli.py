"""The synthetic and verify command groups: what they write and how they exit."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from trading_research import cli_synthetic
from trading_research.cli import app
from trading_research.synthetic.verify import CheckResult

runner = CliRunner()


def test_the_synthetic_commands_write_what_they_say(tmp_path) -> None:
    out = tmp_path / "flow"
    result = runner.invoke(
        app, ["synthetic", "order-flow", "--out", str(out), "--days", "1", "--null"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads((out / "truth.json").read_text())["planted"]["adverse_bp"] == 0.0
    assert runner.invoke(app, ["validate-data", "--input", str(out)]).exit_code == 0
    listed = runner.invoke(app, ["synthetic", "--help"]).output
    for name in ("book-days", "break-days", "cross-section", "order-flow", "funding"):
        assert name in listed


def test_every_generator_command_writes_a_truth(tmp_path) -> None:
    for command, extra in (
        ("book-days", ["--days", "2", "--session-s", "600"]),
        ("cross-section", ["--days", "2", "--instruments", "6"]),
        ("funding", ["--days", "30"]),
    ):
        out = tmp_path / command
        result = runner.invoke(app, ["synthetic", command, "--out", str(out), *extra])
        assert result.exit_code == 0, result.output
        assert json.loads((out / "truth.json").read_text())["source"] == "synthetic"


def test_verify_reversion_writes_its_page_and_exits_by_the_result(tmp_path) -> None:
    result = runner.invoke(
        app, ["verify", "reversion", "--paths", "1", "--days", "6", "--out", str(tmp_path)]
    )
    page = (tmp_path / "reversion" / "page.md").read_text()
    payload = json.loads((tmp_path / "reversion" / "result.json").read_text())
    assert result.exit_code == (0 if payload["passed"] else 1)
    assert "| relative: planted reversion found |" in page
    assert payload["seeds"]["paths"] == [20200106]


def test_a_failed_check_exits_non_zero(tmp_path, monkeypatch) -> None:
    def failing(**_: object) -> CheckResult:
        result = CheckResult("simulator", "t", "p", "a", "b", "c")
        result.add("finds it", "1", "0", "t >= 3", False)
        return result

    monkeypatch.setattr("trading_research.synthetic.verify.verify_simulator", failing)
    result = runner.invoke(app, ["verify", "simulator", "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert (tmp_path / "simulator" / "page.md").exists()


def test_the_command_line_registers_both_groups() -> None:
    names = {group.name for group in app.registered_groups}
    assert {"synthetic", "verify"} <= names
    assert cli_synthetic.synthetic_app.info.name == "synthetic"
