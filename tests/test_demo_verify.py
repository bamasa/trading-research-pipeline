"""``trp demo --verify`` and ``verify all --quick``: the checks at the quick sizes, their table on the page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.synthetic import verify
from trading_research.walkthrough import demo

runner = CliRunner()


def _fake(what: str, calls: list[tuple[str, dict[str, Any]]], fail: bool) -> Any:
    def check(**kwargs: Any) -> verify.CheckResult:
        calls.append((what, kwargs))
        result = verify.CheckResult(what, what, "planted", "proves", "not", f"verify {what}")
        result.add(f"{what}: criterion", "1", "1", "exact", not fail)
        result.add(f"{what}: measured", "1", "1", "info", True, binding=False)
        result.seconds = 1.0
        return result

    return check


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    checks = {w: _fake(w, calls, fail=w == "detector") for w in verify.CHECKS}
    monkeypatch.setattr(verify, "CHECKS", checks)
    return calls


def test_quick_names_every_check_and_only_shrinks_sizes() -> None:
    assert set(verify.QUICK) == set(verify.CHECKS)
    sizes = {"days", "paths"}
    assert all(set(k) <= sizes for k in verify.QUICK.values())


def test_verify_all_quick_passes_the_quick_sizes(
    fakes: list[tuple[str, dict[str, Any]]], tmp_path: Path
) -> None:
    result = runner.invoke(app, ["verify", "all", "--quick", "--out", str(tmp_path)])
    assert result.exit_code == 1  # the fake detector fails
    for what, kwargs in fakes:
        assert {k: v for k, v in kwargs.items() if k in ("days", "paths")} == verify.QUICK[what]
    assert (tmp_path / "all" / "page.md").exists()


def test_the_demo_page_gets_one_row_per_check(
    fakes: list[tuple[str, dict[str, Any]]], tmp_path: Path
) -> None:
    checks = demo.verify_quick(tmp_path / "verify", 20200106, report=lambda _line: None)
    assert [w for w, _ in fakes] == list(verify.CHECKS)
    section = demo.verify_section(checks, tmp_path / "verify", 20200106)
    assert "## Verify: the components on planted answers" in section
    assert "| detector | FAIL | 0 of 1 | detector: criterion |" in section
    assert "| edge | pass | 1 of 1 | none |" in section
    assert "seed 20200106" in section and "`verify/all/page.md`" in section


def test_demo_takes_verify_and_the_config_carries_it() -> None:
    result = runner.invoke(app, ["demo", "--help"], terminal_width=200)
    assert "--verify" in result.output
    assert demo.DemoConfig(verify=True).verify and not demo.DemoConfig().verify
