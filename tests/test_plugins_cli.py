"""``trading-research strategies``: the commands a reader of docs/extending.md runs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.plugins import KINDS, conformance, registry
from trading_research.plugins.market import demo_market

runner = CliRunner()

#: The smallest market the commands accept, to keep these tests quick.
SMALL = ["--steps", "24000"]


def test_list_shows_every_kind_and_the_worked_example() -> None:
    result = runner.invoke(app, ["strategies", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    for name in ("book_imbalance", "reversion", "logistic", "generated", "walked_taker", "maker"):
        assert name in result.output
    only = runner.invoke(app, ["strategies", "list", "--kind", "execution"], env={"COLUMNS": "200"})
    assert "taker" in only.output and "logistic" not in only.output


def test_show_states_the_declarations() -> None:
    result = runner.invoke(app, ["strategies", "show", "book_imbalance"])
    assert result.exit_code == 0, result.output
    assert "horizon      30" in result.output
    assert "lookback     0" in result.output
    assert "contrib" in result.output
    bigger = runner.invoke(app, ["strategies", "show", "book_imbalance", "-p", "hold=60"])
    assert "horizon      60" in bigger.output


def test_unknown_names_and_kinds_are_refused() -> None:
    assert runner.invoke(app, ["strategies", "show", "no_such_rule"]).exit_code == 2
    assert runner.invoke(app, ["strategies", "list", "--kind", "sizing"]).exit_code == 2
    assert runner.invoke(app, ["strategies", "check"]).exit_code == 2


@pytest.fixture
def contrib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Every kind's contrib folder redirected to an empty package under ``tmp_path``."""
    root = tmp_path / "src"
    tag = uuid.uuid4().hex[:8]  # a package name imported by no earlier test
    for kind in KINDS:
        folder = root / f"contrib_{kind}_{tag}"
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("")
        monkeypatch.setitem(registry.CONTRIB, kind, f"contrib_{kind}_{tag}")
    monkeypatch.syspath_prepend(str(root))
    registry.discover(force=True)
    yield tmp_path
    monkeypatch.undo()
    registry.discover(force=True)


@pytest.mark.parametrize("kind", KINDS)
def test_new_writes_a_plugin_that_passes_check_as_written(contrib: Path, kind: str) -> None:
    name = f"fresh_{kind}"
    tests = contrib / "tests"
    result = runner.invoke(app, ["strategies", "new", name, "--kind", kind, "--tests", str(tests)])
    assert result.exit_code == 0, result.output
    [module] = (contrib / "src").glob(f"contrib_{kind}_*/{name}.py")
    test = tests / f"test_plugins_contrib_{name}.py"
    assert module.exists() and test.exists()
    assert f'register_{kind}("{name}")' in module.read_text()
    assert f'registry.get("{name}", "{kind}")' in test.read_text()

    registry.discover(force=True)
    spec = registry.get(name, kind)
    assert spec.origin == "contrib"
    setup = conformance.make_setup(executions=["taker"], market=demo_market(steps=24_000, peers=1))
    checked = conformance.check(spec, setup)
    assert checked.passed, [(r.check, r.detail) for r in checked.results]

    again = runner.invoke(app, ["strategies", "new", name, "--kind", kind, "--tests", str(tests)])
    assert again.exit_code == 2


def test_new_refuses_a_name_that_is_not_a_module_name(contrib: Path) -> None:
    result = runner.invoke(app, ["strategies", "new", "Book-Imbalance", "--tests", str(contrib)])
    assert result.exit_code == 2
    taken = runner.invoke(app, ["strategies", "new", "reversion", "--tests", str(contrib)])
    assert taken.exit_code == 2


def test_check_writes_the_page(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "strategies",
            "check",
            "book_imbalance",
            "-e",
            "taker,passive",
            *SMALL,
            "-o",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    page = (tmp_path / "book_imbalance" / "page.md").read_text()
    assert "passes every check" in page
    table = pd.read_csv(tmp_path / "book_imbalance" / "execution.csv")
    assert list(table["execution"]) == ["taker", "passive"]


def test_backtest_runs_a_strategy_and_a_model(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["strategies", "backtest", "book_imbalance", *SMALL, "-o", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "book_imbalance" / "backtest"
    table = pd.read_csv(out / "results.csv")
    assert list(table["execution"]) == ["taker", "passive"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["plugin"]["origin"] == "contrib"
    assert manifest["decision_rows"] > 0 and manifest["fee_tier"] == "base"

    model = runner.invoke(
        app, ["strategies", "backtest", "order_flow", "-e", "taker", *SMALL, "-o", str(tmp_path)]
    )
    assert model.exit_code == 0, model.output
    feature = runner.invoke(app, ["strategies", "backtest", "book", "-o", str(tmp_path)])
    assert feature.exit_code == 2


def test_backtest_reads_a_dataset_written_by_generate_demo_data(tmp_path: Path) -> None:
    data = tmp_path / "demo"
    made = runner.invoke(app, ["generate-demo-data", "--output", str(data), "--steps", "24000"])
    assert made.exit_code == 0, made.output
    result = runner.invoke(
        app,
        [
            "strategies",
            "backtest",
            "book_imbalance",
            "-i",
            str(data),
            "-e",
            "taker",
            "-o",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "book_imbalance" / "backtest" / "manifest.json").read_text())
    assert str(data) in manifest["data"]


def test_the_worker_checks_one_plugin_in_its_own_process(tmp_path: Path) -> None:
    from trading_research.plugins import worker

    row = worker.run("label", "directional", "built-in", tmp_path, steps=24_000)
    assert row["passed"] == "yes", row
    assert (tmp_path / "directional" / "page.md").exists()
