"""The command tree: groups by step, the flat names kept as hidden aliases, help as a map."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import typer.main
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.cli_common import MAP, ORDER

REPO = Path(__file__).resolve().parent.parent
runner = CliRunner()

#: Every command the flat app had before the tree, which tests and documents call.
FLAT = (
    "generate-demo-data", "validate-data", "download", "prepare", "select", "train", "predict",
    "backtest", "retrain-search", "breaks", "exit-search", "screen", "review", "discover",
    "download-book", "describe-schema", "reproduce", "mm-backtest",
)  # fmt: skip

GROUPS = {
    "data": ("download", "download-book", "generate-demo-data", "validate-data", "describe-schema"),
    "regimes": ("breaks",),
    "instruments": ("discover", "screen"),
    "signal": ("prepare", "select", "train", "predict"),
    "backtest": ("backtest", "mm-backtest", "exit-search", "retrain-search"),
}


def root():  # type: ignore[no-untyped-def]
    return typer.main.get_command(app)


def test_every_flat_command_still_exists_and_only_review_and_reproduce_show() -> None:
    commands = root().commands  # type: ignore[attr-defined]
    for name in FLAT:
        assert name in commands, name
        if name == "backtest":  # the group of that name stands in for it, below
            continue
        assert commands[name].hidden == (name not in ("review", "reproduce")), name


def test_the_backtest_group_is_still_the_taker_backtest_without_a_subcommand(
    tmp_path: Path,
) -> None:
    grouped = root().commands["backtest"]  # type: ignore[attr-defined]
    flat_options = [p.opts for p in grouped.commands["backtest"].params]
    assert [p.opts for p in grouped.params] == flat_options
    result = runner.invoke(app, ["backtest", "--predictions", str(tmp_path / "none")])
    assert result.exit_code == 1 and "predictions.parquet" in result.output


def test_each_group_holds_the_flat_command_itself() -> None:
    flat = {info.name: info.callback for info in app.registered_commands}
    groups = {g.typer_instance.info.name: g.typer_instance for g in app.registered_groups}
    commands = root().commands  # type: ignore[attr-defined]
    for group, names in GROUPS.items():
        held = groups[group].registered_commands  # type: ignore[union-attr]
        assert tuple(info.name for info in held) == names
        assert all(info.callback is flat[info.name] for info in held)
        clicked = commands[group].commands
        for name in names:
            source = commands[name] if name != group else clicked[name]
            assert [p.opts for p in clicked[name].params] == [p.opts for p in source.params]


def test_the_grouped_and_the_flat_name_print_the_same() -> None:
    flat = runner.invoke(app, ["describe-schema", "bars"])
    grouped = runner.invoke(app, ["data", "describe-schema", "bars"])
    assert flat.exit_code == grouped.exit_code == 0
    assert flat.output == grouped.output


def test_the_help_lists_the_steps_in_the_order_the_pipeline_runs() -> None:
    result = runner.invoke(app, ["--help"], terminal_width=200)
    assert result.exit_code == 0
    rows = [line for line in result.output.splitlines() if line.startswith("│")]
    lines = [line.strip("│ ").split(" ")[0] for line in rows]
    shown = [name for name in lines if name in ORDER]
    assert [ORDER.index(name) for name in shown] == sorted(ORDER.index(n) for n in shown)
    assert {"data", "regimes", "instruments", "signal", "backtest", "review"} <= set(shown)
    assert "data → regimes" in result.output and "verdict → study → agent" in result.output
    assert MAP.startswith("data → regimes")
    for hidden in ("download-book", "mm-backtest", "exit-search", "generate-demo-data"):
        assert hidden not in lines


def test_trp_and_trading_research_are_the_same_application() -> None:
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["trp"] == scripts["trading-research"] == "trading_research.cli:app"


def test_the_package_runs_as_a_module_with_every_group() -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "COLUMNS": "200"}
    done = subprocess.run(
        [sys.executable, "-m", "trading_research", "--help"],
        cwd=REPO, capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    for group in GROUPS:
        assert f" {group} " in done.stdout


#: Every top-level group the ``trp`` script exposes, and the panel its help sits in.
EVERY_GROUP = {
    "data": "The pipeline, step by step",
    "regimes": "The pipeline, step by step",
    "instruments": "The pipeline, step by step",
    "signal": "The pipeline, step by step",
    "search": "The pipeline, step by step",
    "backtest": "The pipeline, step by step",
    "review": "The pipeline, step by step",
    "strategies": "Extend",
    "synthetic": "Check",
    "verify": "Check",
    "study": "Registered studies",
    "agent": "Registered studies",
    "paper": "Deploy",
    "demo": "End to end",
    "reproduce": "End to end",
    "fees": "End to end",
    "catalog": "End to end",
}


def test_trp_exposes_every_group_in_its_panel_in_pipeline_order() -> None:
    assert tuple(EVERY_GROUP) == ORDER
    result = runner.invoke(app, ["--help"], terminal_width=200)
    assert result.exit_code == 0
    panel, where, seen = "", {}, []
    for line in result.output.splitlines():
        if line.startswith("╭─"):
            panel = line.strip("╭─ ╮").strip("─ ")
        elif line.startswith("│") and panel != "Options":
            name = line.strip("│ ").split(" ")[0]
            if name in EVERY_GROUP:
                where[name] = panel
                seen.append(name)
    assert seen == list(ORDER)
    for name, prefix in EVERY_GROUP.items():
        assert where[name].startswith(prefix), (name, where[name])
    assert "agent → paper" in result.output
