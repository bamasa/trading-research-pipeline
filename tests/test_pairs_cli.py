"""``trp pairs`` and ``trp verify pairs`` on synthetic data: what they write and how they exit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.synthetic import verify

runner = CliRunner()
A, B, C, D = "SYNTHAUSDT", "SYNTHBUSDT", "SYNTHCUSDT", "SYNTHDUSDT"


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("pairs") / "data"
    done = runner.invoke(app, ["pairs", "synthetic", "--out", str(root), "--days", "60"])
    assert done.exit_code == 0, done.output
    done = runner.invoke(
        app,
        ["pairs", "synthetic", "--out", str(root), "--days", "60", "--variant", "independent",
         "--seed", "7", "-s", C, "-s", D],
    )  # fmt: skip
    assert done.exit_code == 0, done.output
    return root


def test_synthetic_writes_bars_in_the_download_layout_and_its_truth(data: Path) -> None:
    for symbol in (A, B, C, D):
        folder = data / symbol / "klines-1h"
        assert len(list(folder.glob("*.parquet"))) == 60
        assert json.loads((folder / "manifest.json").read_text())["source"] == "synthetic"
    truth = json.loads((data / "truth.json").read_text())
    assert truth["planted"]["variant"] == "independent" and truth["null"] is True


def test_screen_ranks_the_planted_pair_first_and_writes_its_page(
    data: Path, tmp_path: Path
) -> None:
    out = tmp_path / "screen"
    done = runner.invoke(
        app,
        ["pairs", "screen", "--root", str(data), "-s", A, "-s", B, "-s", C, "-s", D,
         "--start", "2020-01-06", "--end", "2020-02-04", "--out", str(out)],
        terminal_width=200,
    )  # fmt: skip
    assert done.exit_code == 0, done.output
    table = pd.read_csv(out / "screen.csv")
    assert len(table) == 6
    assert (table.iloc[0]["a"], table.iloc[0]["b"]) == (A, B) and table.iloc[0]["passes"]
    assert not table.iloc[1:]["passes"].any()
    page = (out / "page.md").read_text()
    assert "# pairs screen" in page and "Holm" in page and "720 bars" in page
    assert json.loads((out / "summary.json").read_text())["passing"] == 1


def test_backtest_writes_trades_days_windows_and_a_page(data: Path, tmp_path: Path) -> None:
    out = tmp_path / "backtest"
    done = runner.invoke(
        app,
        ["pairs", "backtest", "--root", str(data), "-s", A, "-s", B, "--fee-bp", "4",
         "--trading", "120", "--out", str(out)],
        terminal_width=200,
    )  # fmt: skip
    assert done.exit_code == 0, done.output
    for name in ("page.md", "trades.csv", "days.csv", "windows.csv", "summary.json"):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["trades"] > 0 and summary["fee_bp_per_fill"] == 4.0
    assert summary["rule_trading"] == 120
    assert "cost per fill at which net is zero" in (out / "page.md").read_text()


def test_backtest_and_screen_refuse_what_they_cannot_read(data: Path, tmp_path: Path) -> None:
    one = runner.invoke(app, ["pairs", "backtest", "--root", str(data), "-s", A])
    assert one.exit_code == 2
    missing = runner.invoke(
        app, ["pairs", "screen", "--root", str(tmp_path), "-s", A, "-s", B,
              "--start", "2020-01-06", "--end", "2020-01-10"],
    )  # fmt: skip
    assert missing.exit_code == 1 and "no 1h bars" in missing.output
    (data / B / "klines-1h" / "2020-01-20.parquet").rename(tmp_path / "moved.parquet")
    try:
        gap = runner.invoke(app, ["pairs", "backtest", "--root", str(data), "-s", A, "-s", B])
        assert gap.exit_code == 1 and "not on the same bars" in gap.output
    finally:
        (tmp_path / "moved.parquet").rename(data / B / "klines-1h" / "2020-01-20.parquet")


def test_pairs_is_a_group_in_the_steps_panel() -> None:
    result = runner.invoke(app, ["--help"], terminal_width=200)
    panel = ""
    for line in result.output.splitlines():
        if line.startswith("╭─"):
            panel = line
        elif line.startswith("│ pairs "):
            assert "The pipeline, step by step" in panel
            break
    else:
        raise AssertionError("pairs is not in the help")
    for command in ("screen", "backtest", "synthetic"):
        shown = runner.invoke(app, ["pairs", command, "--help"], terminal_width=200)
        assert shown.exit_code == 0 and "--out" in shown.output


def test_verify_all_runs_the_pairs_check_and_quick_shrinks_it() -> None:
    assert "pairs" in verify.CHECKS and verify.QUICK["pairs"] == {"paths": 1}


def test_verify_pairs_passes_and_writes_its_page(tmp_path: Path) -> None:
    done = runner.invoke(app, ["verify", "pairs", "--paths", "1", "--out", str(tmp_path)])
    assert done.exit_code == 0, done.output
    result = json.loads((tmp_path / "pairs" / "result.json").read_text())
    assert result["passed"] is True
    checks = {c["check"]: c["result"] for c in result["criteria"]}
    for name in (
        "screen: the planted pair ranks first",
        "screen: independent walks do not pass after Holm",
        "backtest: planted, positive before costs",
        "backtest: independent walks, not positive after costs",
        "backtest: after the break, not positive after costs",
        "no decision reads a later bar",
    ):
        assert checks[name] == "pass", name
