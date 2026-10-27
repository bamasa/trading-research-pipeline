"""The parity guarantee on the synthetic demo day: streaming features equal batch features.

The day is made by the ``generate-demo-data`` command and replayed through the
paper loop's interface. The batch side is ``features.build`` for the registry,
``build_timed`` for the timed set, and the ``prepare`` stage itself for the
prepared set; the streaming side holds only each set's declared window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.live.parity import ROUNDING_SENSITIVE, parity
from trading_research.live.replay import open_day

runner = CliRunner()


def demo_day(path: Path, steps: int) -> Path:
    result = runner.invoke(
        app, ["generate-demo-data", "--output", str(path), "--steps", str(steps)]
    )
    assert result.exit_code == 0, result.output
    return path


def test_streaming_features_equal_batch_features_at_every_boundary(tmp_path: Path) -> None:
    """A day shorter than the prepared set's declared look-back: every feature, every boundary."""
    day = open_day(demo_day(tmp_path / "demo", 1_500))
    report = parity(day, subsample=15, workdir=tmp_path / "work")
    table = report.table
    assert set(table["set"]) == {"registry", "timed", "prepared"}
    assert table.groupby("set")["boundaries"].max().to_dict() == {
        "prepared": 100,
        "registry": 1_500,
        "timed": 1_500,
    }
    assert (table["nan_mismatch"] == 0).all()
    assert table["within_tolerance"].all(), table[~table["within_tolerance"]]
    assert (table["compared"] > 0).sum() >= 0.9 * len(table)


def test_the_parity_command_prints_the_table_and_exits_zero(tmp_path: Path) -> None:
    path = demo_day(tmp_path / "demo", 600)
    out = tmp_path / "parity.csv"
    result = runner.invoke(
        app,
        ["paper", "parity", "--day", str(path), "--subsample", "20", "--sets", "registry,timed",
         "--output", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert out.exists() and "registry" in result.output


@pytest.mark.slow
def test_parity_on_the_default_demo_day_beyond_the_declared_look_back(tmp_path: Path) -> None:
    """20,000 rows, longer than the prepared set's 13,810-row look-back: the windows are cut.

    Everything is held to the tolerance except the rolling dispersions and their
    z-scores, whose deviation must stay at the rounding level of the input.
    """
    day = open_day(demo_day(tmp_path / "demo", 20_000))
    report = parity(day, subsample=50, workdir=tmp_path / "work")
    table = report.table
    assert report.ok
    held = table[~table["feature"].str.contains(ROUNDING_SENSITIVE)]
    assert held["within_tolerance"].all()
    dispersions = table[table["feature"].str.contains(r"_std\d+$")]
    assert float(dispersions["max_abs_dev"].max()) < 1e-8
    assert np.isfinite(table["max_abs_dev"]).all()
