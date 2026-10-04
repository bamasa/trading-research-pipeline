"""``trading-research mm-backtest``: a synthetic day end to end, and the held-out gate.

The command is exercised the way the README tells a reader to run it, on a day
directory built here in the layout the downloaders write. The gate is checked
before anything is read: a held-out day is refused without the flag, refused
with it unless the values are the frozen ones, and with both still goes through
the ledger, which here has no registration to check and refuses too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.market_making.synthetic import SYNTHETIC_DAY, random_market

runner = CliRunner()
REPO = Path(__file__).resolve().parent.parent
DAY = SYNTHETIC_DAY.isoformat()


@pytest.fixture(scope="module")
def day_dir(tmp_path_factory: pytest.TempPathFactory, write_market_day) -> Path:
    root = tmp_path_factory.mktemp("mm_cli")
    events = random_market(5, n_snapshots=86_400, snapshot_ms=1000, prints_per_snapshot=0.3)
    write_market_day(events, root)
    return root


def _roots(root: Path) -> list[str]:
    return [
        "--book",
        str(root / "book"),
        "--trades",
        str(root / "trades"),
        "--funding",
        str(root / "funding"),
    ]


def _output(result) -> str:  # type: ignore[no-untyped-def]
    return result.output + (getattr(result, "stderr", "") or "")


def test_mm_backtest_runs_a_synthetic_day_and_writes_its_tables(day_dir: Path) -> None:
    out = day_dir / "out"
    result = runner.invoke(
        app,
        [
            "mm-backtest", "--symbol", "FUZZUSDT", "--start", DAY, "--end", DAY,
            "--strategy", "s0", "--clip-notional", "250", "--clip-touch-share", "1",
            *_roots(day_dir), "--output", str(out),
        ],
    )  # fmt: skip
    assert result.exit_code == 0, _output(result)
    assert "Decomposition" in result.output and "outside the study" in result.output
    (run,) = list(out.iterdir())
    days = pd.read_csv(run / "days.csv")
    assert days["status"].tolist() == ["ok"] and days["fills"].iloc[0] > 0
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["plan"] == "S0" and manifest["access"] == "outside the study"
    assert manifest["summary"]["net_per_day"] == pytest.approx(days["net"].iloc[0])


def test_held_out_days_are_refused_without_the_flag(tmp_path: Path) -> None:
    args = ["mm-backtest", "--start", "2024-02-26", "--end", "2024-02-27", "--strategy", "s1"]
    result = runner.invoke(app, [*args, "--clip-notional", "20", "--sigma-ref", "8"])
    assert result.exit_code == 2
    assert "allow-heldout" in _output(result)

    hand_chosen = runner.invoke(app, [*args, "--clip-notional", "20", "--allow-heldout"])
    assert hand_chosen.exit_code == 2
    assert "--frozen" in _output(hand_chosen)


def test_with_the_flag_the_ledger_still_decides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # a repository without the registration's document
    config = REPO / "configs" / "mm_prereg.yaml"
    result = runner.invoke(
        app,
        [
            "mm-backtest", "--start", "2024-02-26", "--end", "2024-02-26", "--strategy", "s1",
            "--frozen", "--params", str(config), "--allow-heldout",
        ],
    )  # fmt: skip
    assert result.exit_code == 2
    assert "no pre-registration document" in _output(result)
    assert not (tmp_path / "experiments").exists()  # nothing recorded, nothing read
