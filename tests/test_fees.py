"""The fee tables, read where they live and listed in one place; ``trp fees`` prints them."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from trading_research import fees
from trading_research.backtest import costs
from trading_research.cli import app

REPO = Path(__file__).resolve().parent.parent


def test_every_bybit_linear_tier_is_listed_as_the_cost_module_holds_it() -> None:
    listed = {r.tier: r for r in fees.code_rows() if r.venue == fees.BYBIT_LINEAR}
    for tier in costs.BYBIT_LINEAR_TIERS:
        row = listed[tier.name]
        assert (row.maker_bp, row.taker_bp, row.group) == (tier.maker_bp, tier.taker_bp, tier.group)
        assert costs.BYBIT_FEE_SOURCE in row.source and costs.BYBIT_FEE_RETRIEVED in row.period
    assert listed["mm_programme"].maker_bp == costs.BYBIT_MM_PROGRAMME_MAKER_BP
    assert math.isnan(listed["mm_programme"].taker_bp)  # not published, never filled in


def test_the_registrations_fee_blocks_are_read_from_their_files() -> None:
    table = fees.table(REPO)
    mm = yaml.safe_load((REPO / "configs/mm_prereg.yaml").read_text())["fees"]["base"]
    row = table[table["source"] == "configs/mm_prereg.yaml fees"].iloc[0]
    assert (row["maker_bp"], row["taker_bp"]) == (mm["maker_bp"], mm["taker_bp"])
    assert row["period"].startswith("2024-")
    spot = yaml.safe_load((REPO / "configs/spot_2026.yaml").read_text())["fees"]["intervals"]
    expected = sum(len(i["tiers"]) for intervals in spot.values() for i in intervals)
    assert table["source"].str.startswith("configs/spot_2026.yaml fees.intervals").sum() == expected
    assert (table["source"] == "configs/short_horizon_2025.yaml common.fees.verdict_tier").any()


def test_a_tier_is_looked_up_or_refused() -> None:
    base = fees.tier("base")
    assert (base.maker_bp, base.taker_bp) == (costs.BYBIT_BASE.maker_bp, costs.BYBIT_BASE.taker_bp)
    with pytest.raises(KeyError, match="no fee tier"):
        fees.tier("vip99")
    assert [r.tier for r in fees.ladder("g1")] == [t.name for t in costs.tiers_for("g1")]


def test_the_command_prints_the_table_and_writes_it(tmp_path: Path) -> None:
    out = tmp_path / "fees.csv"
    result = CliRunner().invoke(app, ["fees", "--venue", fees.BYBIT_LINEAR, "--csv", str(out)])
    assert result.exit_code == 0, result.output
    written = pd.read_csv(out)
    assert set(written["venue"]) == {fees.BYBIT_LINEAR}
    assert "base" in result.output and "rebate" in result.output
    assert CliRunner().invoke(app, ["fees", "--venue", "nowhere"]).exit_code == 1
