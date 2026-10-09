"""The finalists page: statistics from the daily series, kills and order."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_research.backtest.costs import tiers_for
from trading_research.evaluation.significance import newey_west_t
from trading_research.market_making.verdicts import KillCheck, day_t
from trading_research.pipeline.regime import Attribution
from trading_research.reporting.finalists import (
    FINALIST_COLUMNS,
    FinalistInputs,
    finalists_table,
    markdown_table,
    tiers_paying,
)
from trading_research.reporting.headlines import HeadlineError

DAILY = np.random.default_rng(4).normal(-1.0, 3.0, 24)


def _kill(name: str, value: float, fired: bool = True) -> KillCheck:
    return KillCheck("x", name, "", value, 0.0, fired)


def test_a_row_carries_the_bars_statistics_and_the_first_kill() -> None:
    draws = np.arange(20, dtype=float)
    attribution = Attribution("pull_4h", 3.0, float(np.quantile(draws, 0.95)), draws, 5)
    row = finalists_table(
        [
            FinalistInputs(
                "S-C", "H", "killed", DAILY, 140,
                kills=(_kill("K-days", 0.4), _kill("K1", -1.0)), mde=2.5, break_even_per_side=1.6,
                attribution=attribution,
            )
        ]
    ).iloc[0]  # fmt: skip
    found = day_t(DAILY)
    assert row["mean"] == pytest.approx(found.mean) and row["t"] == pytest.approx(found.t)
    assert row["newey_west_t"] == pytest.approx(newey_west_t(DAILY).t)
    assert row["median_day"] == pytest.approx(np.median(DAILY)) and row["days"] == 24
    assert (row["first_kill"], row["kill_value"]) == ("K1", -1.0)
    assert row["tiers_paying"] == 1  # only pro6_g1 charges less than 1.6 bp a side
    assert row["regime_difference"] == 3.0 and row["unit"] == "bp of traded notional"


def test_tiers_paying_reads_the_side_of_the_break_even() -> None:
    tiers = tiers_for("g1")
    assert tiers_paying("S25", 5.6) == len(tiers)
    assert tiers_paying("S25", 1.5) == 0
    assert tiers_paying("S-P", 0.1) == sum(t.maker_bp < 0.1 for t in tiers)
    assert tiers_paying("S-M", math.inf) == len(tiers)
    assert tiers_paying("S-M", -math.inf) == 0 and tiers_paying("S25", math.nan) == 0


def test_the_page_is_ordered_by_block_then_family() -> None:
    rows = [
        FinalistInputs("S25-Z", "H", "killed", DAILY, 120, kills=(_kill("K1", -1.0),)),
        FinalistInputs("S-M", "F", "inconclusive", DAILY, 120),
        FinalistInputs("S25", "H", "candidate (H before 2025-08-01)", np.abs(DAILY) + 1, 300),
        FinalistInputs("S-P", "H", "not admitted on D (i)", [], 0),
    ]
    table = finalists_table(rows)
    assert tuple(table.columns) == FINALIST_COLUMNS
    assert list(zip(table["block"], table["slot"], strict=True)) == [
        ("H", "S25"), ("H", "S-P"), ("H", "S25-Z"), ("F", "S-M"),
    ]  # fmt: skip
    empty = table.loc[table["slot"] == "S-P"].iloc[0]
    assert empty["days"] == 0 and math.isnan(empty["median_day"]) and empty["first_kill"] == ""
    assert table.loc[table["slot"] == "S-M", "unit"].iloc[0] == "per 100 USDT of clip a day"
    with pytest.raises(HeadlineError):
        finalists_table([rows[0], rows[0]])
    with pytest.raises(HeadlineError):
        finalists_table([FinalistInputs("S-C", "D", "killed", DAILY, 1)])


def test_the_markdown_table() -> None:
    table = finalists_table([FinalistInputs("S-C", "H", "not admitted on D (i, ii)", [], 0)])
    text = markdown_table(table[["slot", "status", "mean", "trades"]])
    assert text.splitlines() == [
        "| slot | status | mean | trades |",
        "|---|---|---|---|",
        "| S-C | not admitted on D (i, ii) |  | 0 |",
    ]
    assert "a \\| b" in markdown_table(table[["slot"]].assign(slot="a | b"))


def test_each_row_carries_its_registered_sentence() -> None:
    table = finalists_table(
        [FinalistInputs("S25-Z", "H", "killed", DAILY, 120, kills=(_kill("K2", -0.5),))]
    )
    assert (
        table["sentence"].iloc[0] == "S25-Z on H: killed; first kill K2 at -0.50 against 0.00 bp."
    )
    with pytest.raises(HeadlineError):
        finalists_table([FinalistInputs("S-C", "H", "killed", DAILY, 120)])
