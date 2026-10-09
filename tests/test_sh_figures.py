"""The search's figures render in both themes from tables shaped like the real ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from _sh_catalogue import catalogue_keys, synthetic_cells
from trading_research.reporting.leaderboard import leaderboard
from trading_research.reporting.marginals import marginals
from trading_research.reporting.nulls import nulls_table, s25_nulls_table
from trading_research.reporting.plots import both_themes
from trading_research.reporting.search_figures import (
    leaderboard_chart,
    marginals_chart,
    picture,
    placebo_search_chart,
)
from trading_research.reporting.search_keys import catalogue_table, neighbourhoods
from trading_research.validation.choice import StratumOutcome
from trading_research.validation.placebo_search import NullDraws, admission_iv


def _drawn(paths: list[Path]) -> None:
    assert [p.name.rsplit("_", 1)[1] for p in paths] == ["light.png", "dark.png"]
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths)


@pytest.fixture(scope="module")
def board() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = catalogue_table(catalogue_keys())
    cells, gross = synthetic_cells(len(table), days=20, seed=2)
    return table, leaderboard(table, cells, neighbourhoods(table), gross_bp=gross)


def test_the_placebo_search_chart(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    a_null = NullDraws(("C", "R"), rng.normal(1.5, 0.4, (999, 2)))
    b_null = NullDraws(("C", "R"), rng.normal(1.5, 0.4, (137, 2)))
    observed = {"C": StratumOutcome(1, 2.8), "R": StratumOutcome(None, 0.0)}
    results = admission_iv(observed, a_null, b_null, refit={"C": 2.0})
    table = nulls_table(results, a_null, b_null, ["a", "b"])
    _drawn(both_themes(lambda p: placebo_search_chart(table, p), tmp_path / "search.png"))
    s25 = s25_nulls_table(
        0.4,
        NullDraws(("C",), rng.normal(size=(999, 1))),
        NullDraws(("C",), rng.normal(size=(137, 1))),
    )
    _drawn(both_themes(lambda p: placebo_search_chart(s25, p, subject="S25"), tmp_path / "s25.png"))


def test_the_leaderboard_and_marginals_charts(
    board: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    table, found = board
    _drawn(both_themes(lambda p: leaderboard_chart(found, p), tmp_path / "board.png"))
    margins = marginals(found, table)
    _drawn(both_themes(lambda p: marginals_chart(margins, p), tmp_path / "margins.png"))
    one = found.loc[found["key"].str.startswith("BTCUSDT|C|")]
    _drawn(both_themes(lambda p: leaderboard_chart(one, p), tmp_path / "board_c.png"))


def test_the_readme_picture_block() -> None:
    block = picture("short_horizon_2025_nulls", "The placebo search")
    assert 'srcset="docs/images/short_horizon_2025_nulls_dark.png"' in block
    assert 'src="docs/images/short_horizon_2025_nulls_light.png"' in block
    assert block.startswith("<picture>") and block.endswith("</picture>")
