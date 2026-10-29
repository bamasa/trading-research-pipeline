"""The search's figures render in both themes from tables shaped like the real ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from _sh_catalogue import catalogue_keys, synthetic_cells
from trading_research.reporting.edge_ladder import edge_ladder
from trading_research.reporting.leaderboard import leaderboard
from trading_research.reporting.marginals import marginals
from trading_research.reporting.nulls import nulls_table, s25_nulls_table
from trading_research.reporting.plots import both_themes
from trading_research.reporting.search_figures import (
    edge_ladder_chart,
    fill_rule_chart,
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


def test_the_edge_ladder_chart_before_and_after_the_reads(tmp_path: Path) -> None:
    full = edge_ladder(
        a0_zero_fee={"crossing": 4.0, "touch": -1.0, "trade_through": 2.0, "queue": -2.0},
        s25_d=(3.0, -8.0),
        s25_h_window=-6.0,
        s25_h_all=-7.0,
        s25_break_even=1.7,
        s25z_net_zero_fee=-1.0,
    )
    _drawn(both_themes(lambda p: edge_ladder_chart(full, p), tmp_path / "ladder.png"))
    _drawn(both_themes(lambda p: edge_ladder_chart(edge_ladder(), p), tmp_path / "empty.png"))


def test_the_fill_rule_chart(tmp_path: Path) -> None:
    def summary(*ratios: float) -> pd.DataFrame:
        return pd.DataFrame({"rule": ["touch", "trade_through", "queue"], "fill_ratio": ratios})

    streams = {"A0 on O": summary(0.8, 0.2, 0.4), "S-P on H": summary(0.7, 0.1, np.nan)}
    _drawn(both_themes(lambda p: fill_rule_chart(streams, p), tmp_path / "fill.png"))
    with pytest.raises(ValueError):
        fill_rule_chart({str(k): summary(1, 1, 1) for k in range(4)}, tmp_path / "x.png")
