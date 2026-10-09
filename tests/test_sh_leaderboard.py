"""The D leaderboard agrees with the registered choice, and its file round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _sh_catalogue import catalogue_keys, synthetic_cells
from trading_research.reporting.leaderboard import (
    LEADERBOARD_COLUMNS,
    day_level_t,
    leaderboard,
    read_leaderboard,
    region_rows,
    s25_rank,
    write_leaderboard,
)
from trading_research.reporting.search_keys import (
    REGIONS,
    S25_KEY,
    catalogue_table,
    in_region,
    neighbourhoods,
)
from trading_research.validation.choice import CellDays, ChoiceError, choose, statistic_a


@pytest.fixture(scope="module")
def catalogue() -> tuple[pd.DataFrame, np.ndarray]:
    table = catalogue_table(catalogue_keys())
    return table, neighbourhoods(table)


@pytest.fixture(scope="module")
def board(catalogue: tuple[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    table, index = catalogue
    cells, gross = synthetic_cells(len(table), seed=3)
    return leaderboard(table, cells, index, gross_bp=gross, ic=np.zeros(len(table)))


def test_the_columns_are_the_registered_ones(board: pd.DataFrame) -> None:
    assert tuple(board.columns) == LEADERBOARD_COLUMNS
    assert len(board) == 2400
    assert sorted(board["rank_overall"]) == list(range(1, 2401))


def test_the_ranks_agree_with_the_choice(catalogue: tuple[pd.DataFrame, np.ndarray]) -> None:
    table, index = catalogue
    cells, _ = synthetic_cells(len(table), seed=5)
    board = leaderboard(table, cells, index)
    for stratum in ("C", "R"):
        members = np.flatnonzero(table["stratum"] == stratum)
        found = choose(cells, members, index)
        mine = board.iloc[members].sort_values("rank_in_stratum")
        assert mine.index[0] == found.peak
        assert mine.loc[mine["admissible"]].index[0] == found.cell
        np.testing.assert_allclose(board["S"].to_numpy()[members], found.s[members])
        assert board.loc[found.cell, "A"] == pytest.approx(statistic_a(cells, index, found.cell))
        assert sorted(mine["rank_in_stratum"]) == list(range(1, len(members) + 1))


def test_s25_carries_its_badge_and_rank(board: pd.DataFrame) -> None:
    badges = board.loc[board["S25_badge"] != ""]
    assert list(badges["key"]) == [S25_KEY]
    rank, n = s25_rank(board)
    assert n == 2400 and badges["S25_badge"].iloc[0] == f"S25, rank {rank} of 2400"
    with pytest.raises(ChoiceError):
        s25_rank(board.loc[board["key"] != S25_KEY])


def test_each_region_reports_its_best_cell(
    board: pd.DataFrame, catalogue: tuple[pd.DataFrame, np.ndarray]
) -> None:
    table, _ = catalogue
    rows = region_rows(board, table).set_index("region")
    assert rows.loc["T1", "cells"] == 16 and rows.loc["T2", "cells"] == 36
    for name, values in REGIONS.items():
        members = board.loc[in_region(table, values)]
        assert rows.loc[name, "rank_overall"] == members["rank_overall"].min()
        assert rows.loc[name, "key"] in set(members["key"])
        assert name in board.loc[board["key"] == rows.loc[name, "key"], "region"].iloc[0]


def test_a_cell_without_a_trade_scores_zero_and_has_no_t() -> None:
    table = catalogue_table([S25_KEY])
    cells = CellDays(np.zeros((1, 4), dtype=int), np.full((1, 4), np.nan), np.full((1, 4), np.nan))
    row = leaderboard(table, cells, np.array([[0]])).iloc[0]
    assert row["net"] == 0.0 and row["S"] == 0.0 and row["A"] == 0.0
    assert np.isnan(row["day_t"]) and not row["admissible"]
    assert np.isnan(day_level_t(np.array([[1.0, 1.0, np.nan]])))[0]


def test_ties_go_to_the_own_mean_then_to_the_earlier_row() -> None:
    keys = catalogue_keys()[:3]
    table = catalogue_table(keys)
    net = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
    cells = CellDays(np.full((3, 2), 6), net, net.copy())
    index = np.array([[0, 1, 2]] * 3)  # every cell has the same S
    board = leaderboard(table, cells, index)
    assert list(board["rank_overall"]) == [2, 1, 3]


def test_the_shapes_must_agree(catalogue: tuple[pd.DataFrame, np.ndarray]) -> None:
    table, index = catalogue
    cells, _ = synthetic_cells(10)
    with pytest.raises(ChoiceError):
        leaderboard(table, cells, index)


def test_the_file_round_trips_with_its_header(board: pd.DataFrame, tmp_path: Path) -> None:
    path = write_leaderboard(board, tmp_path / "lb.csv", planned=2400, scored=2400, skipped=0)
    assert path.read_text().startswith("# N planned: 2400\n# N scored: 2400\n# N skipped: 0\n")
    assert "selected, biased upward" in path.read_text().splitlines()[3]
    counts, back = read_leaderboard(path)
    assert counts == {"planned": 2400, "scored": 2400, "skipped": 0}
    assert list(back["key"]) == list(board["key"])
    assert list(back["region"]) == list(board["region"])
    assert list(back["S25_badge"]) == list(board["S25_badge"])
    np.testing.assert_allclose(back["S"], board["S"], rtol=1e-5)
    with pytest.raises(ChoiceError):
        write_leaderboard(board, tmp_path / "x.csv", planned=2400, scored=2399, skipped=0)
