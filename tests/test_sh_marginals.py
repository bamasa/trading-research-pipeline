"""Per-axis marginals of the leaderboard, on boards whose medians are known."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _sh_catalogue import catalogue_keys
from trading_research.reporting.marginals import (
    MARGINAL_COLUMNS,
    axis_spread,
    marginals,
    write_marginals,
)
from trading_research.reporting.search_keys import catalogue_table
from trading_research.validation.choice import ChoiceError


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return catalogue_table(catalogue_keys())


def _board(table: pd.DataFrame, net: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"key": table["key"], "gross": net + 11.0, "net": net, "G": -net})


def test_the_regime_rows_are_one_per_window_in_window_order(table: pd.DataFrame) -> None:
    found = marginals(_board(table, table["regime_h"].to_numpy()), table)
    assert tuple(found.columns) == MARGINAL_COLUMNS
    regime = found.loc[(found["scope"] == "all") & (found["axis"] == "regime")]
    assert list(regime["value"]) == ["none", "pull_4h", "pull_24h", "pull_120h"]
    assert list(regime["net"]) == [0.0, 4.0, 24.0, 120.0]
    assert list(regime["gross"]) == [11.0, 15.0, 35.0, 131.0]
    assert list(regime["G"]) == [0.0, -4.0, -24.0, -120.0]
    assert list(regime["cells"]) == [600] * 4


def test_each_stratum_reports_its_own_axes(table: pd.DataFrame) -> None:
    found = marginals(_board(table, table["hold_s"].to_numpy()), table)
    c = found.loc[found["scope"] == "C"]
    r = found.loc[found["scope"] == "R"]
    assert set(c["axis"]) == {
        "model",
        "inputs",
        "gate",
        "hold_s",
        "trades_per_day",
        "label_fee_bp",
        "regime",
    }
    assert set(r["axis"]) == {"model", "gate", "hold_s", "k", "beta_bp", "regime"}
    assert list(r.loc[r["axis"] == "k", "value"]) == ["0", "0.5", "1"]
    assert list(r.loc[r["axis"] == "k", "cells"]) == [288] * 3
    assert list(c.loc[c["axis"] == "model", "cells"]) == [768, 768]
    holds = c.loc[c["axis"] == "hold_s"]
    assert list(holds["value"]) == ["5", "20", "50", "120"]
    assert list(holds["net"]) == [5.0, 20.0, 50.0, 120.0]
    strata = found.loc[found["axis"] == "stratum"]
    assert list(strata["value"]) == ["C", "R"] and list(strata["cells"]) == [1536, 864]


def test_how_far_each_axis_moves_the_median(table: pd.DataFrame) -> None:
    spread = axis_spread(marginals(_board(table, table["regime_h"].to_numpy()), table))
    by = spread.set_index(["scope", "axis"])["net_range"]
    assert by[("all", "regime")] == 120.0
    assert by[("C", "hold_s")] == 0.0


def test_a_missing_number_gives_nan_and_mismatched_keys_are_refused(table: pd.DataFrame) -> None:
    board = _board(table, np.zeros(len(table)))
    board["gross"] = np.nan
    assert marginals(board, table)["gross"].isna().all()
    with pytest.raises(ChoiceError):
        marginals(board.iloc[::-1], table)


def test_the_file_carries_the_development_label(table: pd.DataFrame, tmp_path: Path) -> None:
    found = marginals(_board(table, table["hold_s"].to_numpy()), table)
    path = write_marginals(found, tmp_path / "m.csv")
    assert path.read_text().splitlines()[0].endswith("never quoted as a result")
    back = pd.read_csv(path, comment="#", dtype={"value": str})
    assert list(back["value"]) == list(found["value"])
    np.testing.assert_allclose(back["net"], found["net"])
