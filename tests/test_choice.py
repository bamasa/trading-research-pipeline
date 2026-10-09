"""The registered choice on D, on synthetic grids whose answer is known."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.verdicts import day_t
from trading_research.validation.choice import (
    TEST_MONTHS,
    CellDays,
    ChoiceError,
    carry_forward,
    cell_day_t,
    cell_means,
    choose,
    neighbourhood_index,
    neighbourhood_median,
    not_admitted,
    rerun,
    slot_conditions,
    statistic_a,
)
from trading_research.validation.search import neighbourhood_scores

#: Stratum C's axes, ordered ones as numbers (regime window in hours).
C_AXES = {
    "model": ["gbm", "logistic"],
    "inputs": ["micro", "micro+tensor"],
    "gate": ["open", "quiet_out", "tight_only"],
    "hold_s": [5, 20, 50, 120],
    "trades_per_day": [25, 50, 100, 240],
    "label_fee_bp": [0, 5.5],
    "regime_h": [0, 4, 24, 120],
}
C_ORDERED = ["hold_s", "trades_per_day", "label_fee_bp", "regime_h"]
C_CATEGORICAL = ["model", "inputs", "gate"]


def _catalogue_c() -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product(*C_AXES.values())), columns=list(C_AXES))


def _grid(sizes: tuple[int, ...]) -> pd.DataFrame:
    rows = list(itertools.product(*(range(n) for n in sizes)))
    return pd.DataFrame(rows, columns=[f"x{i}" for i in range(len(sizes))]).astype(float)


def test_the_shapes_must_agree() -> None:
    with pytest.raises(ChoiceError):
        CellDays(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 4)))


def test_a_ledger_pivots_onto_the_grid() -> None:
    rows = pd.DataFrame(
        {
            "key": ["a", "b", "a"],
            "day": ["d1", "d1", "d2"],
            "trades": [3, 0, 2],
            "net_bp": [1.5, 0.0, -2.0],
            "G_bp": [0.5, 0.0, -1.0],
        }
    )
    cells = CellDays.from_ledger(rows, ["a", "b"], ["d1", "d2", "d3"])
    assert cells.trades.tolist() == [[3, 2, 0], [0, 0, 0]]
    assert cells.net_bp[0, :2].tolist() == [1.5, -2.0]
    assert np.isnan(cells.net_bp[1]).all()
    assert np.isnan(cells.g_bp[0, 2])
    with pytest.raises(ChoiceError):
        CellDays.from_ledger(rows, ["a"], ["d1", "d2"])


def test_cell_statistics_match_the_verdict_functions() -> None:
    values = np.random.default_rng(0).normal(0.2, 1.0, (6, 30))
    values[2, ::3] = np.nan
    for row in range(6):
        assert cell_day_t(values)[row] == pytest.approx(day_t(values[row]).t)
        assert cell_means(values)[row] == pytest.approx(np.nanmean(values[row]))


def test_empty_and_one_day_cells_score_zero() -> None:
    values = np.full((3, 5), np.nan)
    values[1, 2] = 4.0
    values[2] = 1.0  # no dispersion
    assert cell_means(values)[0] == 0.0
    assert cell_day_t(values).tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_index_equals_neighbourhood_scores(seed: int) -> None:
    rng = np.random.default_rng(seed)
    table = _grid((4, 3, 2))
    table["group"] = rng.integers(0, 2, len(table))
    table = table.drop_duplicates(["group", "x0", "x1", "x2"]).reset_index(drop=True)
    table["v"] = rng.normal(size=len(table))
    index = neighbourhood_index(table, ["x0", "x1", "x2"], by=["group"])
    ours = neighbourhood_median(table["v"].to_numpy(), index)
    for _, part in table.groupby("group"):
        theirs = neighbourhood_scores(part, ["x0", "x1", "x2"], "v")
        np.testing.assert_allclose(ours[part.index], theirs.to_numpy())


def test_s25_has_twenty_four_neighbours() -> None:
    table = _catalogue_c()
    assert len(table) == 1536
    index = neighbourhood_index(table, C_ORDERED, by=C_CATEGORICAL)
    s25 = table.index[
        (table.model == "gbm")
        & (table.inputs == "micro")
        & (table.gate == "open")
        & (table.hold_s == 50)
        & (table.trades_per_day == 240)
        & (table.label_fee_bp == 0)
        & (table.regime_h == 0)
    ][0]
    near = table.loc[index[s25][index[s25] >= 0]]
    assert len(near) == 24
    assert set(near.hold_s) == {20, 50, 120}
    assert set(near.regime_h) == {0, 4}
    assert set(near.model) == {"gbm"} and set(near.gate) == {"open"}
    assert (index >= 0).sum(axis=1).max() == 3 * 3 * 2 * 3  # label fee has two values


def test_string_axes_are_refused() -> None:
    table = pd.DataFrame({"regime": ["none", "pull_4h", "pull_120h"]})
    with pytest.raises(ChoiceError, match="numeric"):
        neighbourhood_index(table, ["regime"])


def _line(net: list[float], trades: list[int], days: int = 10) -> CellDays:
    """A one-axis grid: cell i has a constant daily net and trade count."""
    net_bp = np.repeat(np.asarray(net, dtype=float)[:, None], days, axis=1)
    count = np.repeat(np.asarray(trades)[:, None], days, axis=1)
    net_bp[count == 0] = np.nan
    return CellDays(count, net_bp, net_bp.copy())


def _line_index(n: int) -> np.ndarray:
    return neighbourhood_index(pd.DataFrame({"x": np.arange(n, dtype=float)}), ["x"])


def test_the_highest_neighbourhood_median_wins() -> None:
    # An isolated spike at 2 loses to the region around 6.
    cells = _line([0, 0, 9, 0, 0, 3, 4, 3, 0, 0], [10] * 10)
    out = choose(cells, np.arange(10), _line_index(10))
    assert out.cell == 6
    assert out.s[2] == 0.0 and out.s[6] == 3.0


def test_the_trade_filter_limits_the_choice_not_the_neighbours() -> None:
    cells = _line([0, 3, 4, 3, 0], [10, 10, 4, 10, 10])
    out = choose(cells, np.arange(5), _line_index(5))
    assert out.peak == 2
    assert out.cell in (1, 3)
    assert not out.admissible[2]
    assert out.s[1] == 3.0  # cell 2 still counts as cell 1's neighbour


def test_ties_go_to_the_cell_s_own_mean() -> None:
    cells = _line([1, 2, 2, 5, 2, 2, 2], [10] * 7)
    out = choose(cells, np.arange(7), _line_index(7))
    assert out.s[2] == out.s[3] == out.s[4] == 2.0
    assert out.cell == 3


def test_a_cell_without_trades_scores_zero_in_the_median() -> None:
    cells = _line([5, 5, 5], [10, 0, 10])
    out = choose(cells, np.arange(3), _line_index(3))
    assert out.s[0] == pytest.approx(2.5)  # median of 5 and 0


def test_no_admissible_cell_is_no_choice() -> None:
    cells = _line([1, 2, 3], [4, 4, 4])
    out = choose(cells, np.arange(3), _line_index(3))
    assert out.cell is None and out.peak is not None


def test_the_choice_stays_inside_its_stratum() -> None:
    cells = _line([9, 9, 1, 1], [10] * 4)
    table = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "stratum": ["C", "C", "R", "R"]})
    index = neighbourhood_index(table, ["x"], by=["stratum"])
    assert choose(cells, np.array([2, 3]), index).cell in (2, 3)


def test_a_is_the_neighbourhood_median_of_the_t_of_g() -> None:
    rng = np.random.default_rng(3)
    g = rng.normal(0.0, 1.0, (5, 40)) + np.array([0.0, 0.1, 0.5, 0.2, 0.0])[:, None]
    cells = CellDays(np.ones_like(g, dtype=int), g, g)
    index = _line_index(5)
    t = [day_t(row).t for row in g]
    assert statistic_a(cells, index, 2) == pytest.approx(float(np.median(t[1:4])))
    assert statistic_a(cells, index, 0) == pytest.approx(float(np.median(t[0:2])))


def test_rerun_chooses_per_stratum_and_keeps_a_fixed_cell() -> None:
    cells = _line([0, 2, 2, 0, 3, 3], [10] * 6)
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 1.0, cells.g_bp.shape)
    cells = CellDays(cells.trades, cells.net_bp, cells.net_bp + noise)
    table = pd.DataFrame({"x": [0.0, 1.0, 2.0] * 2, "s": ["C"] * 3 + ["R"] * 3})
    index = neighbourhood_index(table, ["x"], by=["s"])
    strata = {"C": np.arange(3), "R": np.arange(3, 6)}
    out = rerun(cells, strata, index)
    assert out["C"].cell == 1 and out["R"].cell == 4
    assert out["C"].a == pytest.approx(statistic_a(cells, index, 1))
    fixed = rerun(cells, strata, index, fixed={"C": 0})
    assert fixed["C"].cell == 0 and fixed["R"].cell == 4
    empty = rerun(CellDays(cells.trades * 0, cells.net_bp, cells.g_bp), strata, index)
    assert empty["C"].cell is None and empty["C"].a == 0.0


DAYS = pd.date_range("2025-02-01", "2025-06-30", freq="D")


def _by_month(values: dict[str, float]) -> np.ndarray:
    """A daily series taking each month's value."""
    return np.array([values[str(day.to_period("M"))] for day in DAYS], dtype=float)


def _two_cells(first: np.ndarray, second: np.ndarray, trades: int = 10) -> CellDays:
    net = np.vstack([first, second])
    return CellDays(np.full(net.shape, trades), net, net.copy())


def test_there_are_150_test_days() -> None:
    assert len(DAYS) == 150


def test_carry_forward_rechooses_each_month_and_skips_the_embargo() -> None:
    first = _by_month({"2025-02": 2, "2025-03": 2, "2025-04": -1, "2025-05": -1, "2025-06": -1})
    first[(DAYS >= "2025-03-01") & (DAYS <= "2025-03-05")] = -2.0  # inside the embargo
    second = _by_month({"2025-02": 0, "2025-03": 0, "2025-04": 1, "2025-05": 1, "2025-06": 1})
    identity = np.arange(2)[:, None]
    out = carry_forward(_two_cells(first, second), DAYS, [0, 1], identity)
    assert out.cells == (0, 0, 0, 1)
    assert out.g == (2.0, -1.0, -1.0, 1.0)
    assert out.mean_g == pytest.approx(0.25)
    assert out.mean_net == pytest.approx(0.25)
    assert out.passes


def test_a_month_without_an_admissible_cell_scores_zero() -> None:
    values = np.ones(len(DAYS))
    out = carry_forward(_two_cells(values, values, trades=1), DAYS, [0, 1], np.arange(2)[:, None])
    assert out.cells == (None,) * 4
    assert out.mean_g == 0.0 and not out.passes


def test_the_dates_must_match_the_days() -> None:
    with pytest.raises(ChoiceError):
        carry_forward(_two_cells(np.ones(3), np.ones(3)), DAYS, [0, 1], np.arange(2)[:, None])


def test_slot_conditions_count_positive_months() -> None:
    months = {"2025-02": 1, "2025-03": 1, "2025-04": -1, "2025-05": 1, "2025-06": 1}
    cells = _two_cells(_by_month(months), -_by_month(months))
    out = slot_conditions(cells, DAYS, np.arange(2)[:, None], 0)
    assert out.monthly == (1.0, 1.0, -1.0, 1.0, 1.0)
    assert out.months_positive == 4 and out.i and out.ii
    worse = slot_conditions(cells, DAYS, np.arange(2)[:, None], 1)
    assert worse.months_positive == 1 and not worse.i and not worse.ii


def test_slot_conditions_use_the_neighbourhood() -> None:
    good = _by_month(dict.fromkeys(TEST_MONTHS, 1.0))
    cells = CellDays(
        np.full((3, len(DAYS)), 10),
        np.vstack([good, -good, -good]),
        np.vstack([good, -good, -good]),
    )
    out = slot_conditions(cells, DAYS, _line_index(3), 0)
    assert out.s == 0.0 and not out.i  # median of 1 and -1
    assert not_admitted(["i", "iv"]) == "not admitted on D (i, iv)"
