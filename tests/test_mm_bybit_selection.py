"""The Bybit 2021 grid, the choice on D and its two nulls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.mm_bybit import development
from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit import selection as sel
from trading_research.validation.placebo_search import null_a_signs


def test_the_grid_has_the_registered_forty_cells_and_keys() -> None:
    cells = q.grid()
    keys = [c.key for c in cells]
    assert len(cells) == 40 == len(set(keys))
    assert q.S0.key == "l0|none" and "l1|vle0.67" in keys and "l2|spread5" in keys
    assert all(q.from_key(k).key == k for k in keys)


def test_neighbourhoods_step_once_on_lambda_and_on_the_gates_parameter() -> None:
    def keys(key: str) -> set[str]:
        return {c.key for c in q.neighbourhood(q.from_key(key))}

    assert keys("l0|none") == {"l0|none", "l0.5|none"}
    assert keys("l1|none") == {"l0.5|none", "l1|none", "l2|none"}
    assert len(keys("l1|vle0.67")) == 9 and "l0.5|vle0.33" in keys("l1|vle0.67")
    assert keys("l0|spread2") == {"l0|spread2", "l0|spread3", "l0.5|spread2", "l0.5|spread3"}
    assert not any("vge" in k for k in keys("l1|vle0.9"))


def test_the_quoter_withdraws_the_adding_side_by_lambda_and_by_its_gate() -> None:
    quantiles = {0.33: 1.0, 0.67: 2.0, 0.9: 3.0}
    calm = q.State(spread=1.0, vol=1.5)
    soft = 0.06
    assert q.wanted(q.S0, 0.06, soft, 1.5, 1.5, calm, quantiles) == (True, True)
    assert q.wanted(q.Cell(1.0), 0.03, soft, 1.5, 1.5, calm, quantiles) == (False, True)
    assert q.wanted(q.Cell(1.0), -0.03, soft, 1.5, 1.5, calm, quantiles) == (True, False)
    assert q.wanted(q.Cell(1.0), 0.02, soft, 1.5, 1.5, calm, quantiles) == (True, True)
    # Twice the reference volatility halves the position at which the side goes.
    assert q.wanted(q.Cell(1.0), 0.02, soft, 3.0, 1.5, calm, quantiles) == (False, True)
    # An undefined volatility reads as the reference.
    assert q.wanted(q.Cell(1.0), 0.03, soft, float("nan"), 1.5, calm, quantiles) == (False, True)
    wide = q.Cell(0.0, "spread", 2.0)
    assert q.wanted(wide, 0.0, soft, 1.5, 1.5, calm, quantiles) == (False, False)
    assert q.wanted(wide, 0.01, soft, 1.5, 1.5, calm, quantiles) == (False, True)
    assert q.wanted(wide, -0.01, soft, 1.5, 1.5, calm, quantiles) == (True, False)
    assert q.wanted(wide, 0.0, soft, 1.5, 1.5, q.State(2.0, 1.5), quantiles) == (True, True)
    assert q.wanted(q.Cell(0.0, "vle", 0.33), 0.0, soft, 1.5, 1.5, calm, quantiles) == (
        False,
        False,
    )
    assert q.wanted(q.Cell(0.0, "vge", 0.33), 0.0, soft, 1.5, 1.5, calm, quantiles) == (True, True)


def test_the_choice_takes_the_admissible_cell_with_the_highest_neighbourhood_median() -> None:
    cells = q.grid()
    keys = [c.key for c in cells]
    nbhd = sel.neighbour_index(cells)
    best = keys.index("l1|vle0.67")
    means = np.full(40, -5.0)
    means[nbhd[best]] = -1.0
    means[best] = 0.0
    y = means[:, None] + np.tile([0.1, -0.1], 15)[None, :]
    admissible = np.ones(40, dtype=bool)
    choice = sel.choose(y, admissible, nbhd, keys)
    # S ties at -1 among the target and its neighbours; the target's own mean breaks it.
    assert keys[choice.index] == "l1|vle0.67" and keys[choice.peak] == "l1|vle0.67"
    assert choice.s == pytest.approx(-1.0)
    assert choice.a == pytest.approx(np.median(sel.t_rows(y)[nbhd[best]]))
    admissible[best] = False
    other = sel.choose(y, admissible, nbhd, keys)
    assert other.index in nbhd[best] and other.index != best
    assert keys[other.peak] == "l1|vle0.67"
    assert sel.choose(y, np.zeros(40, dtype=bool), nbhd, keys).index == -1


def test_ties_go_to_the_cells_own_mean_then_to_the_keys_byte_order() -> None:
    cells = q.grid()
    keys = [c.key for c in cells]
    nbhd = sel.neighbour_index(cells)
    y = np.zeros((40, 10))
    choice = sel.choose(y, np.ones(40, dtype=bool), nbhd, keys)
    assert keys[choice.index] == min(keys)
    y[keys.index("l2|vge0.9")] += 1e-9
    nudged = sel.choose(y, np.ones(40, dtype=bool), nbhd, keys)
    assert keys[nudged.index] == "l2|vge0.9"


def test_a_row_without_dispersion_has_a_t_of_zero() -> None:
    assert sel.t_rows(np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])).tolist() == [
        0.0,
        pytest.approx(2.0 / (1.0 / np.sqrt(3))),
    ]
    assert sel.t_rows(np.ones((2, 1))).tolist() == [0.0, 0.0]


def test_null_b_uses_one_balanced_pattern_over_every_shift() -> None:
    pattern = sel.null_b_pattern(90)
    assert sorted(set(pattern.tolist())) == [-1.0, 1.0] and pattern.sum() == 0
    expected = np.where(np.random.default_rng(0).permutation(90) % 2 == 0, 1.0, -1.0)
    assert np.array_equal(pattern, expected)
    assert list(sel.null_b_shifts(90)) == list(range(7, 84))


def test_the_nulls_rerun_the_choice_and_count_replications_at_or_above_a() -> None:
    cells = q.grid()
    keys = [c.key for c in cells]
    nbhd = sel.neighbour_index(cells)
    rng = np.random.default_rng(1)
    hourly = rng.normal(0.05, 1.0, size=(40, 20, 24))
    y = hourly.sum(axis=2)
    admissible = np.ones(40, dtype=bool)
    observed = sel.choose(y, admissible, nbhd, keys).a
    a = sel.null_a(hourly, admissible, nbhd, keys, observed, draws=50)
    flat = hourly.reshape(40, -1)
    first = (flat * null_a_signs(0, 480)).reshape(40, 20, 24).sum(axis=2)
    assert a.a[0] == pytest.approx(sel.choose(first, admissible, nbhd, keys).a)
    assert a.count == int((a.a >= observed).sum())
    assert a.p == pytest.approx((1 + a.count) / 51)
    b = sel.null_b(y, admissible, nbhd, keys, observed)
    assert len(b.a) == len(sel.null_b_shifts(20))
    assert b.p == pytest.approx((1 + b.count) / (1 + len(b.a)))


def _d_rows(n_days: int = 12, seed: int = 2) -> pd.DataFrame:
    """Primary rows for every cell over a few days, in the shape the D stage writes."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range("2021-01-01", periods=n_days).strftime("%Y-%m-%d"):
        for cell in q.grid():
            hours = rng.normal(0.0, 1.0, size=24)
            row = {
                "day": day, "cell": cell.key, "variant": "primary", "y": float(hours.sum()),
                "hourly_y": hours.tolist(), "passive_fills": 25 + int(cell.lam * 10),
                "markout_usdt": {"1": -0.1, "5": -0.2, "30": -0.3}, "clip_notional": 300.0,
                "market_markout_bp": {"1": 0.1, "5": 0.2, "30": 0.3},
            }  # fmt: skip
            for name in development.SERIES:
                row.setdefault(name, float(rng.normal()))
            row["passive_turnover"] = 7.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_the_d_record_holds_the_choice_the_nulls_and_every_cells_daily_series() -> None:
    record = development.develop(_d_rows(n_days=20), null_a_draws=20)
    assert record["usable_days"] == 20 and len(record["cells"]) == 40
    assert len(record["cells"]["l0|none"]["daily"]["y"]) == 20
    assert record["admission"]["shifts_B"] == 7
    assert record["chosen_cell"] in record["cells"]
    assert set(record["admission"]) >= {"S", "A", "p_A", "p_B", "admitted", "reasons"}
    assert record["admission"]["draws_A"] == 20
    assert set(record["mde"]) == {"F", "S0"}
    assert record["mde"]["S0"] == pytest.approx(3.84 * record["sigma_d"]["S0"] / np.sqrt(84))
    assert set(record["decomposition"]["S0"]) >= {"making_y", "rebate_y", "inventory_y"}
