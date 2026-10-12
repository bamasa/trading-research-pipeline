"""Stage D's reported numbers: the power of the bar, S25's DSR and the fits that lacked a class."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trading_research.evaluation.significance import power_at_bar
from trading_research.pipeline.short_horizon import search
from trading_research.validation.choice import CellDays

C_KEY = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none"
R_KEY = "BTCUSDT|R|bound_ridge|micro|open|hold_s=5|k=0|beta_bp=0|regime=none"


def test_the_power_of_the_bar_is_taken_against_the_carry_forward_estimate() -> None:
    """The chosen C cell trades on 3 of 4 days with nets 1, 3, 5: sigma_D 2, n_eff 33."""
    grid = SimpleNamespace(keys=[C_KEY, R_KEY], strata={"C": np.array([0]), "R": np.array([1])})
    trades = np.array([[2, 0, 1, 4], [0, 0, 0, 0]])
    net = np.array([[1.0, np.nan, 3.0, 5.0], [np.nan] * 4])
    cells = CellDays(trades, net, net.copy())
    choice = {"C*": {"cell": C_KEY}, "R*": {"cell": None}}
    carry = {"C": SimpleNamespace(mean_net=1.5), "R": SimpleNamespace(mean_net=0.0)}
    found = search.power(cells, choice, carry, grid)  # type: ignore[arg-type]
    assert list(found) == ["S-C"]
    row = found["S-C"]
    assert row["sigma_D"] == pytest.approx(2.0) and row["n_eff"] == pytest.approx(33.0)
    assert row["power"] == pytest.approx(power_at_bar(1.5, 2.0, 33.0))
    assert 0.0 < row["power"] < 1.0


def test_the_fits_that_lacked_a_class_are_listed_with_the_side(tmp_path: Path) -> None:
    folds = tmp_path / "A" / "folds"
    folds.mkdir(parents=True)
    fits = {"BTCUSDT|C|gbm|micro|hold_s=5|label_fee_bp=0": {"classes_present": [-1, 0]},
            "BTCUSDT|C|logistic|micro|hold_s=5|label_fee_bp=0": {"classes_present": [-1, 0, 1]},
            "BTCUSDT|C|gbm|micro|hold_s=20|label_fee_bp=0": {"note": "too few rows or classes"},
            "BTCUSDT|R|bound_ridge|micro|hold_s=5": {"rows": 10}}  # fmt: skip
    (folds / "2025-03.json").write_text(json.dumps({"fits": fits}), encoding="utf-8")
    found = search.lacking_a_class(SimpleNamespace(work=tmp_path))  # type: ignore[arg-type]
    by = {row["configuration"]: row for row in found}
    assert set(by) == {
        "BTCUSDT|C|gbm|micro|hold_s=5|label_fee_bp=0",
        "BTCUSDT|C|gbm|micro|hold_s=20|label_fee_bp=0",
    }
    assert by["BTCUSDT|C|gbm|micro|hold_s=5|label_fee_bp=0"]["side"] == ["long"]
    assert by["BTCUSDT|C|gbm|micro|hold_s=5|label_fee_bp=0"]["missing"] == ["buy"]
    assert by["BTCUSDT|C|gbm|micro|hold_s=20|label_fee_bp=0"]["missing"] is None


def test_s25s_fixed_cell_has_its_deflated_sharpe_ratio_beside_the_chosen_cells() -> None:
    from trading_research.reporting.search_keys import S25_KEY

    rng = np.random.default_rng(0)
    keys = [S25_KEY, C_KEY, R_KEY]
    grid = SimpleNamespace(keys=keys, strata={"C": np.array([0, 1]), "R": np.array([2])},
                           index=np.array([[0, 1], [1, 0], [2, -1]]),
                           position=lambda key: keys.index(key) if key in keys else None)  # fmt: skip
    net = rng.normal(0.5, 1.0, (3, 30))
    cells = CellDays(np.ones((3, 30), dtype=np.int64), net, net + 0.1)
    found = search.overfitting(grid, cells, {"C": 1, "R": None}, 50)  # type: ignore[arg-type]
    assert {"S25/G", "S25/net", "C*/G", "C*/net"} <= set(found["dsr"])
