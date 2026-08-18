"""Tests for the target menu and the regression models fitted on it.

Targets are the one place in the pipeline where reading forward is correct, so
the tests check the opposite property from the feature tests: that each target
reads *exactly* as far ahead as it declares. A target reading further than its
horizon would defeat the purge, which is derived from that number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.labels.targets import (
    TARGETS,
    build_target,
    describe_targets,
    triple_barrier,
)
from trading_research.models.base import CLASSES
from trading_research.models.regression import (
    MAX_CONFIDENCE,
    NEUTRAL,
    RidgeBaseline,
    edge_to_proba,
)
from trading_research.pipeline.stages import build_model

COST = 11.0


def frame(moves: list[float], mid: list[float] | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"forward_bp": moves})
    out["mid"] = mid if mid is not None else 100.0
    return out


# ---------------------------------------------------------------------------
# The menu
# ---------------------------------------------------------------------------


def test_every_target_declares_its_kind_and_horizon() -> None:
    for name in TARGETS:
        _, spec = build_target(
            name, frame([1.0] * 200, list(np.linspace(100, 101, 200))), cost_bp=COST, horizon=24
        )
        assert spec.kind in {"classification", "regression"}
        assert spec.horizon == 24


def test_an_unknown_target_lists_the_menu() -> None:
    with pytest.raises(KeyError, match="magnitude"):
        build_target("wishful", frame([1.0]), cost_bp=COST)


def test_the_menu_is_documented() -> None:
    table = describe_targets()
    assert set(table["name"]) == set(TARGETS)
    assert table["description"].str.len().min() > 10


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------


def test_direction_ignores_moves_that_would_not_pay() -> None:
    y, _ = build_target("direction", frame([5.0, -5.0, 20.0, -20.0]), cost_bp=COST)
    assert y.tolist() == [0.0, -0.0, 1.0, -1.0]


# ---------------------------------------------------------------------------
# magnitude and net_pnl
# ---------------------------------------------------------------------------


def test_magnitude_keeps_the_size_the_classes_throw_away() -> None:
    y, spec = build_target("magnitude", frame([12.0, 60.0]), cost_bp=COST)
    assert spec.kind == "regression"
    assert y.tolist() == [12.0, 60.0]


def test_net_pnl_is_what_a_long_actually_makes() -> None:
    """A 12 bp move against an 11 bp round trip is a one basis point trade."""
    y, _ = build_target("net_pnl", frame([12.0, 11.0, 5.0]), cost_bp=COST)
    assert y.tolist() == [1.0, 0.0, -6.0]


def test_net_pnl_and_direction_agree_on_what_is_worth_taking() -> None:
    moves = [30.0, 12.0, 5.0, -5.0, -12.0, -30.0]
    net, _ = build_target("net_pnl", frame(moves), cost_bp=COST)
    side, _ = build_target("direction", frame(moves), cost_bp=COST)
    # Wherever a long nets more than nothing, the three-class target says buy.
    assert ((net > 0) == (side > 0)).all()


# ---------------------------------------------------------------------------
# triple_barrier
# ---------------------------------------------------------------------------


def test_the_barrier_touched_first_is_the_label() -> None:
    # Rises through +100 bp on the second step, never falls.
    path = [100.0, 100.5, 101.0, 101.0, 101.0]
    y = triple_barrier(frame([0.0] * 5, path), cost_bp=20.0, horizon=3)
    assert y.iloc[0] == 1.0


def test_a_path_that_touches_neither_barrier_is_the_clock() -> None:
    path = [100.0, 100.01, 100.02, 100.01, 100.0]
    y = triple_barrier(frame([0.0] * 5, path), cost_bp=50.0, horizon=3)
    assert y.iloc[0] == 0.0


def test_the_stop_wins_when_the_price_falls_first() -> None:
    path = [100.0, 99.0, 101.0, 101.0]
    y = triple_barrier(frame([0.0] * 4, path), cost_bp=50.0, horizon=3)
    assert y.iloc[0] == -1.0


def test_it_reads_exactly_its_horizon_and_no_further() -> None:
    """A target reading past its horizon would defeat the purge, which is
    derived from that number."""
    near = [100.0, 100.0, 100.0, 100.0, 105.0]  # the move is five steps away
    y = triple_barrier(frame([0.0] * 5, near), cost_bp=50.0, horizon=2)
    assert y.iloc[0] == 0.0
    y_longer = triple_barrier(frame([0.0] * 5, near), cost_bp=50.0, horizon=4)
    assert y_longer.iloc[0] == 1.0


def test_asymmetric_barriers_are_honoured() -> None:
    """A tight stop turns a trade that would have paid into one that was stopped.

    The path dips 15 bp and then rises 30. With wide symmetric barriers nothing
    is touched and the clock decides; with a 10 bp stop the dip ends it first,
    and the label says so.
    """
    path = [100.0, 99.85, 100.30, 100.30]

    wide = triple_barrier(frame([0.0] * 4, path), cost_bp=50.0, horizon=3)
    assert wide.iloc[0] == 0.0

    tight_stop = triple_barrier(
        frame([0.0] * 4, path), cost_bp=50.0, horizon=3, take_profit_bp=25.0, stop_loss_bp=10.0
    )
    assert tight_stop.iloc[0] == -1.0


def test_a_path_target_says_so() -> None:
    _, spec = build_target(
        "triple_barrier", frame([0.0] * 60, list(np.linspace(100, 101, 60))), cost_bp=COST
    )
    assert spec.needs_path


def test_the_path_target_needs_the_path() -> None:
    with pytest.raises(KeyError, match="mid"):
        triple_barrier(pd.DataFrame({"forward_bp": [1.0] * 30}), cost_bp=COST, horizon=5)


def test_the_barrier_label_catches_moves_the_endpoint_misses() -> None:
    """A path that touches the target and comes back is a winning trade under a
    take-profit rule and neutral under a clock. Only one of these can tell."""
    there_and_back = [100.0, 100.5, 100.0]
    barrier = triple_barrier(frame([0.0] * 3, there_and_back), cost_bp=20.0, horizon=2)
    endpoint, _ = build_target("direction", frame([0.0, 0.0, 0.0]), cost_bp=20.0)
    assert barrier.iloc[0] == 1.0
    assert endpoint.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------


def test_a_zero_edge_maps_to_no_opinion() -> None:
    proba = edge_to_proba(np.array([0.0]), scale_bp=COST)
    assert np.allclose(proba, NEUTRAL)


def test_a_large_edge_approaches_the_ceiling_without_reaching_it() -> None:
    proba = edge_to_proba(np.array([1000.0]), scale_bp=COST)
    assert proba[0, CLASSES.index(1)] < MAX_CONFIDENCE + 1e-9
    assert proba[0, CLASSES.index(1)] > 0.85


def test_the_mapping_is_monotonic() -> None:
    """Sweeping a confidence threshold must be equivalent to sweeping an
    expected-value threshold, or the conversion loses information."""
    edges = np.linspace(-40, 40, 200)
    buy = edge_to_proba(edges, scale_bp=COST)[:, CLASSES.index(1)]
    rising = buy[edges >= 0]
    assert np.all(np.diff(rising) >= -1e-12)


def test_a_negative_edge_points_the_other_way() -> None:
    proba = edge_to_proba(np.array([-20.0]), scale_bp=COST)
    assert proba[0, CLASSES.index(-1)] > proba[0, CLASSES.index(1)]


def test_converted_probabilities_sum_to_one() -> None:
    proba = edge_to_proba(np.linspace(-50, 50, 100), scale_bp=COST)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 2000
    signal = rng.normal(size=n)
    return (
        pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}),
        pd.Series(signal * 8.0 + rng.normal(0, 3, n)),
    )


def test_a_regressor_recovers_a_linear_relationship(regression_data) -> None:
    x, y = regression_data
    model = RidgeBaseline(scale_bp=COST).fit(x, y)
    assert np.corrcoef(model.predict_edge_bp(x), y)[0, 1] > 0.8


def test_a_regressor_reports_basis_points_as_well_as_probabilities(regression_data) -> None:
    x, y = regression_data
    model = RidgeBaseline(scale_bp=COST).fit(x, y)
    assert model.predict_edge_bp(x).std() > 1.0
    assert np.allclose(model.predict_proba(x).sum(axis=1), 1.0)


def test_a_regressor_refuses_a_target_that_is_almost_all_missing(regression_data) -> None:
    x, y = regression_data
    y = y.copy()
    y.iloc[50:] = np.nan
    with pytest.raises(ValueError, match="usable rows"):
        RidgeBaseline().fit(x, y)


def test_regressors_are_addressable_by_name() -> None:
    assert build_model("ridge").name == "ridge"
    assert build_model("xgboost_regressor").name == "xgboost_regressor"
