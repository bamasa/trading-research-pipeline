"""Tests for the market gate, the adaptive clock, marker exits and ensembles.

The gate's one dangerous property is where its levels come from: fitted on the
window being traded, it would be reading the answer. That is checked directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.execution import ThinningRules, thin
from trading_research.backtest.gating import GATE_GRID, MarketGate, realised_volatility_bp
from trading_research.labels.directional import BUY
from trading_research.models.base import CLASSES, ClassPrior
from trading_research.models.ensemble import EnsembleModel
from trading_research.models.linear import LogisticBaseline
from trading_research.pipeline.stages import StageError, build_model


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "log_mid_vol50": np.abs(rng.normal(0, 1, n)),
            "spread_bp_now": np.abs(rng.normal(2, 1, n)),
        }
    )


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_an_open_gate_lets_everything_through(frame) -> None:
    gate = MarketGate()
    assert gate.is_open
    assert gate.mask(frame, gate.thresholds(frame)).all()


def test_a_volatility_gate_keeps_the_busier_share(frame) -> None:
    gate = MarketGate(min_volatility_quantile=0.7)
    kept = gate.mask(frame, gate.thresholds(frame))
    assert 0.28 < kept.mean() < 0.32


def test_a_spread_gate_drops_the_widest(frame) -> None:
    gate = MarketGate(max_spread_quantile=0.6)
    kept = gate.mask(frame, gate.thresholds(frame))
    assert 0.58 < kept.mean() < 0.62
    assert frame.loc[kept, "spread_bp_now"].max() <= frame["spread_bp_now"].quantile(0.6)


def test_levels_come_from_one_frame_and_apply_to_another(frame) -> None:
    """The property that keeps the gate honest: thresholds are fitted on the
    training window and carried forward, not recomputed on what is traded."""
    gate = MarketGate(min_volatility_quantile=0.5)
    levels = gate.thresholds(frame)

    calmer = frame.assign(log_mid_vol50=frame["log_mid_vol50"] * 0.1)
    kept = gate.mask(calmer, levels)
    # Applying training levels to a quieter period keeps almost nothing, which
    # is the intended behaviour. Refitting would keep half of it regardless.
    assert kept.mean() < 0.1


def test_a_missing_volatility_column_does_not_silently_open_the_gate(frame) -> None:
    """An earlier version of this test asserted the opposite of its own name.

    It pinned ``thresholds() == {}`` for a missing column — and an empty dict
    masks nothing, which *is* the silently-open gate. Three instrument-wide
    searches ran with a misnamed volatility column before anyone noticed,
    because this test blessed the behaviour. A gate asked to filter on a column
    the frame does not have must refuse.
    """
    gate = MarketGate(min_volatility_quantile=0.9, volatility_column="not_a_column")
    with pytest.raises(KeyError, match="not_a_column"):
        gate.thresholds(frame)


def test_impossible_quantiles_are_rejected() -> None:
    with pytest.raises(ValueError, match="min_volatility_quantile"):
        MarketGate(min_volatility_quantile=1.5)
    with pytest.raises(ValueError, match="max_spread_quantile"):
        MarketGate(max_spread_quantile=-0.1)


def test_the_grid_starts_from_no_gate() -> None:
    """Without the ungated baseline the table cannot say whether gating helped."""
    assert GATE_GRID[0].is_open


def test_realised_volatility_rises_with_the_path(frame) -> None:
    calm = 100 * np.exp(np.cumsum(np.full(500, 1e-6)))
    wild = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 1e-3, 500)))
    assert np.nanmean(realised_volatility_bp(wild, 50)) > np.nanmean(
        realised_volatility_bp(calm, 50)
    )


# ---------------------------------------------------------------------------
# Adaptive clock
# ---------------------------------------------------------------------------


def test_a_confident_entry_is_held_longer() -> None:
    n = 40
    forward, spread = np.full(n, 20.0), np.full(n, 2.0)
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY

    sure = np.full(n, 0.95)
    barely = np.full(n, 0.51)
    rules = ThinningRules(hold_periods=10, hold_scale_by_confidence=1.0)

    held_long = thin(
        decision, forward, spread, rules, mid=np.full(n, 100.0), p_buy=sure, p_sell=np.full(n, 0.01)
    )
    held_short = thin(
        decision,
        forward,
        spread,
        rules,
        mid=np.full(n, 100.0),
        p_buy=barely,
        p_sell=np.full(n, 0.01),
    )
    assert held_long[0].exit_index > held_short[0].exit_index


def test_a_zero_scale_keeps_the_fixed_clock() -> None:
    n = 20
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY
    fixed = thin(decision, np.full(n, 20.0), np.full(n, 2.0), ThinningRules(hold_periods=5))
    assert fixed[0].exit_index == 5


def test_a_negative_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="hold_scale_by_confidence"):
        ThinningRules(hold_periods=5, hold_scale_by_confidence=-1.0)


# ---------------------------------------------------------------------------
# Marker exits
# ---------------------------------------------------------------------------


def test_a_marker_closes_the_position() -> None:
    n = 10
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY
    marker = np.zeros(n, dtype=bool)
    marker[3] = True

    trades = thin(
        decision,
        np.full(n, 20.0),
        np.full(n, 2.0),
        ThinningRules(hold_periods=8),
        mid=np.full(n, 100.0),
        exit_when=marker,
    )
    assert trades[0].exit_reason == "marker"
    assert trades[0].exit_index == 3


def test_a_marker_that_never_fires_leaves_the_clock_in_charge() -> None:
    n = 10
    decision = np.zeros(n, dtype=int)
    decision[0] = BUY
    trades = thin(
        decision,
        np.full(n, 20.0),
        np.full(n, 2.0),
        ThinningRules(hold_periods=4),
        mid=np.full(n, 100.0),
        exit_when=np.zeros(n, dtype=bool),
    )
    assert trades[0].exit_reason == "clock"


def test_a_marker_does_not_close_a_position_that_is_not_open() -> None:
    n = 10
    decision = np.zeros(n, dtype=int)
    decision[5] = BUY
    marker = np.ones(n, dtype=bool)
    marker[5:] = False
    trades = thin(
        decision,
        np.full(n, 20.0),
        np.full(n, 2.0),
        ThinningRules(hold_periods=3),
        mid=np.full(n, 100.0),
        exit_when=marker,
    )
    assert len(trades) == 1
    assert trades[0].entry_index == 5


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


@pytest.fixture
def training() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(2)
    n = 3000
    signal = rng.normal(size=n)
    label = np.where(signal > 1.5, 1.0, np.where(signal < -1.5, -1.0, 0.0))
    return pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}), pd.Series(label)


def test_an_ensemble_averages_its_members(training) -> None:
    x, y = training
    a, b = LogisticBaseline().fit(x, y), ClassPrior().fit(x, y)
    both = EnsembleModel([LogisticBaseline(), ClassPrior()]).fit(x, y)
    expected = (a.predict_proba(x) + b.predict_proba(x)) / 2
    assert np.allclose(both.predict_proba(x), expected, atol=1e-9)


def test_ensemble_probabilities_still_sum_to_one(training) -> None:
    x, y = training
    proba = EnsembleModel([LogisticBaseline(), ClassPrior()]).fit(x, y).predict_proba(x)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba.shape[1] == len(CLASSES)


def test_weights_are_normalised(training) -> None:
    x, y = training
    weighted = EnsembleModel([LogisticBaseline(), ClassPrior()], weights=[3.0, 1.0]).fit(x, y)
    assert weighted.weights.tolist() == [0.75, 0.25]


def test_an_ensemble_needs_more_than_one_member() -> None:
    with pytest.raises(ValueError, match="at least two"):
        EnsembleModel([LogisticBaseline()])


def test_mismatched_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="2 weights for 3 members"):
        EnsembleModel([LogisticBaseline(), ClassPrior(), LogisticBaseline()], weights=[1.0, 1.0])


def test_an_ensemble_is_addressable_by_name() -> None:
    model = build_model("ensemble")
    assert [m.name for m in model.members] == ["logistic", "xgboost"]


def test_an_unknown_name_lists_the_ensembles_too() -> None:
    with pytest.raises(StageError, match="ensemble"):
        build_model("magic")
