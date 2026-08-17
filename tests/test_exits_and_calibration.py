"""Tests for path-dependent exits and probability calibration.

The exit tests build price paths by hand, because the rules are about *when*
something happens and an off-by-one there changes the outcome of every trade
while leaving the summary looking reasonable.

The calibration tests check the two properties that decide whether it is worth
having: that it is fitted on data the model did not train on, and that it is
monotonic — which is why it cannot change a threshold strategy's trades, only
the number the threshold is written as.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.execution import ThinningRules, thin, trades_to_frame
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.models.base import CLASSES
from trading_research.models.calibration import (
    CalibratedModel,
    expected_calibration_error,
    reliability,
)
from trading_research.models.linear import LogisticBaseline


def path(prices: list[float]) -> np.ndarray:
    return np.array(prices, dtype="float64")


def constant(n: int, value: float) -> np.ndarray:
    return np.full(n, value)


# ---------------------------------------------------------------------------
# Take-profit
# ---------------------------------------------------------------------------


def test_take_profit_closes_on_the_first_bar_that_clears_it() -> None:
    # From 100.0 the path reaches +5 bp, +10 bp and +20 bp. The threshold sits
    # between two of them rather than exactly on one: 100.10 / 100 - 1 is
    # 9.999999999 in binary, so a test written on the boundary would fail for
    # reasons that have nothing to do with the rule.
    mid = path([100.0, 100.05, 100.10, 100.20])
    decision = np.array([BUY, HOLD, HOLD, HOLD])
    trades = thin(
        decision,
        constant(4, 20.0),
        constant(4, 2.0),
        ThinningRules(hold_periods=3, take_profit_bp=8.0),
        mid=mid,
    )
    assert len(trades) == 1
    assert trades[0].exit_index == 2  # the first bar past +8 bp
    assert trades[0].exit_reason == "take_profit"
    assert trades[0].move_bp == pytest.approx(10.0, abs=0.1)


def test_take_profit_that_is_never_reached_leaves_the_clock_in_charge() -> None:
    mid = path([100.0, 100.01, 100.02, 100.03])
    trades = thin(
        np.array([BUY, HOLD, HOLD, HOLD]),
        constant(4, 3.0),
        constant(4, 2.0),
        ThinningRules(hold_periods=3, take_profit_bp=500.0),
        mid=mid,
    )
    assert trades[0].exit_reason == "clock"
    assert trades[0].exit_index == 3


def test_a_short_takes_profit_when_the_price_falls() -> None:
    mid = path([100.0, 99.9, 99.0])
    trades = thin(
        np.array([SELL, HOLD, HOLD]),
        constant(3, -100.0),
        constant(3, 2.0),
        ThinningRules(hold_periods=2, take_profit_bp=50.0),
        mid=mid,
    )
    assert trades[0].exit_reason == "take_profit"
    assert trades[0].move_bp > 0  # a short profits from the fall


# ---------------------------------------------------------------------------
# Stop-loss
# ---------------------------------------------------------------------------


def test_stop_loss_closes_on_the_first_bar_that_breaches_it() -> None:
    mid = path([100.0, 99.95, 99.5, 99.0])
    trades = thin(
        np.array([BUY, HOLD, HOLD, HOLD]),
        constant(4, -100.0),
        constant(4, 2.0),
        ThinningRules(hold_periods=3, stop_loss_bp=25.0),
        mid=mid,
    )
    assert trades[0].exit_index == 2
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].move_bp < 0


def test_a_stop_realises_a_loss_the_clock_would_have_recovered() -> None:
    """The cost of a stop, in one test: it converts a dip into a realised loss."""
    mid = path([100.0, 99.0, 100.5])  # dips, then recovers above entry
    signals = np.array([BUY, HOLD, HOLD])

    stopped = thin(
        signals,
        constant(3, 50.0),
        constant(3, 2.0),
        ThinningRules(hold_periods=2, stop_loss_bp=50.0),
        mid=mid,
    )
    patient = thin(
        signals, constant(3, 50.0), constant(3, 2.0), ThinningRules(hold_periods=2), mid=mid
    )

    assert stopped[0].move_bp < 0
    assert patient[0].move_bp > 0


# ---------------------------------------------------------------------------
# Both, and bookkeeping
# ---------------------------------------------------------------------------


def test_take_profit_wins_when_both_would_trigger_on_one_bar() -> None:
    """Bar data cannot say which came first; the optimistic resolution is stated."""
    mid = path([100.0, 101.0])
    trades = thin(
        np.array([BUY, HOLD]),
        constant(2, 100.0),
        constant(2, 2.0),
        ThinningRules(hold_periods=1, take_profit_bp=50.0, stop_loss_bp=1.0),
        mid=mid,
    )
    assert trades[0].exit_reason == "take_profit"


def test_path_exits_without_a_price_path_are_refused() -> None:
    """Silently ignoring the rule would report a backtest that never applied it."""
    with pytest.raises(ValueError, match="needs the price path"):
        thin(
            np.array([BUY, HOLD]),
            constant(2, 10.0),
            constant(2, 2.0),
            ThinningRules(hold_periods=1, take_profit_bp=10.0),
        )


def test_negative_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="take_profit_bp"):
        ThinningRules(hold_periods=1, take_profit_bp=-5.0)
    with pytest.raises(ValueError, match="stop_loss_bp"):
        ThinningRules(hold_periods=1, stop_loss_bp=0.0)


def test_exit_reason_is_recorded_for_every_trade() -> None:
    """A rule that never fires and one that always fires summarise identically."""
    mid = path([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
    trades = thin(
        np.array([BUY, HOLD, HOLD, BUY, HOLD, HOLD]),
        constant(6, 50.0),
        constant(6, 2.0),
        ThinningRules(hold_periods=2, take_profit_bp=30.0),
        mid=mid,
    )
    frame = trades_to_frame(trades)
    assert "exit_reason" in frame.columns
    assert (
        frame["exit_reason"]
        .isin({"clock", "take_profit", "stop_loss", "reversal", "end_of_data"})
        .all()
    )


def test_exits_do_not_change_behaviour_when_unset() -> None:
    """Adding the option must not alter results for anyone not using it."""
    decision = np.array([BUY, HOLD, HOLD, SELL, HOLD, HOLD])
    forward, spread = constant(6, 20.0), constant(6, 2.0)
    rules = ThinningRules(hold_periods=2, cooldown_periods=1)

    without = thin(decision, forward, spread, rules)
    with_path = thin(decision, forward, spread, rules, mid=path([100.0] * 6))
    assert [t.entry_index for t in without] == [t.entry_index for t in with_path]
    assert [t.move_bp for t in without] == [t.move_bp for t in with_path]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@pytest.fixture
def skewed() -> tuple[pd.DataFrame, pd.Series]:
    """A target as unbalanced as the real one, where miscalibration shows up."""
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)
    label = np.where(signal > 2.0, 1.0, np.where(signal < -2.0, -1.0, 0.0))
    return pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}), pd.Series(label)


def test_calibrated_probabilities_still_sum_to_one(skewed) -> None:
    x, y = skewed
    model = CalibratedModel(LogisticBaseline(), method="isotonic").fit(x, y)
    proba = model.predict_proba(x)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all()


def test_calibration_preserves_the_ordering(skewed) -> None:
    """The property that decides what calibration can and cannot do here.

    Both methods are monotonic, so a strategy that trades whenever a score
    clears a swept threshold selects exactly the same trades before and after.
    Calibration is worth having for sizing and for comparing models — not for
    changing this pipeline's decisions.
    """
    x, y = skewed
    base = LogisticBaseline().fit(x.iloc[: int(len(x) * 0.8)], y.iloc[: int(len(y) * 0.8)])
    raw = base.predict_proba(x)[:, CLASSES.index(1)]

    calibrated = CalibratedModel(LogisticBaseline(), method="isotonic").fit(x, y)
    cooked = calibrated.predict_proba(x)[:, CLASSES.index(1)]

    # Isotonic is non-decreasing: wherever the raw score is higher, the
    # calibrated one is not lower.
    order = np.argsort(raw)
    assert np.all(np.diff(cooked[order]) >= -1e-9)


def test_calibration_uses_a_holdout_the_model_did_not_see(skewed) -> None:
    """Calibrating on training data teaches a correction that fits nowhere else."""
    x, y = skewed
    model = CalibratedModel(LogisticBaseline(), holdout_fraction=0.25)
    model.fit(x, y)
    described = model.describe()
    assert described["holdout_fraction"] == 0.25
    assert described["calibrated_classes"]


def test_too_little_data_to_calibrate_is_refused(skewed) -> None:
    x, y = skewed
    with pytest.raises(ValueError, match="calibration split"):
        CalibratedModel(LogisticBaseline()).fit(x.head(150), y.head(150))


def test_invalid_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="isotonic"):
        CalibratedModel(LogisticBaseline(), method="magic")


def test_reliability_table_compares_predicted_with_observed(skewed) -> None:
    x, y = skewed
    proba = CalibratedModel(LogisticBaseline()).fit(x, y).predict_proba(x)
    table = reliability(proba, y.to_numpy(), cls=0, bins=5)
    assert {"predicted", "observed", "n"} <= set(table.columns)
    assert table["n"].sum() == len(x)


def test_calibration_reduces_the_calibration_error(skewed) -> None:
    """The measurement that justifies the module existing at all."""
    x, y = skewed
    split = int(len(x) * 0.8)
    train_x, train_y = x.iloc[:split], y.iloc[:split]

    # The uncalibrated model, fitted on the same rows, is the thing to beat.
    # A constant-output baseline would not do: its scores fall in one bucket,
    # and the error is undefined rather than large.
    raw_model = LogisticBaseline().fit(train_x, train_y)
    raw_error = expected_calibration_error(raw_model.predict_proba(x), y.to_numpy(), cls=0)

    calibrated = CalibratedModel(LogisticBaseline(), method="isotonic").fit(train_x, train_y)
    cooked_error = expected_calibration_error(calibrated.predict_proba(x), y.to_numpy(), cls=0)

    assert cooked_error <= raw_error
