"""Tests for trade thinning and the pipeline stages.

The thinning tests matter because the rules are stateful and a forward pass is
easy to get subtly wrong: an off-by-one in when a position closes changes the
trade count by a factor of the holding period, and the resulting profit still
looks like a plausible number.

The stage tests pin the property the whole file layout exists for: a stage that
fits anything reads only the window it was given, so later data is not merely
unused — it was never loaded.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, score, thin, trades_to_frame
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.pipeline.stages import (
    StageError,
    add_label,
    build_model,
    feature_columns,
    load_prepared,
)

COSTS = TakerCosts(fee_bp_per_side=5.0, slippage_bp=0.5)


def signals(pattern: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pattern)
    return (
        np.array(pattern),
        np.full(n, 20.0),  # every trade would move 20 bp in its favour
        np.full(n, 2.0),  # constant spread
    )


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------


def test_a_single_signal_makes_one_trade() -> None:
    decision, forward, spread = signals([HOLD, BUY, HOLD, HOLD, HOLD])
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=2))
    assert len(trades) == 1
    assert trades[0].entry_index == 1
    assert trades[0].direction == BUY


def test_signals_during_a_position_are_ignored() -> None:
    """Twelve hundred overlapping positions for one view is the failure this prevents."""
    decision, forward, spread = signals([BUY] * 10)
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=5))
    # Entries at 0 and 5 only: the rest arrive while a position is open.
    assert [t.entry_index for t in trades] == [0, 5]


def test_cooldown_delays_the_next_entry() -> None:
    decision, forward, spread = signals([BUY] * 12)
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=2, cooldown_periods=3))
    # Open at 0, close at 2, free from 5; open at 5, close at 7, free from 10.
    assert [t.entry_index for t in trades] == [0, 5, 10]


def test_zero_cooldown_reopens_immediately() -> None:
    decision, forward, spread = signals([BUY] * 9)
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=3, cooldown_periods=0))
    assert [t.entry_index for t in trades] == [0, 3, 6]


def test_opposite_signal_is_ignored_by_default() -> None:
    decision, forward, spread = signals([BUY, SELL, SELL, SELL, HOLD])
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=4))
    assert len(trades) == 1
    assert trades[0].direction == BUY


def test_reversal_closes_early_when_enabled() -> None:
    decision, forward, spread = signals([BUY, HOLD, SELL, HOLD, HOLD, HOLD])
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=5, allow_reversal=True))
    assert len(trades) == 2
    assert trades[0].exit_index == 2


def test_a_row_without_an_outcome_is_not_entered() -> None:
    decision = np.array([BUY, BUY])
    forward = np.array([np.nan, 20.0])
    spread = np.array([2.0, 2.0])
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=1))
    assert [t.entry_index for t in trades] == [1]


def test_thinning_reduces_the_trade_count() -> None:
    decision, forward, spread = signals([BUY] * 100)
    many = thin(decision, forward, spread, ThinningRules(hold_periods=1))
    few = thin(decision, forward, spread, ThinningRules(hold_periods=10, cooldown_periods=10))
    assert len(few) < len(many)


def test_invalid_rules_are_rejected() -> None:
    with pytest.raises(ValueError, match="hold_periods"):
        ThinningRules(hold_periods=0)
    with pytest.raises(ValueError, match="cooldown_periods"):
        ThinningRules(hold_periods=1, cooldown_periods=-1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_scoring_charges_one_round_trip_per_trade() -> None:
    decision, forward, spread = signals([BUY, HOLD, HOLD])
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=2))
    result = score(trades, COSTS)
    assert result["trades"] == 1.0
    assert result["gross_per_trade_bp"] == pytest.approx(20.0)
    assert result["net_per_trade_bp"] == pytest.approx(20.0 - 13.0)


def test_a_short_earns_on_a_falling_price() -> None:
    decision = np.array([SELL, HOLD])
    trades = thin(decision, np.full(2, -20.0), np.full(2, 2.0), ThinningRules(hold_periods=1))
    assert score(trades, COSTS)["gross_per_trade_bp"] == pytest.approx(20.0)


def test_no_trades_scores_as_nothing_rather_than_failing() -> None:
    result = score([], COSTS)
    assert result["trades"] == 0.0
    assert result["net_bp"] == 0.0
    assert np.isnan(result["net_per_trade_bp"])


def test_empty_trade_frame_keeps_its_columns() -> None:
    """A downstream concat must not break just because a fold traded nothing."""
    frame = trades_to_frame([])
    assert list(frame.columns) == [
        "entry_index",
        "exit_index",
        "direction",
        "entry_spread_bp",
        "move_bp",
    ]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def make_prepared(directory, days: int = 5, rows: int = 400) -> None:
    """Write a small prepared dataset, one file per day."""
    rng = np.random.default_rng(0)
    directory.mkdir(parents=True, exist_ok=True)
    for d in range(days):
        day = date(2024, 2, 1 + d)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(f"{day}T00:00", periods=rows, freq="1min", tz="UTC"),
                "feat_a": rng.normal(size=rows),
                "feat_b": rng.normal(size=rows),
                "forward_bp": rng.normal(0, 10, rows),
                "spread_bp_now": np.full(rows, 2.0),
            }
        )
        frame.to_parquet(directory / f"{day}.parquet", index=False)


def test_load_prepared_reads_only_the_requested_days(tmp_path) -> None:
    """The guarantee the file layout exists for: later data is never loaded."""
    prepared = tmp_path / "prepared"
    make_prepared(prepared, days=5)

    frame = load_prepared(prepared, date(2024, 2, 1), date(2024, 2, 3))
    days = set(frame["timestamp"].dt.date)
    assert days == {date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 3)}
    assert max(days) < date(2024, 2, 4)


def test_load_prepared_without_a_range_reads_everything(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    make_prepared(prepared, days=3)
    assert len(set(load_prepared(prepared)["timestamp"].dt.date)) == 3


def test_an_empty_range_is_refused_rather_than_silently_empty(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    make_prepared(prepared, days=3)
    with pytest.raises(StageError, match="no prepared days"):
        load_prepared(prepared, date(2024, 5, 1), date(2024, 5, 2))


def test_missing_prepared_data_is_reported(tmp_path) -> None:
    with pytest.raises(StageError, match="run prepare first"):
        load_prepared(tmp_path / "nothing")


def test_feature_columns_exclude_bookkeeping(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    make_prepared(prepared, days=1)
    frame = load_prepared(prepared)
    assert feature_columns(frame) == ["feat_a", "feat_b"]


def test_label_is_added_from_the_threshold(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    make_prepared(prepared, days=1)
    frame = add_label(load_prepared(prepared), threshold_bp=5.0)
    moved = frame["forward_bp"].abs() > 5.0
    assert (frame.loc[moved, "label"] != 0).all()
    assert (frame.loc[~moved, "label"] == 0).all()


def test_label_sign_follows_the_move(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    make_prepared(prepared, days=1)
    frame = add_label(load_prepared(prepared), threshold_bp=5.0)
    assert (frame.loc[frame["label"] == BUY, "forward_bp"] > 0).all()
    assert (frame.loc[frame["label"] == SELL, "forward_bp"] < 0).all()


def test_models_are_addressable_by_name() -> None:
    """A config names a model as a string; an unknown one must fail loudly."""
    assert build_model("logistic").name == "logistic"
    assert build_model("always_hold").name == "always_hold"
    with pytest.raises(StageError, match="unknown model"):
        build_model("magic")


def test_a_position_open_at_the_end_is_still_counted() -> None:
    """Found by these tests: open positions used to vanish with the loop.

    The outcome is known — forward_bp at entry covers the whole holding period,
    and rows without an outcome are never entered — so dropping it would
    silently lose the trades nearest the end of every block.
    """
    decision, forward, spread = signals([HOLD, BUY, HOLD])
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=10))
    assert len(trades) == 1
    assert trades[0].entry_index == 1
    assert trades[0].move_bp == pytest.approx(20.0)


def test_the_final_position_is_not_double_counted() -> None:
    """A position that closes normally must not also be closed again at the end."""
    decision, forward, spread = signals([BUY, HOLD, HOLD, HOLD, HOLD])
    trades = thin(decision, forward, spread, ThinningRules(hold_periods=2))
    assert len(trades) == 1
