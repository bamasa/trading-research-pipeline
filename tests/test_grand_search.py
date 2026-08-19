"""The grand search's own machinery, none of which had a test until it broke.

Every invariant here was violated in production first: apply windows overlapped
fivefold, the breaks policy trained across the break it responded to, sampled
configurations crashed on constructors that rejected their keywords, and the
resampler invented sixteen days of a hundred. The tests pin the fixes.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path as _Path

import numpy as np
import pandas as pd
import pytest

# The experiments directory is a sibling of src, not a package the project
# installs — scripts there are run as modules from the repository root. Tests
# reach them the same way.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from experiments.grand_search import (
    GRID_SECONDS,
    HYPERPARAMETERS,
    Config,
    draw,
    refit_points,
    rules_for,
    to_grid,
)
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.pipeline.stages import build_model


def _config(**overrides) -> Config:
    base = {
        "plane": "micro",
        "model": "logistic",
        "target": "direction",
        "horizon": 24,
        "hold": 24,
        "cooldown": 0,
        "exit": "clock",
        "objective": "net_bp",
        "refit": "1d",
        "train_days": 10,
        "gate": "open",
    }
    base.update(overrides)
    return Config(**base)


# ---------------------------------------------------------------------------
# Refit windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("refit", ["1d", "2d", "5d"])
def test_apply_windows_tile_the_span_exactly_once(refit: str) -> None:
    """Every row is traded by exactly one model.

    The version before this one applied every refit for a fixed five days, so a
    daily cadence traded each row five times with five different models.
    """
    spans = refit_points(_config(refit=refit), 1000, 100_000, breaks=[])
    assert spans[0].at == 1000
    assert spans[-1].until == 100_000
    for a, b in pairwise(spans):
        assert a.until == b.at, "windows must neither overlap nor leave gaps"


def test_breaks_policy_tiles_and_clamps_training_to_the_previous_break() -> None:
    breaks = [30_000, 55_000]
    spans = refit_points(_config(refit="breaks"), 10_000, 90_000, breaks)

    assert [s.at for s in spans] == [10_000, 30_000, 55_000]
    assert spans[-1].until == 90_000
    for a, b in pairwise(spans):
        assert a.until == b.at
    # The refit at a break must not train on data from before the break it is
    # responding to — that data is what the break declared stale.
    assert [s.floor for s in spans] == [0, 10_000, 30_000]


def test_fixed_cadence_may_train_across_the_whole_past() -> None:
    spans = refit_points(_config(refit="2d"), 50_000, 90_000, breaks=[60_000])
    assert all(s.floor == 0 for s in spans), "a fixed cadence makes no staleness claim"


def test_breaks_outside_the_span_are_ignored() -> None:
    spans = refit_points(_config(refit="breaks"), 40_000, 60_000, breaks=[10_000, 75_000])
    assert [s.at for s in spans] == [40_000]
    assert spans[0].until == 60_000


# ---------------------------------------------------------------------------
# The time grid
# ---------------------------------------------------------------------------


def _book(timestamps: list[str]) -> pd.DataFrame:
    stamps = pd.to_datetime(timestamps)
    n = len(stamps)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "bid_price_0": np.linspace(100.0, 101.0, n),
            "ask_price_0": np.linspace(100.1, 101.1, n),
            "bid_size_0": np.ones(n),
            "ask_size_0": np.ones(n),
        }
    )


def test_grid_rows_are_evenly_spaced_within_a_day() -> None:
    book = _book([f"2024-02-01 00:00:{s:02d}.4" for s in range(0, 59, 3)])
    grid = to_grid(book)
    deltas = grid["timestamp"].diff().dropna().dt.total_seconds()
    assert (deltas == GRID_SECONDS).all()


def test_a_missing_day_stays_missing() -> None:
    """Resampling must not invent days that were never downloaded.

    The whole-span version forward-filled a stale book across a two-day gap,
    which manufactured sixteen days of BTCUSDT out of nothing.
    """
    book = pd.concat(
        [
            _book([f"2024-02-01 00:00:{s:02d}" for s in range(0, 50, 5)]),
            _book([f"2024-02-04 00:00:{s:02d}" for s in range(0, 50, 5)]),
        ],
        ignore_index=True,
    )
    grid = to_grid(book)
    days = set(grid["timestamp"].dt.date.astype(str))
    assert days == {"2024-02-01", "2024-02-04"}


def test_quiet_stretches_carry_the_last_book_forward() -> None:
    book = _book(["2024-02-01 00:00:00", "2024-02-01 00:00:31"])
    grid = to_grid(book)
    # Rows between the two updates repeat the first book rather than vanishing.
    assert len(grid) == 7
    assert (grid["bid_price_0"].iloc[:-1] == 100.0).all()


# ---------------------------------------------------------------------------
# Sampled configurations must be constructible
# ---------------------------------------------------------------------------


def test_every_sampled_configuration_builds_its_model() -> None:
    """The search must not sample configurations that crash on construction.

    Before this test a third of the tree-regressor draws and most of the
    composite draws died on unexpected keyword arguments, the search recorded
    them as skipped, and whole regions of the space were silently never
    searched.
    """
    for config in draw(200, seed=99):
        build_model(config.model, **config.kwargs())  # must not raise
        rules_for(config)  # neither must the exit mapping


def test_rule_backed_models_never_run_feature_selection() -> None:
    """A selector can drop the very column a rule reads by name."""
    from experiments.grand_search import RULE_BACKED, RULES

    for config in draw(300, seed=5):
        if config.model in RULES or config.model in RULE_BACKED:
            assert config.select == "all", config.model


def test_hyperparameter_families_match_their_constructors() -> None:
    from experiments.grand_search import FAMILY

    for model, family in FAMILY.items():
        for params in HYPERPARAMETERS[family]:
            build_model(model, **params)  # must not raise


# ---------------------------------------------------------------------------
# Scaled holds are scored over the rows actually held
# ---------------------------------------------------------------------------


def test_scaled_hold_is_scored_on_the_path_not_the_fixed_horizon() -> None:
    """A position held longer than ``hold`` must be credited its own move.

    The price rises 1 bp per row. Forward over the fixed two-row horizon says
    +2 bp; a confident position held for four rows made +4. The bug credited it
    +2 — the move of a trade nobody took.
    """
    n = 12
    mid = 100.0 * (1.0 + 1e-4) ** np.arange(n)
    forward = np.full(n, 2.0)
    decision = np.zeros(n, dtype=int)
    decision[0] = 1
    p_buy = np.full(n, 1.0)
    rules = ThinningRules(hold_periods=2, hold_scale_by_confidence=1.0)

    trades = thin(decision, forward, np.ones(n), rules, mid=mid, p_buy=p_buy, p_sell=1.0 - p_buy)
    assert len(trades) == 1
    held = trades[0].exit_index - trades[0].entry_index
    assert held > 2, "full confidence must lengthen the hold"
    expected = (mid[trades[0].exit_index] / mid[0] - 1.0) * 1e4
    assert trades[0].move_bp == pytest.approx(expected)
    assert trades[0].move_bp != pytest.approx(2.0)


def test_unscaled_clock_exit_still_scores_the_fixed_horizon() -> None:
    """Anyone not using the scaling sees exactly the old behaviour."""
    n = 8
    mid = np.full(n, 100.0)
    forward = np.full(n, 20.0)
    decision = np.zeros(n, dtype=int)
    decision[0] = 1
    rules = ThinningRules(hold_periods=2)

    trades = thin(decision, forward, np.ones(n), rules, mid=mid)
    assert trades[0].move_bp == pytest.approx(20.0)
