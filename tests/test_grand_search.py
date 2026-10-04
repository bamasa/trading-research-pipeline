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


# ---------------------------------------------------------------------------
# Execution as an axis: the default is the taker every result was measured with
# ---------------------------------------------------------------------------


def _execution_data(with_market: bool = False):
    """Four days of a persistent random walk on the grid, with the micro plane."""
    from experiments.grand_search import ROWS_PER_DAY, Data
    from trading_research.pipeline.execution import GridMarket

    n, rng = int(ROWS_PER_DAY * 4), np.random.default_rng(7)
    steps = rng.normal(0, 1.5e-4, n)
    for i in range(1, n):
        steps[i] += 0.15 * steps[i - 1]
    mid = 100.0 * np.exp(np.cumsum(steps))
    spread_bp = 1.0 + rng.gamma(2.0, 0.5, n)
    log_mid = pd.Series(np.log(mid))
    frame = pd.DataFrame({"mid": mid})
    frame["spread_bp"] = spread_bp
    frame["queue_imbalance"] = np.tanh(rng.normal(0, 0.5, n) + 2000 * np.r_[steps[1:], 0.0])
    for span in (20, 50):
        frame[f"log_mid_ret{span}"] = (log_mid.diff(span) * 1e4).to_numpy()
    frame["log_mid_vol50"] = (log_mid.diff().rolling(50).std() * 1e4).to_numpy()
    frame["imbalance_change_10"] = frame["queue_imbalance"].diff(10)
    frame["mid_reversal_10"] = ((log_mid.diff(10) - log_mid.diff(20) / 2) * 1e4).to_numpy()
    frame["size_shock_10"] = rng.normal(0, 0.1, n)
    market = None
    if with_market:
        half = mid * spread_bp / 2e4
        market = GridMarket(
            "SYNTH",
            np.arange(n, dtype=np.int64) * GRID_SECONDS * 10**9,
            mid - half,
            mid + half,
            rng.uniform(1, 20, n),
            rng.uniform(1, 20, n),
            sell_at_bid=rng.exponential(5.0, n),
            buy_at_ask=rng.exponential(5.0, n),
        )
    return Data({4000: frame}, spread_bp, mid, market=market)


def test_execution_is_not_a_drawn_axis_so_draws_are_unchanged() -> None:
    """Labels of the first draws, recorded before execution became an axis."""
    from experiments.grand_search import EXECUTION_SPACE, SPACE

    assert not set(EXECUTION_SPACE) & set(SPACE)
    labels = [c.label() for c in draw(4, 20240819)]
    assert labels == [
        "bound_mixed/touch/net_pnl/h60/hold60/cd120/flip3/net_per_trade_bp/5d/tr20/tight_only/top40/nw4000/[default]",
        "ensemble/depth/direction/h12/hold240/cd240/take_profit_stop3/net_bp/2d/tr20/quiet_out/top16/nw16000/[default]",
        "momentum/micro/direction/h120/hold240/cd480/confidence25/net_per_trade_bp/1d/tr20/tight_only/all/nw4000/[default]",
        "xgboost_regressor/micro/forward_smoothed/h60/hold120/cd0/trailing12/net_bp/1d/tr20/tight_only/top40/nw16000/[default]",
    ]
    assert all(c.execution_variant == ("taker", "none") for c in draw(50, 3))


#: (model, exit, hold, gate, objective) -> (trades, net bp summed), recorded on
#: this fixture before the taker path was routed through pipeline.execution.
PINNED = [
    (("momentum", "clock", 24, "open", "net_bp"), (78, -988.6205514340611)),
    (("order_flow", "clock", 24, "open", "net_bp"), (225, -2385.55053425401)),
    (("order_flow", "take_profit_stop", 60, "open", "net_per_trade_bp"), (191, -2225.407826662346)),
    (("order_flow", "trailing", 60, "quiet_out", "net_per_trade_bp"), (266, -2922.7627150229036)),
]


@pytest.mark.parametrize(("spec", "expected"), PINNED)
def test_the_default_execution_reproduces_the_taker_results(spec, expected) -> None:
    from experiments.grand_search import run_block

    model, exit_, hold, gate, objective = spec
    config = _config(
        model=model, exit=exit_, hold=hold, cooldown=hold, gate=gate, objective=objective,
        train_days=2, exit_level=6.0,
    )  # fmt: skip
    data = _execution_data()
    trades = run_block(config, data, [(0, len(data.mid))], [])
    assert len(trades) == expected[0]
    assert float(trades["net_bp"].sum()) == pytest.approx(expected[1], rel=1e-12)
    assert "filled" not in trades.columns


def test_each_configuration_has_seven_execution_variants_taker_first() -> None:
    from experiments.grand_search import execution_variants

    config = _config()
    variants = execution_variants(config)
    assert len(variants) == 7 and variants[0] == config
    assert len({v.label() for v in variants}) == 7
    assert variants[0].label() == config.label() and "/" + "taker" not in config.label()


def test_passive_and_guarded_variants_run_on_the_grid() -> None:
    from experiments.grand_search import run_block

    data = _execution_data(with_market=True)
    span = [(0, len(data.mid))]
    base = _config(model="order_flow", train_days=2, cooldown=24)
    passive = run_block(
        _config(model="order_flow", train_days=2, cooldown=24, execution="passive_entry"),
        data, span, [],
    )  # fmt: skip
    assert passive["filled"].any() and not passive["filled"].all()
    assert (passive.loc[~passive["filled"], "net_bp"] == 0.0).all()
    breaks = list(range(40_000, len(data.mid), 2_000))
    plain = run_block(base, data, span, breaks)
    pulled = run_block(
        _config(model="order_flow", train_days=2, cooldown=24, regime_policy="guard_pull"),
        data, span, breaks,
    )  # fmt: skip
    assert 0 < len(pulled) < len(plain)
    with pytest.raises(ValueError, match="event-time data"):
        run_block(
            _config(model="order_flow", train_days=2, execution="market_maker"), data, span, []
        )


def test_the_last_rung_is_expanded_and_a_failing_variant_is_recorded() -> None:
    from types import SimpleNamespace

    from experiments.grand_search import expand_last_rung

    config = _config()

    def score(variant, budget):
        if variant.execution == "market_maker":
            raise ValueError("no prints")
        bonus = 1.0 if variant.execution_variant == ("passive_entry", "guard_pull") else 0.0
        return {"trades": 40.0, "net_per_trade_bp": -2.0 + bonus}

    outcome = SimpleNamespace(
        rungs=[SimpleNamespace(budget=24, results=[(config, score(config, 24))])]
    )
    best, metrics, table = expand_last_rung(outcome, score)
    assert best.execution_variant == ("passive_entry", "guard_pull")
    assert metrics["net_per_trade_bp"] == -1.0
    assert table["skipped"].notna().sum() == 3
