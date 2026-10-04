"""Execution as an axis: each mode reproduces the machinery it wraps.

The taker mode must be :func:`thin` with :class:`TakerCosts`, to the bit, or
every earlier result in the repository stops reproducing the moment it is
routed through the new interface. The passive mode must be
:func:`maker.simulate` and :func:`maker.score`. The market maker must run the
event-time simulator end to end on a synthetic market, under the same
permission rules as the study.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.backtest.maker import MakerCosts, PostingRules, score, simulate
from trading_research.market_making import prereg
from trading_research.market_making.synthetic import SYNTHETIC_DAY, random_market
from trading_research.pipeline import market_making as mm
from trading_research.pipeline.execution import (
    ATTEMPT_COLUMNS,
    VALID_PAIRS,
    ExecutionMode,
    GridMarket,
    SignalStream,
    check_pair,
    execute,
    guarded_rows,
)

GRID_NS = 5 * 1_000_000_000


def _grid(n: int = 4_000, seed: int = 3) -> tuple[GridMarket, np.ndarray]:
    """A random walk on a five-second grid, with prints at the touch and a signal."""
    rng = np.random.default_rng(seed)
    tick = 0.01
    mid_ticks = 10_000 + np.cumsum(rng.integers(-1, 2, n))
    half = rng.integers(1, 3, n)
    bid, ask = (mid_ticks - half) * tick, (mid_ticks + half) * tick
    start = pd.Timestamp(SYNTHETIC_DAY.isoformat()).value
    market = GridMarket(
        "FUZZUSDT",
        start + np.arange(n, dtype=np.int64) * GRID_NS,
        bid,
        ask,
        rng.uniform(1, 20, n),
        rng.uniform(1, 20, n),
        sell_at_bid=rng.exponential(3.0, n) * (rng.random(n) < 0.4),
        buy_at_ask=rng.exponential(3.0, n) * (rng.random(n) < 0.4),
        tick=tick,
    )
    decision = np.where(rng.random(n) < 0.03, rng.choice([-1, 1], n), 0)
    return market, decision


def test_the_taker_mode_is_thin_with_taker_costs() -> None:
    market, decision = _grid()
    stream = SignalStream.from_grid("FUZZUSDT", market.labels_ns, decision, hold_rows=24)
    result = execute(stream, ExecutionMode.TAKER, market=market)

    costs = TakerCosts()
    forward = market.forward_bp(24)
    expected_decision = decision.copy()
    expected_decision[~np.isfinite(forward)] = 0
    trades = thin(
        expected_decision, np.nan_to_num(forward), market.spread_bp, ThinningRules(hold_periods=24)
    )
    gross = np.array([t.direction * t.move_bp for t in trades])
    cost = np.array([float(costs.round_trip_bp(t.entry_spread_bp)) for t in trades])
    assert len(trades) > 20
    assert result.attempts["entry_index"].tolist() == [t.entry_index for t in trades]
    np.testing.assert_array_equal(result.attempts["net_bp"].to_numpy(), gross - cost)
    assert result.attempts["filled"].all()
    assert list(result.attempts.columns[: len(ATTEMPT_COLUMNS)]) == list(ATTEMPT_COLUMNS)
    assert result.summary["net_per_attempt_bp"] == pytest.approx(float(np.mean(gross - cost)))


def test_the_passive_mode_is_maker_simulate_and_score() -> None:
    market, decision = _grid(seed=4)
    stream = SignalStream.from_grid("FUZZUSDT", market.labels_ns, decision, hold_rows=12)
    rules = PostingRules(timeout_rows=12, hold_rows=12)
    costs = MakerCosts()
    result = execute(stream, "passive_entry", market=market, posting=rules, maker=costs)

    attempts = simulate(
        decision,
        market.bid,
        market.ask,
        market.bid_size,
        market.ask_size,
        market.sell_at_bid,
        market.buy_at_ask,
        rules,
        tick=market.tick,
    )
    expected = score(attempts, costs, market.spread_bp)
    assert result.summary["attempts"] == expected["attempts"]
    assert result.summary["fills"] == expected["fills"]
    assert 0 < expected["fills"] < expected["attempts"]
    assert result.summary["net_bp"] == pytest.approx(expected["net_bp"], abs=1e-9)
    assert result.summary["net_per_attempt_bp"] == pytest.approx(expected["net_per_attempt_bp"])
    assert (result.attempts.loc[~result.attempts["filled"], "net_bp"] == 0.0).all()


def test_a_signal_never_acts_before_it_is_known() -> None:
    labels = np.arange(10, dtype=np.int64) * GRID_NS
    stream = SignalStream(
        "X", np.array([GRID_NS * 2 + 1, GRID_NS * 5]), np.array([1, -1]), np.ones(2), GRID_NS
    )
    decision = stream.on_grid(labels)
    assert decision.tolist() == [0, 0, 0, 1, 0, -1, 0, 0, 0, 0]


def test_the_guard_opens_strictly_after_the_flag_and_closes_with_its_window() -> None:
    labels = np.arange(10, dtype=np.int64) * GRID_NS
    guarded = guarded_rows(labels, np.array([2 * GRID_NS]), 2 * GRID_NS)
    assert guarded.tolist() == [False, False, False, True, True, False, False, False, False, False]

    market, decision = _grid()
    stream = SignalStream.from_grid("FUZZUSDT", market.labels_ns, decision, hold_rows=6)
    flags = market.labels_ns[::200]
    plain = execute(stream, "taker", market=market)
    pulled = execute(stream, "taker", market=market, regime_policy="guard_pull", flags_ns=flags)
    assert len(pulled.attempts) < len(plain.attempts)


def test_only_the_registered_pairs_exist() -> None:
    assert len(VALID_PAIRS) == 7
    for mode in ("taker", "passive_entry"):
        with pytest.raises(ValueError, match="widen"):
            check_pair(mode, "guard_widen")
    assert check_pair("market_maker", "guard_widen") is ExecutionMode.MARKET_MAKER
    with pytest.raises(ValueError):
        check_pair("market_maker", "sometimes")


# ---------------------------------------------------------------------------
# The market maker
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
DAYS = [SYNTHETIC_DAY, SYNTHETIC_DAY + timedelta(days=1)]


@pytest.fixture(scope="module")
def market_root(tmp_path_factory: pytest.TempPathFactory, write_market_day) -> Path:
    """Two whole synthetic days, so the registered warm-up and day-end flatten apply."""
    root = tmp_path_factory.mktemp("mm")
    for seed, day in enumerate(DAYS):
        events = random_market(
            seed, n_snapshots=86_400, snapshot_ms=1000, prints_per_snapshot=0.3, day=day
        )
        write_market_day(events, root)
    return root


def _sources(root: Path) -> mm.MakerSources:
    return mm.MakerSources((root / "book",), root / "trades", root / "funding")


def test_the_market_maker_runs_end_to_end_on_a_synthetic_market(market_root: Path) -> None:
    plan = mm.MakerPlan("S0", mm.SimConfig(clip_notional=250.0, clip_touch_share=1e9))
    result = execute(
        None, "market_maker", symbol="FUZZUSDT", days=DAYS, plan=plan, sources=_sources(market_root)
    )
    assert result.mode is ExecutionMode.MARKET_MAKER
    assert result.days is not None and (result.days["status"] == "ok").all()
    assert not result.days["excluded"].any()
    assert len(result.attempts) == 2 and result.attempts["filled"].all()
    net = result.days["net"].to_numpy()
    turnover = (result.days["maker_turnover"] + result.days["taker_turnover"]).to_numpy()
    np.testing.assert_allclose(result.attempts["net_bp"], net / turnover * 1e4)
    assert result.summary["net_per_day"] == pytest.approx(net.mean())
    terms = sum(result.summary[t] for t in mm.TERMS)
    assert terms - 2 * result.summary["fees"] == pytest.approx(result.summary["net_per_day"])


def test_a_guard_policy_and_a_signal_lean_run_through_the_same_path(market_root: Path) -> None:
    config = mm.SimConfig(clip_notional=250.0, clip_touch_share=1e9)
    params = {"skew_bp": 2, "k": 0.0, "min_edge_bp": 0.0}
    plan = mm.MakerPlan("S1", config, params, sigma_ref=5.0)
    start = pd.Timestamp(SYNTHETIC_DAY.isoformat()).value
    flags = start + np.arange(1, 24, dtype=np.int64) * 3600 * 1_000_000_000
    pulled = execute(
        None, "market_maker", symbol="FUZZUSDT", days=DAYS, plan=plan,
        sources=_sources(market_root), regime_policy="guard_pull", flags_ns=flags,
        guard_window_min=30.0,
    )  # fmt: skip
    plain = execute(
        None, "market_maker", symbol="FUZZUSDT", days=DAYS, plan=plan, sources=_sources(market_root)
    )
    assert pulled.days is not None and plain.days is not None
    assert pulled.days["bid_quoted_s"].sum() < plain.days["bid_quoted_s"].sum()

    stream = SignalStream(
        "FUZZUSDT", flags, np.where(np.arange(len(flags)) % 2, 1, -1), np.full(len(flags), 5.0),
        600 * 1_000_000_000,
    )  # fmt: skip
    leaning = execute(
        stream, "market_maker", days=DAYS, plan=mm.MakerPlan("S1", config, params, 5.0, lean=1.0),
        sources=_sources(market_root),
    )  # fmt: skip
    assert leaning.days is not None
    assert not leaning.days[["fills", "net"]].equals(plain.days[["fills", "net"]])


def test_the_frozen_plan_is_the_registered_one() -> None:
    config = REPO / "configs" / "mm_prereg.yaml"
    s1 = mm.frozen_plan("S1", "BICOUSDT", path=config)
    assert s1.params == {"skew_bp": 2, "k": 0.5, "min_edge_bp": 4}
    assert s1.config.clip_notional == pytest.approx(20.0434)
    assert s1.config.soft_limit_clips == 6.0
    assert s1.sigma_ref == pytest.approx(7.92472)
    s3 = mm.frozen_plan("S3", "BICOUSDT", path=config)
    assert (s3.params["guards"], s3.params["action"], s3.params["guard_window_min"]) == (
        "S1",
        "widen",
        15,
    )
    assert mm.make_quoter(s3).name == "S3"
    assert mm.frozen_plan("S0", "XRPUSDT", path=config).params == {}
    with pytest.raises(ValueError, match="no frozen clip"):
        mm.frozen_plan("S1", "DOGEUSDT", path=config)


def test_held_out_days_are_refused_and_opened_only_by_the_ledger(tmp_path: Path) -> None:
    assert isinstance(mm.access_for([SYNTHETIC_DAY]), mm.OutsideStudy)
    assert mm.access_for([date(2024, 2, 3)]).blocks == frozenset({"D"})
    held = [date(2024, 2, 26)]
    with pytest.raises(prereg.HeldOutLocked, match="allow-heldout"):
        mm.access_for(held)
    # Allowed, the ledger still decides: here there is no registration to check.
    with pytest.raises(prereg.HeldOutLocked, match="no pre-registration document"):
        mm.access_for(held, allow_heldout=True, repo=tmp_path)
    with pytest.raises(prereg.HeldOutLocked, match="mix"):
        mm.access_for([SYNTHETIC_DAY, date(2024, 2, 3)])
    with pytest.raises(prereg.HeldOutLocked, match="H and F only"):
        mm.access_for([date(2024, 3, 10)], allow_heldout=True, repo=tmp_path)
