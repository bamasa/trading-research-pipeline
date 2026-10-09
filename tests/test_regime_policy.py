"""The regime-policy axis: names, pairings, the guard on a decision stream, attribution."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_research.pipeline import market_making as mm
from trading_research.pipeline import regime
from trading_research.pipeline.execution import VALID_PAIRS, GridMarket, SignalStream, execute
from trading_research.pipeline.regime import RegimePolicy

H = regime.NS_PER_HOUR


def test_policy_names_round_trip_and_map_to_the_execution_layer() -> None:
    for name in ("none", "pull_4h", "pull_24h", "pull_120h", "widen_24h"):
        assert RegimePolicy.parse(name).name == name
    pull = RegimePolicy.parse("pull_24h")
    assert pull.execution_policy == "guard_pull"
    assert pull.window_ns == 24 * H
    assert pull.guard_window_min == 1440.0
    assert RegimePolicy.parse("widen_4h").execution_policy == "guard_widen"
    assert RegimePolicy().execution_policy == "none"


def test_meaningless_policies_are_refused() -> None:
    for bad in ("pull", "pull_h", "pull_4m", "stop_4h", "none_4h"):
        with pytest.raises(ValueError):
            RegimePolicy.parse(bad)
    with pytest.raises(ValueError, match="means nothing"):
        RegimePolicy("pull", 0)
    with pytest.raises(ValueError, match="means nothing"):
        RegimePolicy("none", 4)


def test_the_axis_as_numbers_is_the_window() -> None:
    assert [regime.window_hours(n) for n in regime.SEARCH_POLICIES] == [0, 4, 24, 120]
    assert regime.SEARCH_POLICIES == ("none", "pull_4h", "pull_24h", "pull_120h")


def test_every_policy_offered_is_a_valid_pair() -> None:
    taker = [p.name for p in regime.policies_for("taker")]
    passive = [p.name for p in regime.policies_for("passive_entry")]
    maker = [p.name for p in regime.policies_for("market_maker")]
    assert taker == passive == ["none", "pull_4h", "pull_24h", "pull_120h"]
    assert maker == [*taker, "widen_4h", "widen_24h", "widen_120h"]
    for mode, names in (("taker", taker), ("passive_entry", passive), ("market_maker", maker)):
        for name in names:
            assert (mode, RegimePolicy.parse(name).execution_policy) in VALID_PAIRS


def test_a_taker_cannot_widen() -> None:
    with pytest.raises(ValueError, match="no spread of its own"):
        RegimePolicy.parse("widen_4h").check("taker")
    assert RegimePolicy.parse("widen_4h").check("market_maker").value == "market_maker"


# ---------------------------------------------------------------------------
# The guard on a decision stream
# ---------------------------------------------------------------------------


def test_pull_blanks_entries_after_the_flag_and_not_at_it() -> None:
    labels = np.arange(10, dtype=np.int64) * H
    decision = np.ones(10, dtype=int)
    flags = np.array([2 * H])
    out = regime.guard_decisions(labels, decision, flags, RegimePolicy.parse("pull_4h"))
    assert out.tolist() == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    assert decision.tolist() == [1] * 10  # a copy
    same = regime.guard_decisions(labels, decision, flags, RegimePolicy())
    assert same.tolist() == decision.tolist()
    with pytest.raises(ValueError, match="only be pulled"):
        regime.guard_decisions(labels, decision, flags, RegimePolicy.parse("widen_4h"))


def test_pull_drops_signals_from_a_stream() -> None:
    stream = SignalStream(
        "BTCUSDT",
        np.array([1, 3, 5, 30], dtype=np.int64) * H,
        np.array([1, -1, 1, -1]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        hold_ns=50 * 1_000_000_000,
    )
    guarded = regime.guard_stream(stream, np.array([2 * H]), RegimePolicy.parse("pull_24h"))
    assert (guarded.ts // H).tolist() == [1, 30]
    assert guarded.direction.tolist() == [1, -1]
    assert guarded.strength.tolist() == [0.1, 0.4]
    assert regime.guard_stream(stream, np.array([2 * H]), RegimePolicy()) is stream


def test_a_resting_entry_is_cancelled_by_a_flag_while_it_rests() -> None:
    posted = np.array([0, 10, 20, 30], dtype=np.int64) * H
    ends = posted + 5 * H
    flags = np.array([3 * H, 25 * H, 30 * H], dtype=np.int64)
    pull = RegimePolicy.parse("pull_4h")
    out = regime.cancel_times(posted, ends, flags, pull)
    assert (out[out >= 0] // H).tolist() == [3, 30]
    assert out.tolist()[1:3] == [-1, -1]  # no flag while resting; the flag came after the end
    assert (regime.cancel_times(posted, ends, flags, RegimePolicy()) == -1).all()
    with pytest.raises(ValueError, match="one end"):
        regime.cancel_times(posted, ends[:2], flags, pull)


# ---------------------------------------------------------------------------
# Regime attribution
# ---------------------------------------------------------------------------

DAYS = [date(2025, 3, 1) + timedelta(days=i) for i in range(10)]


def test_random_flags_fall_on_usable_hours_and_are_seeded() -> None:
    hours = regime.usable_hours(DAYS)
    assert len(hours) == 240
    sets = regime.random_flag_sets(DAYS, 3, sets=20)
    assert len(sets) == 20
    assert all(len(np.unique(s)) == 3 and np.isin(s, hours).all() for s in sets)
    again = regime.random_flag_sets(DAYS, 3, sets=200)
    assert all(np.array_equal(a, b) for a, b in zip(sets, again[:20], strict=True))
    with pytest.raises(ValueError, match="do not fit"):
        regime.random_flag_sets(DAYS[:1], 25)


def test_the_flag_count_is_what_falls_inside_the_days() -> None:
    hours = regime.usable_hours(DAYS)
    flags = np.array([hours[0] - H, hours[5], hours[-1], hours[-1] + H])
    assert regime.flags_inside(flags, DAYS) == 2


def test_attribution_compares_the_guard_with_random_pauses() -> None:
    hours = regime.usable_hours(DAYS)
    bad_hour = hours[50]

    def score(flags: np.ndarray) -> float:
        """A strategy that loses 10 on one hour unless a flag pulled it just before."""
        pulled = ((bad_hour - flags) > 0) & ((bad_hour - flags) <= 4 * H)
        return 0.0 if pulled.any() else -10.0

    real = np.array([bad_hour - 2 * H])
    result = regime.regime_attribution(score, real, -10.0, DAYS, policy="pull_4h")
    assert result.difference == 10.0
    assert result.n_flags == 1
    assert len(result.draws) == regime.TAKER_SETS
    assert result.exceeds
    assert result.row()["exceeds_random"] is True
    useless = regime.regime_attribution(score, np.array([hours[200]]), -10.0, DAYS, sets=20)
    assert useless.difference == 0.0
    assert not useless.exceeds


# ---------------------------------------------------------------------------
# Through the execution layer
# ---------------------------------------------------------------------------

GRID_NS = 5 * 1_000_000_000


def _market(n: int = 6_000, seed: int = 5) -> tuple[GridMarket, np.ndarray]:
    rng = np.random.default_rng(seed)
    mid = 10_000 + np.cumsum(rng.integers(-1, 2, n))
    labels = pd.Timestamp("2025-03-01").value + np.arange(n, dtype=np.int64) * GRID_NS
    market = GridMarket(
        "BTCUSDT", labels, (mid - 1) * 0.1, (mid + 1) * 0.1, np.ones(n), np.ones(n), tick=0.1
    )
    decision = np.where(rng.random(n) < 0.03, rng.choice([-1, 1], n), 0)
    return market, decision


def test_executing_under_a_pull_equals_executing_the_guarded_stream() -> None:
    market, decision = _market()
    stream = SignalStream.from_grid("BTCUSDT", market.labels_ns, decision, hold_rows=10)
    flags = market.labels_ns[[1_000, 4_500]]
    pull = RegimePolicy.parse("pull_4h")
    pulled = regime.execute_with_policy(stream, "taker", pull, flags, market=market)
    by_hand = execute(regime.guard_stream(stream, flags, pull), "taker", market=market)
    assert len(pulled.attempts) < len(execute(stream, "taker", market=market).attempts)
    pd.testing.assert_frame_equal(pulled.attempts, by_hand.attempts)
    plain = regime.execute_with_policy(stream, "taker", RegimePolicy(), None, market=market)
    assert len(plain.attempts) > len(pulled.attempts)


def test_executing_refuses_a_pairing_or_missing_flags() -> None:
    market, decision = _market(n=500)
    stream = SignalStream.from_grid("BTCUSDT", market.labels_ns, decision, hold_rows=10)
    with pytest.raises(ValueError, match="no spread of its own"):
        regime.execute_with_policy(stream, "taker", RegimePolicy.parse("widen_4h"), None)
    with pytest.raises(ValueError, match="needs the flag times"):
        regime.execute_with_policy(stream, "taker", RegimePolicy.parse("pull_4h"), None)


def test_the_maker_plan_takes_its_guard_window_in_minutes() -> None:
    plan = mm.MakerPlan(
        "S1", mm.SimConfig(clip_notional=250.0), {"skew_bp": 1, "k": 0, "min_edge_bp": 0}
    )
    widened = regime.maker_plan(plan, RegimePolicy.parse("widen_24h"))
    assert widened.regime_policy == "guard_widen"
    assert widened.guard_window_min == 1440.0
    assert regime.maker_plan(widened, RegimePolicy()).regime_policy == "none"
