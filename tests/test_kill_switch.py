"""The kill switch on a strategy's own results: quiet on an edge, fast on its loss, causal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.validation.kill_switch import kill_switch


def _returns(before: float, after: float, change: int, n: int = 600, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    mean = np.where(np.arange(n) < change, before, after)
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(mean + rng.standard_normal(n), index=index)


def test_an_edge_that_holds_rarely_raises_an_alarm() -> None:
    """A +0.2 sigma edge over 528 monitored periods: about one false alarm in ten runs."""
    alarms = sum(bool(kill_switch(_returns(0.2, 0.2, 600, seed=s)).events) for s in range(200))
    assert alarms / 200 <= 0.15


def test_a_lost_edge_is_caught_soon_after_it_is_lost_and_the_switch_keeps_what_was_made() -> None:
    delays = []
    for seed in range(200):
        returns = _returns(0.2, -1.0, 300, seed=10_000 + seed)
        result = kill_switch(returns)
        stop = result.first_stop
        assert stop is not None
        if stop.position >= 300:
            delays.append(stop.position - 300)
        after = result.equity.iloc[stop.position :]
        assert (after == after.iloc[0]).all()
        assert result.equity.iloc[-1] > returns.cumsum().iloc[-1]
    assert len(delays) >= 180 and float(np.median(delays)) <= 12


def test_no_decision_reads_a_later_period() -> None:
    returns = _returns(0.2, -1.0, 300, seed=7)
    events = [(e.kind, e.position) for e in kill_switch(returns, restart=True).events]
    for cut in (150, 310, 450):
        changed = returns.copy()
        changed.iloc[cut:] = np.random.default_rng(cut).standard_normal(len(returns) - cut) * 9
        again = [(e.kind, e.position) for e in kill_switch(changed, restart=True).events]
        assert [e for e in again if e[1] < cut] == [e for e in events if e[1] < cut], cut


def test_a_paper_recovery_restarts_the_strategy_only_when_asked() -> None:
    rng = np.random.default_rng(3)
    mean = np.r_[np.full(200, 0.2), np.full(100, -1.0), np.full(300, 1.0)]
    returns = pd.Series(mean + rng.standard_normal(600))
    kinds = [e.kind for e in kill_switch(returns, restart=True).events]
    assert kinds[:2] == ["stop", "restart"]
    assert [e.kind for e in kill_switch(returns).events] == ["stop"]


@pytest.mark.parametrize("bad", [{"warm_up": 1}, {"k": -0.1}, {"h": 0.0}])
def test_bad_settings_and_missing_returns_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        kill_switch(_returns(0.2, 0.2, 600), **bad)
    with pytest.raises(ValueError, match="finite"):
        kill_switch(pd.Series([0.1, np.nan, 0.2] * 40))


def test_a_stop_keeps_the_positions_already_open_and_takes_no_new_ones() -> None:
    returns = _returns(0.2, -1.0, 300, seed=11)
    booked = returns.shift(-1).fillna(0.0)  # each period's entries close in the next one
    result = kill_switch(returns, booked=booked)
    stop = result.first_stop
    assert stop is not None
    kept = booked.iloc[: stop.position + 1].sum()
    assert result.equity.iloc[-1] == pytest.approx(kept)
    with pytest.raises(ValueError, match="booked"):
        kill_switch(returns, booked=booked.iloc[:-1])
