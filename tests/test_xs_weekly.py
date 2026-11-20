"""The weekly cross-section study on planted panels: universe, look-ahead, the book's accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.xs_weekly import features, portfolio, rankers
from trading_research.xs_weekly.panel import Panel

C, D = 30, 700


def planted(strength: float, seed: int = 0) -> Panel:
    """Daily log returns whose next week leans against the last 28 days' relative move."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.03, (D, C))
    past = pd.DataFrame(r).rolling(28).sum().shift(1).fillna(0).to_numpy()
    r -= strength * (past - past.mean(axis=1, keepdims=True)) / 28
    close = 10 * np.exp(np.cumsum(r, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    volume = np.tile(np.linspace(1e6, 3e7, C), (D, 1))
    late = np.full(D, np.nan)
    close[:200, 0], open_[:200, 0] = late[:200], late[:200]  # listed late
    return Panel(pd.date_range("2021-01-01", periods=D, freq="1D", tz="UTC"),
                 tuple(f"C{i}USDT" for i in range(C)), open_, close, volume,
                 np.where(np.isfinite(close), 1.0, np.nan))  # fmt: skip


def test_the_universe_needs_history_and_takes_the_most_traded() -> None:
    p = planted(0.0)
    p.quote_volume[:, :] = np.tile(np.arange(C, dtype=float) + 1, (D, 1))
    uni = features.universe(p)
    assert not uni[: features.MIN_HISTORY].any()
    assert not uni[: 200 + features.MIN_HISTORY, 0].any() and uni[400, 0] == (C <= 100)
    features.UNIVERSE, saved = 5, features.UNIVERSE
    try:
        assert set(np.flatnonzero(features.universe(p)[400])) == set(range(C - 5, C))
    finally:
        features.UNIVERSE = saved


def test_inputs_read_nothing_from_the_decision_day_on() -> None:
    p = planted(0.5)
    t = 400
    later = Panel(
        p.days, p.symbols, p.open.copy(), p.close.copy(), p.quote_volume.copy(), p.funding_bp.copy()
    )
    later.close[t:] *= 2
    later.open[t + 1 :] *= 2
    later.quote_volume[t:] *= 5
    later.funding_bp[t:] = 9.0
    a, b = features.build(p), features.build(later)
    np.testing.assert_array_equal(a.tab[: t + 1], b.tab[: t + 1])
    np.testing.assert_array_equal(a.universe[: t + 1], b.universe[: t + 1])
    np.testing.assert_array_equal(
        features.windows(a, t, np.arange(C)), features.windows(b, t, np.arange(C))
    )


def test_the_target_runs_open_to_open_a_week_and_a_stopped_coin_exits_at_its_last_close() -> None:
    p = planted(0.0)
    p.open[450:, 3], p.close[450:, 3] = np.nan, np.nan
    f = features.build(p)
    assert f.gross[300, 5] == pytest.approx(np.log(p.open[307, 5] / p.open[300, 5]))
    assert f.gross[445, 3] == pytest.approx(np.log(p.close[449, 3] / p.open[445, 3]))


def test_the_book_pays_costs_only_on_changes_and_funding_by_side() -> None:
    p = planted(0.0)
    f = features.build(p)
    idx = np.arange(C)
    fixed = np.arange(C, dtype=float)
    weeks = portfolio.book(f, [(400, idx, fixed), (407, idx, fixed)])
    first_cost = 2 * portfolio.LEGS * (0.5 / portfolio.LEGS) * portfolio.SIDE_BP / 1e4
    assert weeks["cost"].iloc[0] == pytest.approx(-first_cost) and weeks["cost"].iloc[1] == 0
    # Every coin's funding is 1 bp a day: longs pay what shorts receive, net zero.
    assert weeks["funding"].abs().max() < 1e-12


def test_a_planted_cross_section_is_ranked_and_a_null_is_not() -> None:
    mondays = np.arange(500, D - 8, 7)
    out = {}
    for strength in (0.9, 0.0):
        f = features.build(planted(strength, seed=2))
        picks = list(rankers.walk(f, "ridge", mondays))
        out[strength] = portfolio.book(f, picks)["ic"].mean()
    assert out[0.9] > 0.05 and abs(out[0.0]) < 0.05
