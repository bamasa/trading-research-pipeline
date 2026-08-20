"""Pairs, cross-section and sizing — the arithmetic each rests on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.strategies.relative import (
    CrossSection,
    RelativeError,
    backtest,
    deviations,
    weights,
)
from trading_research.strategies.sizing import (
    ConfidenceScaled,
    FractionalKelly,
    SizingError,
    VolatilityTarget,
    apply_scale,
)
from trading_research.strategies.spread import (
    SpreadError,
    fit_hedge,
    fit_ou,
    rank_pairs,
    signal,
    z_score,
)

# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


def test_the_hedge_recovers_a_known_ratio() -> None:
    rng = np.random.default_rng(0)
    log_b = np.cumsum(rng.normal(0, 1e-3, 5_000))
    log_a = 0.4 + 1.7 * log_b + rng.normal(0, 1e-5, 5_000)

    hedge = fit_hedge(log_a, log_b)
    assert hedge.beta == pytest.approx(1.7, abs=0.01)
    assert hedge.alpha == pytest.approx(0.4, abs=0.01)


def test_a_mean_reverting_residual_has_a_finite_half_life() -> None:
    """An AR(1) with a known coefficient has a known half-life."""
    rng = np.random.default_rng(1)
    b = 0.9
    values = np.zeros(20_000)
    for i in range(1, len(values)):
        values[i] = b * values[i - 1] + rng.normal(0, 0.01)

    ou = fit_ou(values)
    assert ou.reverts
    assert ou.half_life == pytest.approx(np.log(2) / -np.log(b), rel=0.15)


def test_a_random_walk_does_not_revert() -> None:
    """The control: no pull means no half-life, however the fit is dressed up."""
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.normal(0, 1e-3, 20_000))
    ou = fit_ou(walk)
    assert not ou.reverts or ou.half_life > 1_000


def test_the_signal_enters_against_the_deviation() -> None:
    z = np.array([0.0, -2.5, -1.5, -0.2, 0.0, 3.0, 1.0, 0.1])
    out = signal(z, entry=2.0, exit_at=0.5)
    # Below the band: buy the cheap leg. Held until the residual has mostly
    # closed, not flipped on every crossing.
    assert out[1] == 1
    assert out[2] == 1
    assert out[3] == 0
    assert out[5] == -1
    assert out[6] == -1
    assert out[7] == 0


def test_an_exit_wider_than_the_entry_is_refused() -> None:
    with pytest.raises(SpreadError, match="must exceed"):
        signal(np.zeros(10), entry=1.0, exit_at=2.0)


def test_z_score_uses_the_fitted_moments_not_the_traded_ones() -> None:
    """A rolling z-score on the traded period would centre on the answer."""
    rng = np.random.default_rng(3)
    fitted = rng.normal(5.0, 2.0, 10_000)
    ou = fit_ou(fitted)
    # A later stretch that has drifted must show up as a deviation, not be
    # silently recentred.
    drifted = np.full(100, 11.0)
    assert np.mean(z_score(drifted, ou)) > 1.0


def test_ranking_refuses_degenerate_input() -> None:
    frame = pd.DataFrame({"a": np.ones(500), "b": np.ones(500)})
    assert rank_pairs(frame).empty


# ---------------------------------------------------------------------------
# Cross-section
# ---------------------------------------------------------------------------


def _panel(n: int = 2_000, names: int = 6, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1e-3, n))
    return pd.DataFrame({f"S{i}": common + np.cumsum(rng.normal(0, 5e-4, n)) for i in range(names)})


def test_demeaning_removes_the_common_move() -> None:
    """A day when everything rose must contribute nothing to the signal."""
    panel = _panel()
    relative = deviations(panel, CrossSection(lookback=20, demean=True))
    assert np.allclose(relative.dropna().sum(axis=1), 0.0, atol=1e-8)

    absolute = deviations(panel, CrossSection(lookback=20, demean=False))
    assert not np.allclose(absolute.dropna().sum(axis=1), 0.0, atol=1e-8)


def test_weights_are_market_neutral_and_normalised() -> None:
    panel = _panel()
    config = CrossSection(lookback=20, basket=2)
    w = weights(deviations(panel, config), config)
    active = w[(w != 0).any(axis=1)]
    assert np.allclose(active.sum(axis=1), 0.0, atol=1e-9)
    assert np.allclose(active.abs().sum(axis=1), 1.0, atol=1e-9)


def test_the_long_side_is_the_laggards() -> None:
    """Reversion, not momentum: the name that fell relative to peers is bought."""
    panel = _panel(names=4)
    # The deviation is a *return* over the lookback, so shifting the whole
    # series by a constant changes nothing. The fall has to happen inside the
    # window: the last twenty rows slope down.
    panel.iloc[-20:, panel.columns.get_loc("S0")] -= np.linspace(0.0, 0.05, 20)
    config = CrossSection(lookback=20, basket=1)
    w = weights(deviations(panel, config), config)
    last = w.iloc[-1]
    assert last["S0"] > 0


def test_a_panel_too_narrow_for_the_basket_is_refused() -> None:
    panel = _panel(names=3)
    config = CrossSection(lookback=20, basket=2)
    with pytest.raises(RelativeError, match="need 4 instruments"):
        weights(deviations(panel, config), config)


def test_every_name_pays_its_own_round_trip() -> None:
    """A basket of ten expresses one view and pays ten costs."""
    panel = _panel(n=1_000, names=8)
    config = CrossSection(lookback=20, basket=4, hold=20)
    free = backtest(panel, config, cost_bp_per_name=0.0)
    charged = backtest(panel, config, cost_bp_per_name=10.0)
    assert len(free) == len(charged)
    # Gross exposure is 1, so a 10 bp per-name charge costs exactly 10 bp.
    assert np.allclose(charged["cost_bp"], 10.0)
    assert np.allclose(free["gross_bp"], charged["gross_bp"])


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_volatility_scaling_shrinks_the_noisy_moments() -> None:
    rule = VolatilityTarget(target_bp=10.0, max_leverage=3.0)
    scale = rule.scale(np.array([5.0, 10.0, 40.0]))
    assert scale[0] > scale[1] > scale[2]
    assert scale[1] == pytest.approx(1.0)


def test_volatility_scaling_is_capped_against_a_quiet_forecast() -> None:
    """The failure mode of every vol-scaled book: calm, then a jump."""
    rule = VolatilityTarget(target_bp=10.0, max_leverage=3.0)
    assert rule.scale(np.array([1e-9]))[0] == 3.0


def test_confidence_scaling_is_a_ramp_with_a_floor() -> None:
    rule = ConfidenceScaled(floor=0.4, ceiling=0.6)
    scale = rule.scale(np.array([0.3, 0.4, 0.5, 0.6, 0.9]))
    assert scale[0] == 0.0
    assert scale[1] == 0.0
    assert scale[2] == pytest.approx(0.5)
    assert scale[3] == pytest.approx(1.0)
    assert scale[4] == pytest.approx(1.0)


def test_kelly_never_reverses_the_trade() -> None:
    """Sizing sets size. A negative edge means stand aside, not go the other way."""
    rule = FractionalKelly()
    assert rule.scale(np.array([-5.0]), np.array([10.0]))[0] == 0.0


def test_kelly_is_capped() -> None:
    rule = FractionalKelly(fraction=1.0, max_leverage=2.0)
    assert rule.scale(np.array([100.0]), np.array([0.1]))[0] == 2.0


def test_scaling_cannot_turn_a_losing_edge_positive() -> None:
    """The identity this module exists to state plainly."""
    rng = np.random.default_rng(5)
    net = rng.normal(-3.0, 20.0, 10_000)
    scale = np.abs(rng.normal(1.0, 0.4, 10_000))
    assert np.mean(apply_scale(net, scale)) < 0


def test_impossible_parameters_are_refused() -> None:
    with pytest.raises(SizingError):
        VolatilityTarget(target_bp=0.0)
    with pytest.raises(SizingError):
        ConfidenceScaled(floor=0.7, ceiling=0.5)
    with pytest.raises(SizingError):
        FractionalKelly(fraction=0.0)
    with pytest.raises(SizingError, match="rows"):
        apply_scale(np.zeros(5), np.zeros(3))
