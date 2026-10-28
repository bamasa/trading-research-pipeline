"""The pairs statistics: hedge ratio, Engle-Granger against known critical values, half-life, Holm."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_research.market_making.verdicts import holm
from trading_research.pairs.stats import (
    PairsError,
    adf_t,
    eg_null,
    engle_granger,
    half_life,
    holm_adjusted,
    mackinnon_critical,
    return_correlation,
)
from trading_research.strategies.spread import fit_hedge
from trading_research.synthetic.pairs import PairSpec, generate


def _legs(spec: PairSpec, rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = generate(spec)
    a, b = list(data.bars)
    la = np.log(data.bars[a]["close"].to_numpy())
    lb = np.log(data.bars[b]["close"].to_numpy())
    return (la[:rows], lb[:rows]) if rows else (la, lb)


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    shocks = rng.standard_normal(n)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + shocks[k]
    return x


def test_the_hedge_ratio_is_the_spread_plugins_and_recovers_the_planted_beta() -> None:
    la, lb = _legs(PairSpec(beta=1.3, seed=5))
    test = engle_granger(la, lb)
    assert test.hedge == fit_hedge(la, lb)
    assert test.hedge.beta == pytest.approx(1.3, abs=0.05)


def test_adf_without_lags_is_the_ordinary_least_squares_t() -> None:
    x = np.cumsum(np.random.default_rng(3).standard_normal(400))
    dx, level = np.diff(x), x[:-1]
    design = np.column_stack([level, np.ones_like(level)])
    beta, rss, *_ = np.linalg.lstsq(design, dx, rcond=None)
    s2 = rss[0] / (len(dx) - 2)
    se = math.sqrt(s2 * np.linalg.inv(design.T @ design)[0, 0])
    assert adf_t(x, lags=0, constant=True)[0] == pytest.approx(beta[0] / se, rel=1e-10)
    assert adf_t(x[None, :], lags=0)[0] == pytest.approx(adf_t(x, lags=0)[0])


def test_the_dickey_fuller_null_matches_mackinnon() -> None:
    walks = np.cumsum(np.random.default_rng(11).standard_normal((4000, 500)), axis=1)
    draws = adf_t(walks, lags=0, constant=True)
    for level in (0.05, 0.10):
        assert np.quantile(draws, level) == pytest.approx(
            mackinnon_critical(1, level, 500), abs=0.06
        )


@pytest.mark.parametrize("lags", [0, 1])
def test_the_engle_granger_null_matches_mackinnon_for_two_series(lags: int) -> None:
    draws = eg_null(500, lags)
    assert len(draws) == 4000 and np.all(np.diff(draws) >= 0)
    for level in (0.05, 0.10):
        assert np.quantile(draws, level) == pytest.approx(
            mackinnon_critical(2, level, 500), abs=0.06
        )
    assert np.quantile(draws, 0.01) == pytest.approx(mackinnon_critical(2, 0.01, 500), abs=0.15)


def test_the_null_is_seeded_and_cached() -> None:
    assert eg_null(300) is eg_null(300)
    assert not eg_null(300).flags.writeable


def test_cointegration_found_on_the_planted_pair_and_not_on_independent_walks() -> None:
    planted = engle_granger(*_legs(PairSpec(seed=1), 720))
    assert planted.t < planted.critical_5 and planted.p_value < 0.001
    found = [
        engle_granger(*_legs(PairSpec(seed=s, variant="independent"), 720)).p_value
        for s in range(20)
    ]
    assert np.mean(np.array(found) <= 0.05) <= 0.15
    assert np.median(found) > 0.2


def test_a_constant_residual_is_refused() -> None:
    lb = np.cumsum(np.random.default_rng(2).normal(0, 0.01, 300))
    with pytest.raises(PairsError, match="move as one"):
        engle_granger(0.5 + 2.0 * lb, lb)
    with pytest.raises(PairsError, match="different length"):
        engle_granger(lb, lb[:-1])


@pytest.mark.parametrize("planted", [6.0, 12.0])
def test_half_life_is_recovered_from_an_ar1(planted: float) -> None:
    phi = 0.5 ** (1.0 / planted)
    assert half_life(_ar1(phi, 20_000, 4)) == pytest.approx(planted, rel=0.1)


def test_half_life_of_a_random_walk_is_long_or_infinite() -> None:
    walk = np.cumsum(np.random.default_rng(8).standard_normal(3000))
    assert half_life(walk) > 50


def test_return_correlation() -> None:
    la, lb = _legs(PairSpec(seed=2))
    assert return_correlation(la, la) == pytest.approx(1.0)
    assert return_correlation(la, lb) > 0.8
    assert math.isnan(return_correlation(np.ones(10), la[:10]))


def test_holm_adjusted_p_values() -> None:
    adjusted = holm_adjusted([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    with_nan = holm_adjusted([0.01, float("nan")])
    assert with_nan[0] == pytest.approx(0.02) and math.isnan(with_nan[1])


def test_holm_adjusted_agrees_with_the_repositorys_holm() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(0, 0.1, rng.integers(1, 12))
        adjusted = holm_adjusted(p)
        passed = holm({str(k): v for k, v in enumerate(p)}, 0.05)
        assert [passed[str(k)] for k in range(len(p))] == list(adjusted <= 0.05)
