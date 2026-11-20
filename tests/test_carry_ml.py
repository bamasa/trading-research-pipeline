"""The carry ML study's pieces on planted markets: no look-ahead, the target, the choosers' past-only fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.carry_ml import evaluate, features, models
from trading_research.strategies.carry import Market

MONTHS = pd.period_range("2021-01", "2023-06", freq="M")


def market(level_bp: float, seed: int, drift: float = 0.0) -> Market:
    """Hourly prices and a funding settlement every eight hours, at about ``level_bp``."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2020-10-01", "2023-07-01", freq="1h", tz="UTC", inclusive="left")
    spot = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(hours)))), index=hours)
    perp = spot * (1 + rng.normal(0.0002, 0.0001, len(hours)))
    settle = hours[hours.hour % 8 == 0]
    trend = drift * np.arange(len(settle)) / len(settle)
    rates = (level_bp + trend + rng.normal(0, 0.5, len(settle))) / 3 / 1e4
    return Market(spot=spot, perp=perp, funding=pd.Series(rates, index=settle))


def readable(_coin: str, _month: pd.Period) -> bool:
    return True


@pytest.fixture(scope="module")
def markets() -> dict[str, Market]:
    return {f"C{i}": market(level_bp=float(i), seed=i) for i in range(6)}


def test_a_months_features_read_nothing_from_the_month(markets: dict[str, Market]) -> None:
    month = pd.Period("2022-03", "M")
    start = month.start_time.tz_localize("UTC")
    mk = markets["C2"]
    later = Market(
        spot=mk.spot.where(mk.spot.index < start, mk.spot * 3),
        perp=mk.perp.where(mk.perp.index < start, mk.perp * 2),
        funding=mk.funding.where(mk.funding.index < start, 0.01),
    )
    before, after = features.row(mk, month), features.row(later, month)
    assert {k: before[k] for k in features.FEATURES if k != "rank_f3"} == {
        k: after[k] for k in features.FEATURES if k != "rank_f3"
    }
    assert after["target"] != before["target"]


def test_the_target_is_the_months_funding_a_day(markets: dict[str, Market]) -> None:
    month = pd.Period("2022-02", "M")
    mk = markets["C3"]
    a, b = month.start_time.tz_localize("UTC"), (month + 1).start_time.tz_localize("UTC")
    expected = mk.funding[(mk.funding.index >= a) & (mk.funding.index < b)].sum() * 1e4 / 28
    assert features.row(mk, month)["target"] == pytest.approx(expected)
    assert features.row(mk, month)["target"] == pytest.approx(3.0, abs=0.2)


def test_models_score_a_month_from_earlier_months_only(markets: dict[str, Market]) -> None:
    rows = features.table(markets, MONTHS, readable)
    month = pd.Period("2022-06", "M")
    changed = rows.copy()
    changed.loc[changed["month"] >= month, "target"] = -100.0
    a = models.scores(rows, "ridge", pd.PeriodIndex([month]))
    b = models.scores(changed, "ridge", pd.PeriodIndex([month]))
    np.testing.assert_allclose(a["score"], b["score"])
    assert models.scores(rows, "ridge", pd.PeriodIndex([MONTHS[5]])).empty  # under 12 months


def test_the_rule_and_ridge_find_the_coin_paying_most(markets: dict[str, Market]) -> None:
    rows = features.table(markets, MONTHS, readable)
    later = MONTHS[pd.Period("2022-06", "M") <= MONTHS]
    for name in ("rule_lb3", "ridge"):
        chosen = evaluate.picks(models.scores(rows, name, later))
        assert set(chosen.values()) == {"C5"}
        quality = evaluate.forecast_quality(models.scores(rows, name, later), rows)
        assert quality.mean() > 0.8


def test_the_replay_earns_the_funding_it_holds(markets: dict[str, Market]) -> None:
    months = pd.period_range("2022-01", "2022-06", freq="M")
    held = evaluate.run(markets, months, dict.fromkeys(months, "C5"), readable)
    low = evaluate.run(markets, months, dict.fromkeys(months, "C1"), readable)
    assert held["return_pct"].sum() > low["return_pct"].sum() > 0


def test_one_sided_and_holm() -> None:
    r = evaluate.one_sided(np.array([1.0, 2.0, 3.0, 2.0]))
    assert r["mean"] == pytest.approx(2.0) and r["p"] < 0.01 and r["months"] == 4
    adj = evaluate.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj == pytest.approx({"a": 0.03, "b": 0.08, "c": 0.5})
