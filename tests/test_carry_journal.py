"""The carry journal: a pick from past funding only, written once, scored after the month."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.live import carry_journal as cj
from trading_research.strategies.carry import UNIVERSE, Market

#: funding per settlement by coin; the rest pay 0.5 bp
RATES = {"LTCUSDT": 3e-4, "FILUSDT": 2e-4}


def market(coin: str, months: Sequence[pd.Period], rate: float | None = None) -> Market:
    a = pd.Timestamp(min(months).start_time).tz_localize("UTC")
    b = pd.Timestamp(max(months).end_time).tz_localize("UTC").floor("1h")
    hours = pd.date_range(a, b, freq="1h")
    price = pd.Series(100.0 + np.arange(len(hours)) * 0.01, index=hours)
    funding = pd.Series(RATES.get(coin, 5e-5) if rate is None else rate,
                        index=hours[hours.hour % 8 == 0])  # fmt: skip
    return Market(price, price * 1.0001, funding)


def loader(coin: str, months: Sequence[pd.Period]) -> Market:
    return market(coin, months)


def test_the_pick_is_the_highest_funding_and_enters_at_the_next_hour(tmp_path: Path) -> None:
    did = cj.step(tmp_path, loader, pd.Timestamp("2026-11-15 10:42", tz="UTC"))
    pick = json.loads((tmp_path / "2026-11.pick.json").read_text())
    assert pick["coin"] == "LTCUSDT"
    assert pick["enters_at"] == "2026-11-15T11:00:00Z"
    assert pick["funding_months"] == ["2026-08", "2026-09", "2026-10"]
    assert list(pick["funding_bp_a_day"])[:2] == ["LTCUSDT", "FILUSDT"]
    assert did == ["2026-11: LTCUSDT from 2026-11-15T11:00:00Z"]
    # a second run the same month does nothing new
    assert cj.step(tmp_path, loader, pd.Timestamp("2026-11-20", tz="UTC")) == []
    assert "| 2026-11 | LTCUSDT |" in (tmp_path / "README.md").read_text()


def test_no_pick_before_the_last_month_is_published(tmp_path: Path) -> None:
    def late(coin: str, months: Sequence[pd.Period]) -> Market:
        mk = market(coin, months)
        cut = pd.Timestamp("2026-11-01", tz="UTC")
        return Market(mk.spot, mk.perp, mk.funding[mk.funding.index < cut])

    did = cj.step(tmp_path, late, pd.Timestamp("2026-12-01 03:00", tz="UTC"))
    assert did == ["2026-12: the last month's funding is not published yet; no pick"]
    assert not (tmp_path / "2026-12.pick.json").exists()


def test_nothing_before_the_first_month(tmp_path: Path) -> None:
    assert cj.step(tmp_path, loader, pd.Timestamp("2026-10-20", tz="UTC")) == []


def test_a_month_is_scored_after_it_ends_and_the_next_one_picked(tmp_path: Path) -> None:
    cj.step(tmp_path, loader, pd.Timestamp("2026-11-15 10:42", tz="UTC"))
    did = cj.step(tmp_path, loader, pd.Timestamp("2026-12-01 10:00", tz="UTC"))
    assert did[0].startswith("2026-11: scored, LTCUSDT")
    assert did[1] == "2026-12: LTCUSDT from 2026-12-01T11:00:00Z"
    s = json.loads((tmp_path / "2026-11.score.json").read_text())
    hours = pd.date_range("2026-11-15 11:00", "2026-11-30 23:00", freq="1h", tz="UTC")
    settlements = int((hours.hour % 8 == 0).sum())
    assert s["rank_among_coins"] == 1 and s["beats_all_coins"]
    assert s["coins_compared"] == len(UNIVERSE)
    # one leg's notional: funding on the perpetual's value, which drifts up a few percent here
    assert s["coin_carry_bp"] == pytest.approx(settlements * 3.0, rel=0.03)
    # the account: half the capital a leg, the open cost of both legs (19 bp of a leg)
    assert s["return_pct"] == pytest.approx((settlements * 3.0 - 19.0) / 2 / 100, rel=0.05)
    held = cj.state(tmp_path)
    assert held.coin == "LTCUSDT" and held.equity == pytest.approx(s["equity_after"])
    with pytest.raises(FileExistsError):
        cj._write_once(tmp_path / "2026-11.score.json", s)


def test_a_kept_coin_pays_no_cost_and_a_switch_pays_both(tmp_path: Path) -> None:
    cj.step(tmp_path, loader, pd.Timestamp("2026-11-30 22:10", tz="UTC"))
    cj.step(tmp_path, loader, pd.Timestamp("2026-12-01 10:00", tz="UTC"))
    kept = json.loads((tmp_path / "2026-12.pick.json").read_text())
    assert kept["coin"] == "LTCUSDT"

    def flipped(coin: str, months: Sequence[pd.Period]) -> Market:
        rate = {"LTCUSDT": 5e-5, "XLMUSDT": 4e-4}.get(coin)
        return market(coin, months, rate)

    cj.step(tmp_path, flipped, pd.Timestamp("2027-01-01 10:20", tz="UTC"))
    dec = json.loads((tmp_path / "2026-12.score.json").read_text())
    jan = json.loads((tmp_path / "2027-01.pick.json").read_text())
    assert jan["coin"] == "XLMUSDT"
    month_hours = pd.date_range("2026-12-01", "2026-12-31 23:00", freq="1h", tz="UTC")
    # December's coin was kept: the whole month at LTC's rate under `flipped`, no cost
    settlements = int((month_hours.hour % 8 == 0).sum())
    assert dec["return_pct"] == pytest.approx(settlements * 0.5 / 2 / 100, rel=0.05)
    cj.step(tmp_path, flipped, pd.Timestamp("2027-02-01 10:20", tz="UTC"))
    jan_s = json.loads((tmp_path / "2027-01.score.json").read_text())
    # January: LTC until 10:00 and closed, XLM opened at 11:00; two legs closed and two opened
    assert jan_s["coin"] == "XLMUSDT"
    assert jan_s["return_pct"] < settlements * 4.0 / 2 / 100
    assert jan_s["return_pct"] > (settlements - 4) * 4.0 / 2 / 100 - 2 * 19 / 2 / 100 - 0.01


def test_a_month_without_its_prices_waits(tmp_path: Path) -> None:
    cj.step(tmp_path, loader, pd.Timestamp("2026-11-15 10:42", tz="UTC"))

    def short(coin: str, months: Sequence[pd.Period]) -> Market:
        mk = market(coin, months)
        cut = pd.Timestamp("2026-11-30 12:00", tz="UTC")
        return Market(mk.spot[mk.spot.index < cut], mk.perp[mk.perp.index < cut], mk.funding)

    did = cj.step(tmp_path, short, pd.Timestamp("2026-12-01 02:00", tz="UTC"))
    assert did[0] == "2026-11: over, but Binance has not published all of it yet; not scored"
    assert not (tmp_path / "2026-11.score.json").exists()
