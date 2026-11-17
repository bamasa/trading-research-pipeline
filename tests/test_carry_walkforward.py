"""The carry replay's accounting, on markets whose answer is known."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments import carry_walkforward as cw


def _market(rate: float) -> cw.Market:
    hours = pd.date_range("2020-12-01", "2021-03-31 23:00", freq="1h", tz="UTC")
    flat = pd.Series(100.0, index=hours)
    settlements = hours[hours.hour % 8 == 0]
    return cw.Market(flat, flat, pd.Series(rate, index=settlements))


def test_the_best_coin_above_the_threshold_is_held_and_paid_its_funding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cw, "FIRST", pd.Period("2021-01", "M"))
    monkeypatch.setattr(cw, "LAST", pd.Period("2021-03", "M"))
    markets = {"AAAUSDT": _market(2e-4), "BBBUSDT": _market(0.0)}
    frame, chosen = cw.run(markets, guarded=False)
    assert chosen["2021-01"] == ["AAAUSDT"]
    q = 1.0 / cw.SLOTS / 2 / 100.0
    january = 31 * 3 * 2e-4 * q * 100.0 - cw.cost(q * 100.0)
    assert frame["return_pct"].iloc[0] == pytest.approx(january * 100)
    # kept in February, so no second entry
    assert frame["return_pct"].iloc[1] == pytest.approx(
        28 * 3 * 2e-4 * q * 100.0 / (1 + january) * 100
    )


def test_a_coin_below_the_threshold_leaves_its_slot_in_cash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cw, "FIRST", pd.Period("2021-01", "M"))
    monkeypatch.setattr(cw, "LAST", pd.Period("2021-01", "M"))
    frame, chosen = cw.run({"AAAUSDT": _market(1e-6)}, guarded=False)
    assert chosen["2021-01"] == [] and frame["return_pct"].iloc[0] == 0.0
