"""The carry engine and the automatic search's rules, on markets whose answer is known."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_research.search_engine import carry as search
from trading_research.strategies.carry import Market, Rule, choose, replay, summary


def _market(rate: float, start: str = "2020-09-01") -> Market:
    hours = pd.date_range(start, "2021-06-30 23:00", freq="1h", tz="UTC")
    flat = pd.Series(100.0, index=hours)
    return Market(flat, flat, pd.Series(rate, index=hours[hours.hour % 8 == 0]))


def _all(_: str, __: pd.Period) -> bool:
    return True


def test_the_rule_holds_the_best_payers_above_its_floor() -> None:
    markets = {"A": _market(3e-4), "B": _market(2e-4), "C": _market(1e-6)}
    month = pd.Period("2021-02", "M")
    assert choose(markets, month, Rule(slots=2), _all) == ["A", "B"]
    assert choose(markets, month, Rule(slots=8), _all) == ["A", "B"]  # C pays under the floor
    assert choose(markets, month, Rule(slots=8, threshold_bp_a_day=0.0), _all) == ["A", "B", "C"]


def test_a_flat_market_earns_exactly_its_funding_less_one_round_trip() -> None:
    markets = {"A": _market(2e-4)}
    months = pd.period_range("2021-01", "2021-02", freq="M")
    frame = replay(markets, months, Rule(slots=1), _all)
    q = 1.0 / 1 / 2 / 100.0
    funding = (31 + 28) * 3 * 2e-4 * q * 100.0
    round_trip = 2 * q * 100.0 * (12 + 7) / 1e4
    assert summary(frame)["total_pct"] == pytest.approx((funding - round_trip) * 100)


def test_the_search_never_reads_a_protected_month_and_reads_its_held_out_months_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not search.readable("BTCUSDT", pd.Period("2024-03", "M"))
    assert search.readable("SOLUSDT", pd.Period("2024-03", "M"))
    assert not search.readable("SOLUSDT", pd.Period("2025-07", "M"))
    assert len(search.GRID) == 16 and search.HELD_OUT[0] > search.DEVELOPMENT[-1]
    (tmp_path / "held_out.json").write_text("{}")
    with pytest.raises(FileExistsError, match="read once"):
        search.run(tmp_path, tmp_path)
