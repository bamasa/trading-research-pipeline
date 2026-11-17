"""The reversion rule with the kill switch inside it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.plugins import get


def _guarded():  # type: ignore[no-untyped-def]
    """The module, imported only after the registry has discovered the built-ins.

    Importing a built-in module before discovery registers its plugins as
    registered at run time, which the catalogue would then report.
    """
    get("reversion_guarded", "strategy")
    from trading_research.plugins.builtins import guarded

    return guarded


def _frame(mid: np.ndarray) -> pd.DataFrame:
    stamps = pd.date_range("2024-03-01 00:00:05", periods=len(mid), freq="5s", tz="UTC")
    return pd.DataFrame({"timestamp": stamps, "mid": mid, "spread_bp": 1.0})


def test_a_paper_trade_counts_in_the_hour_it_closes_and_pays_its_costs() -> None:
    mid = np.full(2000, 100.0)
    mid[720:] = 101.0  # +100 bp for a long held across 01:00
    decisions = np.zeros(2000, dtype=np.int64)
    decisions[700] = 1
    guarded = _guarded()
    closed, opened = guarded.paper_hourly(decisions, _frame(mid), hold=120, cooldown=0)
    expected = 100.0 - 1.0 - 2 * (guarded.TAKER_BP + guarded.SLIPPAGE_BP)
    assert closed.sum() == pytest.approx(expected) and opened.sum() == pytest.approx(expected)
    assert closed.idxmax() > opened.idxmax()  # opened in the first hour, closed in the second


def test_the_switch_stops_new_entries_once_the_paper_account_keeps_losing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 720 * 200  # two hundred hours
    rng = np.random.default_rng(0)
    steps = np.where(np.arange(rows) < rows // 2, 0.0, -2e-5)  # the edge turns into a loss
    mid = 100.0 * np.exp(np.cumsum(steps + rng.normal(0, 1e-5, rows)))
    frame = _frame(mid)
    always_long = np.zeros(rows, dtype=np.int64)
    always_long[::240] = 1
    guarded = _guarded()
    monkeypatch.setattr(guarded.Reversion, "decide", lambda _self, f: always_long[: len(f)])
    strategy = guarded.ReversionGuarded()
    strategy.live_from = frame["timestamp"].iloc[0]
    out = strategy.decide(frame)
    assert strategy.stopped_at is not None
    stopped = pd.DatetimeIndex(frame["timestamp"]) > strategy.stopped_at
    assert stopped.any() and not out[stopped].any() and out[~stopped].any()
