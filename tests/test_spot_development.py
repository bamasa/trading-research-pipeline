"""The D procedure's summaries: sigma_D and the MDE, Null A, the usable days."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from trading_research.spot import access as sa
from trading_research.spot import development as dv
from trading_research.spot import statuses as st


def slot(values: list[float]) -> st.SlotDays:
    n = len(values)
    days = [date(2026, 6, 1 + k) for k in range(n)]
    nan = np.full(n, np.nan)
    return st.SlotDays("S25-BB", days, np.array(values), np.ones(n), nan, nan,
                       np.full((200, n), np.nan), nan, nan, nan, nan, nan)  # fmt: skip


def test_sigma_and_the_mde_over_the_days_with_a_trade() -> None:
    s = slot([1.0, 3.0, np.nan, 5.0])
    found = dv.sigma_and_mde(s)
    assert found["sigma_d"] == 2.0 and found["share_days_with_trade"] == 0.75
    assert math.isclose(found["mde"], 3.84 * 2.0 / math.sqrt(75 * 0.75))
    assert math.isnan(dv.sigma_and_mde(slot([np.nan, 1.0]))["sigma_d"])


def test_null_a_compares_each_draw_with_the_true_mean() -> None:
    s = slot([1.0, 1.0])
    records = {d.isoformat(): {"rows": {"S25-BB": {"null_a": [0.0, 2.0, None]}}} for d in s.days}
    found = dv.null_a(records, s, 3)
    assert found["true_mean"] == 1.0 and found["p"] == (1 + 1) / (1 + 3)


def test_usable_days_on_either_venue() -> None:
    days = [date(2026, 5, 1), date(2026, 5, 2)]
    record = {"days": {"2026-05-01": {sa.BINANCE: {"reasons": []}, sa.BYBIT: {"reasons": ["x"]}},
                       "2026-05-02": {sa.BINANCE: {"reasons": ["y"]}, sa.BYBIT: {"reasons": ["x"]}}}}  # fmt: skip
    assert dv.usable_days(record, days) == [date(2026, 5, 1)]
    assert dv.usable_days(record, days, sa.BYBIT) == []
