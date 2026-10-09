"""The break-detector view: guard windows, and the chart in both themes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.reporting.break_view import END_WITHOUT_P, break_view_chart, guard_windows

BLOCKS = {
    "W": (date(2024, 12, 1), date(2024, 12, 31)),
    "D": (date(2025, 1, 1), date(2025, 6, 30)),
    "E": (date(2025, 7, 1), date(2025, 7, 7)),
    "H": (date(2025, 7, 8), date(2025, 8, 20)),
    "F": (date(2025, 8, 21), date(2025, 11, 16)),
    "P": (date(2025, 11, 17), date(2025, 12, 31)),
}


def _t(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def test_guard_windows_follow_each_alarm_merge_and_stop_at_the_end() -> None:
    spans = guard_windows(
        [
            "2025-03-01 10:00",
            "2025-03-01 12:00",
            "2025-04-01 00:00",
            "2025-11-16 22:00",
            "2025-12-01",
        ],
        4,
        END_WITHOUT_P,
    )
    assert spans == [
        (_t("2025-03-01 10:00"), _t("2025-03-01 16:00")),
        (_t("2025-04-01 00:00"), _t("2025-04-01 04:00")),
        (_t("2025-11-16 22:00"), END_WITHOUT_P),
    ]
    assert guard_windows([], 24, END_WITHOUT_P) == []


@pytest.mark.parametrize("p_opened", [False, True])
def test_the_view_is_drawn_to_f_or_to_p(tmp_path: Path, p_opened: bool) -> None:
    pytest.importorskip("matplotlib")
    from trading_research.reporting.plots import both_themes

    hours = pd.date_range("2024-12-01", "2025-12-31 23:00", freq="h", tz="UTC")
    close = pd.Series(
        1e5 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.003, len(hours)))), index=hours
    )
    alarms = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-03-10 05:00", "2025-09-02 08:00", "2025-12-05 00:00"], utc=True
            ),
            "statistic": ["dependence", "scale", "scale"],
        }
    )
    written = both_themes(
        lambda p: break_view_chart(
            close, alarms, p, blocks=BLOCKS, p_opened=p_opened, b="2025-09-02 08:00",
            guards={"S-C": 24}, cusum_restarts=["2025-07-08", "2025-08-21"],
        ),
        tmp_path / "break.png",
    )  # fmt: skip
    assert all(p.exists() and p.stat().st_size > 5_000 for p in written)
    empty = break_view_chart(
        close, alarms.iloc[:0], tmp_path / "none.png", blocks=BLOCKS, p_opened=False
    )
    assert empty.exists()
