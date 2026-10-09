"""The regime map: segments cut at alarms and block changes, every one reported."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.reporting.regime_map import (
    MIN_DAYS_FOR_T,
    SEGMENT_COLUMNS,
    map_alarms,
    regime_map,
    regime_map_chart,
    regime_maps,
)
from trading_research.validation.regime_flags import FoldFlags, walk_forward_months


def _daily(first: date, values: list[float]) -> pd.Series:
    return pd.Series(values, index=[first + timedelta(days=i) for i in range(len(values))])


def _stamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def test_an_alarm_cuts_at_the_first_day_that_begins_after_it() -> None:
    daily = _daily(date(2025, 3, 1), [1.0] * 5 + [-1.0] * 15)
    table = regime_map(daily, [_stamp("2025-03-05 13:00")], slot="S-C")
    assert list(table.columns) == list(SEGMENT_COLUMNS)
    first, second = table.to_dict("records")
    assert (first["start"], first["end"], first["usable_days"]) == (
        date(2025, 3, 1),
        date(2025, 3, 5),
        5,
    )
    assert first["opened_by"] == "start" and pd.isna(first["alarm"])
    assert second["start"] == date(2025, 3, 6)
    assert second["opened_by"] == "alarm"
    assert second["alarm"] == _stamp("2025-03-05 13:00")
    assert second["cumulative_net"] == pytest.approx(-15.0)
    assert np.isnan(first["t"])  # five days carry no t
    assert second["usable_days"] >= MIN_DAYS_FOR_T


def test_an_alarm_at_midnight_puts_that_day_after_it() -> None:
    daily = _daily(date(2025, 3, 1), [1.0, 2.0, 3.0])
    table = regime_map(daily, [_stamp("2025-03-02 00:00")])
    assert table["usable_days"].tolist() == [1, 2]


def test_the_t_is_reported_from_ten_usable_days() -> None:
    rng = np.random.default_rng(0)
    values = list(rng.normal(1.0, 1.0, 12))
    table = regime_map(_daily(date(2025, 3, 1), values), [])
    (row,) = table.to_dict("records")
    expected = np.mean(values) / (np.std(values, ddof=1) / np.sqrt(12))
    assert row["t"] == pytest.approx(expected)
    assert row["mean"] == pytest.approx(np.mean(values))


def test_unusable_days_are_skipped_and_two_alarms_in_a_day_cut_once() -> None:
    daily = _daily(date(2025, 3, 1), [1.0, np.nan, 2.0, 3.0])
    alarms = [_stamp("2025-03-02 05:00"), _stamp("2025-03-02 09:00")]
    table = regime_map(daily, alarms)
    assert table["usable_days"].tolist() == [1, 2]
    assert table.loc[1, "alarm"] == alarms[1]


def test_a_block_change_cuts_and_says_so() -> None:
    daily = _daily(date(2025, 6, 28), [1.0] * 6)
    blocks = {d: ("D" if d <= date(2025, 6, 30) else "H") for d in daily.index}
    table = regime_map(daily, [], blocks=blocks)
    assert table["block"].tolist() == ["D", "H"]
    assert table["opened_by"].tolist() == ["start", "block"]
    assert table["alarm"].isna().all()


def test_every_slot_is_mapped_and_nothing_empty_is_kept() -> None:
    slots = {
        "S25": _daily(date(2025, 3, 1), [1.0] * 4),
        "S-P": _daily(date(2025, 3, 1), [np.nan] * 4),
    }
    table = regime_maps(slots, [_stamp("2025-03-02 12:00")])
    assert table["slot"].tolist() == ["S25", "S25"]
    assert regime_maps({}, []).empty


def test_the_map_reads_each_months_own_alarms_then_the_frozen_ones() -> None:
    march, april = walk_forward_months("2025-03", "2025-04")
    lead_in = pd.DataFrame({"timestamp": [_stamp("2025-02-27 00:00"), _stamp("2025-03-09 00:00")]})
    foldwise = [
        FoldFlags(march, np.ones(24), lead_in),
        FoldFlags(april, np.ones(24), pd.DataFrame({"timestamp": [_stamp("2025-04-02 00:00")]})),
    ]
    frozen = pd.DataFrame({"timestamp": [_stamp("2025-03-10 00:00"), _stamp("2025-08-01 00:00")]})
    alarms = map_alarms(foldwise, frozen, frozen_from="2025-07-01")
    assert list(alarms) == [
        _stamp("2025-03-09 00:00"),
        _stamp("2025-04-02 00:00"),
        _stamp("2025-08-01 00:00"),
    ]


def test_the_chart_is_drawn(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    slots = {"S25": _daily(date(2025, 2, 1), list(np.linspace(-1, 1, 40)))}
    out = regime_map_chart(
        slots,
        [_stamp("2025-02-10 05:00")],
        tmp_path / "regime_map_taker.png",
        blocks={
            "D": (date(2025, 2, 1), date(2025, 2, 28)),
            "H": (date(2025, 3, 1), date(2025, 3, 12)),
        },
    )
    assert out.exists() and out.stat().st_size > 0
