"""Beside R: the second detector on the 1 s grid, with its restarts, and the mean channel."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_research.pipeline.short_horizon import second_detector as sd
from trading_research.validation.changepoint import detect


def _book(day: date, seconds: int, *, spread_after: int | None = None) -> pd.DataFrame:
    """A book updated twice a second for ``seconds``, its spread doubling after
    ``spread_after`` seconds; one row of the next day's first second appended."""
    start = pd.Timestamp(day.isoformat(), tz="UTC")
    rng = np.random.default_rng(day.toordinal())
    n = 2 * seconds
    stamps = pd.to_datetime(start.value + np.arange(n, dtype=np.int64) * 500_000_000, utc=True)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    half = np.where(np.arange(n) // 2 < (spread_after or seconds + 1), 0.005, 0.05)
    half = half * rng.uniform(0.8, 1.2, n)
    frame = pd.DataFrame(
        {"timestamp": stamps, "bid_price_0": mid - half, "ask_price_0": mid + half}
    )
    late = frame.iloc[[-1]].assign(
        timestamp=pd.Timestamp(start.value + 86_400_300_000_000, tz="UTC")
    )
    return pd.concat([frame, late], ignore_index=True)


def test_a_day_is_gridded_at_one_second_on_its_own_hours_labelled_by_the_bins_end() -> None:
    day = date(2025, 8, 20)
    grid = sd.grid_day(_book(day, 3_600), day)
    start = pd.Timestamp(day.isoformat(), tz="UTC").value
    assert len(grid) == 3_600 and grid["timestamp"].iloc[0] == start + 10**9
    assert grid["timestamp"].max() < start + 86_400 * 10**9 + 10**9  # the next day's row is cut
    assert (grid["spread_bp"] > 0).all()


def test_the_stream_runs_the_detector_at_its_defaults_and_labels_the_archive_change() -> None:
    """Two days as one stream; the spread widens tenfold at 08:20 on 2025-08-20,
    after the reference blocks: the alarm, within a day of 2025-08-21, is labelled."""
    first, second = date(2025, 8, 19), date(2025, 8, 20)
    grids = [sd.grid_day(_book(first, 86_400), first),
             sd.grid_day(_book(second, 86_000, spread_after=30_000), second)]  # fmt: skip
    found = sd.run_stream(grids, "H")
    grid = pd.concat(grids, ignore_index=True)
    expected = detect(grid["mid"].to_numpy(), grid["spread_bp"].to_numpy())
    assert found["rows"] == len(grid) and len(found["alarms"]) == len(expected) >= 1
    stamp = pd.Timestamp(found["alarms"][0]["timestamp"])
    assert stamp.value == grid["timestamp"].iloc[expected[0].index - 1]
    assert all(a["label"] == sd.ARCHIVE_CHANGE_LABEL for a in found["alarms"])
    short = sd.run_stream([grid.iloc[:1_000]], "H")
    assert short["alarms"] == [] and "not monitored" in short["note"]
    assert not sd.at_archive_change(pd.Timestamp("2025-08-18T12:00", tz="UTC"))


def test_each_blocks_read_runs_its_own_span_restarting_at_h_and_f() -> None:
    calendar = {"D": (date(2025, 1, 2),), "H": (date(2025, 7, 9),), "F": (date(2025, 8, 22),),
                "P": (date(2025, 11, 18),)}  # fmt: skip
    study = SimpleNamespace(scale=SimpleNamespace(days=lambda b: calendar[b]))
    assert sd.span_days(study, "F-P") == [date(2025, 8, 22), date(2025, 11, 18)]
    assert sd.span_days(study, "H") == [date(2025, 7, 9)]
    assert sd.SPAN_OF_BLOCK == {"H": "H", "F": "F", "P": "F-P"}


def test_the_mean_channel_is_each_windows_first_step_at_its_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_research.validation import regime_flags as rf

    frame = pd.DataFrame({"timestamp": pd.date_range("2025-08-01", periods=4, freq="h", tz="UTC"),
                          "window": [0, 0, 1, 1], "odds_mean": [9.5, 10.0, 1.0, 9.0]})  # fmt: skip
    monkeypatch.setattr(rf, "run_detector", lambda *_, **__: SimpleNamespace(frame=frame))
    found = sd.mean_channel(pd.Series(dtype=float), date(2025, 8, 20), [1.0] * 24)
    assert [(r["window"], r["odds_mean"]) for r in found] == [(0, 9.5), (1, 9.0)]


def test_a_read_runs_the_detector_over_its_own_days_p_after_f(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P's span runs on from F (F's days first); a stand-in for H (the rehearsal) reads
    only its own planned days, never H's, and a stand-in for P on days outside P (the
    reader-test days) reads its own days, never F's."""
    from trading_research.pipeline.short_horizon import flags

    read: list[date] = []
    monkeypatch.setattr(flags, "hourly_close", lambda *_, **__: pd.Series(dtype=float))
    monkeypatch.setattr(sd, "mean_channel", lambda *_: [])
    monkeypatch.setattr(sd, "cached_grid", lambda _, day, __: read.append(day))
    calendar = {"F": (date(2025, 8, 22), date(2025, 8, 23))}
    study = SimpleNamespace(symbol="BTCUSDT", roots=SimpleNamespace(bars=None),
                            scale=SimpleNamespace(days=lambda b: calendar[b]))  # fmt: skip

    def block(name: str, days: tuple[date, ...]) -> SimpleNamespace:
        return SimpleNamespace(name=name, end=None, study=study, access=None, days_planned=days,
                               value=lambda _: [1.0] * 24)  # fmt: skip

    found = sd.beside_r(block("P", (date(2025, 11, 18),)))
    assert read == [date(2025, 8, 22), date(2025, 8, 23), date(2025, 11, 18)]
    assert found["second_detector"]["span"] == "F-P" and found["second_detector"]["days"] == 3
    read.clear()
    sd.beside_r(block("H", (date(2025, 6, 29), date(2025, 6, 30))))
    assert read == [date(2025, 6, 29), date(2025, 6, 30)]
    read.clear()
    found = sd.beside_r(block("P", (date(2026, 1, 5), date(2026, 1, 6))))
    assert read == [date(2026, 1, 5), date(2026, 1, 6)] and found["second_detector"]["days"] == 2
