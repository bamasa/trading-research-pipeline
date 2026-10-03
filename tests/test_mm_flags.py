"""Regime flags: when each detector's flag is known, causality, and the placebo.

A synthetic mid with a volatility break is run through both detectors on
right-labelled grids. Each flag must be effective at the label of the row that
closed the evidence for it, plus the decision latency; the flags found on a
prefix of the data must be exactly the flags of the whole that fall inside the
prefix; and the shifted placebo must keep the count and the spacing.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("structural_break")

from trading_research.market_making import flags, prereg
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.market_making.quoters import FLAG_SIGNAL
from trading_research.validation import changepoint, structural_breaks

FIRST = date(2024, 2, 1)
SECONDS = int(2.5 * 86_400)


def grids(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A one-second and a one-minute right-labelled grid with a volatility jump."""
    rng = np.random.default_rng(seed)
    scale = np.where(np.arange(SECONDS) < 1.4 * 86_400, 1.0, 3.0)
    log_mid = np.cumsum(rng.normal(0.0, 2e-5, SECONDS) * scale)
    mid = 1.0 * np.exp(log_mid)
    spread = 0.0002 * (1 + (np.arange(SECONDS) > 1.6 * 86_400))
    labels = pd.Timestamp(FIRST, tz="UTC") + pd.to_timedelta(np.arange(1, SECONDS + 1), unit="s")
    one_second = pd.DataFrame(
        {"timestamp": labels, "bid_price_0": mid - spread / 2, "ask_price_0": mid + spread / 2}
    )
    one_minute = one_second.iloc[59::60].reset_index(drop=True)
    return one_second, one_minute


SPAN = (day_start_ns(FIRST), day_start_ns(FIRST) + 3 * 86_400 * NS_PER_S)


@pytest.fixture(scope="module")
def timeline() -> flags.FlagTimeline:
    one_second, one_minute = grids()
    return flags.flags_from_grids("SYNTHUSDT", one_second, one_minute, span=SPAN)


def test_both_detectors_flag_the_break(timeline: flags.FlagTimeline) -> None:
    detectors = set(timeline.detector.tolist())
    assert detectors == {"changepoint", "structural_breaks"}
    assert np.all(np.diff(timeline.effective_ns) >= 0)


def test_flags_are_effective_only_after_the_block_that_triggered_them_closes(
    timeline: flags.FlagTimeline,
) -> None:
    one_second, one_minute = grids()
    labels = pd.DatetimeIndex(one_second["timestamp"]).as_unit("ns").asi8
    mid = ((one_second["bid_price_0"] + one_second["ask_price_0"]) / 2).to_numpy()
    spread = ((one_second["ask_price_0"] - one_second["bid_price_0"]) / mid * 1e4).to_numpy()
    expected = sorted(
        int(labels[b.index - 1]) + flags.DECISION_LATENCY_NS
        for b in changepoint.detect(mid, spread)
    )
    mine = timeline.effective_ns[timeline.detector == "changepoint"]
    assert sorted(mine.tolist()) == expected
    # A block of 2000 rows that closed at label L is known at L, never before:
    # each flag sits exactly on a block boundary, plus the latency.
    rows = (mine - flags.DECISION_LATENCY_NS - labels[0]) // NS_PER_S + 1
    assert np.all(rows % 2000 == 0)

    close = pd.Series(
        ((one_minute["bid_price_0"] + one_minute["ask_price_0"]) / 2).to_numpy(),
        index=pd.DatetimeIndex(one_minute["timestamp"]),
    )
    returns = structural_breaks.log_returns(close)
    spec = structural_breaks.MonitorSpec(
        history_len=250,
        online_len=60,
        statistics=("scale", "dependence"),
        threshold={"scale": 7.77, "dependence": 7.65},
    )
    found = structural_breaks.detect_breaks(returns, spec, on_progress=None).breaks
    stamps = pd.DatetimeIndex(returns.index).as_unit("ns").asi8
    expected_sb = sorted(int(stamps[b.index]) + flags.DECISION_LATENCY_NS for b in found)
    assert (
        sorted(timeline.effective_ns[timeline.detector == "structural_breaks"].tolist())
        == expected_sb
    )
    assert len(expected_sb) > 0


def test_flags_found_on_a_prefix_are_the_flags_of_the_whole(
    timeline: flags.FlagTimeline,
) -> None:
    one_second, one_minute = grids()
    cut = int(1.9 * 86_400)
    cut_label = one_second["timestamp"].iloc[cut - 1]
    prefix = flags.flags_from_grids(
        "SYNTHUSDT",
        one_second.iloc[:cut],
        one_minute[one_minute["timestamp"] <= cut_label],
        span=SPAN,
    )
    limit = pd.Timestamp(cut_label).as_unit("ns").value + flags.DECISION_LATENCY_NS
    inside = timeline.effective_ns <= limit
    assert inside.sum() > 0 and (~inside).sum() > 0
    np.testing.assert_array_equal(prefix.effective_ns, timeline.effective_ns[inside])
    np.testing.assert_array_equal(prefix.statistic, timeline.statistic[inside])


def test_shifted_flags_preserve_count_and_spacing(timeline: flags.FlagTimeline) -> None:
    start, end = timeline.span
    length = end - start

    def circular_gaps(stamps: np.ndarray) -> list[int]:
        ordered = np.sort(stamps)
        return sorted([*np.diff(ordered).tolist(), int(ordered[0] + length - ordered[-1])])

    for seed in range(5):
        offset = timeline.placebo_offset(seed)
        assert 6 * 3600 * NS_PER_S <= offset <= length - 6 * 3600 * NS_PER_S
        moved = timeline.shifted(offset)
        assert len(moved) == len(timeline)
        assert ((moved.effective_ns >= start) & (moved.effective_ns < end)).all()
        assert circular_gaps(moved.effective_ns) == circular_gaps(timeline.effective_ns)
        assert sorted(moved.detector.tolist()) == sorted(timeline.detector.tolist())
    assert timeline.placebo_offset(3) == timeline.placebo_offset(3)


def test_the_guard_tape_and_the_share_of_time_guarded() -> None:
    start = day_start_ns(FIRST)
    stamps = start + np.array([3600, 3600, 3900, 20_000]) * NS_PER_S
    timeline = flags.FlagTimeline(
        "X",
        stamps,
        np.array(["a"] * 4, dtype=object),
        np.array(["s"] * 4, dtype=object),
        (start, start + 86_400 * NS_PER_S),
    )
    tape = timeline.tape()
    assert tape.name == FLAG_SIGNAL
    assert len(tape.ts) == 3  # two flags at one instant are one entry
    np.testing.assert_allclose(tape.values, tape.ts / NS_PER_S)
    # Windows of 10 minutes: [3600, 4200) and [3900, 4500) merge, plus [20000, 20600).
    assert flags.guard_share(timeline, 10) == pytest.approx((900 + 600) / 86_400)
    assert timeline.rate_per_day() == pytest.approx(4.0)
    assert flags.flag_counts(timeline) == {"a:s": 4}


def test_reading_the_book_for_a_held_out_day_is_refused_unopened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("a file was opened")

    monkeypatch.setattr(pd, "read_parquet", never)
    days = [FIRST + timedelta(days=i) for i in range(25)] + [date(2024, 2, 26)]
    with pytest.raises(prereg.HeldOutLocked):
        flags.build_flags(
            "BICOUSDT",
            days,
            book_roots=[tmp_path],
            access=prereg.development_access(),  # type: ignore[list-item]
        )
