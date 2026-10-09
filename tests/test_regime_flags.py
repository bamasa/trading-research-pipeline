"""The guard's flags: an hourly stream, an hour profile fitted before each fold, no look-ahead.

The detector runs on a small setting here (60-hour history, 24-hour windows,
fixed thresholds) so that a fold of one day can be checked hour by hour in a
few seconds; the registered setting differs only in its lengths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.validation import regime_flags as rf
from trading_research.validation.structural_breaks import MonitorSpec

UTC = "UTC"
SMALL = MonitorSpec(
    history_len=60,
    online_len=24,
    min_history=50,
    statistics=("scale", "dependence"),
    threshold={"scale": 8.0, "dependence": 7.0},
)
FOLD = rf.Fold(pd.Timestamp("2024-12-08", tz=UTC), pd.Timestamp("2024-12-09", tz=UTC))


def _stamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=UTC)


def _closes(*, break_at: str | None = "2024-12-08 06:00", seed: int = 3) -> pd.Series:
    """Hourly closes from the stream's start with a daily volatility cycle.

    The afternoon hours are 1.8 times as rough as the morning ones; with
    ``break_at`` every return after it is five times as rough again.
    """
    rng = np.random.default_rng(seed)
    labels = pd.date_range("2024-12-01 01:00", "2024-12-10 00:00", freq="h", tz=UTC)
    hours = rf.hours_of_day(labels)
    sd = 0.002 * np.where(hours >= 13, 1.8, 1.0)
    if break_at is not None:
        sd = np.where(labels > _stamp(break_at), 5.0 * sd, sd)
    steps = rng.normal(0.0, 1.0, len(labels)) * sd
    steps[0] = 0.0
    return pd.Series(50_000.0 * np.exp(np.cumsum(steps)), index=labels, name="close")


# ---------------------------------------------------------------------------
# The hourly stream
# ---------------------------------------------------------------------------


def _bars(labels: pd.DatetimeIndex) -> pd.DataFrame:
    """Bars stamped as the archive stamps them: one millisecond before the hour."""
    return pd.DataFrame(
        {
            "timestamp": labels - pd.Timedelta(1, unit="ms"),
            "close": np.linspace(100.0, 110.0, len(labels)),
        }
    )


def test_bar_closes_land_on_the_hour_boundary() -> None:
    labels = pd.date_range("2024-12-01 01:00", periods=6, freq="h", tz=UTC)
    stream = rf.hourly_stream(_bars(labels), end="2024-12-01 06:00")
    assert list(stream.close.index) == list(labels)
    assert stream.missing.empty


def test_a_missing_hour_is_carried_forward_and_counted() -> None:
    labels = pd.date_range("2024-12-01 01:00", periods=6, freq="h", tz=UTC)
    bars = _bars(labels).drop(index=3)
    stream = rf.hourly_stream(bars, end="2024-12-01 06:00")
    assert list(stream.missing) == [labels[3]]
    assert stream.close.iloc[3] == stream.close.iloc[2]
    returns = np.diff(np.log(stream.close.to_numpy()))
    assert returns[2] == 0.0
    assert returns[3] > 0.0  # the move over the gap enters the next return


def test_the_stream_refuses_what_it_cannot_carry() -> None:
    labels = pd.date_range("2024-12-01 01:00", periods=6, freq="h", tz=UTC)
    with pytest.raises(rf.FlagError, match="first bar"):
        rf.hourly_stream(_bars(labels).drop(index=0), end="2024-12-01 06:00")
    doubled = pd.concat([_bars(labels), _bars(labels[:1])], ignore_index=True)
    with pytest.raises(rf.FlagError, match="same hour"):
        rf.hourly_stream(doubled, end="2024-12-01 06:00")
    with pytest.raises(rf.FlagError, match="'close'"):
        rf.hourly_stream(_bars(labels).drop(columns="close"), end="2024-12-01 06:00")


def test_a_return_belongs_to_the_hour_its_bar_spans() -> None:
    labels = pd.DatetimeIndex([_stamp("2024-12-01 01:00"), _stamp("2024-12-02 00:00")])
    assert rf.hours_of_day(labels).tolist() == [0, 23]
    early = pd.DatetimeIndex([_stamp("2024-12-01 00:59:59.999")])
    assert rf.hours_of_day(early).tolist() == [0]


# ---------------------------------------------------------------------------
# The hour profile
# ---------------------------------------------------------------------------


def test_the_profile_recovers_the_daily_cycle() -> None:
    rng = np.random.default_rng(0)
    labels = pd.date_range("2024-12-01 01:00", periods=24 * 400, freq="h", tz=UTC)
    scale = 0.001 * (1.0 + rf.hours_of_day(labels) / 10.0)
    returns = pd.Series(rng.normal(0.0, 1.0, len(labels)) * scale, index=labels)
    profile = rf.fit_hour_profile(returns, "2024-12-01", labels[-1])
    expected = 0.001 * (1.0 + np.arange(24) / 10.0)
    np.testing.assert_allclose(profile, expected, rtol=0.12)
    flat = rf.normalise_by_hour(returns, profile)
    by_hour = flat.groupby(rf.hours_of_day(labels)).std()
    np.testing.assert_allclose(by_hour.to_numpy(), 1.0, rtol=0.15)


def test_the_profile_reads_no_bar_at_or_after_its_end() -> None:
    returns = np.log(_closes()).diff().dropna()
    end = FOLD.start
    profile = rf.fit_hour_profile(returns, rf.STREAM_START, end)
    changed = returns.copy()
    changed[changed.index > end] *= 100.0
    np.testing.assert_array_equal(rf.fit_hour_profile(changed, rf.STREAM_START, end), profile)
    changed[changed.index == end] *= 100.0  # the last bar before the fold is inside
    assert not np.array_equal(rf.fit_hour_profile(changed, rf.STREAM_START, end), profile)


def test_an_empty_hour_is_refused() -> None:
    returns = np.log(_closes()).diff().dropna()
    with pytest.raises(rf.FlagError, match="UTC hour"):
        rf.fit_hour_profile(returns, "2024-12-01", "2024-12-01 12:00")
    with pytest.raises(rf.FlagError, match="24 positive"):
        rf.normalise_by_hour(returns, np.ones(23))


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def test_the_walk_forward_months_are_february_to_june() -> None:
    folds = rf.walk_forward_months()
    assert [f.name for f in folds] == ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    february = folds[0]
    assert february.end == _stamp("2025-03-01")
    assert len(february.hours) == 28 * 24
    assert february.hours[0] == february.start
    assert february.guard_from == _stamp("2025-01-27")


def test_a_fold_refuses_bounds_off_the_hour() -> None:
    with pytest.raises(rf.FlagError, match="hour boundaries"):
        rf.Fold(_stamp("2025-02-01 00:30"), _stamp("2025-03-01"))
    with pytest.raises(rf.FlagError, match="end after"):
        rf.Fold(_stamp("2025-03-01"), _stamp("2025-02-01"))


def _alarms(*stamps: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(stamps), utc=True),
            "statistic": "scale",
            "direction": 1,
            "excursion": 1.2,
            "odds_mean": 0.0,
        }
    )


def test_the_guard_reads_the_lead_in_and_the_map_reads_the_month() -> None:
    fold = rf.walk_forward_months("2025-03", "2025-03")[0]
    alarms = _alarms(
        "2025-02-20 00:00",  # before the lead-in: not used
        "2025-02-25 00:00",  # the lead-in's first instant: guards March's first hours
        "2025-03-10 05:00",
        "2025-04-01 00:00",  # the close of March's last hour
    )
    flags = rf.FoldFlags(fold, np.ones(24), alarms)
    assert list(flags.guard_alarms["timestamp"]) == list(alarms["timestamp"][1:])
    assert list(flags.month_alarms["timestamp"]) == [_stamp("2025-03-10 05:00")]
    assert flags.flags_ns().dtype == np.int64
    assert flags.flags_ns()[0] == _stamp("2025-02-25").value


# ---------------------------------------------------------------------------
# The detector on the stream
# ---------------------------------------------------------------------------

pytest.importorskip("structural_break")


def test_a_planted_break_in_the_fold_is_flagged_after_it() -> None:
    flags = rf.fold_flags(_closes(), FOLD, spec=SMALL)
    assert len(flags.month_alarms) >= 1
    first = flags.month_alarms.iloc[0]
    assert first["timestamp"] > _stamp("2024-12-08 06:00")
    assert first["statistic"] in ("scale", "dependence")
    assert np.isfinite(first["odds_mean"])
    quiet = rf.fold_flags(_closes(break_at=None), FOLD, spec=SMALL)
    assert quiet.month_alarms.empty


def test_the_fold_profile_is_fitted_before_the_fold() -> None:
    closes = _closes()
    returns = np.log(closes).diff().dropna()
    flags = rf.fold_flags(closes, FOLD, spec=SMALL)
    expected = rf.fit_hour_profile(returns, rf.STREAM_START, FOLD.start)
    np.testing.assert_array_equal(flags.profile, expected)


def test_a_longer_run_reproduces_a_shorter_one() -> None:
    closes = _closes()
    profile = rf.fit_hour_profile(np.log(closes).diff().dropna(), rf.STREAM_START, FOLD.start)
    short = rf.alarm_table(rf.run_detector(closes, profile, end="2024-12-09", spec=SMALL))
    long = rf.alarm_table(rf.run_detector(closes, profile, end="2024-12-10", spec=SMALL))
    assert rf.unreproduced_alarms(short, long, through="2024-12-09").empty


def test_frozen_flags_use_the_recorded_profile_when_given() -> None:
    closes = _closes()
    span = (_stamp("2024-12-01"), _stamp("2024-12-08"))
    fitted, alarms = rf.frozen_flags(closes, end="2024-12-10", profile_span=span, spec=SMALL)
    again, same = rf.frozen_flags(closes, end="2024-12-10", profile=fitted, spec=SMALL)
    np.testing.assert_array_equal(fitted, again)
    pd.testing.assert_frame_equal(alarms, same)


# ---------------------------------------------------------------------------
# The flag-causality test
# ---------------------------------------------------------------------------


def test_the_fold_wise_flags_pass_the_causality_test_hour_by_hour() -> None:
    result = rf.check_flag_causality(_closes(), FOLD, run=rf.FoldRunner(spec=SMALL))
    assert not result.leaks
    assert result.hours_checked == 24
    assert "causal over 24 hours" in str(result)


class _PeekingRun:
    """A flag at every hour whose *next* return is large: the look-ahead to catch."""

    def __call__(self, close: pd.Series, fold: rf.Fold) -> pd.DataFrame:
        returns = np.log(close).diff()
        ahead = returns.shift(-1).abs()
        stamps = ahead.index[(ahead > 4.0 * returns.abs().median()).to_numpy()]
        return _alarms(*[str(s) for s in stamps])


def test_a_flag_that_reads_the_next_bar_fails() -> None:
    result = rf.check_flag_causality(_closes(), FOLD, run=_PeekingRun())
    assert result.leaks
    assert "LEAK" in str(result)
    first = result.failures[0]
    assert first.original != first.mutated
    with pytest.raises(rf.FlagLeakError, match="read later bars"):
        rf.assert_flags_causal(_closes(), [FOLD], run=_PeekingRun())


def test_a_profile_fitted_through_the_fold_is_caught() -> None:
    """The quieter leak: the hour profile fitted on the fold's own hours."""

    def through_the_fold(close: pd.Series, fold: rf.Fold) -> pd.DataFrame:
        returns = np.log(close[close.index <= fold.end]).diff().dropna()
        profile = rf.fit_hour_profile(returns, rf.STREAM_START, fold.end)
        result = rf.run_detector(close, profile, end=fold.end, spec=SMALL)
        return rf.alarm_table(result)

    result = rf.check_flag_causality(_closes(), FOLD, run=through_the_fold)
    assert result.leaks
    assert all(not f.original for f in result.failures)  # flags appear that were not there


def test_the_mutation_changes_only_what_comes_after() -> None:
    closes = _closes()
    tau = _stamp("2024-12-08 05:00")
    mutated = rf.mutate_after(closes, tau, np.random.default_rng(0))
    pd.testing.assert_series_equal(mutated[mutated.index <= tau], closes[closes.index <= tau])
    assert (mutated[mutated.index > tau] != closes[closes.index > tau]).all()
    assert (mutated > 0).all()


def test_workers_give_the_same_answer() -> None:
    hours = list(FOLD.hours[::6])
    one = rf.check_flag_causality(_closes(), FOLD, run=_PeekingRun(), hours=hours)
    two = rf.check_flag_causality(_closes(), FOLD, run=_PeekingRun(), hours=hours, workers=2)
    assert [f.tau for f in one.failures] == [f.tau for f in two.failures]
    assert one.hours_checked == two.hours_checked == 4


# ---------------------------------------------------------------------------
# What is recorded beside the flags
# ---------------------------------------------------------------------------


def test_the_alarm_rate_counts_alarms_in_the_span_per_day() -> None:
    alarms = _alarms("2025-01-03 00:00", "2025-01-20 10:00", "2025-02-01 00:00")
    assert rf.alarm_rate(alarms, "2025-01-01", "2025-02-01") == pytest.approx(2 / 31)
    with pytest.raises(rf.FlagError, match="span"):
        rf.alarm_rate(alarms, "2025-02-01", "2025-01-01")


def test_a_missing_or_extra_alarm_is_not_a_reproduction() -> None:
    earlier = _alarms("2025-01-03 00:00", "2025-01-20 10:00")
    later = _alarms("2025-01-03 00:00", "2025-01-25 10:00", "2025-03-01 00:00")
    diff = rf.unreproduced_alarms(earlier, later, through="2025-02-01")
    assert diff["run"].tolist() == ["earlier only", "later only"]
    assert rf.unreproduced_alarms(earlier, earlier, through="2025-02-01").empty


def test_the_two_flag_lists_are_compared_month_by_month() -> None:
    fold = rf.walk_forward_months("2025-03", "2025-03")[0]
    mine = rf.FoldFlags(fold, np.ones(24), _alarms("2025-03-02 00:00", "2025-03-09 00:00"))
    frozen = _alarms("2025-03-02 00:00", "2025-03-20 00:00", "2025-04-02 00:00")
    table = rf.foldwise_against_frozen([mine], frozen)
    assert table.to_dict("records") == [
        {"month": "2025-03", "both": 1, "foldwise_only": 1, "frozen_only": 1}
    ]
