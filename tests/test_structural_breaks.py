"""Does the whitened detector find a break that is there, stay quiet when none is, and never look ahead?

Fixtures are synthetic return series with a planted scale or dependence break,
built here as ``test_changepoint.py`` builds its random walks. The library
behind the detector is an optional extra, so a checkout without it skips this
module rather than failing; CI installs every extra and runs it.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("structural_break")

from structural_break import WHITE_ODDS_CHANNELS, channel_names

from trading_research.validation.changepoint import ChangepointError
from trading_research.validation.structural_breaks import (
    CHANNELS,
    KNOWN_STATISTICS,
    RECORDED_THRESHOLDS,
    MonitorSpec,
    calibrate,
    detect_breaks,
    family_odds,
    log_returns,
    read_prices,
    segment_returns,
    whiten_and_stream,
)

#: Thresholds for the small settings the tests walk, so that no test calibrates.
#: Deliberately a little below the recorded daily ones: the fixtures are short
#: and the point of each test is the attribution, not the alarm rate.
LEVELS = {"scale": 8.0, "dependence": 7.0, "mean": 8.0}


def _ar1(rng: np.random.Generator, phi: float, n: int, sd: float = 1.0) -> np.ndarray:
    x = np.zeros(n)
    e = rng.normal(0.0, sd, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _spec(history: int = 300, online: int = 100, **kwargs: object) -> MonitorSpec:
    return MonitorSpec(history_len=history, online_len=online, threshold=LEVELS, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The column map
# ---------------------------------------------------------------------------


def test_column_map_matches_the_installed_width() -> None:
    """Every channel read exists upstream, and the frame carries exactly the named ones."""
    names = channel_names(odds=True)
    assert len(names) == WHITE_ODDS_CHANNELS
    assert set(CHANNELS.values()) <= set(names)

    rng = np.random.default_rng(0)
    frame = whiten_and_stream(rng.standard_normal(365), rng.standard_normal(40))
    assert list(frame.columns) == ["step", "value", *CHANNELS]
    assert len(frame) == 40
    assert list(frame["step"]) == list(range(40))
    assert np.isfinite(frame.to_numpy()).all()


def test_a_scale_break_lands_in_the_scale_column() -> None:
    """The behavioural pin for the index map: a position alone pins nothing."""
    rng = np.random.default_rng(1)
    history = rng.standard_normal(1000)
    quiet = rng.standard_normal(300)
    loud = rng.standard_normal(300) * 2.0
    odds = family_odds(whiten_and_stream(history, np.concatenate([quiet, loud])))

    before, after = odds.iloc[200:300], odds.iloc[500:600]
    assert after["odds_scale"].mean() > before["odds_scale"].mean() + 5.0
    assert after["direction_scale"].iloc[-1] == +1
    assert after["odds_dependence"].mean() < before["odds_dependence"].mean() + 2.0


def test_a_dependence_break_at_constant_variance_is_attributed_to_dependence() -> None:
    """AR(1) φ 0 → 0.6 with the innovation sd matched so the marginal variance is unchanged."""
    rng = np.random.default_rng(2)
    history = rng.standard_normal(1000)
    quiet = rng.standard_normal(300)
    shifted = _ar1(rng, 0.6, 300, sd=np.sqrt(1 - 0.6**2))
    odds = family_odds(whiten_and_stream(history, np.concatenate([quiet, shifted])))

    before, after = odds.iloc[200:300], odds.iloc[500:600]
    assert after["odds_dependence"].mean() > before["odds_dependence"].mean() + 5.0
    assert after["direction_dependence"].iloc[-1] == +1
    assert after["odds_scale"].mean() < before["odds_scale"].mean() + 3.0


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_quiet_series_produces_no_breaks(seed: int) -> None:
    """A series whose behaviour never changes should not be cut.

    The recorded thresholds are 99th percentiles per window and family, so the
    procedure fires on about one quiet window in fifty; five seeds of three
    windows passing is the expected behaviour rather than a guarantee.
    """
    rng = np.random.default_rng(seed)
    result = detect_breaks(rng.standard_normal(365 + 270), MonitorSpec(365, 90), on_progress=None)
    assert result.breaks == []
    assert result.thresholds == {
        k: v for k, v in RECORDED_THRESHOLDS[(365, 90)].items() if k != "mean"
    }


def test_break_is_found_after_it_happens_never_before() -> None:
    rng = np.random.default_rng(3)
    planted = 500
    series = np.concatenate([rng.standard_normal(planted), rng.standard_normal(300) * 2.0])
    result = detect_breaks(series, _spec(), on_progress=None)

    assert result.breaks, "a doubling of the scale should be detected"
    first = result.breaks[0]
    assert first.statistic == "scale"
    assert first.direction == +1
    assert first.index >= planted
    assert (result.frame["flag"] != "").sum() == len(result.breaks)


def test_truncating_the_series_does_not_change_earlier_breaks() -> None:
    """The causality property: results must not shift as data arrives."""
    rng = np.random.default_rng(4)
    series = np.concatenate([rng.standard_normal(500), rng.standard_normal(400) * 2.5])
    full = detect_breaks(series, _spec(), on_progress=None)
    cut = 700
    truncated = detect_breaks(series[:cut], _spec(), on_progress=None)
    assert full.breaks, "the fixture should contain a break"
    assert [b.index for b in full.breaks if b.index < cut] == [b.index for b in truncated.breaks]


def test_no_history_straddles_a_flagged_break() -> None:
    """After a break the next history starts at the break, and grows from min_history."""
    rng = np.random.default_rng(5)
    series = np.concatenate([rng.standard_normal(500), rng.standard_normal(500) * 2.5])
    spec = _spec(history=300, online=100, min_history=100)
    result = detect_breaks(series, spec, on_progress=None)

    assert result.breaks
    at = result.breaks[0].index
    later = [w for w in result.windows if w[1] > at]
    assert later, "monitoring should resume after the break"
    assert all(history_start >= at for history_start, _, _ in later)
    assert later[0] == (at, at + 100, min(at + 200, len(series)))
    # The frame says how short the history was, rather than pretending otherwise.
    resumed = result.frame[result.frame["index"] >= at + 100]
    assert int(resumed["history_len"].iloc[0]) == 100


def test_windows_roll_every_online_len_without_a_break() -> None:
    rng = np.random.default_rng(6)
    spec = MonitorSpec(history_len=300, online_len=100, threshold=1e9)
    result = detect_breaks(rng.standard_normal(650), spec, on_progress=None)
    assert result.windows == [(0, 300, 400), (100, 400, 500), (200, 500, 600), (300, 600, 650)]
    assert list(result.frame["index"]) == list(range(300, 650))
    assert (result.frame["history_len"] == 300).all()


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def test_segments_cover_the_series_exactly_once() -> None:
    rng = np.random.default_rng(7)
    series = np.concatenate([rng.standard_normal(500), rng.standard_normal(400) * 2.5])
    cut = segment_returns(series, _spec(), minimum_rows=50, on_progress=None)
    assert cut.breaks, "the fixture should contain a break"
    assert cut.bounds[0][0] == 0
    assert cut.bounds[-1][1] == len(series)
    for (_, end), (start, _) in pairwise(cut.bounds):
        assert end == start
    assert sum(cut.lengths()) == len(series)
    assert [b.index for b in cut.breaks] == [start for start, _ in cut.bounds[1:]]


def test_short_segments_are_merged() -> None:
    """Nothing survives a minimum longer than the series, so it stays whole."""
    rng = np.random.default_rng(8)
    series = np.concatenate([rng.standard_normal(500), rng.standard_normal(400) * 2.5])
    cut = segment_returns(series, _spec(), minimum_rows=10_000, on_progress=None)
    assert len(cut) == 1
    assert cut.breaks == []
    assert cut.bounds == [(0, len(series))]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_calibrated_threshold_grows_with_quantile_and_online_length() -> None:
    """The running maximum of the odds can only grow with the window, and a
    higher quantile of it is a higher threshold."""
    short = calibrate(200, 40, null_paths=20, statistics=("scale",), seed=0)
    long = calibrate(200, 120, null_paths=20, statistics=("scale",), seed=0)
    median = calibrate(200, 40, null_paths=20, statistics=("scale",), quantile=0.5, seed=0)
    assert long["scale"] > short["scale"]
    assert short["scale"] > median["scale"]
    assert set(calibrate(200, 20, null_paths=2)) == set(KNOWN_STATISTICS)


def test_threshold_mapping_is_respected_and_unknown_statistic_is_named() -> None:
    rng = np.random.default_rng(9)
    series = rng.standard_normal(400)

    with pytest.raises(ChangepointError, match="momentum"):
        MonitorSpec(history_len=300, statistics=("momentum",))
    with pytest.raises(ChangepointError, match="at least"):
        detect_breaks(series[:200], _spec(), on_progress=None)
    with pytest.raises(ChangepointError, match="NaN"):
        detect_breaks(np.concatenate([series, [np.nan]]), _spec(), on_progress=None)
    with pytest.raises(ChangepointError, match="no threshold"):
        detect_breaks(
            series, MonitorSpec(history_len=300, threshold={"scale": 8.0}), on_progress=None
        )

    # A single number applies to every family; a mapping applies per family.
    one = detect_breaks(series, MonitorSpec(300, 100, threshold=7.5), on_progress=None)
    assert one.thresholds == {"scale": 7.5, "dependence": 7.5}
    per = detect_breaks(
        series, MonitorSpec(300, 100, threshold={"scale": 9.0, "dependence": 6.0}), on_progress=None
    )
    assert per.thresholds == {"scale": 9.0, "dependence": 6.0}


# ---------------------------------------------------------------------------
# Prices in, figure and CLI out
# ---------------------------------------------------------------------------


def _prices_csv(path: Path, n: int = 700, seed: int = 10) -> Path:
    """Daily closes whose volatility triples about two thirds of the way through."""
    rng = np.random.default_rng(seed)
    cut = int(n * 0.65)
    returns = np.concatenate([rng.normal(0, 0.01, cut), rng.normal(0, 0.03, n - cut)])
    close = 100.0 * np.exp(np.cumsum(returns))
    stamps = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    pd.DataFrame({"timestamp": stamps.strftime("%Y-%m-%d"), "close": close}).to_csv(
        path, index=False
    )
    return path


def test_prices_csv_is_read_and_turned_into_labelled_returns(tmp_path: Path) -> None:
    close = read_prices(_prices_csv(tmp_path / "prices.csv", n=10))
    assert close.index.tz is not None
    returns = log_returns(close)
    assert len(returns) == 9
    assert returns.index[0] == close.index[1]

    (tmp_path / "bad.csv").write_text("date,price\n2024-01-01,1\n", encoding="utf-8")
    with pytest.raises(ChangepointError, match="timestamp"):
        read_prices(tmp_path / "bad.csv")


def test_cli_breaks_runs_on_a_csv(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from trading_research.cli import app

    out = tmp_path / "breaks"
    result = CliRunner().invoke(
        app,
        [
            "breaks",
            "--prices",
            str(_prices_csv(tmp_path / "prices.csv")),
            "--history",
            "300",
            "--online",
            "100",
            "--threshold",
            "scale=8,dependence=7",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Breaks flagged" in result.output
    assert "scale" in result.output
    breaks = pd.read_csv(out / "breaks.csv")
    # The volatility triples at return 454 (index 455 of the closes). The flag
    # must land within a few weeks of it; which family clears first is not
    # pinned, because the dependence alternative also rewards the large
    # consecutive products a scale increase produces (see the module docstring).
    first = int(breaks["index"].iloc[0])
    assert 454 <= first <= 485, breaks
    assert set(breaks["statistic"]) <= {"scale", "dependence"}
    assert (out / "steps.csv").exists()
    assert (out / "thresholds.json").exists()
    assert (out / "manifest.json").exists()


def test_plot_renders_both_themes(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from trading_research.reporting import plots

    close = read_prices(_prices_csv(tmp_path / "prices.csv"))
    returns = log_returns(close)
    result = detect_breaks(returns, _spec(), on_progress=None)
    written = plots.both_themes(
        lambda path: plots.structural_breaks(
            result.annotate(returns), path, thresholds=result.thresholds, history_len=300
        ),
        tmp_path / "breaks.png",
    )
    assert [p.name for p in written] == ["breaks_light.png", "breaks_dark.png"]
    assert all(p.stat().st_size > 1_000 for p in written)
