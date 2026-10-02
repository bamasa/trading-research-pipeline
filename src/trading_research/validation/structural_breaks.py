"""Breaks in a return series, read on the stream whitened by its own history.

The second regime detector in this project, and the reason there are two.
``changepoint.py`` reads order-book statistics in non-overlapping blocks at tick
frequency and needs a mid and a spread; it says *when the book changed*. This
one reads a return series at any frequency — daily bars, hourly bars, the
five-second grid — and says *which thing about the returns changed*: their
scale, their dependence on their own past, or their mean. The distinction
matters because §27's market state turned on exactly that difference: the
reversion paid where the index's returns were negatively autocorrelated and
not where they were merely volatile, and a detector that could not tell the
two apart would have flagged the wrong days.

The detector
------------
From `adia-structural-break <https://github.com/bamasa/adia-structural-break>`_,
the author's solution to the ADIA Lab Structural Break Challenge, Real-Time
Edition (CrunchDAO, 2026), installed as the ``breaks`` extra rather than
copied. ``WhiteMonitor`` fits the history once — an AR(p ≤ 12) by BIC, a
conditional scale with its memory chosen by quasi-likelihood, and the
empirical distribution of the standardised innovations — and maps every online
point through that fit to a normal score. Under the null the scores are i.i.d.
N(0, 1) whatever the history looked like, so every test downstream runs on the
same footing: CUSUMs, generalised likelihood ratios over dyadic windows, and
the Shiryaev-Roberts odds of a change read here, one mixture per family —
variance up, variance down, mean, lag-one dependence. The ``scale`` statistic
is the larger of the two variance mixtures, with the direction of whichever
won.

Protocol
--------
A rolling history/online walk. The monitor is fitted on the trailing
``history_len`` returns and streams the next ``online_len``; the first step on
which a family's log-odds clears its threshold is a break, attributed to the
family with the largest excursion when several clear at once. The label is
the first alternative to clear, not a diagnosis: the dependence odds are
computed at a fixed marginal variance, so a sharp volatility increase also
feeds them through large consecutive products and can clear them a few steps
before the scale odds. For a retraining trigger the timing is what counts;
read ``cusum_scale`` and ``glr_dependence`` (scale-invariant) in the frame to
tell the two apart. After a break
the history is re-anchored *at the break*: monitoring resumes ``min_history``
returns later on an expanding history capped at ``history_len``, and the
frame's ``history_len`` column says how short it was, rather than pretending
the year before the break still describes the market. Without a break the
window rolls forward every ``online_len`` steps and the monitor is refitted,
so the odds (which accumulate) and the whitening (which ages) never run
unbounded.

Where the threshold comes from
------------------------------
The textbook argument says the Shiryaev-Roberts statistic is a martingale with
E[R_t] = t under the null, so "flag when R_t ≥ A" false-alarms about once every
A steps and the threshold follows from the alarm rate wanted. That argument
does not survive the whitening. The ECDF truncates the tails of the scores, and
a truncated score is sub-fair for every variance-up alternative: measured
E[R_t] / t for the variance-up mixture on i.i.d. histories fell from 0.44 at
t = 50 to 0.10 at t = 600, while the variance-down mixture is heavy-tailed in
the other direction. So the identity is stated and then not used.

Instead, as ``changepoint.py`` calibrated its CUSUM against forty synthetic
random walks, the thresholds here are null quantiles. :func:`calibrate` draws
``null_paths`` i.i.d. Gaussian histories of ``history_len`` points, streams
``online_len`` further Gaussian points through each, and takes the 99th
percentile of the running maximum of each family's log-odds. Because the scores
are history-free under the null, one calibration serves every window of a walk
and the result can be recorded. Measured with ``seed=0``, 200 paths:

====================  =======  ============  =======
history, online        scale    dependence    mean
====================  =======  ============  =======
365, 90 (daily year)     9.09          7.93     8.93
250, 60                  7.77          7.65     8.40
====================  =======  ============  =======

A setting not in the table is calibrated once per run, which takes about
``null_paths x online_len`` milliseconds, and the run says so. Measured on the
recorded daily setting with the two default families: a scale break of
sd x 1.6 planted at step 30 of the fourth window (return 665 of 965) was found
in 39 of 40 synthetic paths at a median lag of 30 steps, with 2 paths flagging
before the break; on 40 null walks of the same length — seven windows, two
families each at the 99th percentile — 3 raised a flag. The constructions are
in ``tests/test_structural_breaks.py``.

Causality
---------
The history ends before the first scored point, and after a break the next
history starts at the break. A break is therefore flagged after it happened,
never before, and the lag is the price of that: compare these breaks against
nothing placed with hindsight, and use them only forward.

``returns[i]`` is the return ending at observation ``i``; when ``returns`` is
a ``Series`` its index labels each return by the close that ends it, and the
frame's ``timestamp`` column carries those labels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.validation.changepoint import (
    Break,
    ChangepointError,
    Segments,
    cut_at_breaks,
)

#: The families worth acting on for a trading model. ``mean`` exists and is off
#: by default: a drift change in returns is weakly identified at any horizon and
#: is not what makes a fitted model stale.
STATISTICS: tuple[str, ...] = ("scale", "dependence")
KNOWN_STATISTICS: tuple[str, ...] = ("scale", "dependence", "mean")

#: Fewer points than this and the AR fit, the scale memory and the ECDF are all
#: being asked for more than the history can say.
MIN_HISTORY: int = 50

#: The channels read from ``WhiteMonitor.update(…)`` with ``odds=True``, by the
#: names ``structural_break.channel_names`` gives them. Resolved by name at run
#: time, so a reordering upstream fails loudly here instead of silently reading
#: a neighbour; a behavioural test raises a scale break and sees it in the scale
#: column, because a name alone pins nothing either.
CHANNELS: dict[str, str] = {
    "cusum_mean": "u_mean_cusum",
    "cusum_scale": "u_scale_cusum",
    "glr_mean": "u_glr_mean_max",
    "glr_scale": "u_glr_scale_max",
    "glr_dependence": "u_glr_dep_max",
    "sr_var_up": "sr_mix_var_up",
    "sr_var_down": "sr_mix_var_down",
    "sr_mean": "sr_mix_mean",
    "sr_dependence": "sr_mix_dep",
    "sr_mean_up": "sr_mean_+1.0",
    "sr_mean_down": "sr_mean_-1.0",
    "sr_dependence_up": "sr_dep_+0.4",
    "sr_dependence_down": "sr_dep_-0.4",
}

#: Null-calibrated thresholds on the family log-odds for the recorded settings,
#: keyed by ``(history_len, online_len)``. The 99th percentile of the running
#: maximum over 200 Gaussian paths, ``seed=0``; the table in the docstring.
RECORDED_THRESHOLDS: dict[tuple[int, int], dict[str, float]] = {
    (365, 90): {"scale": 9.09, "dependence": 7.93, "mean": 8.93},
    (250, 60): {"scale": 7.77, "dependence": 7.65, "mean": 8.40},
}

#: The daily-year setting, which is what the CLI and the experiment run.
DEFAULT_THRESHOLDS: dict[str, float] = RECORDED_THRESHOLDS[(365, 90)]

INSTALL_HINT = (
    "regime monitoring needs the structural-break library, an optional dependency. "
    "Install with: uv sync --extra breaks"
)


def _library() -> tuple[Any, list[str]]:
    """The detector class and its channel names; the only place the import happens."""
    try:
        from structural_break import WhiteMonitor, channel_names
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(INSTALL_HINT) from exc
    return WhiteMonitor, list(channel_names(odds=True))


def _check_statistics(statistics: Sequence[str]) -> None:
    for name in statistics:
        if name not in KNOWN_STATISTICS:
            raise ChangepointError(f"unknown statistic {name!r}; known: {KNOWN_STATISTICS}")
    if not statistics:
        raise ChangepointError("no statistics to monitor")


@dataclass(frozen=True)
class MonitorSpec:
    """How the rolling history/online protocol walks a series."""

    history_len: int
    online_len: int | None = None
    """Steps streamed before the monitor is refitted. Default ``history_len // 4``."""
    min_history: int | None = None
    """Shortest history accepted after a break. Default ``history_len // 3``."""
    statistics: tuple[str, ...] = STATISTICS
    threshold: float | Mapping[str, float] | None = None
    """One number for every family, a mapping per family, or ``None`` for the
    recorded calibration (calibrating once if the setting is not recorded)."""
    minimum_gap: int = 0
    """Returns that must pass after a break before another can be flagged."""

    def __post_init__(self) -> None:
        if self.history_len < MIN_HISTORY:
            raise ChangepointError(
                f"history_len must be at least {MIN_HISTORY} returns, got {self.history_len}"
            )
        if self.online_len is not None and self.online_len < 1:
            raise ChangepointError(f"online_len must be positive, got {self.online_len}")
        if self.min_history is not None and self.min_history < MIN_HISTORY:
            raise ChangepointError(
                f"min_history must be at least {MIN_HISTORY} returns, got {self.min_history}"
            )
        _check_statistics(self.statistics)

    @property
    def online(self) -> int:
        return self.online_len if self.online_len is not None else max(self.history_len // 4, 1)

    @property
    def minimum_history(self) -> int:
        if self.min_history is not None:
            return self.min_history
        return max(self.history_len // 3, MIN_HISTORY)


@dataclass
class MonitorResult:
    """Per-step channels over the whole walk, the breaks flagged, the thresholds used."""

    frame: pd.DataFrame
    """One row per monitored step: ``step``, ``index``, ``timestamp`` (when the
    returns were labelled), ``value``, ``window``, ``history_len``,
    ``odds_scale``, ``odds_dependence``, ``odds_mean``, ``cusum_scale``,
    ``glr_dependence``, ``excursion``, ``flag`` (the family, or ``""``),
    ``direction``."""
    breaks: list[Break]
    thresholds: dict[str, float]
    windows: list[tuple[int, int, int]] = field(default_factory=list)
    """``(history_start, online_start, online_end)`` per window, in return positions."""
    total: int = 0
    """Length of the series walked, so the segments can close it."""

    def segments(self, minimum_rows: int) -> Segments:
        """The series cut at its breaks, exactly as ``changepoint.segment`` cuts."""
        return cut_at_breaks(self.breaks, self.total, minimum_rows=minimum_rows)

    def table(self) -> pd.DataFrame:
        """The breaks as a table: when, what moved, how far past the threshold."""
        rows = []
        for b in self.breaks:
            at = self.frame.loc[self.frame["index"] == b.index].iloc[-1]
            threshold = self.thresholds[b.statistic]
            row: dict[str, object] = {"index": b.index}
            if "timestamp" in self.frame.columns:
                row["timestamp"] = at["timestamp"]
            row.update(
                {
                    "statistic": b.statistic,
                    "direction": b.direction,
                    "log_odds": b.excursion * threshold,
                    "threshold": threshold,
                    "excursion": b.excursion,
                    "history_len": int(at["history_len"]),
                    "step_in_window": int(at["step"]),
                }
            )
            rows.append(row)
        columns = ["index", "statistic", "direction", "log_odds", "threshold", "excursion"]
        columns += ["history_len", "step_in_window"]
        if "timestamp" in self.frame.columns:
            columns.insert(1, "timestamp")
        return pd.DataFrame(rows, columns=columns)

    def annotate(self, returns: pd.Series | np.ndarray) -> pd.DataFrame:
        """The whole series with the monitor's columns beside it.

        ``NaN`` where the monitor was not watching — the first history and the
        stretch after each break while the new history grows — so a reader can
        see where the protocol was blind as well as where it fired. The
        thresholds ride along as constant columns, which makes the table enough
        to redraw the figure on its own.
        """
        values, labels = _as_returns(returns, None)
        if len(values) != self.total:
            raise ChangepointError(
                f"annotate needs the series that was walked ({self.total} returns), "
                f"got {len(values)}"
            )
        out = pd.DataFrame({"log_return": values})
        if labels is not None:
            out.insert(0, "timestamp", labels)
        carried = [
            "window",
            "history_len",
            "odds_scale",
            "odds_dependence",
            "odds_mean",
            "cusum_scale",
            "glr_dependence",
            "excursion",
            "flag",
            "direction",
        ]
        monitored = self.frame.set_index("index")[carried]
        joined = out.join(monitored, how="left")
        joined["flag"] = joined["flag"].fillna("")
        joined["direction"] = joined["direction"].fillna(0).astype("int64")
        for name, value in self.thresholds.items():
            joined[f"threshold_{name}"] = value
        return joined


def whiten_and_stream(
    history: np.ndarray | Sequence[float], online: np.ndarray | Sequence[float]
) -> pd.DataFrame:
    """One fit, one pass: the named channels for every point of ``online``.

    A thin wrapper around ``structural_break.WhiteMonitor``. Returns ``step``,
    ``value`` and one column per entry of :data:`CHANNELS`, one row per online
    point. The ``sr_*`` columns are log-odds, clipped upstream to [-50, 200].
    """
    monitor_cls, names = _library()
    past = np.asarray(history, dtype="float64")
    future = np.asarray(online, dtype="float64")
    if past.ndim != 1 or future.ndim != 1:
        raise ChangepointError("history and online must be one-dimensional")
    if len(past) < MIN_HISTORY:
        raise ChangepointError(f"need at least {MIN_HISTORY} history points, got {len(past)}")
    if not np.isfinite(past).all() or not np.isfinite(future).all():
        raise ChangepointError("history and online must be finite")

    positions: dict[str, int] = {}
    for ours, theirs in CHANNELS.items():
        if theirs not in names:
            raise ChangepointError(
                f"the installed structural-break has no channel {theirs!r}; "
                f"its channel_names(odds=True) give {len(names)} names"
            )
        positions[ours] = names.index(theirs)

    monitor = monitor_cls(past, odds=True)
    rows = np.empty((len(future), len(names)), dtype="float64")
    for i, x in enumerate(future):
        rows[i] = monitor.update(float(x))

    frame = pd.DataFrame({"step": np.arange(len(future)), "value": future})
    for ours, position in positions.items():
        frame[ours] = rows[:, position]
    return frame


def family_odds(channels: pd.DataFrame) -> pd.DataFrame:
    """Log-odds and direction per family from the channel frame.

    ``scale`` is the larger of the variance-up and variance-down mixtures, with
    direction +1 when up won; ``dependence`` and ``mean`` are their family
    mixtures, with the direction of the stronger signed alternative.
    """
    out = pd.DataFrame(index=channels.index)
    out["odds_scale"] = np.maximum(channels["sr_var_up"], channels["sr_var_down"])
    out["direction_scale"] = np.where(channels["sr_var_up"] >= channels["sr_var_down"], 1, -1)
    out["odds_dependence"] = channels["sr_dependence"]
    out["direction_dependence"] = np.where(
        channels["sr_dependence_up"] >= channels["sr_dependence_down"], 1, -1
    )
    out["odds_mean"] = channels["sr_mean"]
    out["direction_mean"] = np.where(channels["sr_mean_up"] >= channels["sr_mean_down"], 1, -1)
    return out


def calibrate(
    history_len: int,
    online_len: int,
    *,
    statistics: Sequence[str] = KNOWN_STATISTICS,
    null_paths: int = 200,
    quantile: float = 0.99,
    seed: int = 0,
) -> dict[str, float]:
    """Null quantile of the running maximum of each family's log-odds.

    ``null_paths`` i.i.d. Gaussian histories of ``history_len`` points, each
    followed by ``online_len`` Gaussian points streamed through the fitted
    monitor; the threshold for a family is the ``quantile`` of the largest
    log-odds it reached. Whitening makes the null nominally history-free, which
    is why one calibration serves every window of a walk and the table in the
    module docstring can be recorded at all; the tail truncation of the ECDF
    depends on ``history_len`` alone, which is why it is a parameter here.
    """
    _check_statistics(statistics)
    if null_paths < 2:
        raise ChangepointError(f"need at least 2 null paths, got {null_paths}")
    if not 0.0 < quantile < 1.0:
        raise ChangepointError(f"quantile must lie in (0, 1), got {quantile}")
    rng = np.random.default_rng(seed)
    maxima = {name: np.empty(null_paths) for name in statistics}
    for k in range(null_paths):
        odds = family_odds(
            whiten_and_stream(rng.standard_normal(history_len), rng.standard_normal(online_len))
        )
        for name in statistics:
            maxima[name][k] = odds[f"odds_{name}"].max()
    return {name: float(np.quantile(values, quantile)) for name, values in maxima.items()}


def thresholds_for(
    spec: MonitorSpec, *, on_progress: Callable[[str], None] | None = print
) -> dict[str, float]:
    """Resolve ``spec.threshold`` into one number per monitored family."""
    if isinstance(spec.threshold, int | float):
        return dict.fromkeys(spec.statistics, float(spec.threshold))
    if isinstance(spec.threshold, Mapping):
        missing = [s for s in spec.statistics if s not in spec.threshold]
        if missing:
            raise ChangepointError(
                f"no threshold for statistic(s) {missing}; given: {sorted(spec.threshold)}"
            )
        return {s: float(spec.threshold[s]) for s in spec.statistics}
    recorded = RECORDED_THRESHOLDS.get((spec.history_len, spec.online))
    if recorded is None:
        if on_progress:
            on_progress(
                f"no recorded calibration for history {spec.history_len}, online {spec.online}; "
                f"calibrating on 200 Gaussian null paths (about {0.2 * spec.online:.0f} s)"
            )
        recorded = calibrate(spec.history_len, spec.online, statistics=spec.statistics)
    return {s: recorded[s] for s in spec.statistics}


def _as_returns(
    returns: pd.Series | np.ndarray | Sequence[float], index: pd.Index | None
) -> tuple[np.ndarray, pd.Index | None]:
    labels: pd.Index | None = index
    if isinstance(returns, pd.Series):
        if labels is None:
            labels = returns.index
        values = returns.to_numpy(dtype="float64")
    else:
        values = np.asarray(returns, dtype="float64")
    if values.ndim != 1:
        raise ChangepointError(f"returns must be one-dimensional, got shape {values.shape}")
    bad = ~np.isfinite(values)
    if bad.any():
        where = np.flatnonzero(bad)[:5].tolist()
        raise ChangepointError(f"returns contain NaN or infinite values at {where}")
    if labels is not None and len(labels) != len(values):
        raise ChangepointError(f"index has {len(labels)} labels for {len(values)} returns")
    return values, labels


def detect_breaks(
    returns: pd.Series | np.ndarray | Sequence[float],
    spec: MonitorSpec,
    *,
    index: pd.Index | None = None,
    on_progress: Callable[[str], None] | None = print,
) -> MonitorResult:
    """Walk the series and flag the breaks; see the module docstring for the protocol.

    ``index`` labels the returns (a ``Series`` brings its own); the labels
    appear as the frame's ``timestamp`` column and in :meth:`MonitorResult.table`.
    ``on_progress`` receives one line if a calibration had to be run.
    """
    values, labels = _as_returns(returns, index)
    total = len(values)
    needed = spec.history_len + 1
    if total < needed:
        raise ChangepointError(f"need at least {needed} returns to monitor, got {total}")
    thresholds = thresholds_for(spec, on_progress=on_progress)
    online_len = spec.online
    min_history = spec.minimum_history

    frames: list[pd.DataFrame] = []
    breaks: list[Break] = []
    windows: list[tuple[int, int, int]] = []
    anchor = 0
    position = spec.history_len
    last_break = -spec.minimum_gap - 1
    while position < total:
        history_start = max(anchor, position - spec.history_len)
        online_end = min(position + online_len, total)
        channels = whiten_and_stream(values[history_start:position], values[position:online_end])
        odds = family_odds(channels)
        positions = np.arange(position, online_end)
        frame = pd.DataFrame(
            {
                "step": channels["step"].to_numpy(),
                "index": positions,
                "value": channels["value"].to_numpy(),
                "window": len(windows),
                "history_len": position - history_start,
                "odds_scale": odds["odds_scale"].to_numpy(),
                "odds_dependence": odds["odds_dependence"].to_numpy(),
                "odds_mean": odds["odds_mean"].to_numpy(),
                "cusum_scale": channels["cusum_scale"].to_numpy(),
                "glr_dependence": channels["glr_dependence"].to_numpy(),
            }
        )
        excursions = np.column_stack(
            [odds[f"odds_{name}"].to_numpy() / thresholds[name] for name in spec.statistics]
        )
        frame["excursion"] = excursions.max(axis=1)
        frame["flag"] = ""
        frame["direction"] = 0

        eligible = positions - last_break >= spec.minimum_gap
        cleared = np.flatnonzero((excursions >= 1.0).any(axis=1) & eligible)
        if cleared.size:
            k = int(cleared[0])
            at = int(positions[k])
            which = int(np.argmax(excursions[k]))
            family = spec.statistics[which]
            direction = int(odds[f"direction_{family}"].iloc[k])
            breaks.append(Break(at, family, direction, float(excursions[k, which])))
            frame = frame.iloc[: k + 1].copy()
            frame.loc[frame.index[-1], "flag"] = family
            frame.loc[frame.index[-1], "direction"] = direction
            windows.append((history_start, position, at + 1))
            anchor = at
            position = at + min_history
            last_break = at
        else:
            windows.append((history_start, position, online_end))
            position = online_end
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    if labels is not None:
        result.insert(2, "timestamp", labels.take(result["index"].to_numpy()))
    return MonitorResult(
        frame=result, breaks=breaks, thresholds=thresholds, windows=windows, total=total
    )


def segment_returns(
    returns: pd.Series | np.ndarray | Sequence[float],
    spec: MonitorSpec,
    *,
    minimum_rows: int,
    index: pd.Index | None = None,
    on_progress: Callable[[str], None] | None = print,
) -> Segments:
    """``detect_breaks`` cut into ``Segments`` as ``changepoint.segment`` does.

    So a refit-at-breaks policy can take either detector: the bounds are
    half-open return positions covering the series exactly once, and a segment
    shorter than ``minimum_rows`` is merged into the one before it.
    """
    return detect_breaks(returns, spec, index=index, on_progress=on_progress).segments(minimum_rows)


# ---------------------------------------------------------------------------
# Prices in, returns out
# ---------------------------------------------------------------------------


def read_prices(path: Path | str) -> pd.Series:
    """A CSV with ``timestamp`` and ``close`` columns, as a close series in UTC.

    The no-network path into the CLI, for a price history from any source.
    Anything the two columns cannot say — a missing column, an unparseable
    date, a non-positive price, two rows for one moment — is refused with the
    reason rather than coerced.
    """
    frame = pd.read_csv(path)
    missing = {"timestamp", "close"} - set(frame.columns)
    if missing:
        raise ChangepointError(
            f"{path}: prices need columns 'timestamp' and 'close'; missing {sorted(missing)}"
        )
    stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if stamps.isna().any():
        raise ChangepointError(
            f"{path}: {int(stamps.isna().sum())} timestamp(s) could not be parsed"
        )
    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().any() or (close <= 0).any():
        raise ChangepointError(f"{path}: close must be a positive number on every row")
    series = pd.Series(close.to_numpy(dtype="float64"), index=pd.DatetimeIndex(stamps))
    series = series.sort_index()
    if series.index.has_duplicates:
        raise ChangepointError(f"{path}: several rows share a timestamp")
    series.name = "close"
    return series


def log_returns(close: pd.Series) -> pd.Series:
    """``log(close_t / close_{t-1})``, each labelled by the close that ends it."""
    values = np.asarray(close, dtype="float64")
    if len(values) < 2:
        raise ChangepointError(f"need at least 2 prices for a return, got {len(values)}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ChangepointError("prices must be finite and positive")
    return pd.Series(np.diff(np.log(values)), index=close.index[1:], name="log_return")
