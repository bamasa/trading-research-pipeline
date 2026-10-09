"""The break-detector view: BTCUSDT's hourly close through the registered span,
with everything the frozen detector did on it.

From 2024-12-01 to 2025-12-31 when P is opened, otherwise to F's last day,
2025-11-16: the blocks shaded and named, every alarm with its channel, the
break ``b`` with F- and F+ either side of it, each guarded slot's guard
windows after the alarms (one lane per slot), the 1 s CUSUM's restarts, and
the public 2025-09-13 daily-bar alarm, labelled as already public. A picture
of what the detector cut, drawn after the cuts were made; it decides nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.reporting.plots import COLOURS, _pyplot

START = pd.Timestamp("2024-12-01", tz="UTC")
END_WITH_P = pd.Timestamp("2026-01-01", tz="UTC")  # exclusive: P's last day is 2025-12-31
END_WITHOUT_P = pd.Timestamp("2025-11-17", tz="UTC")  # exclusive: F's last day is 2025-11-16
PUBLIC_ALARM = pd.Timestamp("2025-09-13", tz="UTC")
CHANNEL_COLOURS = {"scale": COLOURS["cost"], "dependence": COLOURS["accent"]}


def _utc(stamp: object) -> pd.Timestamp:
    value = pd.Timestamp(stamp)  # type: ignore[arg-type]
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def guard_windows(
    alarms: Sequence[object], window_h: int, end: object
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Each alarm's guard window ``(alarm, alarm + window]``, clipped at ``end``
    and merged where two overlap."""
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    stop = _utc(end)
    for start in sorted(_utc(a) for a in alarms):
        if start >= stop:
            continue
        finish = min(start + pd.Timedelta(int(window_h), unit="h"), stop)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], finish))
        else:
            spans.append((start, finish))
    return spans


def break_view_chart(
    close: pd.Series,
    alarms: pd.DataFrame,
    path: Path | str,
    *,
    blocks: Mapping[str, tuple[date, date]],
    p_opened: bool,
    b: object | None = None,
    guards: Mapping[str, int] | None = None,
    cusum_restarts: Sequence[object] = (),
    public_alarm: object = PUBLIC_ALARM,
    title: str = "",
) -> Path:
    """The view. ``close`` is the hourly close indexed by UTC time, ``alarms``
    the frozen run's alarm table (``regime_flags.alarm_table``), ``guards``
    each guarded slot's window in hours, ``b`` the first alarm in F."""
    plt = _pyplot()
    from matplotlib.ticker import FuncFormatter  # after _pyplot: matplotlib is optional

    end = END_WITH_P if p_opened else END_WITHOUT_P
    series = close.copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    series = series[(series.index >= START) & (series.index < end)]
    lanes = [*(guards or {}), "1 s CUSUM restarts"]
    fig, (top, lane) = plt.subplots(
        2,
        1,
        figsize=(11.0, 4.6 + 0.3 * len(lanes)),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 0.5 + 0.3 * len(lanes)]},
    )
    top.plot(
        series.index, series.to_numpy(dtype=np.float64), color=COLOURS["signal"], linewidth=0.9
    )
    top.set_yscale("log")
    plain = FuncFormatter(lambda value, _: f"{value:,.0f}")
    top.yaxis.set_major_formatter(plain)
    top.yaxis.set_minor_formatter(plain)
    top.tick_params(axis="y", which="minor", labelsize=7)
    top.set_ylabel("BTCUSDT hourly close, USDT (log)")
    for k, (name, (first, last)) in enumerate(blocks.items()):
        left, right = _utc(first), _utc(last) + pd.Timedelta(1, unit="D")
        if left >= end:
            continue
        right = min(right, end)
        for ax in (top, lane):
            ax.axvspan(
                left, right, color=COLOURS["neutral"], alpha=(0.10, 0.04)[k % 2], linewidth=0
            )
        top.text(
            left + (right - left) / 2,
            1.01,
            name,
            transform=top.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOURS["neutral"],
        )
    stamps = [_utc(t) for t in alarms["timestamp"]] if len(alarms) else []
    channels = list(alarms["statistic"]) if "statistic" in alarms else [""] * len(stamps)
    for stamp, channel in zip(stamps, channels, strict=True):
        if START <= stamp < end:
            colour = CHANNEL_COLOURS.get(str(channel), COLOURS["neutral"])
            top.axvline(stamp, color=colour, linewidth=1.0, alpha=0.9)
            top.text(
                stamp,
                0.97,
                f" {channel}",
                transform=top.get_xaxis_transform(),
                rotation=90,
                fontsize=7,
                color=colour,
                va="top",
                ha="left",
            )
    if b is not None:
        split = _utc(b)
        top.axvline(split, color=COLOURS["edge"], linewidth=2.2)
        f_span = blocks.get("F")
        if f_span is not None:
            for text, x in (
                ("F-", _utc(f_span[0]) + (split - _utc(f_span[0])) / 2),
                ("F+", split + (_utc(f_span[1]) - split) / 2),
            ):
                top.text(
                    x, 0.06, text, transform=top.get_xaxis_transform(), ha="center", fontsize=9
                )
        top.text(
            split,
            0.06,
            " b",
            transform=top.get_xaxis_transform(),
            fontsize=9,
            color=COLOURS["edge"],
        )
    public = _utc(public_alarm)
    if START <= public < end:
        top.axvline(public, color=COLOURS["neutral"], linestyle=":", linewidth=1.2)
        top.text(
            public,
            0.80,
            " public daily-bar alarm\n (already known)",
            transform=top.get_xaxis_transform(),
            fontsize=7,
            color=COLOURS["neutral"],
            va="top",
            bbox={"facecolor": plt.rcParams["axes.facecolor"], "alpha": 0.85, "edgecolor": "none"},
        )
    for row, window in enumerate((guards or {}).values()):
        spans = guard_windows(stamps, window, end)
        lane.broken_barh([(s, f - s) for s, f in spans], (row - 0.35, 0.7), color=COLOURS["accent"])
    restarts = [_utc(t) for t in cusum_restarts]
    lane.scatter(
        restarts, [len(lanes) - 1] * len(restarts), marker="|", s=120, color=COLOURS["neutral"]
    )
    labels = [f"{n}: {w} h guard" for n, w in (guards or {}).items()] + [lanes[-1]]
    lane.set_yticks(range(len(lanes)), labels, fontsize=7.5)
    lane.set_ylim(len(lanes) - 0.5, -0.5)
    for ax in (top, lane):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    top.set_xlim(START, end)
    fig.suptitle(
        title or "The break detector on BTCUSDT: alarms, b, and the guard windows",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110)
    plt.close(fig)
    return target
