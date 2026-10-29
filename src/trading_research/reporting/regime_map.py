"""The regime map: each slot's daily net cut at the detector's alarms.

A description, never a status. On the development period a slot's
walk-forward daily net is cut at each test month's fold-wise alarms; on the
held-out blocks at the frozen detector's. The alarms cut the series before any
number is computed for a segment, every segment is reported, and none is
selected or enters a rule: a positive stretch between two alarms appears here
as what it is.

Cutting at day resolution
-------------------------
An alarm is effective at an hour's close, and a daily series has one value a
day, so a segment starts on the first day that begins at or after its alarm;
the alarm's own day, which began before it, closes the segment before. A
change of block cuts too (the blocks are read in different sessions and the
flags change source between them), and the table says which of the two opened
each segment. A segment with fewer than :data:`MIN_DAYS_FOR_T` usable days is
reported and drawn but carries no t.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.market_making.verdicts import day_t
from trading_research.reporting.plots import COLOURS, _finish, _pyplot
from trading_research.validation.regime_flags import FoldFlags

#: Fewer usable days than this and a segment's t is not reported.
MIN_DAYS_FOR_T = 10

SEGMENT_COLUMNS: tuple[str, ...] = (
    "slot",
    "segment",
    "block",
    "start",
    "end",
    "opened_by",
    "alarm",
    "usable_days",
    "mean",
    "t",
    "cumulative_net",
)


def _utc_index(stamps: Sequence[pd.Timestamp] | pd.Series | pd.Index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(stamps), dtype=object), utc=True))
    return index.sort_values()


def map_alarms(
    foldwise: Sequence[FoldFlags], frozen: pd.DataFrame, *, frozen_from: pd.Timestamp | str
) -> pd.DatetimeIndex:
    """The alarms that cut the map: each test month's own, then the frozen run's.

    ``frozen_from`` is the first instant the frozen flags apply (the first day
    after the development period); frozen alarms before it are not used.
    """
    start = pd.Timestamp(frozen_from)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    stamps = [t for flags in foldwise for t in flags.month_alarms["timestamp"]]
    later = pd.to_datetime(frozen["timestamp"], utc=True)
    stamps += list(later[(later >= start).to_numpy()])
    return _utc_index(stamps)


def _days(daily: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    values = pd.Series(daily.to_numpy(dtype="float64"), index=pd.to_datetime(daily.index))
    values = values[np.isfinite(values.to_numpy())].sort_index()
    index = pd.DatetimeIndex(values.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return index.normalize(), values.to_numpy()


def regime_map(
    daily: pd.Series,
    alarms: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    blocks: Mapping[date, str] | pd.Series | None = None,
    slot: str = "",
    min_days_for_t: int = MIN_DAYS_FOR_T,
) -> pd.DataFrame:
    """One row per segment of ``daily`` between alarms (and block changes).

    ``daily`` is the slot's net per usable day, indexed by day; a non-finite
    day is not usable. ``blocks`` maps each day to its block. Columns:
    :data:`SEGMENT_COLUMNS`; ``alarm`` is the latest alarm that opened the
    segment, ``cumulative_net`` the segment's summed daily net.
    """
    days, values = _days(daily)
    if len(days) == 0:
        return pd.DataFrame(columns=list(SEGMENT_COLUMNS))
    cuts = _utc_index(list(alarms))
    crossed = np.asarray(cuts.searchsorted(days, side="right"), dtype=np.int64)
    labels = pd.Series(blocks) if blocks is not None else pd.Series(dtype=object)
    if len(labels):
        labels.index = pd.DatetimeIndex(pd.to_datetime(labels.index)).normalize()
        if labels.index.tz is None:
            labels.index = labels.index.tz_localize("UTC")
    block = [str(labels.get(d, "")) for d in days]
    rows: list[dict[str, object]] = []
    first = 0
    for i in range(1, len(days) + 1):
        if i < len(days) and crossed[i] == crossed[first] and block[i] == block[first]:
            continue
        part = values[first:i]
        stats = day_t(part)
        opened = (
            "start" if not rows else "alarm" if crossed[first] > crossed[first - 1] else "block"
        )
        by_alarm = opened == "alarm"
        rows.append(
            {
                "slot": slot,
                "segment": len(rows),
                "block": block[first],
                "start": days[first].date(),
                "end": days[i - 1].date(),
                "opened_by": opened,
                "alarm": cuts[crossed[first] - 1] if by_alarm else pd.NaT,
                "usable_days": len(part),
                "mean": float(part.mean()),
                "t": stats.t if len(part) >= min_days_for_t else float("nan"),
                "cumulative_net": float(part.sum()),
            }
        )
        first = i
    return pd.DataFrame(rows, columns=list(SEGMENT_COLUMNS))


def regime_maps(
    slots: Mapping[str, pd.Series],
    alarms: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    *,
    blocks: Mapping[date, str] | pd.Series | None = None,
) -> pd.DataFrame:
    """:func:`regime_map` for every slot (and measured row), stacked."""
    parts = [regime_map(s, alarms, blocks=blocks, slot=name) for name, s in slots.items()]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=list(SEGMENT_COLUMNS))
    return pd.concat(parts, ignore_index=True)


#: The shading of alternate blocks, so neighbouring blocks stay distinguishable.
_BLOCK_ALPHA = (0.10, 0.04)


def regime_map_chart(
    slots: Mapping[str, pd.Series],
    alarms: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    path: Path | str,
    *,
    blocks: Mapping[str, tuple[date, date]] | None = None,
    title: str = "",
) -> Path:
    """Cumulative daily net through the year for one execution path's slots.

    One line per slot over its usable days, a vertical line at every alarm
    that cuts the map, and each block shaded and named, its span from
    ``blocks`` (name to first and last day). Drawn for every segment, chosen
    for none.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    palette = [COLOURS["signal"], COLOURS["edge"], COLOURS["accent"], COLOURS["cost"]]
    for k, (name, daily) in enumerate(slots.items()):
        days, values = _days(daily)
        if len(days) == 0:
            continue
        ax.plot(days, np.cumsum(values), linewidth=1.2, color=palette[k % len(palette)], label=name)
    for k, (name, (first, last)) in enumerate((blocks or {}).items()):
        start = pd.Timestamp(first, tz="UTC")
        stop = pd.Timestamp(last, tz="UTC") + pd.Timedelta(1, unit="D")
        ax.axvspan(start, stop, color=COLOURS["neutral"], alpha=_BLOCK_ALPHA[k % 2], linewidth=0)
        ax.text(
            start + (stop - start) / 2, 0.98, name, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color=COLOURS["neutral"],
        )  # fmt: skip
    stamps = _utc_index(list(alarms))
    for stamp in stamps:
        ax.axvline(stamp, color=COLOURS["cost"], linewidth=0.9, alpha=0.8, linestyle="--")
    ax.axhline(0.0, color=COLOURS["neutral"], linewidth=0.8)
    ax.set_ylabel("cumulative daily net")
    if slots:
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.text(
        0.995, 0.02, f"{len(stamps)} alarm(s), dashed; shaded: blocks; no segment selected",
        transform=ax.transAxes, ha="right", fontsize=8, color=COLOURS["neutral"],
    )  # fmt: skip
    fig.autofmt_xdate()
    return _finish(
        fig, ax, title or "The regime map: cumulative daily net, cut at alarms", Path(path)
    )
