"""The break detector's flags: fold-wise on D, frozen on H, F and P.

Every flag comes from one hourly stream of Binance BTCUSDT closes from
2024-12-01, read through the access the stage holds (the development access
reads bars to 2025-06-30 only). On D, test month m's flags come from one run
over the stream to m's end with the hour profile fitted before m
(``validation/regime_flags.fold_flags``), and the guard uses its alarms from
120 hours before m. On a held-out block (and in the dry run) the flags are the
frozen detector's, its profile fitted on all of D and recorded in the
amendment. Fold runs are kept on disk, so a resumed stage does not rerun them.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import study_access as sa
from trading_research.data import study_loader as sl
from trading_research.validation import regime_flags as rf

STREAM_START = date(2024, 12, 1)


def hourly_close(symbol: str, root: Path, end: date, access: sa.StudyAccess) -> pd.Series:
    """Hourly closes from the stream's start through ``end``'s last hour."""
    days = [STREAM_START + timedelta(days=i) for i in range((end - STREAM_START).days + 1)]
    bars = sl.load_bars(symbol, days, root, access=access)
    label_end = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    return rf.hourly_stream(bars, end=label_end).close


def month_fold(month: str) -> rf.Fold:
    folds = {f.name: f for f in rf.walk_forward_months()}
    return folds[month]


def _alarms_from(record: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(record, columns=list(rf.ALARM_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _alarms_to(alarms: pd.DataFrame) -> list[dict[str, Any]]:
    out = alarms.loc[:, list(rf.ALARM_COLUMNS)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).map(lambda t: t.isoformat())
    return [
        {str(k): (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in out.to_dict("records")
    ]


def fold_flags(close: pd.Series, month: str, directory: Path) -> dict[str, Any]:
    """Month ``month``'s run: its profile, every alarm, and the guard's flags in ns."""
    path = directory / f"{month}.json"
    if path.exists():
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    found = rf.fold_flags(close, month_fold(month))
    record = {
        "month": month,
        "profile": [float(v) for v in found.profile],
        "alarms": _alarms_to(found.alarms),
        "guard_flags_ns": [int(v) for v in found.flags_ns()],
    }
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def saved_fold_flags(month: str, directory: Path) -> dict[str, Any]:
    """A month's run as stage A kept it."""
    loaded: dict[str, Any] = json.loads((directory / f"{month}.json").read_text(encoding="utf-8"))
    return loaded


def guard_flags(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["guard_flags_ns"], dtype=np.int64)


def frozen(close: pd.Series, end: date, profile: list[float] | None = None) -> dict[str, Any]:
    """The frozen detector over the stream to ``end``: its profile (fitted on D
    unless given) and its alarms, with their times in ns."""
    stamp = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    fitted, alarms = rf.frozen_flags(
        close, end=stamp, profile=None if profile is None else np.asarray(profile, dtype=np.float64)
    )
    return {
        "profile": [float(v) for v in fitted],
        "alarms": _alarms_to(alarms),
        "flags_ns": [int(v) for v in rf.flag_times(alarms)],
    }


def alarms_frame(record: dict[str, Any]) -> pd.DataFrame:
    return _alarms_from(record["alarms"])


def causality(
    close: pd.Series, hours_per_fold: int | None, workers: int, folds: int | None = None
) -> list[dict[str, Any]]:
    """The flag-causality test in every fold (all hours, or evenly spaced ones)."""
    out = []
    for fold in rf.walk_forward_months()[:folds]:
        hours = None
        if hours_per_fold is not None:
            every = fold.hours
            hours = list(every[np.linspace(0, len(every) - 1, hours_per_fold).round().astype(int)])
        found = rf.check_flag_causality(close, fold, hours=hours, workers=workers)
        out.append(
            {
                "fold": fold.name,
                "hours": found.hours_checked,
                "leaks": found.leaks,
                "detail": str(found),
            }
        )
    return out
