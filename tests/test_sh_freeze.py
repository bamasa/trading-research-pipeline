"""Stage E's detector record on D, and the read's check that a later run reproduces it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pandas as pd

from trading_research.pipeline.short_horizon import flags, freeze, reads
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import DRY_RUN

ALARMS = [
    {"timestamp": "2024-12-20T05:00:00+00:00", "statistic": "scale", "direction": "up"},
    {"timestamp": "2025-03-10T04:00:00+00:00", "statistic": "mean", "direction": "down"},
    {"timestamp": "2025-07-02T01:00:00+00:00", "statistic": "scale", "direction": "up"},
]


def _frozen(*_: Any, **__: Any) -> dict[str, Any]:
    return {"alarms": ALARMS, "profile": [1.0] * 24}


def test_alarms_d_holds_the_registered_span_only(tmp_path: Path) -> None:
    study = Study(tmp_path / "repo", DRY_RUN, work_root=tmp_path / "work")
    with (
        mock.patch.object(flags, "hourly_close", return_value=pd.Series(dtype=float)),
        mock.patch.object(flags, "frozen", _frozen),
        mock.patch.object(flags, "saved_fold_flags", return_value={"guard_flags_ns": []}),
    ):
        record = freeze.detector_record(study)
    assert [a["timestamp"] for a in record["alarms_D"]] == ["2025-03-10T04:00:00+00:00"]
    assert record["alarm_rate_D"] == 1 / 181


def test_a_later_run_reproduces_d_when_its_december_alarms_are_left_aside() -> None:
    recorded = [{"timestamp": "2025-03-10T04:00:00+00:00", "channel": "mean"}]
    registration = SimpleNamespace(value=lambda _: recorded)
    assert not reads.r_void(registration, ALARMS)  # December and July are outside D
    assert reads.r_void(registration, ALARMS[:1])  # D's alarm not reproduced
