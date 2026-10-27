"""The study's context: its calendar, the disk check, and day jobs that resume."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import signal as sg
from trading_research.spot_maker import standin
from trading_research.spot_maker.context import MakerStudy, compute_days, signal_days

REPO = Path(__file__).resolve().parents[1]
CALLS: list[str] = []


def _registered(target: Path) -> Path:
    for relative in (rg.CONFIG_PATH, rg.DOCUMENT_PATH):
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target / relative)
    return target


def _job(repo: Path, day: date, access: Any, binding: Any, fees: Any) -> dict[str, Any]:
    CALLS.append(day.isoformat())
    return {"day": day.isoformat(), "stamp": access.stamp, "fees": sorted(fees)[:1]}


def test_the_calendar_is_the_registered_one_unless_a_dry_run_names_days(tmp_path: Path) -> None:
    study = MakerStudy(_registered(tmp_path))
    assert len(study.days("H")) == 75 and study.days(bk.READER_TEST) == bk.READER_TEST_DAYS
    dry = MakerStudy(tmp_path, calendar={"H": (date(2026, 8, 8),)})
    assert dry.days("H") == (date(2026, 8, 8),)
    assert study.table().exclusion(bk.INSTRUMENT, date(2026, 8, 8)) is None
    with pytest.raises(ValueError):
        study.require_source()


def test_the_disk_check_is_recorded_and_refuses_below_the_margin(tmp_path: Path) -> None:
    assert MakerStudy(tmp_path, disk=lambda _: 50.0).disk_check("0")["passed"]
    with pytest.raises(RuntimeError):
        MakerStudy(tmp_path, disk=lambda _: 10.5).disk_check("1")
    found = json.loads((tmp_path / "data/spot_maker_2026/disk.json").read_text("utf-8"))
    assert [f["stage"] for f in found] == ["0", "1"] and found[1]["needed_gb"] == 11.0


def test_day_jobs_resume_from_the_records_kept(tmp_path: Path) -> None:
    study = MakerStudy(_registered(tmp_path))
    days, folder = [date(2026, 5, 1), date(2026, 5, 2)], tmp_path / "work"
    CALLS.clear()
    first = compute_days(study, days[:1], bk.development_access(), folder, _job)
    again = compute_days(study, days, bk.development_access(), folder, _job)
    assert CALLS == ["2026-05-01", "2026-05-02"] and again[0] == first[0]
    assert again[1]["stamp"] == "development"


def test_the_signal_alone_on_stand_in_days(tmp_path: Path) -> None:
    days = [date(2026, 4, 30), date(2026, 5, 1)]
    source, repo = tmp_path / "source", _registered(tmp_path / "repo")
    binding = standin.write_source(source, days)
    sg.copy_signal(source, repo, binding)
    cp.copy_days(source, repo, days, bk.development_access())
    study = MakerStudy(repo, binding=binding)
    found = signal_days(study, [days[1], date(2026, 5, 2)], bk.development_access(), tmp_path / "s")
    assert found[0]["usable"] and found[0]["warm"] and found[0]["decisions"] > 0
    assert found[1] == {"day": "2026-05-02", "reasons": ["no_prints_file"],
                        "stamp": "development", "usable": False}  # fmt: skip
