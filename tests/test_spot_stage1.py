"""Stage 1: the code-freeze note, the gap threshold, and the cut the leakage checks need."""

from __future__ import annotations

import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.data import fetch_log as fl
from trading_research.spot import archives as ar
from trading_research.spot import stage1
from trading_research.spot.context import SpotStudy
from trading_research.spot.dry_run import code_freeze_note, commit


def test_the_code_freeze_names_a_commit_after_which_no_study_code_changed(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    study = SpotStudy(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError, match=r"no docs/spot_2026_code_freeze\.md"):
        stage1.code_freeze(study)
    code_freeze_note(tmp_path)
    assert len(stage1.code_freeze(study)["commit"]) == 40
    (tmp_path / "src" / "code.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="changed after the code-freeze commit"):
        stage1.code_freeze(study)
    commit(tmp_path, "a change after the freeze")
    with pytest.raises(RuntimeError, match="correction"):
        stage1.code_freeze(study)


def test_the_gap_threshold_reads_every_replayed_d_day(tmp_path: Path) -> None:
    log = fl.FetchLog(tmp_path / "log.json")
    days = [date(2026, 5, d) for d in range(1, 21)]
    for k, day in enumerate(days):
        record = log.get(ar.BYBIT_BOOK.name, day)
        record.status, record.stats = fl.ON_DISK, {"sequence_gaps": k % 4}
    log.get(ar.BYBIT_BOOK.name, date(2026, 5, 21)).status = fl.UNREADABLE
    threshold, counts = stage1.gap_threshold(log, [*days, date(2026, 5, 21)])
    assert len(counts) == 20 and threshold == 3
    with pytest.raises(RuntimeError, match="no D day"):
        stage1.gap_threshold(log, [date(2026, 6, 1)])


def test_the_cut_falls_strictly_inside_an_interval_with_two_prints() -> None:
    rows = pd.DataFrame({"timestamp": pd.to_datetime(np.arange(0, 2_000, 10), unit="ms", utc=True)})
    stamps = np.array([5, 15, 1_505, 1_506, 1_507])
    prints = pd.DataFrame({"timestamp": pd.to_datetime(stamps, unit="ms", utc=True)})
    found = stage1.cut_slice(rows, prints, 100)
    assert found is not None
    kept, kept_prints, cut = found
    k = int(len(kept) * cut)
    assert k == 151 and len(kept) == 200 and len(kept_prints) == 5
    assert stage1.cut_slice(rows, prints.iloc[:3], 100) is None
