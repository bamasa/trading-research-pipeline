"""Copies inside the access, the copy log, and a block's days loaded warm or cold."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from trading_research.market_making.prereg import HeldOutLocked, sha256_bytes
from trading_research.spot.costs import FeeTable
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import standin

REPO = Path(__file__).resolve().parents[1]
DAYS = [date(2026, 4, 30), date(2026, 5, 1), date(2026, 5, 3), date(2026, 8, 8)]


def _table() -> FeeTable:
    config = yaml.safe_load((REPO / "configs/spot_maker_2026.yaml").read_text(encoding="utf-8"))
    return FeeTable.from_config(config["fees"])


def test_copies_are_checked_logged_and_refused_outside_the_access(tmp_path: Path) -> None:
    source, repo = tmp_path / "source", tmp_path / "repo"
    standin.write_source(source, DAYS)
    access = bk.development_access()
    found = cp.copy_days(source, repo, [date(2026, 5, 1), date(2026, 5, 2)], access)
    assert found["2026-05-01"]["status"] == "copied" and found["2026-05-02"]["status"] == "absent"
    copied = bk.day_file(repo, date(2026, 5, 1)).read_bytes()
    assert sha256_bytes(copied) == found["2026-05-01"]["sha256"]
    assert set(cp.copy_log(repo)["files"]) == {"2026-05-01", "2026-05-02"}
    with pytest.raises(HeldOutLocked):
        cp.copy_days(source, repo, [date(2026, 8, 8)], access)
    assert not bk.files_of_days(repo, [date(2026, 8, 8)])
    bk.day_file(repo, date(2026, 5, 1)).write_bytes(b"changed")
    with pytest.raises(HeldOutLocked):
        cp.copy_days(source, repo, [date(2026, 5, 1)], access)
    with pytest.raises(HeldOutLocked):
        cp.copy_days(source, source, [date(2026, 5, 1)], access)


def test_walk_marks_warm_cold_and_excluded_days(tmp_path: Path) -> None:
    source, repo = tmp_path / "source", tmp_path / "repo"
    standin.write_source(source, DAYS)
    access = bk.development_access()
    cp.copy_days(source, repo, DAYS[:3] + [date(2026, 5, 2)], access)
    found = {d.day: d for d in cp.walk(repo, DAYS[1:3] + [date(2026, 5, 2)], access, _table())}
    assert found[date(2026, 5, 1)].warm  # W's last day before D's first
    assert found[date(2026, 5, 2)].reasons == ("no_prints_file",)
    third = found[date(2026, 5, 3)]
    assert not third.warm and third.scored_from > third.tape().start_ns
    assert found[date(2026, 5, 1)].carried is not None
    with pytest.raises(ValueError):
        _ = found[date(2026, 5, 2)].frame
