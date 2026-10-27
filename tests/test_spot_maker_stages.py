"""The run log, the code-freeze and configuration gates, and the stages' order."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot.state import StageOrderError
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import stages as sm
from trading_research.spot_maker.context import MakerStudy
from trading_research.spot_maker.dry_run import commit, make_repository
from trading_research.spot_maker.runner import run_stage

REPO = Path(__file__).resolve().parents[1]


def _repository(tmp_path: Path) -> MakerStudy:
    repo = tmp_path / "repo"
    repo.mkdir()
    make_repository(REPO, repo)
    (repo / "src").mkdir()
    (repo / "src" / "code.py").write_text("x = 1\n", encoding="utf-8")
    commit(repo, "code")
    return MakerStudy(repo, disk=lambda _: 1_000.0)


def _freeze(study: MakerStudy, sha: str) -> None:
    note = Path(study.repo) / rg.CODE_FREEZE_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(f"# Code freeze\n\nCode-freeze commit: `{sha}`\n", encoding="utf-8")
    commit(study.repo, "note")


def test_the_run_log_appends_a_row_per_computation(tmp_path: Path) -> None:
    sm.log_run(tmp_path, "D1", "D", [date(2026, 5, 1)], "every run")
    sm.log_run(tmp_path, "D1", "D", [date(2026, 5, 2)])
    rows = json.loads((tmp_path / rg.RUN_LOG_PATH).read_text("utf-8"))
    assert [r["days"] for r in rows] == [["2026-05-01"], ["2026-05-02"]] and rows[0]["run"] == "D1"


def test_the_code_freeze_gate(tmp_path: Path) -> None:
    study = _repository(tmp_path)
    with pytest.raises(HeldOutLocked):
        sm.require_code_freeze(study)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=study.repo, capture_output=True,
                          text=True, check=True).stdout.strip()  # fmt: skip
    _freeze(study, head[:12])
    assert sm.require_code_freeze(study) == head
    sm.require_config_pinned(study, head)
    (Path(study.repo) / "src" / "code.py").write_text("x = 2\n", encoding="utf-8")
    commit(study.repo, "a change to the code")
    with pytest.raises(HeldOutLocked, match="changed since the code freeze"):
        sm.require_code_freeze(study)


def test_the_configuration_may_change_only_in_its_placeholders(tmp_path: Path) -> None:
    study = _repository(tmp_path)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=study.repo, capture_output=True,
                          text=True, check=True).stdout.strip()  # fmt: skip
    config = Path(study.repo) / rg.CONFIG_PATH
    text = config.read_text("utf-8")
    config.write_text(text.replace("  date: null  #", "  date: 2026-10-28  #"), encoding="utf-8")
    sm.require_config_pinned(study, head)
    config.write_text(text.replace("clip_btc: 0.010", "clip_btc: 0.020"), encoding="utf-8")
    with pytest.raises(HeldOutLocked):
        sm.require_config_pinned(study, head)


def test_the_stages_run_in_order(tmp_path: Path) -> None:
    study = _repository(tmp_path)
    with pytest.raises(StageOrderError):
        run_stage(study, "E")
    with pytest.raises(StageOrderError):
        run_stage(study, "1")
    study.state.begin("0")
    study.state.finish("0", {"passes": False})
    with pytest.raises(StageOrderError, match="did not pass"):
        run_stage(study, "1")
