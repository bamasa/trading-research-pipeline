"""The entry point end to end on the synthetic venue: every stage, then H through the ledger.

One dry run is shared by the module. It runs the entry point itself,
``python -m experiments.spot_2026 --dry-run --keep <dir>``, in a fresh
interpreter, since fitting XGBoost in a process that has imported torch is not
safe and this test process may have. The dry run builds a temporary repository
holding the registration as registered, answers every request from the
synthetic venue, runs stages 0, 1, D and the amendment in order, then opens,
commits, reads and closes H (and P when a slot is a candidate on H). The
assertions read what it left behind.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trading_research.market_making.prereg import HeldOutLocked, sha256_bytes
from trading_research.spot import registration as sr
from trading_research.spot import runlog
from trading_research.spot import synthetic as syn
from trading_research.spot.context import SpotStudy
from trading_research.spot.dry_run import binding_of
from trading_research.spot.ledger import file_sha256
from trading_research.spot.plan import DRY_RUN
from trading_research.spot.runner import run_stage
from trading_research.spot.state import STAGES, StageOrderError
from trading_research.spot.statuses import check_status

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dry(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    keep = tmp_path_factory.mktemp("spot_dry_run")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    command = [sys.executable, "-m", "experiments.spot_2026", "--dry-run", "--keep", str(keep)]
    done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=3_000,
                          check=False, env=env)  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    repo = keep / "repo"
    study = SpotStudy(repo, DRY_RUN, binding=binding_of(repo), disk=lambda _: 1_000.0)
    return {"report": json.loads(done.stdout), "study": study}


def test_every_stage_completes_in_order_and_nothing_reached_the_network(
    dry: dict[str, Any],
) -> None:
    study, report = dry["study"], dry["report"]
    assert report["stages"] == list(STAGES)
    assert all(study.state.complete(stage) for stage in (*STAGES, "H"))
    assert report["venue"]["requests"] > 0 and report["venue"]["unknown"] == 0


def test_a_complete_stage_computes_nothing_and_an_early_one_is_refused(dry: dict[str, Any]) -> None:
    study = dry["study"]
    attempts = study.state.entries()["D"]["attempts"]
    assert run_stage(study, "D") == study.state.record("D")
    assert study.state.entries()["D"]["attempts"] == attempts
    other = SpotStudy(study.repo, DRY_RUN, run=2)
    with pytest.raises(StageOrderError, match="needs stage 0"):
        run_stage(other, "1")
    assert not other.state.entries()


def test_the_amendment_froze_the_configuration_and_the_run_log(dry: dict[str, Any]) -> None:
    study = dry["study"]
    registration = sr.Registration.load(repo=study.repo).require_frozen()
    document = (study.repo / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
    assert sr.frozen_hash_line(registration.text) in document
    assert registration.sha256 == dry["report"]["amendment"]["frozen_config_sha256"]
    manifest = runlog.manifest(study.repo)
    assert registration.trials == {"sha256": manifest["sha256"], "rows": manifest["rows"]}
    stages = set(runlog.rows(study.repo)["stage"])
    assert stages == {"1", "D", "amend"}
    assert registration.walk_levels in (10, 200) and registration.max_sequence_gaps is not None
    assert set(registration.frozen_slot_files_sha256) == {
        "S25-PF",
        "S25-BB",
        "S25-T",
        "S25-PB",
        "S25-PFT",
    }


def test_h_was_read_once_through_the_ledger(dry: dict[str, Any]) -> None:
    study = dry["study"]
    ledger = study.ledger()
    entry = ledger.committed_entries()["H"][0]
    results = study.repo / sr.RESULTS_PATHS["H"]
    assert entry["results_sha256"] == file_sha256(results) and entry["fetch"]
    found = json.loads(results.read_text(encoding="utf-8"))
    assert found["stamp"] == "first read" and set(found["statuses"]) == set(sr.FAMILY)
    assert all(check_status(v["status"]) for v in found["statuses"].values())
    sentences = found["measurements"]["sentences"]
    assert len(sentences["each_slot_on_H"]) == 3 and sentences["S25-PF"] and sentences["S25-T"]
    put_back = found["measurements"]["gap_days_put_back"]["S25-BB"]
    assert put_back["put_back"] == 1
    with syn.serving(syn.SpotVenue(syn.SpotMarket(frozenset()))), pytest.raises(HeldOutLocked):
        ledger.open("H")
    assert dry["report"]["H_first_process"]["next"].startswith("commit")


def test_p_follows_h_and_the_page_is_written_from_the_results(dry: dict[str, Any]) -> None:
    study, report = dry["study"], dry["report"]
    on_h = json.loads((study.repo / sr.RESULTS_PATHS["H"]).read_text())
    if not any(v["status"].startswith("candidate") for v in on_h["statuses"].values()):
        assert report["reads"]["P"]["opened"] is False
    assert report["pages"]["written"] and report["pages"]["changed_on_rewrite"] == []
    page = (study.repo / report["pages"]["written"][0]).read_text(encoding="utf-8")
    assert "| S25-PF |" in page and "Edge ladder" in page
    assert sha256_bytes(page.encode())
