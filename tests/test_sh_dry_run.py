"""The entry point end to end on a synthetic venue: every stage, then H, F and P through the ledger.

One dry run is shared by the module. It runs the entry point itself,
``python -m experiments.short_horizon_2025 --dry-run --keep <dir>``, in a fresh
interpreter: the network never shares a process with XGBoost, and this test
process has imported torch for the network's own tests. The dry run builds a
temporary repository holding the registration, answers every request from a
synthetic venue, runs stages 0 to E in order, then opens, commits, reads and
closes each block. The assertions read what it left behind.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trading_research.pipeline import short_horizon_registration as shr
from trading_research.pipeline.short_horizon import runner
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import DRY_RUN
from trading_research.pipeline.short_horizon.state import STAGES, StageOrderError
from trading_research.reporting.headlines import check_status

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None, reason="the network's fits need 'deep'"
    ),
]
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dry(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    keep = tmp_path_factory.mktemp("dry_run")
    command = [sys.executable, "-m", "experiments.short_horizon_2025", "--dry-run",
               "--keep", str(keep), "--workers", "4"]  # fmt: skip
    done = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=1500, check=False
    )
    assert done.returncode == 0, done.stderr[-3000:]
    study = Study(keep / "repo", DRY_RUN)
    return {"report": json.loads(done.stdout), "study": study}


def tracked(repo: Path) -> set[str]:
    done = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True)
    return set(done.stdout.split())


def test_every_stage_completes_in_the_registered_order(dry: dict[str, Any]) -> None:
    study = dry["study"]
    assert dry["report"]["stages"] == list(STAGES)
    assert all(study.state.complete(stage) for stage in STAGES)
    assert study.state.complete("H") and study.state.complete("F")
    assert study.state.complete("P") or dry["report"]["reads"]["P"]["opened"] is False


def test_nothing_was_asked_of_the_network(dry: dict[str, Any]) -> None:
    assert dry["report"]["venue"]["requests"] > 0 and dry["report"]["venue"]["unknown"] == 0


def test_a_complete_stage_returns_its_record_and_computes_nothing(dry: dict[str, Any]) -> None:
    study = dry["study"]
    attempts = study.state.entries()["B"]["attempts"]
    assert runner.run_stage(study, "B") == study.state.record("B")
    assert study.state.entries()["B"]["attempts"] == attempts


def test_a_stage_refuses_to_run_before_the_stages_it_reads(dry: dict[str, Any]) -> None:
    other = Study(dry["study"].repo, DRY_RUN, run=2)
    with pytest.raises(StageOrderError, match="needs stage 0"):
        runner.run_stage(other, "A")
    assert not other.state.entries()


def test_stage_0_timed_the_code_and_passed_the_parity_gate(dry: dict[str, Any]) -> None:
    record = dry["study"].record("0")
    assert record["parity"]["ok"] and record["parity"]["runs"]
    assert record["catalogue_sha256"] == shr.REGISTERED_CATALOGUE_SHA256
    assert record["projected_hours"]["total"] > 0 and record["replay_memory"]["measured"]


def test_the_trial_ledger_is_committed_with_its_sidecar(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon_ledger import (
        trial_ledger_paths,
        trial_ledger_sha256,
    )

    study = dry["study"]
    paths = trial_ledger_paths(study.repo)
    sidecar = json.loads((study.repo / shr.TRIALS_SIDECAR_PATH).read_text(encoding="utf-8"))
    assert sidecar["sha256"] == trial_ledger_sha256(paths) == study.record("D")["ledger_sha256"]
    assert sidecar["n_trials"] == len(study.keys)
    assert {p.relative_to(study.repo).as_posix() for p in paths} <= tracked(study.repo)


def test_stage_d_chose_admitted_and_wrote_its_files(dry: dict[str, Any]) -> None:
    study = dry["study"]
    record = study.record("D")
    assert set(record["statuses_on_D"]) == {"S-C", "S-R"}
    for status in record["statuses_on_D"].values():
        assert status == "admitted" or check_status(status)
    for name in ("leaderboard.csv", "marginals.csv", "nulls.csv"):
        assert study.results(name).exists()


def test_the_amendment_freezes_every_placeholder(dry: dict[str, Any]) -> None:
    from trading_research.market_making.prereg import sha256_bytes

    study = dry["study"]
    record = study.record("E")
    assert record["dry_run"]["ran"] and record["dry_run"]["parity_repeated_ok"]
    registration = shr.Registration.load(repo=study.repo).require_frozen()
    document = (study.repo / shr.DOCUMENT_PATH).read_text(encoding="utf-8")
    assert f"Frozen configuration sha256: `{sha256_bytes(registration.text.encode())}`" in document
    assert registration.weights_sha256 and registration.finalist_files_sha256


def test_each_read_closes_its_session_with_registered_statuses(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon.reads import read_results
    from trading_research.pipeline.short_horizon_ledger import Ledger

    study = dry["study"]
    entries = Ledger(repo=study.repo, upstream=None).committed_entries()
    for block in ("H", "F"):
        entry = entries[block][-1]
        assert entry["read"] == "first read" and entry["fetch"] and entry["results_sha256"]
        results = read_results(study, block)
        assert set(results["statuses"]) == {"S25", "S-C", "S-R", "S-P", "S-M", "S25-Z"}
        for status in results["statuses"].values():
            check_status(status)
        assert results["stamp"] == "first read" and results["sentences"]
