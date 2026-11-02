"""The entry point end to end on stand-ins: every stage, then H through the ledger.

One dry run is shared by the module. It runs the entry point itself,
``python -m experiments.spot_maker_2026 --dry-run --keep <dir>``, in a fresh
interpreter: a temporary repository holding the registration as registered, a
stand-in of the spot study's checkout (the synthetic venue's prints converted by
the spot study's parser, a stand-in model in place of S25-PF), stages 0, 1, D
and the amendment in order, then H opened, copied, committed, computed and
closed. The assertions read what it left behind, and every gate is tried again
on it.
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
from trading_research.spot_maker import reads as rd
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.context import STAGES, MakerStudy
from trading_research.spot_maker.dry_run import DRY_CALENDAR
from trading_research.spot_maker.ledger import MakerLedger
from trading_research.spot_maker.signal import SignalBinding
from trading_research.spot_maker.standin import STAND_IN_THETA

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dry(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    keep = tmp_path_factory.mktemp("spot_maker_dry_run")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    command = [
        sys.executable,
        "-m",
        "experiments.spot_maker_2026",
        "--dry-run",
        "--keep",
        str(keep),
    ]
    done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=3_000,
                          check=False, env=env)  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    repo = keep / "repo"
    names = {rg.SIGNAL_SLOT_FILE: "slot", rg.SIGNAL_FIT: "fit"}
    digests = {
        names[p.name]: sha256_bytes(p.read_bytes()) for p in (repo / rg.SIGNAL_DIR).iterdir()
    }
    binding = SignalBinding(digests["slot"], digests["fit"], STAND_IN_THETA)
    study = MakerStudy(repo, keep / "source", binding, DRY_CALENDAR, disk=lambda _: 1_000.0)
    return {"report": json.loads(done.stdout), "study": study}


def test_every_stage_completed_in_order_and_stage_0_passed(dry: dict[str, Any]) -> None:
    study, report = dry["study"], dry["report"]
    assert report["stages"] == list(STAGES) and report["stage_0_passes"]
    record = study.state.record("0")
    for day in record["reader_test"].values():
        assert day["checks"]["fill_rule_orders"]["violations"] == 0
        assert day["checks"]["fill_rule_orders"]["orders"] > 0 and day["checks"]["voids"] == []
    assert set(record["timings"]) >= {"copy_s", "features_s_a_day", "runs_s_a_day"}


def test_the_amendment_froze_the_configuration_it_records(dry: dict[str, Any]) -> None:
    study = dry["study"]
    registration = study.registration()
    assert registration.frozen and registration.amendment_hash() == registration.sha256
    record = registration.value(("compute", "record"))
    assert record["trials"]["rows"] == 3 and len(record["disk"]) == 3  # stage 1, D, calibration
    calibration = registration.value(("calibration", "result"))
    assert (
        calibration["status"] == rg.CALIBRATION_NOT_RUN
        and calibration["label"] == rg.CALIBRATION_LABEL
    )
    manifest = json.loads((study.repo / rg.TRIALS_MANIFEST_PATH).read_text("utf-8"))
    assert manifest == record["trials"]


def test_h_was_read_once_through_the_ledger(dry: dict[str, Any]) -> None:
    study, report = dry["study"], dry["report"]
    entries = MakerLedger(study.repo, study.binding).committed_entries()
    (entry,) = entries["H"]
    results = study.repo / rg.RESULTS_PATHS["H"]
    assert entry["results_sha256"] == sha256_bytes(results.read_bytes())
    assert entry["copies"]["days"] == len(DRY_CALENDAR["H"]) and len(entry["processes"]) == 2
    assert set(report["statuses"]) == set(rg.FAMILY)
    page = (study.repo / rg.PAGE_PATH).read_text("utf-8")
    assert "## H (first read)" in page and "Every kill" in page
    assert page.count("(uncalibrated fill rule)") == len(rg.FAMILY)


def test_every_gate_refuses_after_the_read(dry: dict[str, Any]) -> None:
    study = dry["study"]
    with pytest.raises(HeldOutLocked, match="already on disk"):
        rd.open_block(study, "H")
    with pytest.raises(HeldOutLocked, match="closed"):
        rd.read_block(study, "H")
    with pytest.raises(HeldOutLocked, match="candidates"):
        rd.open_block(study, "P")
    wrong = MakerLedger(study.repo, MakerStudy(study.repo).binding)
    with pytest.raises(HeldOutLocked):
        wrong.open("H", force=True)
