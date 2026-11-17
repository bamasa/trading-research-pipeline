"""The entry point end to end on the synthetic venue: stages 0, 1, D, the amendment, then H.

One dry run is shared by the module. It runs the entry point itself,
``python -m experiments.mm_bybit_2021 --dry-run --keep <dir>``, in a fresh
interpreter: a temporary repository holding the registration as registered, every
request answered by the synthetic venue, H opened through the ledger. The
assertions read what it left behind. It is a test of the code, never a result.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trading_research.mm_bybit import config

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dry(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    keep = tmp_path_factory.mktemp("mm_bybit_dry_run")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    command = [sys.executable, "-m", "experiments.mm_bybit_2021", "--dry-run", "--workers", "4",
               "--keep", str(keep)]  # fmt: skip
    done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=3_000,
                          check=False, env=env)  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    results = keep / "experiments/results"
    return {
        "report": json.loads(done.stdout),
        "repo": keep,
        **{name: json.loads((results / f"mm_bybit_2021_{name}.json").read_text())
           for name in ("stage0", "stage1", "D", "H", "ledger")},
    }  # fmt: skip


def test_every_stage_ran_and_h_was_read_once_through_the_ledger(dry: dict[str, Any]) -> None:
    assert set(dry["report"]) == {"F", "S0"}
    reads = dry["ledger"]["reads"]
    assert len(reads) == 1 and not reads[0]["second_read"]
    results = dry["repo"] / "experiments/results/mm_bybit_2021_H.json"
    assert reads[0]["results_sha256"] == config.sha256_bytes(results.read_bytes())
    log = subprocess.run(["git", "-C", str(dry["repo"]), "log", "--format=%s"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    assert log.splitlines()[0].startswith("MM Bybit 2021 ledger: H opened at")


def test_the_amendment_froze_the_configuration_it_records(dry: dict[str, Any]) -> None:
    repo = dry["repo"]
    text = (repo / config.CONFIG_PATH).read_text()
    document = (repo / config.DOCUMENT_PATH).read_text()
    assert config.is_frozen(text)
    assert config.frozen_sha256(document) == config.sha256_bytes(text.encode())
    config.check_registered(text, document)
    assert dry["H"]["frozen_configuration_sha256"] == config.frozen_sha256(document)


def test_stage_0_proved_the_reader_on_the_reader_test_days_only(dry: dict[str, Any]) -> None:
    found = dry["stage0"]["reader_test_result"]
    assert sorted(found) == ["2020-12-15", "2020-12-16"]
    assert {v["direction"] for v in found.values()} == {"ascending", "descending"}
    assert all(v["status"] == "usable" and v["tick"] == 0.5 for v in found.values())


def test_d_ran_every_cell_and_the_choice_is_recorded(dry: dict[str, Any]) -> None:
    d = dry["D"]
    assert d["usable_days"] == 90 and len(d["cells"]) == 40
    assert d["chosen_cell"] in d["cells"]
    assert all("touch_mean_y" in c and "latency10_mean_y" in c for c in d["cells"].values())
    assert dry["stage1"]["sigma_ref_bp"] > 0


def test_h_reports_every_status_kill_and_measurement(dry: dict[str, Any]) -> None:
    h = dry["H"]
    assert h["usable_days"] == 84 and h["stamp"] == "first read"
    s0 = h["primaries"]["S0"]
    assert next(k["kill"] for k in s0["kills"]) == "K1"
    m = s0["measurements"]
    for key in ("net_bp_per_fill", "ex_rebate_bp_per_fill", "making_bp_per_fill",
                "mean_negative_control_y", "touch_mean_y", "latency10_mean_y",
                "stress_2021-05-19", "y_by_month", "markout_bp", "market_markout_bp"):  # fmt: skip
        assert key in m
    assert len(s0["sentences"]) == 3
    assert (dry["repo"] / "experiments/results/mm_bybit_2021_H_table.csv").exists()


def test_no_raw_file_is_left_on_disk(dry: dict[str, Any]) -> None:
    data = dry["repo"] / "data"
    assert not [p for p in data.rglob("*") if p.suffix in (".gz", ".csv")]
