"""The 2025 study's ledger: opening H, and the read session it binds.

Each test builds a small repository holding the frozen configuration, the
registration with its amendment, the frozen files and the trial ledger, and
commits it; nothing touches the real repository or any remote.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from trading_research.data import study_access as sa
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.pipeline import short_horizon_ledger as shl
from trading_research.pipeline import short_horizon_registration as shr

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / shr.CONFIG_PATH).read_text(encoding="utf-8")
DOCUMENT = (ROOT / shr.DOCUMENT_PATH).read_text(encoding="utf-8")
CELL = "hold_s=50|trades_per_day=240|label_fee_bp=0"
FROZEN_FILES = {
    shl.WEIGHTS_DIR / "gbm_h50": b"weights",
    shl.FINALISTS_DIR / "S25.json": b"{}",
    shl.FITTED_VALUES_PATH: b'{"touch": 1}',
}


def parquet_bytes(rows: int) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table({"run": list(range(rows))}), sink)
    return bytes(sink.getvalue().to_pybytes())


TRIALS = parquet_bytes(10)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Study", "-c", "user.email=study", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


def commit(repo: Path, message: str = "step") -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").strip()


def frozen_values() -> dict[tuple[str, ...], Any]:
    digests = {path: sha(data) for path, data in FROZEN_FILES.items()}
    return {
        ("data", "max_sequence_gaps"): 3,
        ("data", "excluded_days_D"): [],
        ("data", "instrument_spec"): {"tick": 0.1, "lot": 0.001},
        ("features", "selected"): ["ofi_1"],
        ("features", "fitted_values_sha256"): digests[shl.FITTED_VALUES_PATH],
        ("models", "weights_sha256"): {"gbm_h50": digests[shl.WEIGHTS_DIR / "gbm_h50"]},
        ("models", "checkpoint_epoch"): {"tcn_h20": 7},
        ("search", "result_trials"): {
            "N_trials": 2400, "N_eff": 300.0, "R_D": 1,
            "ledger_sha256": sha(TRIALS), "ledger_rows": 10, "hand_runs": [],
        },
        ("search", "result_choice"): {"C*": {"cell": "x"}},
        ("search", "result_admission"): {"S-C": {"decision": "not admitted on D (i)"}},
        ("search", "result_nulls"): {"p": 1},
        ("slots", "S25", "thresholds"): {CELL: {"theta_long": 0.6, "theta_short": 0.6}},
        ("slots", "S-C", "thresholds"): {CELL: {"theta_long": 0.5, "theta_short": 0.5}},
        ("metrics", "mde", "sigma_d"): {"S25": 4.0},
        ("metrics", "mde", "value"): {"S25": 2.0},
        ("measurements", "result_O"): {"A0": "t"},
        ("detector", "flags_D_foldwise"): {"2025-02": []},
        ("detector", "hour_profile"): [1.0] * 24,
        ("detector", "alarm_rate_D"): 0.0,
        ("detector", "alarms_D"): [],
        ("detector", "r_untestable_declared"): False,
        ("compute", "record"): {"workers": 4},
        ("amendment", "date"): date(2026, 11, 1),
        ("artefacts", "frozen_finalist_files_sha256"): {"S25": digests[shl.FINALISTS_DIR / "S25.json"]},
    }  # fmt: skip


def write(repo: Path, relative: Path | str, data: bytes) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, ".gitignore", b"models/\ndata/\n")
    for relative, data in FROZEN_FILES.items():
        write(repo, relative, data)
    write(repo, shr.TRIALS_PATH, TRIALS)
    write(repo, shr.TRIALS_MANIFEST_PATH, json.dumps(manifest()).encode())
    frozen = shr.freeze_text(TEXT, frozen_values())
    write(repo, shr.CONFIG_PATH, frozen.encode())
    amendment = DOCUMENT.replace("None yet.", shr.frozen_hash_line(frozen))
    write(repo, shr.DOCUMENT_PATH, amendment.encode())
    write(repo, "src/study.py", b"x = 1\n")
    commit(repo, "registered and frozen")
    return repo


def manifest(rows: int = 10, digest: str | None = None) -> dict[str, Any]:
    name = shr.TRIALS_PATH.name
    digest = digest or sha(TRIALS)
    part = {"file": name, "sha256": digest, "rows": rows}
    return {"file": name, "parts": [part], "sha256": digest, "rows": rows,
            "catalogue_sha256": shr.REGISTERED_CATALOGUE_SHA256}  # fmt: skip


def ledger(repo: Path, **changes: Any) -> shl.Ledger:
    settings: dict[str, Any] = {"upstream": None, "require_locked_library": False}
    settings["environment"] = lambda: "env-1"
    return shl.Ledger(repo=repo, **{**settings, **changes})


def open_h(repo: Path, **changes: Any) -> dict[str, Any]:
    return ledger(repo, **changes).open("H", catalogue_sha256=shr.REGISTERED_CATALOGUE_SHA256)


def test_h_opens_once_and_records_its_entry(repo: Path) -> None:
    head = git(repo, "rev-parse", "HEAD").strip()
    entry = open_h(repo)
    assert (entry["block"], entry["read"], entry["commit"]) == ("H", "first read", head)
    assert entry["environment"] == "env-1"
    assert entry["config_sha256"] == sha((repo / shr.CONFIG_PATH).read_bytes())
    assert ledger(repo).entries()["H"] == [entry]
    with pytest.raises(HeldOutLocked, match="not committed"):
        open_h(repo)
    commit(repo, "H opened")
    with pytest.raises(HeldOutLocked, match="second read needs force"):
        open_h(repo)
    again = ledger(repo).open("H", catalogue_sha256=shr.REGISTERED_CATALOGUE_SHA256, force=True)
    assert again["read"] == "second read"


def test_h_needs_the_amendment_and_a_clean_tree(repo: Path) -> None:
    write(repo, "notes.txt", b"a note")
    with pytest.raises(HeldOutLocked, match="changes besides the ledger"):
        open_h(repo)
    (repo / "notes.txt").unlink()
    write(repo, shr.DOCUMENT_PATH, DOCUMENT.encode())
    commit(repo, "the amendment withdrawn")
    with pytest.raises(HeldOutLocked, match="no amendment records"):
        open_h(repo)
    with pytest.raises(ValueError, match="opens"):
        ledger(repo).open("D")


def test_h_refuses_frozen_files_that_moved(repo: Path) -> None:
    write(repo, shl.WEIGHTS_DIR / "gbm_h50", b"retrained")
    with pytest.raises(HeldOutLocked, match="gbm_h50 has sha256"):
        open_h(repo)
    write(repo, shl.WEIGHTS_DIR / "gbm_h50", b"weights")
    (repo / shl.FITTED_VALUES_PATH).unlink()
    with pytest.raises(HeldOutLocked, match="missing"):
        open_h(repo)


def test_h_refuses_a_manifest_unlike_the_amendments(repo: Path) -> None:
    assert not git(repo, "ls-files", shr.TRIALS_PATH.as_posix())  # the parts are never committed
    write(repo, shr.TRIALS_MANIFEST_PATH, json.dumps(manifest(rows=11)).encode())
    commit(repo, "a row more")
    with pytest.raises(HeldOutLocked, match="manifest records"):
        open_h(repo)


def test_h_checks_the_parts_on_disk_against_the_committed_manifest(repo: Path) -> None:
    part = repo / shr.TRIALS_PATH
    part.write_bytes(parquet_bytes(11))
    with pytest.raises(HeldOutLocked, match="differs from its sha256"):
        open_h(repo)
    part.unlink()
    with pytest.raises(HeldOutLocked, match="not on disk"):
        open_h(repo)
    part.write_bytes(TRIALS)
    extra = write(repo, part.with_name(f"{part.stem}.part001.parquet"), TRIALS)
    with pytest.raises(HeldOutLocked, match="does not list"):
        open_h(repo)
    extra.unlink()
    assert open_h(repo)["block"] == "H"


def test_h_refuses_a_misrecorded_part_and_a_missing_manifest(repo: Path) -> None:
    record = manifest()
    record["parts"][0]["rows"] = 9
    write(repo, shr.TRIALS_MANIFEST_PATH, json.dumps(record).encode())
    commit(repo, "a part's rows misrecorded")
    with pytest.raises(HeldOutLocked, match="another row count"):
        open_h(repo)
    (repo / shr.TRIALS_MANIFEST_PATH).unlink()
    commit(repo, "no manifest")
    with pytest.raises(HeldOutLocked, match="no trial-ledger manifest"):
        open_h(repo)


def test_h_refuses_another_catalogue_and_an_unlocked_library(repo: Path) -> None:
    with pytest.raises(HeldOutLocked, match="catalogue"):
        ledger(repo).open("H", catalogue_sha256="0" * 64)
    with pytest.raises(HeldOutLocked, match="locked commit"):
        open_h(repo, require_locked_library=True)


def test_h_refuses_while_its_files_or_e_bars_are_on_disk(repo: Path) -> None:
    write(repo, "data/book/BTCUSDT/2025-06-30.parquet", b"D")
    bars = write(repo, "data/bars/BTCUSDT/klines-1h/2025-07-03.parquet", b"E")
    with pytest.raises(HeldOutLocked, match="already on disk"):
        open_h(repo)
    bars.unlink()
    write(repo, "data/funding/BTCUSDT/2025-08-20.parquet", b"H")
    with pytest.raises(HeldOutLocked, match="already on disk"):
        open_h(repo)
    assert len(ledger(repo).files_on_disk("H")) == 1
    assert sa.opening_days("H")[sa.FUNDING][-1] == date(2025, 8, 20)


# ---------------------------------------------------------------------------
# Read sessions
# ---------------------------------------------------------------------------


def opened_h(repo: Path) -> str:
    """H opened and its entry committed; returns the session's commit."""
    entry = open_h(repo)
    commit(repo, "H opened")
    return str(entry["commit"])


def test_a_session_starts_only_from_the_committed_entry(repo: Path) -> None:
    open_h(repo)
    with pytest.raises(HeldOutLocked, match="not in HEAD's ledger"):
        ledger(repo).session("H")
    commit(repo, "H opened")
    session = ledger(repo).session("H", workers=4)
    entry = ledger(repo).entries()["H"][-1]
    assert session.key == ("H", entry["commit"], entry["config_sha256"], "env-1")
    assert session.stamp == "first read" and session.access.session == session.key
    assert session.access.allows("BTCUSDT", date(2025, 7, 8), sa.BOOK)
    assert session.access.allows("BTCUSDT", date(2025, 7, 1), sa.BARS)
    assert not session.access.allows("BTCUSDT", date(2025, 8, 21), sa.BOOK)
    assert entry["processes"][0]["workers"] == 4
    assert session.resumptions == 0
    ledger(repo).session("H", workers=2)
    assert session.resumptions == 1


def test_a_resumption_at_another_commit_or_environment_is_refused(repo: Path) -> None:
    opened_h(repo)
    with pytest.raises(HeldOutLocked, match="environment differs"):
        ledger(repo, environment=lambda: "env-2").session("H")
    write(repo, "src/study.py", b"x = 2\n")
    with pytest.raises(HeldOutLocked, match="changes besides the ledger"):
        ledger(repo).session("H")
    commit(repo, "a fix")
    with pytest.raises(HeldOutLocked, match="second read"):
        ledger(repo).session("H")


def test_the_entry_must_have_reached_the_upstream(repo: Path) -> None:
    before = git(repo, "rev-parse", "HEAD").strip()
    opened_h(repo)
    upstream = "refs/remotes/origin/main"
    git(repo, "update-ref", upstream, before)
    with pytest.raises(HeldOutLocked, match="has not reached"):
        ledger(repo, upstream=upstream).session("H")
    with pytest.raises(HeldOutLocked, match="has not reached"):
        ledger(repo, upstream="@{upstream}").session("H")
    git(repo, "update-ref", upstream, "HEAD")
    assert ledger(repo, upstream=upstream).session("H").stamp == "first read"


def test_a_read_that_fails_twice_alike_or_resumes_too_often_must_close(repo: Path) -> None:
    opened_h(repo)
    session = ledger(repo).session("H")
    assert session.record_failure("B", ValueError("bad row")) is None
    assert session.record_failure("A", ValueError("bad row")) is None
    failure = session.record_failure("B", ValueError("bad row"))
    assert failure is not None and failure.stage == "B"
    assert failure.exception == "ValueError: bad row"
    fresh = ledger(repo)
    for _ in range(shl.MAX_RESUMPTIONS):
        fresh.session("H")
    assert session.resumptions == shl.MAX_RESUMPTIONS
    entries = fresh.entries()
    entries["H"][-1]["failures"] = []
    fresh._write(entries)
    assert session.must_close() is None
    fresh.session("H")
    reason = session.must_close()
    assert reason is not None and "resumptions" in reason.reason


def test_the_fetch_is_recorded_and_committed_before_computing(repo: Path) -> None:
    opened_h(repo)
    session = ledger(repo).session("H")
    with pytest.raises(ValueError, match="download time"):
        session.record_fetch({"book/2025-07-08": {"sha256": "a" * 64}})
    record = {"sha256": "a" * 64, "downloaded_at": "2026-11-02T10:00:00Z"}
    session.record_fetch({"book/2025-07-08": record})
    with pytest.raises(HeldOutLocked, match="fetch records are not committed"):
        session.require_fetch_committed()
    commit(repo, "H fetched")
    session.require_fetch_committed()
    assert ledger(repo).entries()["H"][-1]["fetch"] == {"book/2025-07-08": record}


def close_h(repo: Path) -> str:
    opened_h(repo)
    session = ledger(repo).session("H")
    write(repo, shr.RESULTS_PATHS["H"], b'{"S25": "killed"}')
    digest = session.close()
    commit(repo, "H read")
    return digest


def test_f_opens_after_h_closes_with_only_results_changed(repo: Path) -> None:
    with pytest.raises(HeldOutLocked, match="no results"):
        ledger(repo).open("F")
    digest = close_h(repo)
    assert ledger(repo).entries()["H"][-1]["results_sha256"] == digest
    with pytest.raises(HeldOutLocked, match="session is closed"):
        ledger(repo).session("H")
    write(repo, "docs/notes.md", b"what H showed")
    write(repo, "README.md", b"readme")
    commit(repo, "reported")
    entry = ledger(repo).open("F")
    assert (entry["block"], entry["read"]) == ("F", "first read")
    git(repo, "checkout", "-q", "--", ".")
    (repo / shr.LEDGER_PATH).write_text(git(repo, "show", f"HEAD:{shr.LEDGER_PATH}"))
    write(repo, "src/study.py", b"x = 3\n")
    commit(repo, "a change after H")
    with pytest.raises(HeldOutLocked, match="changed since H's read"):
        ledger(repo).open("F")


def test_f_refuses_a_results_file_changed_after_the_read(repo: Path) -> None:
    close_h(repo)
    write(repo, shr.RESULTS_PATHS["H"], b'{"S25": "candidate"}')
    commit(repo, "edited")
    with pytest.raises(HeldOutLocked, match="changed after H's read closed"):
        ledger(repo).open("F")


def test_p_opens_after_f_and_only_for_a_qualifying_slot(repo: Path) -> None:
    close_h(repo)
    ledger(repo).open("F")
    commit(repo, "F opened")
    session = ledger(repo).session("F")
    assert session.access.allows("BTCUSDT", date(2025, 11, 16), sa.BOOK)
    assert not session.access.allows("BTCUSDT", date(2025, 11, 17), sa.BOOK)
    write(repo, shr.RESULTS_PATHS["F"], b'{"S-C": "candidate"}')
    session.close()
    commit(repo, "F read")
    with pytest.raises(HeldOutLocked, match="qualifies"):
        ledger(repo).open("P")
    entry = ledger(repo).open("P", qualifying=["S-C"])
    assert entry["qualifying"] == ["S-C"]


def test_only_results_docs_outside_the_registration_and_the_readme_may_change() -> None:
    assert shl._diff_allowed_after_a_read("experiments/results/x.json")
    assert shl._diff_allowed_after_a_read("docs/short_horizon_2025_defects.md")
    assert shl._diff_allowed_after_a_read("README.md")
    assert not shl._diff_allowed_after_a_read("docs/preregistration/short_horizon_2025.md")
    assert not shl._diff_allowed_after_a_read("src/trading_research/x.py")


def test_the_environment_fingerprint_and_the_trial_ledger_parts(tmp_path: Path) -> None:
    fingerprint = shl.environment_fingerprint()
    assert len(fingerprint) == 64 and fingerprint == shl.environment_fingerprint()
    assert shl.library_commit("pytest") is None
    assert shl.library_commit("no-such-distribution") is None
    parts = [write(tmp_path, f"t.part{k}.parquet", bytes([k]) * 3) for k in range(3)]
    found = shl.trial_ledger_parts(tmp_path, Path("t.parquet"))
    assert found == parts
    assert shl.trial_ledger_sha256(found) == sha(b"".join(p.read_bytes() for p in parts))
    write(tmp_path, "t.parquet", b"one")
    assert shl.trial_ledger_parts(tmp_path, Path("t.parquet")) == [tmp_path / "t.parquet", *parts]
    assert shl.parquet_rows(write(tmp_path, "r.parquet", TRIALS)) == 10


def test_a_process_refuses_frozen_files_that_moved_after_the_open(repo: Path) -> None:
    opened_h(repo)
    write(repo, shl.FINALISTS_DIR / "S25.json", b'{"theta": 0.7}')
    with pytest.raises(HeldOutLocked, match=r"S25\.json has sha256"):
        ledger(repo).session("H")
