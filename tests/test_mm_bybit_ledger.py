"""H's ledger: the only way to an access for H, and what it refuses."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from trading_research.mm_bybit import access as acc
from trading_research.mm_bybit import archive, config, ledger

REPO = Path(__file__).resolve().parent.parent


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def registered(tmp_path: Path) -> Path:
    """A repository holding the registration as registered, committed."""
    repo = tmp_path / "repo"
    for rel in (config.CONFIG_PATH, config.DOCUMENT_PATH):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, repo / rel)
    (repo / ".gitignore").write_text("/data/\n")
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "test")
    git(repo, "config", "user.email", "ledger-test@localhost")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "registration")
    return repo


def amended(repo: Path) -> str:
    """Freeze every placeholder, record the hash in the document, commit; return it."""
    freeze_commit = git(repo, "rev-parse", "HEAD").strip()
    path = repo / config.CONFIG_PATH
    names = [p[-1] for p in config.placeholder_lines(path.read_text())]
    values: dict[str, object] = {name: {"stand_in": True} for name in names}
    values["compute"] = {"code_freeze_commit": freeze_commit, "d_runs": 1}
    path.write_text(config.freeze_text(path.read_text(), values))
    sha = config.sha256_bytes(path.read_bytes())
    document = repo / config.DOCUMENT_PATH
    document.write_text(document.read_text() + f"\nFrozen configuration sha256: `{sha}`\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "amendment")
    return sha


def test_h_opens_once_records_and_commits_the_entry_and_a_second_read_needs_force(
    tmp_path: Path,
) -> None:
    repo = registered(tmp_path)
    sha = amended(repo)
    head = git(repo, "rev-parse", "HEAD").strip()
    access = ledger.open_h(repo, tmp_path / "data")
    assert access.permits(acc.BLOCKS["H"][0]) and not access.permits(acc.BLOCKS["D"][0])
    assert not access.second_read
    entry = ledger.read(repo)["reads"][0]
    assert entry["commit"] == head and entry["configuration_sha256"] == sha
    assert "H opened at" in git(repo, "log", "-1", "--format=%s")
    assert git(repo, "status", "--porcelain") == ""
    with pytest.raises(ledger.HeldOutLocked, match="second read"):
        ledger.open_h(repo, tmp_path / "data")
    again = ledger.open_h(repo, tmp_path / "data", force=True)
    assert again.second_read and ledger.read(repo)["reads"][-1]["second_read"]


def test_h_stays_locked_before_the_amendment(tmp_path: Path) -> None:
    repo = registered(tmp_path)
    with pytest.raises(ledger.HeldOutLocked, match="no frozen configuration"):
        ledger.open_h(repo, tmp_path / "data")


def test_h_stays_locked_on_a_dirty_tree_or_a_changed_configuration(tmp_path: Path) -> None:
    repo = registered(tmp_path)
    amended(repo)
    (repo / "notes.txt").write_text("a change\n")
    with pytest.raises(ledger.HeldOutLocked, match="not clean"):
        ledger.open_h(repo, tmp_path / "data")
    (repo / "notes.txt").unlink()
    path = repo / config.CONFIG_PATH
    path.write_text(path.read_text().replace("pause_s: 5", "pause_s: 6"))
    with pytest.raises(ledger.HeldOutLocked, match="not the frozen one"):
        ledger.open_h(repo, tmp_path / "data")


def test_h_stays_locked_when_study_code_changed_after_the_freeze(tmp_path: Path) -> None:
    repo = registered(tmp_path)
    amended(repo)
    code = repo / "src/trading_research/mm_bybit/late.py"
    code.parent.mkdir(parents=True)
    code.write_text("CHANGED = True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "a late change")
    with pytest.raises(ledger.HeldOutLocked, match="study code changed"):
        ledger.open_h(repo, tmp_path / "data")


def test_h_stays_locked_while_a_file_of_h_is_on_disk(tmp_path: Path) -> None:
    repo = registered(tmp_path)
    amended(repo)
    data = tmp_path / "data"
    archive.write_status(data, acc.BLOCKS["H"][0], "no_prints_file", "HTTP 404")
    with pytest.raises(ledger.HeldOutLocked, match="already on disk"):
        ledger.open_h(repo, data)


def test_the_close_writes_the_results_sha256_into_the_last_read(tmp_path: Path) -> None:
    repo = registered(tmp_path)
    amended(repo)
    ledger.open_h(repo, tmp_path / "data")
    results = repo / "experiments/results/mm_bybit_2021_H.json"
    results.write_text("{}\n")
    ledger.close_h(repo, results)
    assert ledger.read(repo)["reads"][-1]["results_sha256"] == config.sha256_bytes(b"{}\n")
