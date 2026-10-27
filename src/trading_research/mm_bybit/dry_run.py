"""The dry run: every stage and H's read end to end on a synthetic venue.

It runs in a temporary git repository holding the registration as registered,
with :class:`~trading_research.mm_bybit.synthetic.FakeArchive` answering every
URL: no request reaches the network and no file of this repository is written.
It is a test of the code, never a result.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.mm_bybit import access as acc
from trading_research.mm_bybit import config, ledger, runner, synthetic


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def venue(n_prints: int = 20_000) -> synthetic.FakeArchive:
    """Synthetic prints for every day of W, D and H, and their settlements."""
    days = [d for block in ("W", "D", "H") for d in acc.block_days(block)]
    files = {
        d: synthetic.csv_gz(synthetic.random_day(d, seed=k, n_prints=n_prints),
                            descending=k % 2 == 1)
        for k, d in enumerate(days)
    }  # fmt: skip
    settlements = pd.concat([synthetic.funding(d) for d in days], ignore_index=True)
    return synthetic.FakeArchive(files, settlements)


def run(source_repo: Path | str, *, keep: Path | None = None, workers: int = 4,
        n_prints: int = 20_000) -> dict[str, Any]:  # fmt: skip
    """Every stage on the synthetic venue; returns the H record."""
    source = Path(source_repo)
    temp = Path(tempfile.mkdtemp(prefix="mm_bybit_dry_"))
    try:
        for rel in (config.CONFIG_PATH, config.DOCUMENT_PATH):
            (temp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source / rel, temp / rel)
        (temp / ".gitignore").write_text("/data/\n")
        _git(temp, "init", "-q")
        _git(temp, "-c", "user.name=dry", "-c", "user.email=dry-run@localhost", "add", ".")
        _git(temp, "-c", "user.name=dry", "-c", "user.email=dry-run@localhost", "commit",
             "-q", "-m", "registration")  # fmt: skip
        data = temp / "data"
        fake = venue(n_prints)
        runner.stage0(temp, data, opener=fake)
        runner.stage1(temp, data, opener=fake)
        runner.develop_d(temp, data, workers=workers)
        freeze_commit = _git(temp, "rev-parse", "HEAD").strip()
        log = temp / runner.RUN_LOG
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("# Dry run log\n\nThe synthetic venue; not a result.\n")
        runner.amend(temp, code_freeze_commit=freeze_commit, day=date(2026, 10, 27))
        _git(temp, "-c", "user.name=dry", "-c", "user.email=dry-run@localhost", "add", ".")
        _git(temp, "-c", "user.name=dry", "-c", "user.email=dry-run@localhost", "commit",
             "-q", "-m", "amendment")  # fmt: skip
        _git(temp, "config", "user.name", "dry")
        _git(temp, "config", "user.email", "dry-run@localhost")
        access = ledger.open_h(temp, data)
        record = runner.read_h(temp, data, access, opener=fake, workers=workers)
        ledger.close_h(temp, runner.record_path(temp, "H"))
        return record
    finally:
        if keep is not None:
            shutil.copytree(temp, keep, dirs_exist_ok=True)
        shutil.rmtree(temp, ignore_errors=True)
