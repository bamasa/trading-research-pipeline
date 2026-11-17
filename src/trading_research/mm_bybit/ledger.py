"""H's ledger: the only way to an access for H, and the record of its read.

:func:`open_h` refuses unless the D amendment records the frozen
configuration's sha256 and the file on disk has it, fully frozen, differing from
the registration only in its placeholders; the tree is clean but for the ledger;
the study code is unchanged since the code-freeze commit the configuration
records; and no file of H is on disk. It then writes the open entry (block,
commit, configuration sha256, UTC time) and commits the ledger locally before
the access is returned. A second read needs ``force=True`` and is stamped.
Nothing is pushed.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from trading_research.market_making.prereg import sha256_bytes
from trading_research.mm_bybit import archive, config, funding
from trading_research.mm_bybit.access import Access, _heldout_access, block_days

LEDGER_PATH = Path("experiments/results/mm_bybit_2021_ledger.json")
CODE_PATHS = ("src/trading_research/mm_bybit", "experiments/mm_bybit_2021.py")


class HeldOutLocked(PermissionError):
    """H may not be opened."""


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if done.returncode:
        raise HeldOutLocked(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def read(repo: Path | str) -> dict[str, Any]:
    path = Path(repo) / LEDGER_PATH
    return json.loads(path.read_text()) if path.exists() else {"reads": []}


def checks(repo: Path | str, data_root: Path | str) -> str:
    """Run every gate; return the frozen configuration's sha256 or raise."""
    root = Path(repo)
    text = (root / config.CONFIG_PATH).read_text()
    document = (root / config.DOCUMENT_PATH).read_text()
    frozen = config.frozen_sha256(document)
    if frozen is None:
        raise HeldOutLocked("the D amendment records no frozen configuration sha256")
    if sha256_bytes(text.encode()) != frozen:
        raise HeldOutLocked("the configuration on disk is not the frozen one")
    config.check_registered(text, document)
    if not config.is_frozen(text):
        raise HeldOutLocked("a placeholder is still null")
    for tracked in (config.CONFIG_PATH, config.DOCUMENT_PATH):
        _git(root, "ls-files", "--error-unmatch", tracked.as_posix())
    committed = _git(root, "show", f"HEAD:{config.CONFIG_PATH.as_posix()}")
    if sha256_bytes(committed.encode()) != frozen:
        raise HeldOutLocked("the committed configuration is not the frozen one")
    dirty = [line[3:] for line in _git(root, "status", "--porcelain").splitlines()]
    if [p for p in dirty if p != LEDGER_PATH.as_posix()]:
        raise HeldOutLocked(f"the tree is not clean: {dirty}")
    freeze = yaml.safe_load(text)["frozen"]["compute"]["code_freeze_commit"]
    changed = _git(root, "diff", "--name-only", f"{freeze}..HEAD", "--", *CODE_PATHS).split()
    if changed:
        raise HeldOutLocked(f"study code changed since the code freeze: {changed}")
    present = [
        d for d in block_days("H")
        if archive.prints_path(data_root, d).exists() or archive.status_path(data_root, d).exists()
    ]  # fmt: skip
    if present or funding.path(data_root, "H").exists():
        raise HeldOutLocked(f"files of H are already on disk: {present[:3]}")
    return frozen


def open_h(repo: Path | str, data_root: Path | str, *, force: bool = False) -> Access:
    """Check every gate, record and commit the open entry, return H's access."""
    root = Path(repo)
    frozen = checks(root, data_root)
    ledger = read(root)
    earlier = [r for r in ledger["reads"] if r["block"] == "H"]
    if earlier and not force:
        raise HeldOutLocked("H has been read; a second read needs force=True")
    entry: dict[str, Any] = {
        "block": "H",
        "commit": _git(root, "rev-parse", "HEAD").strip(),
        "configuration_sha256": frozen,
        "opened_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "second_read": bool(earlier),
    }
    ledger["reads"].append(entry)
    path = root / LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=1) + "\n")
    _git(root, "add", LEDGER_PATH.as_posix())
    _git(
        root, "commit", "-q", "-m", f"MM Bybit 2021 ledger: H opened at {str(entry['commit'])[:7]}"
    )
    return _heldout_access(second_read=bool(earlier))


def close_h(repo: Path | str, results: Path | str) -> None:
    """Write the results file's sha256 into the last H entry."""
    root = Path(repo)
    ledger = read(root)
    ledger["reads"][-1]["results_sha256"] = sha256_bytes(Path(results).read_bytes())
    (root / LEDGER_PATH).write_text(json.dumps(ledger, indent=1) + "\n")
