"""A search run on disk: its registration, one directory per step, and the ledger of its one read.

``<run>/registration.json`` is written by the first step before any day of
the dataset is read: the fee, the grids, the methods and models, the execution
family, the kill conditions, the placebos and Holm's m. Its sha256 is checked
by every later step, so nothing that decides a status moves once the search
has looked at data.

``<run>/<n>-<step>/`` holds each step's tables, a ``manifest.json`` (the
pipeline's :class:`~trading_research.pipeline.stages.StageManifest`) and a
``chosen.json``: what the step chose and everything the next step needs. A
step reads only the previous step's ``chosen.json``, so a reader can stop at
any step, look, and run the next one alone.

``<run>/ledger.json`` is the held-out block's ledger, the protocol of
:class:`~trading_research.market_making.prereg.HeldOutLedger` on a search's
own files: the read is recorded before the block is opened, with the
registration's and the frozen choices' sha256; a second read is refused unless
asked for, and then stamped "second read".
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_research.pipeline.stages import StageManifest

#: The steps of a search, in order; a step reads the one before it.
STEPS: tuple[str, ...] = ("instruments", "frequency", "method", "model", "execution", "read")


class SearchError(RuntimeError):
    """A step cannot run: an earlier step is missing, or the registration changed."""


class ReadRefused(SearchError):
    """The held-out block was asked for a second time, or before its choices were frozen."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def head_commit(repo: Path | None = None) -> str | None:
    """The repository's HEAD, or None outside a git checkout."""
    root = repo or Path(__file__).resolve().parents[3]
    done = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return done.stdout.strip() or None


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


@dataclass(frozen=True)
class SearchRun:
    """The directory of one search."""

    root: Path

    @property
    def registration_path(self) -> Path:
        return self.root / "registration.json"

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.json"

    def step_dir(self, step: str) -> Path:
        return self.root / f"{STEPS.index(step) + 1}-{step}"

    # -- the registration --------------------------------------------------

    def register(self, values: dict[str, Any]) -> str:
        """Write the registration once; refuse a different one over it."""
        payload = {**values, "written": utc_now(), "commit": head_commit()}
        if self.registration_path.exists():
            raise SearchError(
                f"{self.registration_path} exists: a search is registered once; start a new run "
                "directory to register another"
            )
        write_json(self.registration_path, payload)
        return sha256_file(self.registration_path)

    def registration(self) -> tuple[dict[str, Any], str]:
        if not self.registration_path.exists():
            raise SearchError(f"no {self.registration_path}: run the instruments step first")
        return json.loads(self.registration_path.read_text()), sha256_file(self.registration_path)

    # -- the steps ---------------------------------------------------------

    def previous(self, step: str) -> dict[str, Any]:
        """The ``chosen.json`` of the step before ``step``, checked against the registration."""
        index = STEPS.index(step)
        before = STEPS[index - 1]
        path = self.step_dir(before) / "chosen.json"
        if not path.exists():
            raise SearchError(f"{step} reads {path}; run `trp search {before}` first")
        chosen: dict[str, Any] = json.loads(path.read_text())
        _, digest = self.registration()
        if chosen.get("registration_sha256") != digest:
            raise SearchError(
                f"{self.registration_path} changed after {before} ran: a registration does not "
                "move once data has been read"
            )
        return chosen

    def complete(self, step: str) -> bool:
        return (self.step_dir(step) / "chosen.json").exists()

    def finish(
        self,
        step: str,
        chosen: dict[str, Any],
        *,
        inputs: dict[str, Any],
        params: dict[str, Any],
        outputs: dict[str, Any],
    ) -> Path:
        """Write the step's ``chosen.json`` and manifest; return its directory."""
        out = self.step_dir(step)
        _, digest = self.registration()
        write_json(out / "chosen.json", {**chosen, "step": step, "registration_sha256": digest})
        StageManifest(stage=f"search {step}", inputs=inputs, params=params, outputs=outputs).write(
            out
        )
        return out

    def status(self) -> dict[str, str]:
        found = {}
        for step in STEPS:
            found[step] = "complete" if self.complete(step) else "not run"
        if self.ledger_path.exists():
            entries = json.loads(self.ledger_path.read_text()).get("H", [])
            found["held-out reads"] = str(len(entries))
        return found

    # -- the ledger --------------------------------------------------------

    def open_held_out(self, frozen: Path, *, second_read: bool = False) -> dict[str, Any]:
        """Record the read before the block is opened; refuse a second unless asked."""
        _, registration = self.registration()
        if not frozen.exists():
            raise ReadRefused(f"no {frozen}: the execution step freezes every choice first")
        ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        previous = ledger.get("H", [])
        if previous and not second_read:
            first = previous[0]
            raise ReadRefused(
                f"the held-out block was read on {first['utc']}; a second read needs "
                "--second-read and is stamped 'second read'"
            )
        entry = {
            "read": "second read" if previous else "first read",
            "utc": utc_now(),
            "registration_sha256": registration,
            "frozen_sha256": sha256_file(frozen),
            "commit": head_commit(),
        }
        ledger["H"] = [*previous, entry]
        write_json(self.ledger_path, ledger)
        return entry

    def close_held_out(self, results: Path) -> None:
        """Add the results file's sha256 to the read just made."""
        ledger = json.loads(self.ledger_path.read_text())
        ledger["H"][-1]["results_sha256"] = sha256_file(results)
        write_json(self.ledger_path, ledger)
