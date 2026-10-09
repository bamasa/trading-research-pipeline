"""The stages' order and record: each runs once, after the stages it reads from.

The registered order is 0, 1, A, B, D, C, E, then the block reads H, F and P.
A stage is **complete** when its record file exists and its entry in
``state.json`` holds that file's sha256; running a complete stage again
returns its record and computes nothing (idempotent). A stage that stopped part
way is started again from its own checkpoints, which each stage keeps under its
directory (resumable). A stage refuses to start until every stage before it is
complete with its record unchanged, and refuses to run again once a later stage
has started, since that stage read its outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The development stages, in the registered order.
STAGES: tuple[str, ...] = ("0", "1", "A", "B", "D", "C", "E")
#: The held-out reads, each after the one before it.
READS: tuple[str, ...] = ("H", "F", "P")
ORDER: tuple[str, ...] = (*STAGES, *READS)


class StageOrderError(RuntimeError):
    """A stage was asked for before the stages it reads from, or after a later one."""


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


@dataclass(frozen=True)
class StageState:
    """The run's ``state.json`` and the stages' record files under ``root``."""

    root: Path

    @property
    def path(self) -> Path:
        return self.root / "state.json"

    def record_path(self, stage: str) -> Path:
        return self.root / stage / "record.json"

    def entries(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        loaded: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        return loaded

    def _write(self, entries: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name("state.json.writing")
        temporary.write_text(_json(dict(entries)), encoding="utf-8")
        temporary.replace(self.path)

    def complete(self, stage: str) -> bool:
        """The stage finished and its record is the one it finished with."""
        entry = self.entries().get(stage, {})
        path = self.record_path(stage)
        return bool(entry.get("sha256")) and path.exists() and _sha256(path) == entry["sha256"]

    def record(self, stage: str) -> dict[str, Any]:
        """A complete stage's record; refused for a stage that is not complete."""
        if not self.complete(stage):
            raise StageOrderError(f"stage {stage} is not complete")
        loaded: dict[str, Any] = json.loads(self.record_path(stage).read_text(encoding="utf-8"))
        return loaded

    def require_ready(self, stage: str) -> None:
        """Every earlier stage complete and unchanged; no later stage started."""
        if stage not in ORDER:
            raise StageOrderError(f"unknown stage {stage!r}; the stages are {ORDER}")
        position = ORDER.index(stage)
        entries = self.entries()
        for earlier in ORDER[:position]:
            if not self.complete(earlier):
                state = "changed since it finished" if earlier in entries else "not run"
                raise StageOrderError(f"stage {stage} needs stage {earlier}, which is {state}")
        later = [s for s in ORDER[position + 1 :] if s in entries]
        if later:
            raise StageOrderError(
                f"stage {stage} cannot run again: stage {later[0]} has started and read its outputs"
            )

    def run(self, stage: str, compute: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
        """Run ``stage`` once: its record if complete, else ``compute`` after the checks."""
        if self.complete(stage):
            return self.record(stage)
        self.require_ready(stage)
        entries = self.entries()
        entry = entries.get(stage, {"attempts": 0})
        entry.update(attempts=int(entry.get("attempts", 0)) + 1, started=_utc(), sha256=None)
        self._write({**entries, stage: entry})
        record = dict(compute())
        path = self.record_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json(record), encoding="utf-8")
        entries = self.entries()
        entries[stage].update(finished=_utc(), sha256=_sha256(path))
        self._write(entries)
        return self.record(stage)
