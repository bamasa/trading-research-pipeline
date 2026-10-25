"""The spot study's stages, their order and their records: each runs once, after the ones it reads.

The registered order is stage 0 (the reader-test days), stage 1 (W and D, after
the code-freeze note), D (the development procedure: leakage, fits, scoring,
sigma_D and the MDEs), the D amendment, then the reads of H and P. A stage is
**complete** when its record file exists and ``state.json`` holds that file's
sha256: running it again returns the record and computes nothing. A stage that
stopped part way starts again from its own checkpoints under its directory. A
stage refuses to start until every stage before it is complete and unchanged,
and refuses to run again once a later stage has started, since that stage read
its outputs. The 2025 study's stage machine is not reused: it is bound to that
study's stages.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The development stages, in the registered order, and the held-out reads.
STAGES: tuple[str, ...] = ("0", "1", "D", "amend")
READS: tuple[str, ...] = ("H", "P")
ORDER: tuple[str, ...] = (*STAGES, *READS)


class StageOrderError(RuntimeError):
    """A stage was asked for before the stages it reads from, or after a later one."""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value: Any) -> Any:
    """``value`` with numpy scalars and arrays, tuples, dates and non-finite floats
    made JSON- and YAML-ready (a non-finite float becomes None)."""
    import math
    from datetime import date

    import numpy as np

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [plain(v) for v in value]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, date):
        return value.isoformat()
    return value


def to_json(value: Any) -> str:
    return json.dumps(plain(value), indent=1, sort_keys=True, default=str) + "\n"


def write_json(path: Path, value: Any) -> str:
    """Write atomically; the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(to_json(value), encoding="utf-8")
    partial.replace(path)
    return file_sha256(path)


@dataclass(frozen=True)
class StageState:
    """The run's ``state.json`` and the stages' records under ``root``."""

    root: Path
    order: tuple[str, ...] = ORDER

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
        write_json(self.path, dict(entries))

    def complete(self, stage: str) -> bool:
        entry = self.entries().get(stage, {})
        path = self.record_path(stage)
        return bool(entry.get("sha256")) and path.exists() and file_sha256(path) == entry["sha256"]

    def record(self, stage: str) -> dict[str, Any]:
        if not self.complete(stage):
            raise StageOrderError(f"stage {stage} is not complete")
        loaded: dict[str, Any] = json.loads(self.record_path(stage).read_text(encoding="utf-8"))
        return loaded

    def require_ready(self, stage: str) -> None:
        """Every earlier stage complete and unchanged; no later stage started."""
        if stage not in self.order:
            raise StageOrderError(f"unknown stage {stage!r}; the stages are {self.order}")
        position = self.order.index(stage)
        entries = self.entries()
        for earlier in self.order[:position]:
            if not self.complete(earlier):
                state = "changed since it finished" if earlier in entries else "not run"
                raise StageOrderError(f"stage {stage} needs stage {earlier}, which is {state}")
        later = [s for s in self.order[position + 1 :] if s in entries]
        if later:
            raise StageOrderError(
                f"stage {stage} cannot run again: stage {later[0]} has started and read its outputs"
            )

    def begin(self, stage: str) -> None:
        self.require_ready(stage)
        entries = self.entries()
        entry = entries.get(stage, {"attempts": 0})
        entry.update(attempts=int(entry.get("attempts", 0)) + 1, started=utc_now(), sha256=None)
        self._write({**entries, stage: entry})

    def finish(self, stage: str, record: Mapping[str, Any]) -> dict[str, Any]:
        digest = write_json(self.record_path(stage), dict(record))
        entries = self.entries()
        entries[stage].update(finished=utc_now(), sha256=digest)
        self._write(entries)
        return self.record(stage)

    def run(self, stage: str, compute: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
        """``stage``'s record if complete, else ``compute`` after the order checks."""
        if self.complete(stage):
            return self.record(stage)
        self.begin(stage)
        return self.finish(stage, compute())

    def status(self) -> dict[str, str]:
        entries = self.entries()
        out = {}
        for stage in self.order:
            if self.complete(stage):
                out[stage] = f"complete {entries[stage].get('finished', '')}"
            elif stage in entries:
                out[stage] = f"started {entries[stage].get('started', '')}, not complete"
            else:
                out[stage] = "not run"
        return out
