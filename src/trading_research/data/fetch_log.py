"""Every attempt at every day, kept on disk, and the two rules that end the asking.

A day whose download fails is asked for again, with backoff, until it is on
disk. Two terminal rules end that, and nothing else does:

- **unreadable**: the archive arrived complete at least twice with the same
  sha256 and still did not decompress, or it decompressed and its replay
  raised (no usable book rows, a line the decoder cannot parse);
- **unfetchable**: the download still fails after at least
  :data:`UNFETCHABLE_ATTEMPTS` attempts spread over at least
  :data:`UNFETCHABLE_SPAN`.

An HTTP 404 is neither: it is a missing plane, recorded at once.

The rules span days of wall time, longer than any one process should be
trusted to live, so the attempts are written to a JSON file after every one,
and a stage resumed after a restart carries on from the same history.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

#: The day's file is on disk.
ON_DISK = "on_disk"
#: The venue answered 404.
MISSING_PLANE = "missing_plane"
UNREADABLE = "unreadable"
UNFETCHABLE = "unfetchable"
PENDING = "pending"

TERMINAL = frozenset({ON_DISK, MISSING_PLANE, UNREADABLE, UNFETCHABLE})
#: Statuses that exclude a day, each counted per block by reason.
EXCLUDING = frozenset({MISSING_PLANE, UNREADABLE, UNFETCHABLE})

#: Attempt outcomes.
OK = "ok"
TRANSIENT = "transient"
#: Arrived whole (no announced length contradicted) and did not decompress.
CORRUPT = "corrupt"
#: Decompressed, and the replay or parser raised on it.
REPLAY_FAILED = "replay_failed"
MISSING = "missing"
OUTCOMES = frozenset({OK, TRANSIENT, CORRUPT, REPLAY_FAILED, MISSING})

UNFETCHABLE_ATTEMPTS = 10
UNFETCHABLE_SPAN = timedelta(hours=48)

#: The first retry waits a minute; each later one three times longer, at most
#: twelve hours, so ten failures span about two days of a long outage.
FIRST_DELAY_S = 60.0
MAX_DELAY_S = 12 * 3600.0


def backoff_s(failures: int) -> float:
    """Seconds to wait after the ``failures``-th failed attempt."""
    if failures < 1:
        return 0.0
    return min(FIRST_DELAY_S * 3.0 ** (failures - 1), MAX_DELAY_S)


@dataclass
class Attempt:
    at: str  # UTC, ISO 8601
    outcome: str
    detail: str = ""
    sha256: str | None = None

    @property
    def when(self) -> datetime:
        return datetime.fromisoformat(self.at)


@dataclass
class DayRecord:
    """One plane of one day: its attempts, its status and what it produced."""

    plane: str
    day: str  # ISO date
    status: str = PENDING
    attempts: list[Attempt] = field(default_factory=list)
    #: sha256 and UTC download time of the file that was used, and its size.
    sha256: str | None = None
    downloaded_at: str | None = None
    size: int | None = None
    #: What the day produced: rows, sequence gaps, message counts, peak memory.
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status in TERMINAL

    def failures(self) -> list[Attempt]:
        return [a for a in self.attempts if a.outcome in (TRANSIENT, CORRUPT)]

    def corrupt_twice(self) -> bool:
        """The same complete archive failed to decompress at least twice."""
        hashes = Counter(a.sha256 for a in self.attempts if a.outcome == CORRUPT and a.sha256)
        return any(n >= 2 for n in hashes.values())

    def unfetchable(self) -> bool:
        failed = self.failures()
        if len(failed) < UNFETCHABLE_ATTEMPTS:
            return False
        return failed[-1].when - failed[0].when >= UNFETCHABLE_SPAN

    def due(self, now: datetime) -> datetime:
        """When the next attempt may start."""
        failed = self.failures()
        if not failed or self.resolved:
            return now
        return failed[-1].when + timedelta(seconds=backoff_s(len(failed)))

    def add(
        self,
        outcome: str,
        *,
        now: datetime,
        detail: str = "",
        sha256: str | None = None,
        size: int | None = None,
    ) -> str:
        """Record one attempt, apply the rules, and return the day's status."""
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown attempt outcome {outcome!r}")
        stamp = now.astimezone(UTC).isoformat()
        self.attempts.append(Attempt(stamp, outcome, detail, sha256))
        if outcome == OK:
            self.status, self.sha256, self.downloaded_at, self.size = ON_DISK, sha256, stamp, size
        elif outcome == MISSING:
            self.status = MISSING_PLANE
        elif outcome == REPLAY_FAILED or (outcome == CORRUPT and self.corrupt_twice()):
            self.status = UNREADABLE
        elif self.unfetchable():
            self.status = UNFETCHABLE
        return self.status

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DayRecord:
        attempts = [Attempt(**a) for a in raw.get("attempts", [])]
        return cls(**{**raw, "attempts": attempts})


class FetchLog:
    """The records of one fetch, one JSON file, rewritten after every change.

    Keyed by plane and day. The file is the fetch's own account of what it did;
    a ledger that records each file's sha256 and download time reads it from
    here.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records: dict[str, DayRecord] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = {k: DayRecord.from_dict(v) for k, v in raw["records"].items()}

    @staticmethod
    def key(plane: str, day: date) -> str:
        return f"{plane}/{day.isoformat()}"

    def get(self, plane: str, day: date) -> DayRecord:
        key = self.key(plane, day)
        if key not in self.records:
            self.records[key] = DayRecord(plane=plane, day=day.isoformat())
        return self.records[key]

    def save(self) -> None:
        """Write atomically: a crash mid-write leaves the previous file whole."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": {k: asdict(v) for k, v in sorted(self.records.items())}}
        partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.part")
        partial.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        partial.replace(self.path)

    def of(self, plane: str | None = None) -> list[DayRecord]:
        return [r for r in self.records.values() if plane is None or r.plane == plane]

    def counts(self, plane: str | None = None) -> Counter[str]:
        """Days by status: what a block reports as excluded, by reason."""
        return Counter(r.status for r in self.of(plane))

    def excluded(self, plane: str | None = None) -> dict[str, str]:
        """``plane/day`` to its excluding status."""
        return {
            self.key(r.plane, date.fromisoformat(r.day)): r.status
            for r in self.of(plane)
            if r.status in EXCLUDING
        }

    def complete(self, plane: str | None = None) -> bool:
        """True when every recorded day is on disk or ended by a terminal rule."""
        return all(r.resolved for r in self.of(plane))
