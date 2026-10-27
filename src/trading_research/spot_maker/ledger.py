"""The maker study's ledger: the one gate to H and P, and the read sessions it binds.

On the spot study's pattern (``spot/ledger.py``), bound to this study's
registration ("Access"); nothing is pushed anywhere.

* :meth:`MakerLedger.open` checks a block's preconditions and appends its open
  entry (block, the session's commit, the configuration's sha256, the
  environment fingerprint, the UTC time) to
  ``experiments/results/spot_maker_2026_ledger.json``; the entry is committed
  before the block is copied.
* :meth:`MakerLedger.session` starts every process of a read: refused unless the
  open entry is in HEAD's ledger and still open, nothing but the ledger changed
  since the session's commit, the configuration is the one the block was opened
  with, the code is the frozen code and the environment the open entry's. It
  returns a :class:`ReadSession` whose access carries the seal.

**H opens only if** the amendment records the configuration's sha256 and the
file has it and is committed unchanged; the code-freeze commit ``compute.record``
names is the note's latest, HEAD descends from it and nothing of the study code
changed since; the configuration changed since only in its placeholders and the
document only by appending one frozen-hash line and no null-hash line; the tree
is clean but for the ledger; the signal's copies are at their sha256s; the run
log is at its frozen sha256 and row count; and no file of H is on disk.

**P opens only** for H's candidates, by its deadline, once H's results file is
committed unchanged, 2026-12-31 is published and, since H's read, only results,
docs and the README changed but for the fee amendment
(:mod:`~trading_research.spot_maker.prospective`); every hash is then read
without the fee amendment's intervals.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_research.market_making.prereg import HeldOutLocked, sha256_bytes, unfrozen_text
from trading_research.spot.ledger import environment_fingerprint
from trading_research.spot.state import write_json
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import prospective as pr
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.signal import SignalBinding, verify_signal

FIRST_READ, SECOND_READ = "first read", "second read"
MAX_RESUMPTIONS = 5


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_freeze_commit(document: str) -> str | None:
    """The note's latest ``Code-freeze commit:`` line."""
    import re

    found = re.findall(r"Code-freeze commit: `([0-9a-f]{7,40})`", document)
    return found[-1] if found else None


def run_log_digest(repo: Path) -> dict[str, Any]:
    """The run log's sha256 and row count (none: no file)."""
    path = Path(repo) / rg.RUN_LOG_PATH
    if not path.exists():
        return {"sha256": None, "rows": 0}
    return {"sha256": sha256_bytes(path.read_bytes()), "rows": len(json.loads(path.read_text()))}


@dataclass
class MakerLedger:
    """The ledger of a checkout, with the signal binding its copies must have."""

    repo: Path
    binding: SignalBinding
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    opener: Any = None

    @property
    def path(self) -> Path:
        return Path(self.repo) / rg.LEDGER_PATH

    def git(self, *args: str, check: bool = True) -> str:
        done = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True)
        if check and done.returncode != 0:
            raise HeldOutLocked(f"git {' '.join(args)} failed: {done.stderr.strip()[:200]}")
        return done.stdout

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()

    def entries(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {}
        loaded: dict[str, list[dict[str, Any]]] = json.loads(self.path.read_text("utf-8"))
        return loaded

    def committed_entries(self) -> dict[str, list[dict[str, Any]]]:
        done = subprocess.run(["git", "show", f"HEAD:{rg.LEDGER_PATH.as_posix()}"], cwd=self.repo,
                              capture_output=True, text=True)  # fmt: skip
        loaded: dict[str, list[dict[str, Any]]] = json.loads(done.stdout) if done.returncode == 0 else {}
        return loaded

    def write(self, entries: Mapping[str, Any]) -> None:
        write_json(self.path, dict(entries))

    def committed_unchanged(self, relative: Path) -> None:
        if self.git("ls-files", relative.as_posix()).strip() != relative.as_posix():
            raise HeldOutLocked(f"{relative} is not committed")
        if self.git("diff", "--name-only", "HEAD", "--", relative.as_posix()).strip():
            raise HeldOutLocked(f"{relative} differs from HEAD")

    def verify_frozen(self, block: str = "H") -> rg.Registration:
        registration = rg.Registration.load(self.repo)
        registration.require_frozen()
        actual = registration.base_sha256 if block == "P" else registration.sha256
        if registration.amendment_hash() != actual:
            raise HeldOutLocked("the amendment does not record the configuration's sha256")
        for relative in (rg.CONFIG_PATH, rg.DOCUMENT_PATH):
            self.committed_unchanged(relative)
        return registration

    def verify_code_freeze(self, registration: rg.Registration) -> str:
        recorded = registration.value(("compute", "record"))["code_freeze"]["commit"]
        note = Path(self.repo) / rg.CODE_FREEZE_NOTE
        latest = code_freeze_commit(note.read_text("utf-8")) if note.exists() else None
        if latest is None or (not str(recorded).startswith(latest) and not latest.startswith(str(recorded))):
            raise HeldOutLocked(f"the code-freeze commit {recorded} is not the note's {latest}")
        if subprocess.run(["git", "merge-base", "--is-ancestor", str(recorded), "HEAD"],
                          cwd=self.repo).returncode != 0:  # fmt: skip
            raise HeldOutLocked(f"HEAD does not descend from the code-freeze commit {recorded}")
        changed = self.git("diff", "--name-only", str(recorded), "--", *rg.CODE_PATHS).strip()
        if changed:
            raise HeldOutLocked(f"the study code changed since the code freeze: {changed.split()[:3]}")
        return str(recorded)

    def verify_pinned(self, registration: rg.Registration, commit: str) -> None:
        """The configuration changed since ``commit`` only in its placeholders; the
        document only by appending one frozen-hash line and no null-hash line."""
        then = self.git("show", f"{commit}:{rg.CONFIG_PATH.as_posix()}")
        if unfrozen_text(then) != unfrozen_text(registration.base_text or registration.text):
            raise HeldOutLocked("the configuration changed outside its placeholders since the freeze")
        before = self.git("show", f"{commit}:{rg.DOCUMENT_PATH.as_posix()}")
        if not registration.document.startswith(before.rstrip("\n")):
            raise HeldOutLocked("the document changed since the freeze other than by appending")
        added = registration.document[len(before.rstrip("\n")) :]
        if len(rg.FROZEN_HASH_PATTERN.findall(added)) != 1 or rg.NULL_HASH_PATTERN.findall(added):
            raise HeldOutLocked("the appended text must hold one frozen-hash line, no null-hash line")

    def verify_tree(self, since: str | None = None) -> None:
        dirty = [line[3:] for line in self.git("status", "--porcelain").splitlines()]
        moved = self.git("diff", "--name-only", f"{since}..HEAD").split() if since else []
        other = [p for p in [*dirty, *moved] if p != rg.LEDGER_PATH.as_posix()]
        if other:
            raise HeldOutLocked(f"the tree holds changes besides the ledger: {other[:3]}")

    def verify_trials(self, registration: rg.Registration) -> None:
        frozen = registration.value(("compute", "record"))["trials"]
        found = run_log_digest(self.repo)
        if (found["sha256"], found["rows"]) != (frozen["sha256"], frozen["rows"]):
            raise HeldOutLocked(f"the run log is {found}, not the frozen {frozen}")
        manifest = Path(self.repo) / rg.TRIALS_MANIFEST_PATH
        self.committed_unchanged(rg.TRIALS_MANIFEST_PATH)
        if json.loads(manifest.read_text("utf-8")) != frozen:
            raise HeldOutLocked("the run log's manifest is not the frozen one")

    def files_on_disk(self, block: str) -> list[Path]:
        return bk.files_of_days(self.repo, bk.block_days(block))

    def open(self, block: str, *, force: bool = False,
             candidates: tuple[str, ...] = ()) -> dict[str, Any]:  # fmt: skip
        """Check ``block``'s preconditions and append its open entry; return it."""
        if block not in bk.HELD_OUT:
            raise ValueError(f"the ledger opens {bk.HELD_OUT}, not {block!r}")
        registration = self.verify_frozen(block)
        self.verify_tree()
        self.verify_pinned(registration, self.verify_code_freeze(registration))
        committed = self.committed_entries()
        if self.entries() != committed:
            raise HeldOutLocked("the ledger has entries that are not committed")
        if block == "P":
            self.verify_p(committed, candidates, registration)
        verify_signal(self.repo, self.binding)
        self.verify_trials(registration)
        on_disk = self.files_on_disk(block)
        if on_disk:
            raise HeldOutLocked(f"{len(on_disk)} files of {block} are already on disk: {on_disk[:2]}")
        previous = committed.get(block, [])
        if previous and not force:
            raise HeldOutLocked(f"{block} was opened on {previous[0]['utc']}: a second read needs force")
        environment = environment_fingerprint()
        if block == "P" and committed["H"][0]["environment"] != environment:
            raise HeldOutLocked("the environment is not the one H was read in")
        entry: dict[str, Any] = {
            "block": block, "read": SECOND_READ if previous else FIRST_READ,
            "commit": self.head(), "config_sha256": registration.sha256,
            "environment": environment, "utc": _utc(self.clock()),
            "candidates": sorted(candidates), "processes": [], "failures": [], "copies": None,
            "results_sha256": None,
        }  # fmt: skip
        self.write({**committed, block: [*previous, entry]})
        return entry

    def verify_p(self, committed: Mapping[str, Any], candidates: tuple[str, ...],
                 registration: rg.Registration) -> None:  # fmt: skip
        """P only for H's first read's candidates, its results committed unchanged, by
        the deadline, 2026-12-31 published, only results and docs changed since H."""
        if not committed.get("H") or not committed["H"][0].get("results_sha256"):
            raise HeldOutLocked("P opens only after H's results are written and committed")
        h = committed["H"][0]
        path = Path(self.repo) / rg.RESULTS_PATHS["H"]
        self.committed_unchanged(rg.RESULTS_PATHS["H"])
        if sha256_bytes(path.read_bytes()) != h["results_sha256"]:
            raise HeldOutLocked("H's results file is not the one its read wrote")
        statuses = json.loads(path.read_text("utf-8"))["statuses"]
        found = tuple(sorted(s for s, v in statuses.items() if v["status"] == "candidate"))
        if not found or tuple(sorted(candidates)) != found:
            raise HeldOutLocked(f"P opens only for H's candidates {found}, not {candidates}")
        added = self.git("log", "--diff-filter=A", "--format=%cI", "--",
                         rg.RESULTS_PATHS["H"].as_posix()).split()  # fmt: skip
        first = datetime.fromisoformat(added[-1]).astimezone(UTC).date()
        changed = self.git("diff", "--name-only", f"{h['commit']}..HEAD").split()
        h_document = self.git("show", f"{h['commit']}:{rg.DOCUMENT_PATH.as_posix()}")
        pr.check_since_h(changed, registration.base_sha256, h["config_sha256"],
                         registration.document, h_document)  # fmt: skip
        pr.check_open(self.clock().astimezone(UTC).date(), first,
                      lambda: pr.published(bk.BLOCKS["P"][1], self.opener))  # fmt: skip

    def session(self, block: str, *, workers: int = 1) -> ReadSession:
        """Start one process of ``block``'s latest read session; return its access."""
        committed = self.committed_entries()
        if not committed.get(block):
            raise HeldOutLocked(f"{block}'s open entry is not in HEAD's ledger: commit it first")
        entry = committed[block][-1]
        if entry.get("results_sha256"):
            raise HeldOutLocked(f"{block}'s read session is closed")
        self.verify_tree(since=entry["commit"])
        registration = self.verify_frozen(block)
        if registration.sha256 != entry["config_sha256"]:
            raise HeldOutLocked(f"the configuration is not the one {block} was opened with")
        self.verify_pinned(registration, self.verify_code_freeze(registration))
        verify_signal(self.repo, self.binding)
        if environment_fingerprint() != entry["environment"]:
            raise HeldOutLocked("the environment is not the one the block was opened in")
        live = self.entries()
        if live[block][-1]["commit"] != entry["commit"] or live[block][-1].get("results_sha256"):
            raise HeldOutLocked("the ledger on disk does not end with HEAD's open entry")
        live[block][-1]["processes"].append({"utc": _utc(self.clock()), "workers": int(workers)})
        self.write(live)
        key = (block, entry["commit"], entry["config_sha256"], entry["environment"])
        access = bk.MakerAccess(bk.opened_blocks(block), entry["read"], key, seal=bk._SEAL)
        return ReadSession(self, block, access)


@dataclass
class ReadSession:
    """One process of a block's read session."""

    ledger: MakerLedger
    block: str
    access: bk.MakerAccess

    def _update(self, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        entries = self.ledger.entries()
        change(entries[self.block][-1])
        self.ledger.write(entries)
        return entries[self.block][-1]

    def record_copies(self, records: Mapping[str, Any]) -> None:
        digest = sha256_bytes(json.dumps(dict(records), sort_keys=True).encode())
        self._update(lambda e: e.update(copies={"days": len(records), "sha256": digest,
                                                "absent": sorted(d for d, r in records.items()
                                                                 if r["status"] == "absent")}))  # fmt: skip

    def copies_committed(self) -> bool:
        committed = self.ledger.committed_entries().get(self.block, [])
        return bool(committed and committed[-1].get("copies"))

    def record_failure(self, stage: str, exc: BaseException) -> bool:
        """Record a failure; True when the session must close as failed."""
        text = f"{type(exc).__name__}: {exc}"[:300]
        entry = self._update(lambda e: e["failures"].append({"stage": stage, "exception": text}))
        same = [f for f in entry["failures"] if (f["stage"], f["exception"]) == (stage, text)]
        return len(same) >= 2 or len(entry["processes"]) > MAX_RESUMPTIONS

    def close(self, results: Path) -> str:
        digest = sha256_bytes(Path(results).read_bytes())
        self._update(lambda e: e.update(results_sha256=digest, closed=_utc(self.ledger.clock())))
        return digest
