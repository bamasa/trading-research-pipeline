"""The spot study's ledger: the one gate to H and P, and the read sessions it binds.

On the 2025 study's pattern, bound to this study's blocks and its registration
(``docs/preregistration/spot_2026.md``, "Access"); nothing is pushed anywhere:

* :meth:`Ledger.open` checks a block's preconditions and appends its **open
  entry** (block, session commit, configuration sha256, environment
  fingerprint, UTC time) to ``experiments/results/spot_2026_ledger.json``. It
  returns no access: the entry is committed before the block's fetch begins.
* :meth:`Ledger.session` starts **every process** of a read session. It refuses
  unless the open entry is in HEAD's ledger, the tree and ``git diff --name-only
  <session's commit>..HEAD`` list nothing but the ledger, the configuration is
  the one the block was opened with, and the environment fingerprint equals H's
  open entry's; it records the process and returns a :class:`ReadSession` whose
  :class:`~trading_research.spot.access.SpotAccess` the readers require.
* The session records the fetch's per-file records and its failures (the same
  stage failing again with the same exception, or five resumptions unfinished,
  closes it as failed) and, at the end, the results file's sha256.

**H** opens only if the amendment records the configuration's sha256, the file
on disk has it and is committed; the study code is the code the amendment froze
(the code-freeze commit its compute record names, nothing of the code changed
since), the configuration changed since that commit only in its placeholders and
the document only by appending one frozen-hash line and no null-hash line (both
checked at every open and every process); the tree is clean but for the ledger; every
frozen fit, frozen slot file and the fitted-values file is at the amendment's
sha256; the 2025 study's files S25-T loads are at the sha256s its configuration
records and that configuration at its frozen sha256; the run log's manifest is
committed with the amendment's sha256 and row count and its parts on disk
(under ``data/``, never committed) match it; and no file of H is on disk.

**P** opens only for a candidate on H, by P's deadline, once H's results file is
committed unchanged, P's last day is published on both venues and the fee
amendment covers P's days; since H's read, only ``README.md``, paths under
``experiments/results/`` and paths under ``docs/`` outside
``docs/preregistration/`` may change, except that the fee amendment adds P's
intervals to the configuration and appends to the document's amendments, the
configuration without them having H's sha256.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from trading_research.market_making.prereg import (
    HeldOutLocked,
    PreRegistrationError,
    unfrozen_text,
)
from trading_research.spot import access as sa
from trading_research.spot import code_freeze as cf
from trading_research.spot import registration as sr
from trading_research.spot import statuses as st

#: Where this study's frozen files live (the ignored ``models/`` directory).
FROZEN_DIR = Path("models/spot_2026")
WEIGHTS_DIR = FROZEN_DIR / "weights"
SLOT_FILES_DIR = FROZEN_DIR / "slots"
FITTED_VALUES_PATH = FROZEN_DIR / "fitted_values.json"

#: A read that has not finished after this many resumptions closes as failed.
MAX_RESUMPTIONS = 5
FIRST_READ, SECOND_READ = "first read", "second read"
#: The planes whose P days must be published before P opens.
PLANES_OF = {name: i.planes for name, i in sa.INSTRUMENTS.items()}


def _now() -> datetime:
    return datetime.now(UTC)


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HeldOutLocked(f"git {' '.join(args)} failed in {repo}: {exc}") from exc
    return done.stdout


def _origin(distribution: Any) -> str:
    """How a distribution was installed: its VCS commit, ``editable`` or ``local``."""
    text = distribution.read_text("direct_url.json")
    if not text:
        return ""
    info = json.loads(text)
    vcs = info.get("vcs_info")
    if isinstance(vcs, Mapping):
        return f"@{vcs.get('commit_id', '')}"
    return "@editable" if info.get("dir_info", {}).get("editable") else "@local"


def environment_fingerprint() -> str:
    """The sha256 of the installed distributions, each ``name==version`` with its
    VCS commit when installed from one (no path enters it)."""
    from importlib.metadata import distributions

    lines = set()
    for distribution in distributions():
        name = str(distribution.metadata["Name"]).lower().replace("_", "-")
        lines.add(f"{name}=={distribution.version}{_origin(distribution)}")
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()


def run_log_parts(repo: Path, trials: Path = sr.TRIALS_PATH) -> list[Path]:
    """The run log's files on disk: the one file, or its parts ``<stem>.part<k>.parquet``."""
    single = repo / trials
    found = [single] if single.exists() else []
    return found + sorted(single.parent.glob(f"{single.stem}.part*{single.suffix}"))


def run_log_sha256(paths: Sequence[Path]) -> str:
    """The sha256 of the parts' bytes in order (the file's own sha256 for one part)."""
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    """A parquet file's row count, from its footer."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    return int(pq.ParquetFile(path).metadata.num_rows)


def _porcelain_paths(text: str) -> list[str]:
    return [
        line[3:].split(" -> ")[-1].strip().strip('"') for line in text.splitlines() if line.strip()
    ]


def _allowed_after_h(path: str) -> bool:
    """What may change between H's read and P's open: results, docs outside the
    registration, and the README."""
    if path == "README.md" or path.startswith("experiments/results/"):
        return True
    return path.startswith("docs/") and not path.startswith("docs/preregistration/")


_SLOT_FILE_2025 = sr.FROZEN_DIR_2025 / "finalists" / "S25.json"
_FITTED_2025 = sr.FROZEN_DIR_2025 / "fitted_values.json"


def files_2025() -> dict[Path, str]:
    """The 2025 study's files S25-T loads, each with the sha256 this registration binds."""
    return {
        _SLOT_FILE_2025: sr.FROZEN_FILES_2025["slot_file_S25"],
        _FITTED_2025: sr.FROZEN_FILES_2025["fitted_values"],
        **{sr.FROZEN_DIR_2025 / "weights" / k: v for k, v in sr.NEIGHBOURHOOD_FITS_2025.items()},
    }


def copy_frozen_2025(
    source: Path, target: Path, wanted: Mapping[Path, str] | None = None
) -> list[Path]:
    """Copy the 2025 study's files S25-T loads from the checkout that holds them,
    each sha256 checked before and after the copy; return the copies."""
    wanted = files_2025() if wanted is None else wanted
    copies = []
    for relative, digest in wanted.items():
        origin, copy = source / relative, target / relative
        if not origin.is_file() or file_sha256(origin) != digest:
            raise HeldOutLocked(
                f"{relative} in {source} is missing or not at its registered sha256"
            )
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, copy)
        if file_sha256(copy) != digest:
            raise HeldOutLocked(f"the copy of {relative} differs from its registered sha256")
        copies.append(copy)
    return copies


@dataclass
class Ledger:
    """The study's one gate to H and P. Paths are relative to ``repo``.

    ``frozen_2025`` is the checkout holding the 2025 study's frozen files (this
    repository when None); ``config_2025_sha256`` and ``files_2025`` are the
    hashes the registration binds them to; ``environment`` computes the
    fingerprint and ``clock`` the time. Each is replaceable in tests.
    """

    repo: Path = Path()
    config: Path = sr.CONFIG_PATH
    document: Path = sr.DOCUMENT_PATH
    ledger: Path = sr.LEDGER_PATH
    trials: Path = sr.TRIALS_PATH
    trials_manifest: Path = sr.TRIALS_MANIFEST_PATH
    weights_dir: Path = WEIGHTS_DIR
    slot_files_dir: Path = SLOT_FILES_DIR
    fitted_values: Path = FITTED_VALUES_PATH
    config_2025: Path = sr.CONFIG_2025_PATH
    config_2025_sha256: str = sr.CONFIG_2025_SHA256
    files_2025: Mapping[Path, str] = field(default_factory=files_2025)
    frozen_2025: Path | None = None
    data: Path = Path("data")
    environment: Callable[[], str] = field(default=environment_fingerprint, repr=False)
    clock: Callable[[], datetime] = field(default=_now, repr=False)

    # -- the record -------------------------------------------------------------

    def entries(self) -> dict[str, list[dict[str, Any]]]:
        """The ledger as it is on disk (committed or not)."""
        path = self.repo / self.ledger
        if not path.exists():
            return {}
        loaded: dict[str, list[dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def committed_entries(self, ref: str = "HEAD") -> dict[str, list[dict[str, Any]]]:
        """The ledger as ``ref`` holds it; empty when ``ref`` has none."""
        try:
            text = _git(self.repo, "show", f"{ref}:{self.ledger.as_posix()}")
        except HeldOutLocked:
            return {}
        loaded: dict[str, list[dict[str, Any]]] = json.loads(text)
        return loaded

    def _write(self, entries: Mapping[str, Any]) -> None:
        path = self.repo / self.ledger
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _head(self) -> str:
        return _git(self.repo, "rev-parse", "HEAD").strip()

    def _tracked(self, path: Path) -> None:
        _git(self.repo, "ls-files", "--error-unmatch", path.as_posix())

    def _committed_unchanged(self, path: Path) -> None:
        self._tracked(path)
        on_disk = (self.repo / path).read_text(encoding="utf-8")
        if _git(self.repo, "show", f"HEAD:{path.as_posix()}") != on_disk:
            raise HeldOutLocked(f"{path} differs from the committed file")

    def today(self) -> date:
        return self.clock().astimezone(UTC).date()

    # -- the checks -------------------------------------------------------------

    def expected_sha256(self) -> str:
        """The frozen configuration's sha256, as the D amendment records it."""
        path = self.repo / self.document
        if not path.exists():
            raise HeldOutLocked(f"no pre-registration document at {self.document}")
        found = sr.FROZEN_HASH_PATTERN.findall(sr.amendments_text(path.read_text(encoding="utf-8")))
        if not found:
            raise HeldOutLocked(
                "no amendment records the frozen configuration's sha256; the D amendment "
                "must be committed before a held-out block is opened"
            )
        if len(set(found)) != 1:
            raise HeldOutLocked(
                f"the amendments record several frozen hashes: {sorted(set(found))}"
            )
        return str(found[0])

    def verify_frozen(self, block: str = "H") -> sr.Registration:
        """The configuration on disk and at HEAD is the frozen one (for P, once the
        fee amendment's intervals are removed), valid and committed."""
        expected = self.expected_sha256()
        try:
            registration = sr.Registration.load(self.config, self.document, repo=self.repo)
            registration.require_frozen()
        except PreRegistrationError as exc:
            raise HeldOutLocked(f"the frozen configuration is not valid: {exc}") from exc
        if block == "H" and registration.p_intervals:
            raise HeldOutLocked("the configuration holds P's intervals before H is read")
        actual = registration.base_sha256 if block == "P" else registration.sha256
        if actual != expected:
            raise HeldOutLocked(
                f"{self.config} has sha256 {actual}, but the amendment froze {expected}"
            )
        self._tracked(self.document)
        self._committed_unchanged(self.config)
        return registration

    def verify_code_freeze(self, registration: sr.Registration) -> str:
        """The study code is the code the amendment froze: the code-freeze commit the
        frozen configuration records is the note's (a later line only with an
        amendment naming it), HEAD descends from it, and nothing of ``src/``,
        ``experiments/spot_2026.py``, ``pyproject.toml`` or ``uv.lock`` changed since."""
        recorded = registration.code_freeze
        if recorded is None:
            raise HeldOutLocked("the amendment records no code-freeze commit")
        document = (self.repo / self.document).read_text(encoding="utf-8")
        try:
            expected = _git(self.repo, "rev-parse", f"{recorded}^{{commit}}").strip()
            return cf.check(self.repo, _git, sr.amendments_text(document), paths=cf.CODE_ONLY,
                            expected=expected)  # fmt: skip
        except cf.CodeFreezeError as exc:
            raise HeldOutLocked(f"the code freeze does not hold: {exc}") from exc

    def verify_pinned(self, registration: sr.Registration, commit: str) -> None:
        """Since the code-freeze commit the configuration changed only in its
        placeholders (and, for P, by the fee amendment's intervals), and the document
        only by appending, with exactly one frozen-hash line and no null-hash line; the
        null-placeholder hash accepted is the one the document held at that commit."""
        then = _git(self.repo, "show", f"{commit}:{self.config.as_posix()}")
        if unfrozen_text(registration.base_text) != unfrozen_text(then):
            raise HeldOutLocked(
                f"{self.config} changed outside its placeholders since the code-freeze commit "
                f"{commit[:7]}"
            )
        before = _git(self.repo, "show", f"{commit}:{self.document.as_posix()}").rstrip("\n")
        now = (self.repo / self.document).read_text(encoding="utf-8")
        if not now.startswith(before):
            raise HeldOutLocked(
                f"{self.document} changed since the code-freeze commit other than by appending"
            )
        appended = now[len(before) :]
        frozen = sr.FROZEN_HASH_PATTERN.findall(appended)
        if len(frozen) != 1 or sr.NULL_HASH_PATTERN.findall(appended):
            raise HeldOutLocked(
                "since the code-freeze commit the document must gain exactly one frozen "
                f"configuration hash and no null-placeholder hash (found {len(frozen)} and "
                f"{len(sr.NULL_HASH_PATTERN.findall(appended))})"
            )
        try:
            sr.Registration.from_text(registration.text, null_sha256=sr.latest_null_sha256(before))
        except PreRegistrationError as exc:
            raise HeldOutLocked(f"the configuration is not the registered one: {exc}") from exc

    def verify_tree(self, since: str | None = None) -> None:
        """Nothing but the ledger file is changed, untracked or, since ``since``, committed."""
        ledger = self.ledger.as_posix()
        status = _git(self.repo, "status", "--porcelain", "--untracked-files=normal")
        others = [p for p in _porcelain_paths(status) if p != ledger]
        if others:
            raise HeldOutLocked(f"the working tree has changes besides the ledger: {others}")
        if since is not None:
            diff = _git(self.repo, "diff", "--name-only", f"{since}..HEAD").split()
            changed = [p for p in diff if p != ledger]
            if changed:
                raise HeldOutLocked(
                    f"{changed} changed since the session's commit {since[:7]}: a resumption "
                    "at another commit is a second read (open the block again with force)"
                )

    def verify_frozen_files(self, registration: sr.Registration) -> None:
        """Every frozen fit, frozen slot file and the fitted-values file at the
        amendment's sha256."""
        weights, slots = registration.weights_sha256, registration.frozen_slot_files_sha256
        fitted = registration.fitted_values_sha256
        if not weights or not slots or not fitted:
            raise HeldOutLocked("the amendment records no frozen fits, slot files or fitted values")
        files = [(self.weights_dir / key, digest) for key, digest in weights.items()]
        files += [(self.slot_files_dir / f"{slot}.json", digest) for slot, digest in slots.items()]
        files.append((self.fitted_values, fitted))
        self._check_files(self.repo, files)

    def verify_2025_files(self) -> None:
        """The 2025 configuration at its frozen sha256, recording the hashes this
        registration binds, and every 2025 file S25-T loads at them."""
        path = self.repo / self.config_2025
        if not path.is_file() or file_sha256(path) != self.config_2025_sha256:
            raise HeldOutLocked(f"{self.config_2025} is not at its frozen sha256")
        self._committed_unchanged(self.config_2025)
        frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
        records = {
            _SLOT_FILE_2025: frozen["artefacts"]["frozen_finalist_files_sha256"].get("S25"),
            _FITTED_2025: frozen["features"]["fitted_values_sha256"],
            **{
                sr.FROZEN_DIR_2025 / "weights" / k: v
                for k, v in frozen["models"]["weights_sha256"].items()
            },
        }
        differ = sorted(str(p) for p, digest in self.files_2025.items() if records.get(p) != digest)
        if differ:
            raise HeldOutLocked(f"the 2025 configuration records other hashes for {differ}")
        root = self.repo if self.frozen_2025 is None else self.frozen_2025
        self._check_files(root, list(self.files_2025.items()))

    @staticmethod
    def _check_files(root: Path, files: Sequence[tuple[Path, str]]) -> None:
        for relative, digest in files:
            target = root / relative
            if not target.is_file():
                raise HeldOutLocked(f"the frozen file {relative} is missing")
            actual = file_sha256(target)
            if actual != digest:
                raise HeldOutLocked(
                    f"{relative} has sha256 {actual}, the registration records {digest}"
                )

    def verify_trials(self, registration: sr.Registration) -> None:
        """The run log's manifest committed unchanged, at the amendment's sha256 and
        rows, and its parts on disk (under ``data/``, never committed) exactly the
        manifest's: each at its sha256 and row count, in order, no other beside them."""
        recorded = registration.trials
        if recorded is None:
            raise HeldOutLocked("the amendment records no run log")
        relative = self.trials_manifest
        if not (self.repo / relative).is_file():
            raise HeldOutLocked(f"no run-log manifest at {relative}")
        self._committed_unchanged(relative)
        manifest = json.loads((self.repo / relative).read_text(encoding="utf-8"))
        if (manifest.get("sha256"), manifest.get("rows")) != (recorded["sha256"], recorded["rows"]):
            raise HeldOutLocked(
                f"the manifest records ({manifest.get('sha256')}, {manifest.get('rows')}), "
                f"the amendment ({recorded['sha256']}, {recorded['rows']})"
            )
        folder = (self.repo / self.trials).parent
        listed = manifest.get("parts") or []
        parts = [folder / part["file"] for part in listed]
        if not parts:
            raise HeldOutLocked("the manifest lists no part of the run log")
        extra = sorted(set(run_log_parts(self.repo, self.trials)) - set(parts))
        if extra:
            raise HeldOutLocked(f"parts on disk the manifest does not list: {extra[:3]}")
        for part, target in zip(listed, parts, strict=True):
            if not target.is_file():
                raise HeldOutLocked(f"the run log's part {part['file']} is not on disk")
            if file_sha256(target) != part["sha256"] or parquet_rows(target) != part["rows"]:
                raise HeldOutLocked(f"the part {part['file']} differs from its sha256 or row count")
        found = (run_log_sha256(parts), sum(int(part["rows"]) for part in listed))
        if found != (manifest["sha256"], manifest["rows"]):
            raise HeldOutLocked(f"the parts give (sha256, rows) {found}, not the manifest's")

    def files_on_disk(self, block: str) -> list[Path]:
        """Files of ``block`` already under the study's data root."""
        return sa.files_of_days(sa.block_days(block), self.repo / self.data)

    def _environment(self, entries: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
        """This process's fingerprint, which must equal H's open entry's."""
        fingerprint = self.environment()
        first = entries.get("H") or []
        if first and first[0]["environment"] != fingerprint:
            raise HeldOutLocked(
                "the environment differs from the one H's open entry records "
                f"({fingerprint[:12]} against {first[0]['environment'][:12]})"
            )
        return fingerprint

    def h_read(self, entries: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        """H's first read, closed, its results file committed unchanged at the
        sha256 the ledger records."""
        reads = [e for e in entries.get("H", []) if e["read"] == FIRST_READ]
        if not reads or not reads[0].get("results_sha256"):
            raise HeldOutLocked("H's read has no results recorded in HEAD's ledger")
        results = sr.RESULTS_PATHS["H"]
        self._committed_unchanged(results)
        if file_sha256(self.repo / results) != reads[0]["results_sha256"]:
            raise HeldOutLocked(f"{results} changed after H's read closed")
        return reads[0]

    def p_deadline(self) -> date:
        """The later of 2027-01-31 and 30 days after H's results file was first committed."""
        path = sr.RESULTS_PATHS["H"].as_posix()
        dates = _git(self.repo, "log", "--diff-filter=A", "--format=%cI", "--", path).split()
        if not dates:
            raise HeldOutLocked(f"{path} is not committed")
        return st.p_deadline(datetime.fromisoformat(dates[-1]).astimezone(UTC).date())

    def _since_h(self, entry: Mapping[str, Any], registration: sr.Registration) -> None:
        """Only results, docs and the README changed since H's read, but for the fee
        amendment: P's intervals added to the configuration (without them it has
        H's sha256) and the document's text of H's read kept as its start."""
        diff = _git(self.repo, "diff", "--name-only", f"{entry['commit']}..HEAD").split()
        fee = {self.config.as_posix(), self.document.as_posix()}
        outside = [p for p in diff if not _allowed_after_h(p) and p not in fee]
        if outside:
            raise HeldOutLocked(f"{outside} changed since H's read; only results may")
        if registration.base_sha256 != entry["config_sha256"]:
            raise HeldOutLocked("the configuration without P's intervals is not H's")
        before = _git(self.repo, "show", f"{entry['commit']}:{self.document.as_posix()}")
        if not (self.repo / self.document).read_text(encoding="utf-8").startswith(before):
            raise HeldOutLocked("the document changed since H's read other than by appending")

    def verify_p(
        self, candidates: Sequence[str], published: Callable[[str, str, date], bool] | None
    ) -> None:
        """P's own conditions: read once for every candidate on H, by the deadline,
        after P's last day is published on both venues."""
        results = json.loads((self.repo / sr.RESULTS_PATHS["H"]).read_text(encoding="utf-8"))
        on_h = st.candidates_in(results)
        if not on_h or set(candidates) != set(on_h):
            raise HeldOutLocked(
                f"P opens once, for every candidate on H ({on_h}), not for {list(candidates)}"
            )
        if self.today() > self.p_deadline():
            raise HeldOutLocked(f"P's deadline {self.p_deadline()} has passed: no P read opens")
        last = sa.BLOCKS["P"][1]
        if published is None or not all(
            published(name, plane, last) for name, planes in PLANES_OF.items() for plane in planes
        ):
            raise HeldOutLocked(f"P's last day {last} is not published on both venues")

    # -- opening a block ----------------------------------------------------------

    def open(
        self,
        block: str,
        *,
        candidates: Sequence[str] = (),
        published: Callable[[str, str, date], bool] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Check ``block``'s preconditions and append its open entry; return the entry.

        No access is returned: the entry is committed before the block's fetch, and
        :meth:`session` issues the access. ``candidates`` names the slots P is read
        for; ``published(instrument, plane, day)`` says whether a venue has
        published a day (P only). A block already opened is refused unless
        ``force``, and the new entry is then a second read.
        """
        if block not in sa.HELD_OUT:
            raise ValueError(f"the ledger opens {sa.HELD_OUT}, not {block!r}")
        if block == "H" and candidates:
            raise ValueError("candidates are named for P only")
        registration = self.verify_frozen(block)
        self.verify_tree()
        self.verify_pinned(registration, self.verify_code_freeze(registration))
        committed = self.committed_entries()
        if self.entries() != committed:
            raise HeldOutLocked("the ledger has entries that are not committed")
        environment = self._environment(committed)
        if block == "P":
            h_entry = self.h_read(committed)
            self._since_h(h_entry, registration)
            self.verify_p(candidates, published)
            uncovered = [
                (name, day) for name in sa.INSTRUMENTS for day in sa.block_days("P")
                if not registration.fee_covered(name, day)
            ]  # fmt: skip
            if uncovered:
                raise HeldOutLocked(f"the fee amendment does not cover P's days: {uncovered[:3]}")
        self.verify_frozen_files(registration)
        self.verify_2025_files()
        self.verify_trials(registration)
        on_disk = self.files_on_disk(block)
        if on_disk:
            raise HeldOutLocked(
                f"{len(on_disk)} files of {block} are already on disk: {on_disk[:3]}"
            )
        previous = committed.get(block, [])
        if previous and not force:
            first = previous[0]
            raise HeldOutLocked(
                f"{block} was opened on {first['utc']} at commit {first['commit'][:7]}; "
                "a second read needs force=True and is stamped 'second read'"
            )
        entry: dict[str, Any] = {
            "block": block,
            "read": SECOND_READ if previous else FIRST_READ,
            "commit": self._head(),
            "config_sha256": registration.sha256,
            "h_config_sha256": registration.base_sha256,
            "environment": environment,
            "utc": _utc(self.clock()),
            "candidates": sorted(candidates),
            "processes": [],
            "failures": [],
            "fetch": {},
            "results_sha256": None,
        }
        self._write({**committed, block: [*previous, entry]})
        return entry

    # -- a process of a read session ---------------------------------------------

    def session(self, block: str, *, workers: int = 1) -> ReadSession:
        """Start one process of ``block``'s latest read session; return its access.

        Refused unless the open entry is in HEAD's ledger, the session is not
        closed, nothing but the ledger changed since the session's commit, the
        configuration and frozen files are the ones the block was opened with, the
        environment is H's and, for P, its deadline has not passed.
        """
        committed = self.committed_entries()
        if not committed.get(block):
            raise HeldOutLocked(f"{block}'s open entry is not in HEAD's ledger: commit it first")
        entry = committed[block][-1]
        if entry.get("results_sha256"):
            raise HeldOutLocked(f"{block}'s read session is closed: its results file is written")
        self.verify_tree(since=entry["commit"])
        registration = self.verify_frozen(block)
        if registration.sha256 != entry["config_sha256"]:
            raise HeldOutLocked(f"the configuration is not the one {block} was opened with")
        self.verify_pinned(registration, self.verify_code_freeze(registration))
        self.verify_frozen_files(registration)
        self.verify_2025_files()
        environment = self._environment(committed)
        if block == "P" and self.today() > self.p_deadline():
            raise HeldOutLocked("P's deadline has passed: its read is not completed")
        on_disk = self.entries()
        live = on_disk[block][-1]
        if (live["utc"], live["commit"]) != (entry["utc"], entry["commit"]) or live.get(
            "results_sha256"
        ):
            raise HeldOutLocked(
                "the ledger on disk does not end with HEAD's open entry, still open"
            )
        live["processes"].append({"utc": _utc(self.clock()), "workers": int(workers)})
        self._write(on_disk)
        key = (block, entry["commit"], entry["config_sha256"], environment)
        grants = sa.grants_for(sa.opened_blocks(block))
        access = sa.SpotAccess(grants, entry["read"], session=key, seal=sa._SEAL)
        return ReadSession(self, block, key, access)


@dataclass(frozen=True)
class ReadFailure:
    """Why a read must close as failed: the stage reached and the exception."""

    stage: str
    exception: str
    reason: str


@dataclass
class ReadSession:
    """One process of a block's read session.

    ``key`` is (block, commit, configuration sha256, environment fingerprint);
    every output the session writes carries it, with :attr:`stamp`. ``access`` is
    what the study's readers require.
    """

    ledger: Ledger
    block: str
    key: tuple[str, ...]
    access: sa.SpotAccess

    @property
    def stamp(self) -> str:
        return self.access.stamp

    def _update(self, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        entries = self.ledger.entries()
        entry = entries[self.block][-1]
        if (entry["commit"], entry["config_sha256"]) != self.key[1:3]:
            raise HeldOutLocked("the ledger's latest entry is not this session's")
        if entry.get("results_sha256"):
            raise HeldOutLocked(f"{self.block}'s read session is closed")
        change(entry)
        self.ledger._write(entries)
        return entry

    def record_fetch(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        """Each fetched file's sha256 and UTC download time, keyed ``plane/day``;
        committed when the fetch ends, before anything is computed."""
        for name, record in records.items():
            if not {"sha256", "downloaded_at"} <= set(record):
                raise ValueError(f"{name} needs its sha256 and its UTC download time")
        self._update(lambda entry: entry["fetch"].update({k: dict(v) for k, v in records.items()}))

    def require_fetch_committed(self) -> None:
        """Refuse to compute until HEAD's ledger holds the fetch's records."""
        committed = self.ledger.committed_entries().get(self.block, [])
        disk = self.ledger.entries()[self.block][-1]
        if not committed or not disk["fetch"] or committed[-1]["fetch"] != disk["fetch"]:
            raise HeldOutLocked(f"{self.block}'s fetch records are not committed")

    @property
    def resumptions(self) -> int:
        """Processes started after the first."""
        return max(0, len(self.ledger.entries()[self.block][-1]["processes"]) - 1)

    def record_failure(self, stage: str, exc: BaseException) -> ReadFailure | None:
        """Record a failed process; return why the read must close, if it must."""
        text = f"{type(exc).__name__}: {exc}"
        stamp = _utc(self.ledger.clock())
        self._update(
            lambda entry: entry["failures"].append(
                {"stage": stage, "exception": text, "utc": stamp}
            )
        )
        return self.must_close()

    def must_close(self) -> ReadFailure | None:
        """A read whose same stage failed again with the same exception, or that has
        not finished after five resumptions, closes: on H every slot not assessed
        (H read failed); on P every candidate not confirmed (P read failed)."""
        entry = self.ledger.entries()[self.block][-1]
        seen: set[tuple[str, str]] = set()
        for failure in entry["failures"]:
            pair = (failure["stage"], failure["exception"])
            if pair in seen:
                return ReadFailure(*pair, "the same stage failed again with the same exception")
            seen.add(pair)
        if self.resumptions > MAX_RESUMPTIONS:
            last = entry["failures"][-1] if entry["failures"] else {"stage": "", "exception": ""}
            return ReadFailure(
                last["stage"],
                last["exception"],
                f"not finished after {MAX_RESUMPTIONS} resumptions",
            )
        return None

    def close(self, results: Path | None = None) -> str:
        """Enter the results file's sha256 into the open entry; the session ends."""
        digest = file_sha256(self.ledger.repo / (results or sr.RESULTS_PATHS[self.block]))
        self._update(lambda entry: entry.update(results_sha256=digest))
        return digest
