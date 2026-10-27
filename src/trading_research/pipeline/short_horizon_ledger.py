"""The 2025 study's ledger: the one gate to H, F and P, and the read sessions it binds.

On the pattern of :mod:`trading_research.market_making.round2`'s ledger, with
the registration's stricter protocol (``## Who may read what``):

* :meth:`Ledger.open` checks a block's preconditions and appends its **open
  entry** (block, session commit, config sha256, environment fingerprint, UTC
  time) to ``experiments/results/short_horizon_2025_ledger.json``. It returns no
  access: the entry is committed and pushed first.
* :meth:`Ledger.session` starts **every process** of a read session. It refuses
  unless the open entry is in HEAD's (and the upstream's) ledger, the tree and
  the diff since the session's commit list nothing but the ledger file, and the
  environment fingerprint equals H's open entry's; it records the process and
  returns a :class:`ReadSession` whose :class:`~trading_research.data.study_access.StudyAccess`
  the readers require.
* The session records the fetch's per-file records, its failures (a read that
  fails twice at the same stage with the same exception, or runs past five
  resumptions, must close as failed) and, at the end, the results file's
  sha256.

H opens only with the frozen configuration recorded by the amendment and
committed, the ``ob200`` reader test recorded with no reader-test day failing
but on its gap count, every frozen weight file, finalist file and the fitted-values file
at the amendment's sha256, the trial ledger's manifest committed with the
amendment's sha256 and row count and every part on disk matching it (the parts
stay under ``data/``, never committed), the catalogue at its registered (or amended)
sha256, and no file of H, nor any hourly bar of E, on disk. F and P open only
after the previous block's results file is committed unchanged, with nothing
but results, docs outside the registration and the README changed since that
read's commit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_research.data import study_access as sa
from trading_research.data.fetch_stage import StagePlan, StudyRoots, files_on_disk
from trading_research.data.study_fetch import file_sha256
from trading_research.market_making.prereg import (
    FROZEN_HASH_PATTERN,
    HeldOutLocked,
    PreRegistrationError,
    _git,
)
from trading_research.pipeline import short_horizon_registration as shr

#: Where the frozen files live (the ignored ``models/`` directory).
FROZEN_DIR = Path("models/short_horizon_2025")
WEIGHTS_DIR = FROZEN_DIR / "weights"
FINALISTS_DIR = FROZEN_DIR / "finalists"
FITTED_VALUES_PATH = FROZEN_DIR / "fitted_values.json"

#: A read that has not finished after this many resumptions closes as failed.
MAX_RESUMPTIONS = 5
FIRST_READ, SECOND_READ = "first read", "second read"


def _now() -> datetime:
    return datetime.now(UTC)


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def library_commit(name: str = shr.LOCKED_LIBRARY) -> str | None:
    """The VCS commit ``name`` is installed from, or None (not installed, or not from VCS)."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        origin = _origin(distribution(name))
    except PackageNotFoundError:
        return None
    if not origin.startswith("@") or origin in ("@editable", "@local"):
        return None
    return origin[1:]


def trial_ledger_parts(repo: Path, trials: Path = shr.TRIALS_PATH) -> list[Path]:
    """The trial ledger's files on disk: the one file, or its parts ``<stem>.part<k>.parquet``."""
    single = repo / trials
    found = [single] if single.exists() else []
    return found + sorted(single.parent.glob(f"{single.stem}.part*{single.suffix}"))


def trial_ledger_sha256(paths: Sequence[Path]) -> str:
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
    paths = []
    for line in text.splitlines():
        if line.strip():
            path = line[3:]
            paths.append(path.split(" -> ")[-1].strip().strip('"'))
    return paths


def _diff_allowed_after_a_read(path: str) -> bool:
    """What may change between a read and the next open: results, docs outside
    the registration, and the README."""
    if path == "README.md" or path.startswith("experiments/results/"):
        return True
    return path.startswith("docs/") and not path.startswith("docs/preregistration/")


@dataclass
class Ledger:
    """The study's one gate to H, F and P. Paths are relative to ``repo``.

    ``upstream`` is the ref an open entry must have reached before a session
    starts (None skips that check); ``environment`` computes the fingerprint;
    ``require_locked_library`` refuses an open unless the detector library is
    installed from its locked commit.
    """

    repo: Path = Path()
    config: Path = shr.CONFIG_PATH
    document: Path = shr.DOCUMENT_PATH
    ledger: Path = shr.LEDGER_PATH
    trials: Path = shr.TRIALS_PATH
    trials_manifest: Path = shr.TRIALS_MANIFEST_PATH
    weights_dir: Path = WEIGHTS_DIR
    finalists_dir: Path = FINALISTS_DIR
    fitted_values: Path = FITTED_VALUES_PATH
    roots: StudyRoots | None = None
    symbol: str = sa.INSTRUMENTS[0]
    upstream: str | None = "@{upstream}"
    require_locked_library: bool = True
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

    def ledger_versions(
        self, ref: str = "HEAD"
    ) -> list[tuple[str, dict[str, list[dict[str, Any]]]]]:
        """Every version of the ledger in ``ref``'s history, newest first, each with
        the commit that wrote it (a commit that deleted the file is skipped)."""
        ledger = self.ledger.as_posix()
        try:
            commits = _git(self.repo, "log", "--format=%H", ref, "--", ledger).split()
        except HeldOutLocked:
            return []
        out = []
        for found in commits:
            try:
                text = _git(self.repo, "show", f"{found}:{ledger}")
            except HeldOutLocked:
                continue
            try:
                out.append((found, json.loads(text)))
            except json.JSONDecodeError:
                continue
        return out

    def erased_entries(self, block: str, kept: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """``block``'s entries some version of the ledger held (in HEAD's history, and
        in the upstream's when one is set) that ``kept`` (HEAD's) lacks: an open entry
        erased by a later commit or a reset."""
        refs = ["HEAD"]
        if self.upstream is not None:
            with contextlib.suppress(HeldOutLocked):
                refs.append(_git(self.repo, "rev-parse", "--verify", "-q", self.upstream).strip())
        have = {(e.get("utc"), e.get("commit")) for e in kept}
        erased: dict[tuple[Any, Any], dict[str, Any]] = {}
        for ref in refs:
            for _, entries in self.ledger_versions(ref):
                for entry in entries.get(block, []):
                    key = (entry.get("utc"), entry.get("commit"))
                    if key not in have:
                        erased.setdefault(key, dict(entry))
        return list(erased.values())

    def _write(self, entries: Mapping[str, Any]) -> None:
        path = self.repo / self.ledger
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _head(self) -> str:
        return _git(self.repo, "rev-parse", "HEAD").strip()

    def _tracked(self, path: Path) -> None:
        _git(self.repo, "ls-files", "--error-unmatch", path.as_posix())

    # -- the checks -------------------------------------------------------------

    def expected_sha256(self) -> str:
        """The frozen configuration's sha256, as the amendment records it."""
        path = self.repo / self.document
        if not path.exists():
            raise HeldOutLocked(f"no pre-registration document at {self.document}")
        text = shr.amendments_text(path.read_text(encoding="utf-8"))
        found = FROZEN_HASH_PATTERN.findall(text)
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

    def verify_frozen(self) -> shr.Registration:
        """The configuration on disk and at HEAD is the frozen one, and valid."""
        expected = self.expected_sha256()
        actual = file_sha256(self.repo / self.config)
        if actual != expected:
            raise HeldOutLocked(
                f"{self.config} has sha256 {actual}, but the amendment froze {expected}"
            )
        try:
            registration = shr.Registration.load(self.config, self.document, repo=self.repo)
            registration.require_frozen()
        except PreRegistrationError as exc:
            raise HeldOutLocked(f"the frozen configuration is not valid: {exc}") from exc
        for tracked in (self.config, self.document):
            self._tracked(tracked)
        committed = _git(self.repo, "show", f"HEAD:{self.config.as_posix()}")
        if hashlib.sha256(committed.encode()).hexdigest() != expected:
            raise HeldOutLocked("the committed configuration is not the frozen one")
        return registration

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

    def verify_frozen_files(self, registration: shr.Registration) -> None:
        """Every frozen weight file, finalist file and the fitted-values file at
        the amendment's sha256."""
        files = [
            (self.weights_dir / key, digest) for key, digest in registration.weights_sha256.items()
        ]
        files += [
            (self.finalists_dir / f"{slot}.json", digest)
            for slot, digest in registration.finalist_files_sha256.items()
        ]
        fitted = registration.fitted_values_sha256
        if not registration.weights_sha256 or not registration.finalist_files_sha256 or not fitted:
            raise HeldOutLocked("the amendment records no weights, finalist files or fitted values")
        files.append((self.fitted_values, fitted))
        for relative, digest in files:
            path = self.repo / relative
            if not path.is_file():
                raise HeldOutLocked(f"the frozen file {relative} is missing")
            actual = file_sha256(path)
            if actual != digest:
                raise HeldOutLocked(
                    f"{relative} has sha256 {actual}, the amendment records {digest}"
                )

    def verify_trials(self, registration: shr.Registration) -> None:
        """The trial ledger's manifest committed unchanged, at the amendment's
        sha256 and rows and the registered catalogue, and the parts on disk (under
        ``data/``, never committed) exactly the manifest's: each at its sha256 and
        row count, in its order, and no other part beside them."""
        result = registration.result_trials
        if result is None:
            raise HeldOutLocked("the amendment records no trial ledger")
        relative = self.trials_manifest.as_posix()
        path = self.repo / self.trials_manifest
        if not path.is_file():
            raise HeldOutLocked(f"no trial-ledger manifest at {relative}")
        self._tracked(self.trials_manifest)
        text = path.read_text(encoding="utf-8")
        if _git(self.repo, "show", f"HEAD:{relative}") != text:
            raise HeldOutLocked(f"{relative} differs from the committed manifest")
        manifest = json.loads(text)
        recorded = (manifest.get("sha256"), manifest.get("rows"), manifest.get("catalogue_sha256"))
        wanted = (result["ledger_sha256"], result["ledger_rows"], registration.catalogue_sha256)
        if recorded != wanted:
            raise HeldOutLocked(
                f"the manifest records (sha256, rows, catalogue) {recorded}, the amendment {wanted}"
            )
        folder = (self.repo / self.trials).parent
        parts = [folder / part["file"] for part in manifest.get("parts", [])]
        if not parts:
            raise HeldOutLocked("the manifest lists no part of the trial ledger")
        extra = sorted(set(trial_ledger_parts(self.repo, self.trials)) - set(parts))
        if extra:
            raise HeldOutLocked(f"parts on disk the manifest does not list: {extra[:3]}")
        for part, target in zip(manifest["parts"], parts, strict=True):
            if not target.is_file():
                raise HeldOutLocked(f"the trial ledger's part {part['file']} is not on disk")
            if file_sha256(target) != part["sha256"]:
                raise HeldOutLocked(f"the part {part['file']} differs from its sha256")
            if parquet_rows(target) != part["rows"]:
                raise HeldOutLocked(f"the part {part['file']} holds another row count")
        total = sum(int(part["rows"]) for part in manifest["parts"])
        digest = trial_ledger_sha256(parts)
        if (digest, total) != (manifest["sha256"], manifest["rows"]):
            raise HeldOutLocked(
                f"the parts give (sha256, rows) ({digest}, {total}), the manifest "
                f"({manifest['sha256']}, {manifest['rows']})"
            )

    def verify_reader_test(self, registration: shr.Registration) -> None:
        """The ``ob200`` replay proven before H: the amendment records the reader
        test, and every reader-test day passed as a block day or failed on its gap
        count alone (a property of the archive, recorded, and the study goes on).
        Any other failure is first checked as a reader fault, corrected under the
        amendment protocol, before H is opened."""
        result = registration.value(("data", "ob200_reader_test_result"))
        if not result or not result.get("days"):
            raise HeldOutLocked("the amendment records no ob200 reader test; H waits for it")
        days = result["days"]
        missing = [d.isoformat() for d in sa.READER_TEST_DAYS if d.isoformat() not in days]
        if missing:
            raise HeldOutLocked(f"the ob200 reader test records no result for {missing}")
        failed = {
            day: list(entry.get("exclusions", []))
            for day, entry in days.items()
            if not entry.get("passes") and not entry.get("fails_on_gaps_alone")
        }
        if failed:
            raise HeldOutLocked(
                f"ob200 reader-test days failed as block days: {failed}; the reader is checked "
                "and a fault corrected under the amendment protocol before H is opened"
            )

    def files_on_disk(self, block: str) -> list[Path]:
        """Files of ``block``'s opening already on disk (for H, E's hourly bars too)."""
        days = sa.opening_days(block)
        plan = StagePlan(
            book=days[sa.BOOK], prints=days[sa.PRINTS], bars=days[sa.BARS], funding=days[sa.FUNDING]
        )
        roots = self.roots or StudyRoots.under(self.repo / "data")
        return files_on_disk(self.symbol, plan, roots)

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

    def _locked_library(self) -> None:
        if not self.require_locked_library:
            return
        commit = library_commit()
        if commit != shr.LOCKED_COMMIT:
            raise HeldOutLocked(
                f"{shr.LOCKED_LIBRARY} is installed from {commit or 'no VCS commit'}, "
                f"not its locked commit {shr.LOCKED_COMMIT}"
            )

    def _after_read(
        self, previous: str, entries: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> None:
        """``previous``'s first read closed, its results committed unchanged, and
        nothing but results, docs and the README changed since its commit."""
        reads = [e for e in entries.get(previous, []) if e["read"] == FIRST_READ]
        if not reads or not reads[0].get("results_sha256"):
            raise HeldOutLocked(f"{previous}'s read has no results recorded in HEAD's ledger")
        entry = reads[0]
        results = shr.RESULTS_PATHS[previous]
        self._tracked(results)
        actual = file_sha256(self.repo / results)
        if actual != entry["results_sha256"]:
            raise HeldOutLocked(f"{results} changed after {previous}'s read closed")
        diff = _git(self.repo, "diff", "--name-only", f"{entry['commit']}..HEAD").split()
        outside = [p for p in diff if not _diff_allowed_after_a_read(p)]
        if outside:
            raise HeldOutLocked(f"{outside} changed since {previous}'s read; only results may")

    # -- opening a block ----------------------------------------------------------

    def open(
        self,
        block: str,
        *,
        catalogue_sha256: str | None = None,
        qualifying: Sequence[str] = (),
        force: bool = False,
    ) -> dict[str, Any]:
        """Check ``block``'s preconditions and append its open entry; return the entry.

        No access is returned: the entry is committed and pushed before the
        block's fetch, and :meth:`session` issues the access. ``catalogue_sha256``
        is the catalogue as the entry point computes it (H only); ``qualifying``
        names the slots P is read for. A block already opened is refused
        unless ``force``, and the new entry is then a second read.
        """
        if block not in sa.HELD_OUT:
            raise ValueError(f"the ledger opens {sa.HELD_OUT}, not {block!r}")
        registration = self.verify_frozen()
        self.verify_tree()
        self._locked_library()
        committed = self.committed_entries()
        if self.entries() != committed:
            raise HeldOutLocked("the ledger has entries that are not committed")
        environment = self._environment(committed)
        if block == "H":
            if catalogue_sha256 != registration.catalogue_sha256:
                raise HeldOutLocked(
                    f"the catalogue's sha256 {catalogue_sha256} is not the registered or "
                    f"amended {registration.catalogue_sha256}"
                )
            self.verify_reader_test(registration)
        else:
            self._after_read(sa.HELD_OUT[sa.HELD_OUT.index(block) - 1], committed)
        self.verify_trials(registration)
        if block == "P" and not qualifying:
            raise HeldOutLocked("P is opened only for a slot that qualifies on F")
        self.verify_frozen_files(registration)
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
        erased = self.erased_entries(block, previous)
        if erased and not force:
            first = erased[-1]
            raise HeldOutLocked(
                f"{block}'s open entry of {first.get('utc')} at commit {str(first.get('commit'))[:7]} "
                "is in the ledger's history but not in HEAD's: a read cannot be erased and opened "
                "again; a second read needs force=True and is stamped 'second read'"
            )
        entry: dict[str, Any] = {
            "block": block,
            "read": SECOND_READ if previous or erased else FIRST_READ,
            "commit": self._head(),
            "config_sha256": registration.sha256,
            "environment": environment,
            "utc": _utc(self.clock()),
            "qualifying": sorted(qualifying),
            "processes": [],
            "failures": [],
            "fetch": {},
            "results_sha256": None,
        }
        self._write({**committed, block: [*previous, entry]})
        return entry

    # -- the open entry on the remote ---------------------------------------------

    def entry_commit(self, block: str, entry: Mapping[str, Any]) -> str:
        """The commit that first wrote ``entry`` into the ledger (HEAD when none did)."""
        for found, entries in reversed(self.ledger_versions("HEAD")):
            if any((e.get("utc"), e.get("commit")) == (entry["utc"], entry["commit"])
                   for e in entries.get(block, [])):  # fmt: skip
                return found
        return self._head()

    def require_pushed(self, block: str, entry: Mapping[str, Any]) -> None:
        """The open entry has reached the upstream: a remote-tracking branch (never a
        local branch set as upstream) whose ledger holds it, and the remote itself,
        asked with ``git ls-remote``, at a tip that contains the commit that wrote
        it. A remote that cannot be reached refuses: the push cannot be confirmed."""
        if self.upstream is None:
            return
        where = f"{block}'s open entry has not reached {self.upstream}"
        try:
            name = _git(self.repo, "rev-parse", "--symbolic-full-name", self.upstream).strip()
        except HeldOutLocked as exc:
            raise HeldOutLocked(f"{where}: no upstream is set") from exc
        if not name.startswith("refs/remotes/"):
            raise HeldOutLocked(
                f"{where}: the upstream {name or self.upstream} is not a remote-tracking branch"
            )
        pushed = self.committed_entries(name).get(block, [])
        if not any(e["utc"] == entry["utc"] and e["commit"] == entry["commit"] for e in pushed):
            raise HeldOutLocked(where)
        remote, _, branch = name.removeprefix("refs/remotes/").partition("/")
        try:
            listed = _git(self.repo, "ls-remote", "--exit-code", remote, f"refs/heads/{branch}")
        except HeldOutLocked as exc:
            raise HeldOutLocked(
                f"{where}: the remote {remote!r} cannot be reached (or has no branch "
                f"{branch!r}) to confirm the push; a session starts only once it can"
            ) from exc
        tip = listed.split()[0]
        written = self.entry_commit(block, entry)
        try:
            _git(self.repo, "merge-base", "--is-ancestor", written, tip)
        except HeldOutLocked as exc:
            raise HeldOutLocked(
                f"{where}: the remote's tip {tip[:7]} does not contain {written[:7]}, the "
                "commit that wrote it (push it, then fetch)"
            ) from exc

    # -- a process of a read session ---------------------------------------------

    def session(self, block: str, *, workers: int = 1) -> ReadSession:
        """Start one process of ``block``'s latest read session; return its access.

        Refused unless the open entry is in HEAD's ledger (and the upstream's),
        the session is not closed, nothing but the ledger changed since the
        session's commit, the frozen configuration and files are unchanged, and
        the environment is H's. The process is recorded with ``workers``.
        """
        committed = self.committed_entries()
        if not committed.get(block):
            raise HeldOutLocked(f"{block}'s open entry is not in HEAD's ledger: commit it first")
        entry = committed[block][-1]
        self.require_pushed(block, entry)
        if entry.get("results_sha256"):
            raise HeldOutLocked(f"{block}'s read session is closed: its results file is written")
        self.verify_tree(since=entry["commit"])
        registration = self.verify_frozen()
        if registration.sha256 != entry["config_sha256"]:
            raise HeldOutLocked(f"the configuration is not the one {block} was opened with")
        self.verify_frozen_files(registration)
        environment = self._environment(committed)
        on_disk = self.entries()
        live = on_disk[block][-1]
        if (live["utc"], live["commit"]) != (entry["utc"], entry["commit"]):
            raise HeldOutLocked("the ledger on disk does not end with HEAD's open entry")
        if live.get("results_sha256"):
            raise HeldOutLocked(f"{block}'s read session is closed: its results file is written")
        live["processes"].append({"utc": _utc(self.clock()), "workers": int(workers)})
        self._write(on_disk)
        key = (block, entry["commit"], entry["config_sha256"], environment)
        grants = sa.grants_for(sa.opened_pairs(block))
        access = sa.StudyAccess(grants, entry["read"], session=key, issuer=sa._ISSUER)
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

    ``key`` is (block, commit, config sha256, environment fingerprint); every
    output the session writes carries it, with :attr:`stamp`. ``access`` is
    what the study's readers require.
    """

    ledger: Ledger
    block: str
    key: tuple[str, ...]
    access: sa.StudyAccess

    @property
    def stamp(self) -> str:
        return self.access.stamp

    def _update(self, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        entries = self.ledger.entries()
        entry = entries[self.block][-1]
        if (entry["commit"], entry["config_sha256"]) != self.key[1:3]:
            raise HeldOutLocked("the ledger's latest entry is not this session's")
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
        self._update(
            lambda entry: entry["failures"].append(
                {"stage": stage, "exception": text, "utc": _utc(self.ledger.clock())}
            )
        )
        return self.must_close()

    def must_close(self) -> ReadFailure | None:
        """A read that failed twice at one stage with one exception, or has not
        finished after five resumptions, closes with every slot not assessed."""
        entry = self.ledger.entries()[self.block][-1]
        seen: set[tuple[str, str]] = set()
        for failure in entry["failures"]:
            pair = (failure["stage"], failure["exception"])
            if pair in seen:
                return ReadFailure(*pair, "the same stage failed again with the same exception")
            seen.add(pair)
        if self.resumptions > MAX_RESUMPTIONS:
            last = entry["failures"][-1] if entry["failures"] else {"stage": "", "exception": ""}
            reason = f"not finished after {MAX_RESUMPTIONS} resumptions"
            return ReadFailure(last["stage"], last["exception"], reason)
        return None

    def close(self, results: Path | None = None) -> str:
        """Enter the results file's sha256 into the open entry; the session ends."""
        path = self.ledger.repo / (results or shr.RESULTS_PATHS[self.block])
        digest = file_sha256(path)
        self._update(lambda entry: entry.update(results_sha256=digest))
        return digest
