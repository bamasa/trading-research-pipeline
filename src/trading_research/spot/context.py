"""One run of the spot study: where it reads and writes, its scale, its venue and its bindings.

A :class:`SpotStudy` is built by the entry point for a repository and a scale.
The registered study reads and writes under the repository: the YAML and the
registration document; ``data/spot/<venue>/<plane>/<symbol>/`` for the converted
days and ``data/spot_2026/`` for the fetch logs, the disk record and the run
log (all ignored by git); ``artifacts/spot_2026/run<n>`` for what a stage keeps
between processes; ``models/spot_2026`` for the frozen files; and
``experiments/results`` for the files the registration names. The dry run is the
same object on a temporary repository with a synthetic venue
(:mod:`trading_research.spot.synthetic`).

**The 2025 binding** (:class:`Binding2025`): the 2025 configuration S25-T's
thresholds are read from, its frozen sha256, and the sha256 of every 2025 file
S25-T loads. The registered binding is the registration's own; the dry run binds
a stand-in made of synthetic files, checked by the same code.
"""

from __future__ import annotations

import subprocess
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

from trading_research.data import fetch_log as fl
from trading_research.spot import archives as ar
from trading_research.spot import ledger as sl
from trading_research.spot import registration as sr
from trading_research.spot import synthetic as syn
from trading_research.spot.costs import FeeTable
from trading_research.spot.fetch_stage import SpotRoots
from trading_research.spot.plan import REGISTERED, Scale
from trading_research.spot.state import StageState

WORK_DIR = Path("artifacts/spot_2026")
RECORDS_DIR = Path("data/spot_2026")
RESULTS_DIR = Path("experiments/results")


def git(repo: Path, *args: str) -> str:
    """A git command's output in ``repo``; raises on failure."""
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


@dataclass(frozen=True)
class Binding2025:
    """The 2025 configuration S25-T reads, its frozen sha256, and the files it loads."""

    config: Path = sr.CONFIG_2025_PATH
    config_sha256: str = sr.CONFIG_2025_SHA256
    files: Mapping[Path, str] = field(default_factory=sl.files_2025)
    #: The checkout holding the files when it is not the study's repository.
    root: Path | None = None


@dataclass
class SpotStudy:
    """The repository, the scale, the D run's number, the worker count and the venue."""

    repo: Path
    scale: Scale = REGISTERED
    run: int = 1
    workers: int = 1
    #: The dry run's venue; None for the registered study, which uses the network.
    venue: syn.SpotVenue | None = None
    binding: Binding2025 = field(default_factory=Binding2025)
    #: Free disk in GB at a path (None: the machine's, ``data/disk.free_gb``).
    disk: Callable[[Path | str], float] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def work(self) -> Path:
        return self.repo / WORK_DIR / f"run{self.run}"

    @property
    def data(self) -> Path:
        return self.repo / "data"

    @property
    def spot_root(self) -> Path:
        return self.data / "spot"

    @property
    def roots(self) -> SpotRoots:
        return SpotRoots.under(self.spot_root)

    @property
    def state(self) -> StageState:
        return StageState(self.work)

    def dir(self, *parts: str) -> Path:
        path = self.work.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def records(self, name: str) -> Path:
        """``data/spot_2026/<name>``: fetch logs and the disk record, never committed."""
        path = self.repo / RECORDS_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def fetch_log(self, stage: int) -> fl.FetchLog:
        return fl.FetchLog(self.records(f"fetch_stage{stage}.json"))

    @property
    def disk_record(self) -> Path:
        return self.records("disk.json")

    def results(self, name: str) -> Path:
        return self.repo / RESULTS_DIR / f"spot_2026_{name}"

    def commit(self) -> str:
        return git(self.repo, "rev-parse", "HEAD").strip()

    @cached_property
    def registration(self) -> sr.Registration:
        """The YAML validated against the registration, null or frozen everywhere."""
        return sr.Registration.load(repo=self.repo).require_consistent()

    def reload(self) -> sr.Registration:
        self.__dict__.pop("registration", None)
        return self.registration

    @property
    def fee_table(self) -> FeeTable:
        return FeeTable.from_config(self.registration.tree["fees"])

    @property
    def opener(self) -> Any:
        """What every request goes through: the venue in a dry run, else urllib."""
        return self.venue if self.venue is not None else urllib.request.urlopen

    def present(self) -> dict[str, frozenset[date]]:
        """The days each plane's archive lists: the registered head check, or the venue's."""
        if self.venue is not None:
            return self.venue.market.present()
        return ar.listed_present(self.repo / ar.HEAD_CHECK)

    @contextmanager
    def serving(self) -> Iterator[None]:
        """Every request answered by the venue (no socket) in a dry run."""
        with syn.serving(self.venue) if self.venue is not None else nullcontext():
            yield

    def ledger(self) -> sl.Ledger:
        b = self.binding
        return sl.Ledger(repo=self.repo, config_2025=b.config, config_2025_sha256=b.config_sha256,
                         files_2025=dict(b.files), frozen_2025=b.root)  # fmt: skip
