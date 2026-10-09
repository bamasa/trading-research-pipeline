"""One run of the study: where it reads and writes, its catalogue, and the days it trusts.

A :class:`Study` is built by the entry point for a repository and a scale. The
registered study reads and writes under the repository: the YAML and the
registration document, ``data/`` for the planes, ``artifacts/short_horizon_2025``
for everything a stage keeps between runs (ignored by git), and
``experiments/results`` for the files the registration names. The dry run is
the same object on a temporary repository with a synthetic venue.

The catalogue is loaded from the YAML and checked against its registered hash
(or the latest amendment's) every time it is read; the cells a run scores are
the catalogue's keys in its byte order, restricted to the scale's holds.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

from trading_research.data import study_access as sa
from trading_research.data.fetch_stage import StudyRoots
from trading_research.pipeline import short_horizon_registration as shr
from trading_research.pipeline.short_horizon.plan import REGISTERED, Scale
from trading_research.pipeline.short_horizon.state import StageState
from trading_research.pipeline.short_horizon.synthetic import Venue
from trading_research.search_engine.catalogue import Catalogue, CatalogueError, parse_key

SYMBOL = sa.INSTRUMENTS[0]
WORK_DIR = Path("artifacts/short_horizon_2025")
RESULTS_DIR = Path("experiments/results")


def git(repo: Path, *args: str) -> str:
    """A git command's output in ``repo``; raises on failure."""
    done = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )  # fmt: skip
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


@dataclass
class Study:
    """The repository, the scale, the D run's number and the worker count."""

    repo: Path
    scale: Scale = REGISTERED
    run: int = 1
    workers: int = 1
    #: The dry run's venue; None for the registered study, which uses the network.
    venue: Venue | None = None
    work_root: Path | None = None
    symbol: str = SYMBOL
    notes: list[str] = field(default_factory=list)

    @property
    def work(self) -> Path:
        return (self.work_root or self.repo / WORK_DIR) / f"run{self.run}"

    @property
    def data(self) -> Path:
        return self.repo / "data"

    @property
    def roots(self) -> StudyRoots:
        return StudyRoots.under(self.data)

    @property
    def state(self) -> StageState:
        return StageState(self.work)

    def dir(self, *parts: str) -> Path:
        path = self.work.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def results(self, name: str) -> Path:
        """``experiments/results/short_horizon_2025_<name>`` in the repository."""
        return self.repo / RESULTS_DIR / f"short_horizon_2025_{name}"

    def commit(self) -> str:
        return git(self.repo, "rev-parse", "HEAD").strip()

    @cached_property
    def registration(self) -> shr.Registration:
        """The YAML, validated against the registration (raises on any departure)."""
        return shr.Registration.load(repo=self.repo)

    def catalogue(self) -> Catalogue:
        """The catalogue from the YAML, refused unless its hash is the registered
        or the latest amended one."""
        catalogue = Catalogue.load(self.repo / shr.CONFIG_PATH)
        document = (self.repo / shr.DOCUMENT_PATH).read_text(encoding="utf-8")
        accepted = shr.latest_catalogue_sha256(document)
        if catalogue.sha256 != accepted:
            raise CatalogueError(
                f"the catalogue's sha256 {catalogue.sha256} is not the accepted {accepted}"
            )
        return catalogue

    @cached_property
    def keys(self) -> tuple[str, ...]:
        """The cells this run scores, in the catalogue's byte order."""
        catalogue = self.catalogue()
        holds = self.scale.holds_s
        keys = tuple(k for k in catalogue.keys if holds is None or parse_key(k).hold_s in holds)
        catalogue.require(keys)
        return keys

    def record(self, stage: str) -> dict[str, Any]:
        return self.state.record(stage)

    def usable_days(self, block: str) -> list[date]:
        """Days of ``block`` stage 1 found usable (W and D only)."""
        days = self.record("1")["days"]
        return [d for d in self.scale.days(block) if days[d.isoformat()]["usable"]]
