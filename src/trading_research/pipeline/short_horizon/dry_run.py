"""The dry run: every stage and every block read on a synthetic venue, in a temporary repository.

The temporary repository holds a copy of the registration (the YAML and the
document) under git, so the study's own gates (the catalogue's hash, the YAML
validated against the registration, the code-freeze note, the ledger's
preconditions for H, F and P, which read git) run exactly as they will on the
real study. Every request is answered by :mod:`.synthetic`'s venue and no
request reaches the network. The two 2024 timing days are put on disk first,
as they already are in the real repository. Nothing is written to the real
repository, and the outputs are discarded unless ``keep`` names a folder.

This is a test of the code. It is not the registered dry run of stage E,
which runs the entry point on D's last 44 days and the reader-test days.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from trading_research.data import fetch_log as fl
from trading_research.data import study_access as sa
from trading_research.data.disk import StageBudget
from trading_research.data.fetch_stage import StagePlan, run_stage
from trading_research.pipeline import short_horizon_registration as shr
from trading_research.pipeline.short_horizon import synthetic as syn
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import DRY_RUN, Scale
from trading_research.pipeline.short_horizon.stage1 import CODE_FREEZE_NOTE
from trading_research.pipeline.short_horizon.state import STAGES

REPO = Path(__file__).resolve().parents[4]
COPIED = (shr.CONFIG_PATH, shr.DOCUMENT_PATH)
GITIGNORE = "/data/\n/artifacts/\n/models/\n"
_IDENTITY = {"GIT_AUTHOR_NAME": "dry run", "GIT_AUTHOR_EMAIL": "dry-run@localhost",
             "GIT_COMMITTER_NAME": "dry run", "GIT_COMMITTER_EMAIL": "dry-run@localhost"}  # fmt: skip
TINY = StageBudget(archives_in_flight_gb=0.01, books_gb=0.01, prints_gb=0.01)


def commit(repo: Path, message: str) -> str:
    """Commit everything in the temporary repository; its new HEAD."""
    env = {**os.environ, **_IDENTITY}
    for args in (["add", "-A"], ["commit", "-q", "--allow-empty", "-m", message]):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


def make_repository(source: Path, target: Path) -> Path:
    """The registration under git, then the code-freeze note naming that commit."""
    target.mkdir(parents=True, exist_ok=True)
    for relative in COPIED:
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target / relative)
    (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (target / "experiments" / "results").mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=target, check=True, capture_output=True)
    frozen = commit(target, "Registration")
    note = target / CODE_FREEZE_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        f"# Code freeze (dry run)\n\nCode-freeze commit: `{frozen}`\n", encoding="utf-8"
    )
    commit(target, "Code-freeze note")
    return target


#: The venue's market for the dry run: trend episodes a model can learn, so that
#: stratum R has an admissible cell; detector bursts on two test days of D, one
#: of H and one of F (the frozen detector alarms on each within its hours), the
#: signal inverted from each burst to the day's end with the spread kept at one
#: tick (a wider spread leaves the models nothing they recognise and the cells
#: stop trading), so a regime guard pays on D and H's random flag sets have a
#: flag to match; and one F day with sequence gaps, excluded and put back. The
#: scale's ridge is held at its intercept, so R* is the network and its control
#: fits and held-out scores run.
BURST_DAYS = frozenset({date(2025, 3, 10), date(2025, 5, 10), date(2025, 7, 21), date(2025, 9, 15)})
GAP_DAYS = frozenset({date(2025, 9, 16)})


def market(scale: Scale) -> syn.SyntheticMarket:
    return syn.SyntheticMarket(market_days(scale), gap_days=GAP_DAYS, episodes_per_day=48,
                               burst_days=BURST_DAYS, burst_hours=(2, 8), burst_spread_ticks=1)  # fmt: skip


def market_days(scale: Scale) -> frozenset[date]:
    """Every day the synthetic venue has a book and prints for."""
    blocks = ("W", "D", "H", "F", "P")
    return frozenset({d for b in blocks for d in scale.days(b)} | set(scale.timing_days))


def place_timing_days(study: Study) -> None:
    """The two 2024 days already on disk, and their archives for the replay's memory.

    One fetch per day: funding is fetched as one span, and the days between the
    two are in no block, so the development access refuses a span over both."""
    log = fl.FetchLog(study.repo / "data" / "timing_fetch_log.json")
    for day in study.scale.timing_days:
        plan = StagePlan(book=(day,), prints=(day,), funding=(day,))
        present = {"ob500": frozenset({day}), "ob200": frozenset(), "prints": frozenset({day})}
        run_stage(study.symbol, plan, study.roots, log, budget=TINY, book_workers=1,
                  print_workers=1, expected_present=present, access=sa.development_access())  # fmt: skip
    assert study.venue is not None
    folder = study.roots.archives.parent / "timing_archives"
    folder.mkdir(parents=True, exist_ok=True)
    for day in study.scale.timing_days:
        archive = study.venue.market.book_archive(day)
        (folder / f"{study.symbol}-{day}-ob500.zip").write_bytes(archive)


def prepare(root: Path, *, source: Path = REPO, scale: Scale = DRY_RUN, workers: int = 1) -> Study:
    """A temporary repository and a study on it, with the venue the run reads."""
    repo = make_repository(source, root / "repo")
    venue = syn.Venue(market(scale), frozenset(scale.days(sa.READER_TEST)))
    study = Study(repo, scale, workers=workers, venue=venue)
    with syn.serving(venue):
        place_timing_days(study)
    return study


def run(study: Study, stages: Sequence[str] = STAGES) -> dict[str, Any]:
    """Every named stage on the synthetic venue; refuses a request it does not know."""
    from trading_research.pipeline.short_horizon.runner import run_stage as run_one

    assert study.venue is not None
    records = {}
    with syn.serving(study.venue):
        for stage in stages:
            records[stage] = run_one(study, stage)
            commit(
                study.repo, f"Dry run: stage {stage}"
            )  # the registration commits each stage's files
    if study.venue.unknown:
        raise RuntimeError(
            f"the dry run asked for what the venue does not serve: {study.venue.unknown[:3]}"
        )
    return records


def read_blocks(study: Study, blocks: Sequence[str] = ("H", "F", "P")) -> dict[str, Any]:
    """Open and read each block as the registration orders it, committing what the
    author would commit by hand between the steps; P only when a slot qualifies."""
    from trading_research.pipeline.short_horizon import reads

    assert study.venue is not None
    records: dict[str, Any] = {}
    with syn.serving(study.venue):
        for block in blocks:
            if block == "P" and not reads.qualifying_for_p(study):
                records["P"] = {"opened": False, "reason": "no slot passed on F"}
                break
            if not study.state.complete(block):
                if not reads.ledger_for(study).committed_entries().get(block):
                    reads.open_block(study, block)
                    commit(study.repo, f"Dry run: {block}'s open entry")
                reads.read_block(study, block, commit=commit)
                commit(study.repo, f"Dry run: {block}'s results")
            records[block] = study.state.record(block)
    if study.venue.unknown:
        raise RuntimeError(
            f"the dry run asked for what the venue does not serve: {study.venue.unknown[:3]}"
        )
    return records


def dry_run(
    keep: Path | None = None, workers: int = 1, stages: Sequence[str] = STAGES
) -> dict[str, Any]:
    """The whole dry run; its files are deleted unless ``keep`` is given."""
    root = keep or Path(tempfile.mkdtemp(prefix="short-horizon-dry-run-"))
    try:
        study = prepare(root, workers=workers)
        records = run(study, stages)
        reads = read_blocks(study)
        statuses = {block: r.get("statuses", r) for block, r in reads.items()}
        assert study.venue is not None
        return {
            "stages": list(records),
            "reads": statuses,
            "venue": {"requests": len(study.venue.asked), "unknown": len(study.venue.unknown)},
            "work": str(study.work) if keep else "discarded",
        }
    finally:
        if keep is None:
            shutil.rmtree(root, ignore_errors=True)
