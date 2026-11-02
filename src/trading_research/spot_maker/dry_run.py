"""The dry run: every stage and the H read on the spot study's synthetic venue, in a temporary repository.

The temporary repository holds the registration as registered under git (the
configuration with every placeholder reset, the document cut before its D
amendment). The spot study's checkout is a stand-in
(:func:`trading_research.spot_maker.standin.write_source`): its prints are the
spot synthetic venue's Binance archives converted by the spot study's parser,
and its S25-PF files a deterministic stand-in model, bound by their own
sha256s. The calendar is a few days of each block. Every gate runs as it will
on the real study: the code-freeze note, the stage order, the frozen hash, the
ledger's open entry and session, the copies inside the sealed access. It is a
test of the code, never a result; the outputs are discarded unless ``keep``
names a folder.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from trading_research.market_making.prereg import unfrozen_text
from trading_research.spot.synthetic import no_network
from trading_research.spot_maker import pages, reads, standin
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.context import STAGES, MakerStudy
from trading_research.spot_maker.runner import run_stage

REPO = Path(__file__).resolve().parents[3]
D_AMENDMENT = re.compile(r"\n### \d{4}-\d{2}-\d{2}: the D amendment\n")
GITIGNORE = "/data/\n/artifacts/\n/models/\n"
_IDENTITY = {"GIT_AUTHOR_NAME": "dry run", "GIT_AUTHOR_EMAIL": "dry-run@localhost",
             "GIT_COMMITTER_NAME": "dry run", "GIT_COMMITTER_EMAIL": "dry-run@localhost"}  # fmt: skip
DRY_CALENDAR: dict[str, tuple[date, ...]] = {
    "reader_test": (date(2026, 3, 3), date(2026, 3, 4)),
    "W": (date(2026, 4, 30),),
    "D": (date(2026, 5, 1), date(2026, 7, 30), date(2026, 7, 31)),
    "H": (date(2026, 8, 8), date(2026, 8, 9)),
    "P": (),
}


def commit(repo: Path, message: str) -> str:
    env = {**os.environ, **_IDENTITY}
    for args in (["add", "-A"], ["commit", "-q", "--allow-empty", "-m", message]):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                          text=True)  # fmt: skip
    return done.stdout.strip()


def registered_document(text: str) -> str:
    """The document as it stood at the code freeze: cut before its D amendment, the
    amendments made before the freeze kept."""
    match = D_AMENDMENT.search(text)
    return text if match is None else text[: match.start()].rstrip("\n") + "\n"


def make_repository(source: Path, target: Path) -> Path:
    """The registration as registered under git."""
    texts = {
        rg.CONFIG_PATH: unfrozen_text((source / rg.CONFIG_PATH).read_text("utf-8")),
        rg.DOCUMENT_PATH: registered_document((source / rg.DOCUMENT_PATH).read_text("utf-8")),
    }
    for relative, text in texts.items():
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_text(text, encoding="utf-8")
    (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (target / "experiments" / "results").mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=target, check=True, capture_output=True)
    commit(target, "Dry run: the registration as registered")
    return target


def code_freeze_note(repo: Path) -> str:
    head = commit(repo, "Dry run: before the code freeze")
    note = repo / rg.CODE_FREEZE_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(f"# Code freeze (dry run)\n\nCode-freeze commit: `{head}`\n", encoding="utf-8")
    return commit(repo, "Dry run: the code-freeze note")


def prepare(root: Path, *, source: Path = REPO) -> MakerStudy:
    days = sorted({d for block in DRY_CALENDAR.values() for d in block})
    binding = standin.write_source(root / "source", days)
    repo = make_repository(source, root / "repo")
    return MakerStudy(repo, root / "source", binding, DRY_CALENDAR, disk=lambda _: 1_000.0)


def run(study: MakerStudy) -> dict[str, Any]:
    records = {}
    for stage in STAGES:
        if stage == "1":
            code_freeze_note(study.repo)
        records[stage] = run_stage(study, stage)
        commit(study.repo, f"Dry run: stage {stage}")
    return records


def read_h(study: MakerStudy) -> dict[str, Any]:
    reads.open_block(study, "H")
    commit(study.repo, "Dry run: H's open entry")
    first = reads.read_block(study, "H")
    commit(study.repo, "Dry run: H's copy records")
    second = reads.read_block(study, "H")
    commit(study.repo, "Dry run: H's results and page")
    return {"first_process": first, "second_process": second}


def dry_run(keep: Path | None = None) -> dict[str, Any]:
    root = keep or Path(tempfile.mkdtemp(prefix="spot-maker-dry-run-"))
    try:
        with no_network():
            study = prepare(root)
            stages = run(study)
            h = read_h(study)
            page = pages.write_page(study)
        results = reads.h_results(study)
        return {"stages": list(stages), "stage_0_passes": stages["0"]["passes"],
                "amendment": stages["amend"]["amendment"], "H": h,
                "statuses": {s: v["status"] for s, v in results["statuses"].items()},
                "headlines": results["headlines"], "page": page.name,
                "repo": str(study.repo) if keep else "discarded"}  # fmt: skip
    finally:
        if keep is None:
            shutil.rmtree(root, ignore_errors=True)
