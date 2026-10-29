"""The study code's freeze: the note's commit, and nothing of the study code changed since.

Registered (the development-period procedure, step 1; "A correction before H"):
the study code is frozen before the stage 1 fetch by a dated note,
``docs/spot_2026_code_freeze.md``, naming a commit on a line ``Code-freeze commit:
`<sha>```. Every change to the study code after it is a correction, made only by a
dated amendment that states it and re-runs D in full; so a later freeze line in
the note counts only when the document's amendments name its commit.

The freeze is checked at the start of stage 1, of D and of the amendment (the
configuration included, since it is frozen only by the amendment), and at every
open of H or P and at the start of every process of a read session (the code
only: the amendment has frozen the configuration, which the ledger pins on its
own). D and the amendment also require the freeze to be the one stage 1 ran
under; the ledger reads it from the frozen configuration (``compute.record``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

CODE_FREEZE_NOTE = "docs/spot_2026_code_freeze.md"
CODE_FREEZE_LINE = re.compile(r"Code-freeze commit: `([0-9a-f]{7,40})`")
CONFIG = "configs/spot_2026.yaml"
#: What the freeze covers: the study code, its configuration and its environment.
STUDY_CODE: tuple[str, ...] = ("src/", "experiments/spot_2026.py", CONFIG, "pyproject.toml",
                               "uv.lock")  # fmt: skip
#: What it covers once the amendment has frozen the configuration.
CODE_ONLY: tuple[str, ...] = tuple(p for p in STUDY_CODE if p != CONFIG)

Git = Callable[..., str]


class CodeFreezeError(RuntimeError):
    """The study code is not the code the freeze names."""


def note_commit(repo: Path, git: Git, amendments: str) -> str:
    """The commit the note's latest line names, resolved; a later line than the first
    must be named by the document's amendments (a dated correction)."""
    path = repo / CODE_FREEZE_NOTE
    if not path.exists():
        raise CodeFreezeError(f"no {CODE_FREEZE_NOTE}: the code freeze is recorded, with a line "
                              "'Code-freeze commit: `<sha>`', before the stage 1 fetch")  # fmt: skip
    git(repo, "ls-files", "--error-unmatch", CODE_FREEZE_NOTE)
    found = CODE_FREEZE_LINE.findall(path.read_text(encoding="utf-8"))
    if not found:
        raise CodeFreezeError(f"{CODE_FREEZE_NOTE} names no 'Code-freeze commit: `<sha>`'")
    latest = found[-1]
    if len(set(found)) > 1 and latest not in amendments:
        raise CodeFreezeError(f"the code-freeze commit {latest} replaces an earlier one, and no "
                              "dated amendment names it (a correction under the protocol)")  # fmt: skip
    return git(repo, "rev-parse", f"{latest}^{{commit}}").strip()


def require_unchanged(repo: Path, git: Git, commit: str, paths: Sequence[str]) -> None:
    """HEAD descends from ``commit``, and ``paths`` changed neither since nor in the tree."""
    git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    changed = git(repo, "diff", "--name-only", f"{commit}..HEAD", "--", *paths).split()
    dirty = git(repo, "status", "--porcelain", "--", *paths).splitlines()
    if changed or dirty:
        raise CodeFreezeError(f"study code changed after the code-freeze commit {commit[:7]}: "
                              f"{(changed + [d[3:] for d in dirty])[:5]}; a change is a correction "
                              "under the amendment protocol")  # fmt: skip


def check(
    repo: Path, git: Git, amendments: str, *, paths: Sequence[str] = STUDY_CODE,
    expected: str | None = None,
) -> str:  # fmt: skip
    """The freeze's commit, refused unless ``paths`` are unchanged since it and, when
    ``expected`` is given, it is that commit (the one stage 1 ran under)."""
    commit = note_commit(repo, git, amendments)
    if expected is not None and commit != expected:
        raise CodeFreezeError(f"the code-freeze commit {commit[:7]} is not the one stage 1 ran "
                              f"under, {expected[:7]}: a new freeze needs a new numbered D run")  # fmt: skip
    require_unchanged(repo, git, commit, paths)
    return commit
