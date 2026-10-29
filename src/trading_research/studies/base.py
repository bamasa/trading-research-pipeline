"""What every registered study offers, and the adapter that offers it through the study's own entry point.

A registered study has one entry point, ``experiments/<name>.py``, whose
subcommands are its stages in the registered order, the opening and reading
of its held-out blocks through its ledger, ``status``, ``pages`` and
``--dry-run``. :class:`EntryPointStudy` runs that entry point exactly as the
author would run it by hand::

    python -m experiments.<name> <stage> [block] [options]

in a fresh interpreter, in the study's repository, with the package on the
path. It passes the arguments through unchanged and reads back the JSON the
entry point prints. It never imports a stage, a ledger or an access of the
study, never opens a block itself, and never edits the entry point: a stage out
of order, a block not granted, a read without its committed open entry is
refused by the entry point, with its own message and its own exit code (2),
and the runner reports that refusal as it came. Running through the runner is
therefore never a way around the ledger, only a shorter way to type the
command.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: The exit code an entry point returns with a refusal (``entry()`` in each).
REFUSED = 2


@dataclass(frozen=True)
class StudyRun:
    """One invocation of a study's entry point, as it came back.

    ``result`` is the JSON the entry point printed, when it printed one;
    ``refused`` is True when it refused (exit code 2 with a ``refused:``
    message on stderr), which is never an error of the runner.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    result: Any = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def refused(self) -> bool:
        return self.returncode == REFUSED and "refused" in self.stderr

    @property
    def command(self) -> str:
        """The command as the author would type it."""
        return " ".join(("python", "-m", *self.argv[2:]))


@runtime_checkable
class RegisteredStudy(Protocol):
    """What the study runner, ``trp study`` and :class:`trading_research.api.Study` need."""

    name: str

    @property
    def registration(self) -> Path:
        """The registration document, ``docs/preregistration/<name>.md``."""
        ...

    @property
    def config(self) -> Path:
        """The registration's YAML, ``configs/<name>.yaml``."""
        ...

    def stages(self) -> tuple[str, ...]:
        """Every step the entry point accepts, in the registered order."""
        ...

    def run(self, stage: str, *args: str, **options: Any) -> StudyRun:
        """Run one step through the entry point."""
        ...

    def status(self) -> StudyRun: ...

    def pages(self) -> StudyRun: ...

    def dry_run(self, keep: Path | None = None, **options: Any) -> StudyRun: ...


def _flag(name: str, value: Any) -> list[str]:
    """``workers=4`` as ``--workers 4``; a True boolean as the bare flag; False and None as nothing."""
    flag = "--" + name.replace("_", "-")
    if value is None or value is False:
        return []
    if value is True:
        return [flag]
    return [flag, str(value)]


@dataclass
class EntryPointStudy:
    """A registered study, through its entry point ``experiments/<name>.py``.

    ``stage_names`` and ``reads`` are the entry point's own (read from the
    study's state module or from the entry point's source, never retyped);
    ``options`` the flags it accepts beyond the stage, documented for the
    command line. ``repo`` is the repository the study lives in.
    """

    name: str
    repo: Path
    stage_names: tuple[str, ...]
    reads: tuple[str, ...]
    options: tuple[str, ...] = ("--workers", "--run")
    description: str = ""
    python: str = field(default_factory=lambda: sys.executable)

    @property
    def registration(self) -> Path:
        return self.repo / "docs" / "preregistration" / f"{self.name}.md"

    @property
    def config(self) -> Path:
        return self.repo / "configs" / f"{self.name}.yaml"

    @property
    def entry_point(self) -> Path:
        return self.repo / "experiments" / f"{self.name}.py"

    def stages(self) -> tuple[str, ...]:
        """The development stages, then each block's open and read, in order."""
        steps = [*self.stage_names]
        for block in self.reads:
            steps += [f"open {block}", f"read {block}"]
        return tuple(steps)

    def argv(self, stage: str, *args: str, **options: Any) -> list[str]:
        """The entry point's arguments for ``stage``: ``"read H"`` and ``("read", "H")`` alike."""
        words = [*stage.split(), *args] if stage else [*args]
        for name, value in options.items():
            words += _flag(name, value)
        return words

    def invoke(self, words: Sequence[str]) -> StudyRun:
        """The entry point in a fresh interpreter, in the study's repository."""
        argv = (self.python, "-m", f"experiments.{self.name}", *words)
        # The study's own package first, then the one running this, then the caller's path.
        here = str(Path(__file__).resolve().parents[2])
        parts = (str(self.repo / "src"), here, os.environ.get("PYTHONPATH"))
        path = os.pathsep.join(p for p in parts if p)
        done = subprocess.run(
            argv,
            cwd=self.repo,
            env={**os.environ, "PYTHONPATH": path},
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            result = json.loads(done.stdout) if done.stdout.strip() else None
        except json.JSONDecodeError:
            result = None
        return StudyRun(tuple(argv), done.returncode, done.stdout, done.stderr, result)

    def run(self, stage: str, *args: str, **options: Any) -> StudyRun:
        return self.invoke(self.argv(stage, *args, **options))

    def status(self) -> StudyRun:
        return self.invoke(["status"])

    def pages(self) -> StudyRun:
        return self.invoke(["pages"])

    def dry_run(self, keep: Path | None = None, **options: Any) -> StudyRun:
        return self.invoke(["--dry-run", *_flag("keep", keep), *self.argv("", **options)])


def declared(
    entry_point: Path, names: Sequence[str] = ("STAGES", "READS")
) -> dict[str, tuple[str, ...]]:
    """Tuples of strings an entry point assigns at module level, read without importing it."""
    found: dict[str, tuple[str, ...]] = {}
    tree = ast.parse(entry_point.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        value = getattr(node, "value", None)
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    literal = ast.literal_eval(value)  # type: ignore[arg-type]
                except ValueError:
                    continue
                if isinstance(literal, tuple) and all(isinstance(v, str) for v in literal):
                    found[target.id] = literal
    return found
