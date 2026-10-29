"""The tool policy: what each role's agent may run, read and write.

Every tool call an agent makes in the research loop is decided by
:meth:`ToolPolicy.decide` before it runs. The orchestrator installs the same
decision as a hook on every tool call and as the SDK's tool callback, so no
allow rule or mode can widen it. The rules are ``CLAUDE.md``'s: the CLI's
commands, the study entry points, ``pytest`` and read-only ``git``, one at a
time; never a push, a remote, an edit of study code or a write under ``data/``;
no command whose dates touch a day a registration protects.

**The protected calendar** is read from every ``configs/*.yaml`` with a
``blocks`` section, in the working tree and on every local branch (a study
registered on another branch protects its days here too): each dated block,
given as a start/end mapping or as a two-date list, whatever its name, unless
it is free: its role is development, warm-up or a descriptive window inside
development, or it reads ``free``, or, given without a role, it is named D or
W, the repository's names for development and warm-up; and each date of any
other list of dates. An archive published by
the month (``fundingRate``, and ``klines``, fetched a month at a time) counts
as its whole month, since the month's file holds every day of it. Only the
owner's grant for the run, ``<config stem>:<block>``, lifts a block, and only
for the study's own entry point and the market-making ``--allow-heldout``: a
plain download fetches free days only, and an entry point's ``read`` also needs
the block's open entry committed in its ledger and its read unfinished.
"""

from __future__ import annotations

import calendar as _calendar
import json
import re
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROLES: tuple[str, ...] = ("researcher", "registrar", "executor", "skeptic", "scribe")
#: The registered studies' single entry points, ``experiments/<name>.py``.
ENTRY_POINTS: tuple[str, ...] = ("short_horizon_2025", "spot_2026")
FREE_ROLES = frozenset({"development", "warm_up", "descriptive_window_inside_D"})
#: CLI commands that fetch the days their ``--start`` and ``--end`` name.
FETCHES = frozenset({"download", "download-book", "breaks", "mm-backtest"})
#: Plain downloads, which fetch free days only: a protected day is reached
#: through its study's entry point and ledger, never by a grant alone.
FREE_DAYS_ONLY = frozenset({"download", "download-book"})
#: Where a study's entry point keeps its ledger, ``{"<block>": [entries]}``.
LEDGER = "experiments/results/{study}_ledger.json"
#: Where a command writes under ``data/`` when no ``--output`` is given.
DATA_OUTPUTS = {
    "download": "data/raw",
    "download-book": "data/book",
    "generate-demo-data": "data/demo",
}
#: Options whose value is a path, which stays inside the repository.
PATH_OPTIONS = ("--input", "--raw", "--book", "--trades", "--funding", "--prepared",
                "--features", "--model", "--predictions", "--prices", "--params", "--output", "-o")  # fmt: skip
MONTHLY_KINDS = frozenset({"fundingRate", "klines"})
DATE_OPTIONS = ("--start", "--end", "--train-start", "--train-end")
#: Hosts the repository's downloaders read market data from.
DATA_HOSTS: tuple[str, ...] = (
    "api.binance.com",
    "api.bybit.com",
    "data.binance.vision",
    "fapi.binance.com",
    "public.bybit.com",
    "quote-saver.bycsi.com",
    "s3-ap-northeast-1.amazonaws.com",
)
GIT_READ = frozenset({"log", "diff", "status", "show"})
#: Directories no role reads or searches by hand: the data, the fitted models,
#: the pipeline's artefacts and git's own store.
UNREAD = frozenset({"data", "models", "artifacts", ".git"})
GIT_WRITES_FILES = ("--output", "--ext-diff", "--textconv")
OPERATORS = ";&|<>()"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def allow(reason: str) -> Decision:
    return Decision(True, reason)


def deny(reason: str) -> Decision:
    return Decision(False, reason)


@dataclass(frozen=True)
class Protected:
    """One protected range of one registration."""

    config: str
    block: str
    first: date
    last: date

    @property
    def key(self) -> str:
        return f"{self.config}:{self.block}"


#: Blocks without a role that are free by the repository's naming.
FREE_NAMES = frozenset({"D", "W"})


def _span(name: str, value: Any) -> tuple[date, date] | bool | None:
    """A block's range; ``True`` when it is free; ``None`` when it is not a range."""
    if isinstance(value, dict):
        role = value.get("role")
        if (
            role in FREE_ROLES
            or value.get("reads") == "free"
            or (role is None and name in FREE_NAMES)
        ):
            return True
        first, last = value.get("start"), value.get("end")
        return (first, last) if isinstance(first, date) and isinstance(last, date) else None
    pair = isinstance(value, list) and len(value) == 2 and all(isinstance(d, date) for d in value)
    if pair and name in FREE_NAMES:
        return True
    return (value[0], value[1]) if pair and value[0] <= value[1] else None


def calendar_of(stem: str, text: str) -> list[Protected]:
    """The ranges one registration's YAML protects."""
    loaded = yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    blocks = loaded.get("blocks") if isinstance(loaded, dict) else None
    found: list[Protected] = []
    for name, value in (blocks or {}).items() if isinstance(blocks, dict) else []:
        span = _span(str(name), value)
        if isinstance(span, tuple):
            found.append(Protected(stem, str(name), *span))
        elif span is None and isinstance(value, list):
            days = sorted(d for d in value if isinstance(d, date))
            found += [Protected(stem, str(name), a, b) for a, b in _runs(days)]
    return found


_BLOBS: dict[str, list[Protected]] = {}


def _git(repo: Path, *args: str) -> str | None:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else None


def branch_configs(repo: Path) -> dict[str, list[tuple[str, str]]]:
    """Each local branch's registration YAMLs, as (stem, blob id)."""
    found: dict[str, list[tuple[str, str]]] = {}
    for branch in (
        _git(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads") or ""
    ).split():
        listing = _git(repo, "ls-tree", branch, "configs/") or ""
        rows = [line.split(None, 3) for line in listing.splitlines()]
        found[branch] = [
            (Path(r[3]).stem, r[2]) for r in rows if len(r) == 4 and r[3].endswith(".yaml")
        ]
    return found


def protected_calendar(repo: Path) -> tuple[Protected, ...]:
    """Every range a registration protects, in the working tree's ``configs/``
    and on every local branch, each range once."""
    found: list[Protected] = []
    for path in sorted((repo / "configs").glob("*.yaml")):
        found += calendar_of(path.stem, path.read_text())
    for _branch, configs in sorted(branch_configs(repo).items()):
        for stem, blob in configs:
            if blob not in _BLOBS:
                _BLOBS[blob] = calendar_of(stem, _git(repo, "cat-file", "blob", blob) or "")
            found += _BLOBS[blob]
    return tuple(dict.fromkeys(found))


def _runs(days: Sequence[date]) -> list[tuple[date, date]]:
    """Consecutive days as ranges."""
    runs: list[tuple[date, date]] = []
    for day in days:
        if runs and (day - runs[-1][1]).days == 1:
            runs[-1] = (runs[-1][0], day)
        else:
            runs.append((day, day))
    return runs


def whole_months(first: date, last: date) -> tuple[date, date]:
    end = _calendar.monthrange(last.year, last.month)[1]
    return first.replace(day=1), last.replace(day=end)


@dataclass(frozen=True)
class Command:
    """One command the policy knows: its kind, its name and its arguments."""

    kind: str  # cli, entry, pytest, git or harness
    name: str  # the subcommand, the entry point's module, or git's verb
    args: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


def words(text: str) -> list[str] | None:
    """The command's words, or ``None`` when the shell would do more than run it."""
    if any(c in text for c in "$`\n\r"):
        return None
    lexer = shlex.shlex(text, posix=True, punctuation_chars=OPERATORS)
    lexer.whitespace_split = True
    try:
        found = list(lexer)
    except ValueError:
        return None
    if any(w and set(w) <= set(OPERATORS) for w in found):
        return None
    return found


def parse_command(text: str, python: Iterable[str] = (), cli: Iterable[str] = ()) -> Command | None:
    """Read one command in any of the prefixes this repository uses, or ``None``.

    ``PYTHONPATH=...`` assignments, ``uv run [--no-sync]``, ``python -m``,
    ``.venv/bin/`` and the interpreter and CLI paths the run names are accepted
    as prefixes; anything else is not a command the policy knows.
    """
    found = words(text)
    if not found:
        return None
    env: list[tuple[str, str]] = []
    while found and "=" in found[0] and not found[0].startswith("-"):
        key, _, value = found.pop(0).partition("=")
        env.append((key, value))
    if found[:2] == ["uv", "run"]:
        found = found[2:]
        if found[:1] == ["--no-sync"]:
            found = found[1:]
    if not found:
        return None
    head, rest = found[0], found[1:]
    pythons = {"python", "python3", ".venv/bin/python", *python}
    if head in {"trading-research", ".venv/bin/trading-research", "trp", ".venv/bin/trp", *cli}:
        return Command("cli", rest[0] if rest else "", tuple(rest[1:]), tuple(env))
    if head in {"pytest", ".venv/bin/pytest"}:
        return Command("pytest", "pytest", tuple(rest), tuple(env))
    if head == "git":
        return Command("git", rest[0] if rest else "", tuple(rest[1:]), tuple(env))
    if head in pythons and len(rest) >= 2 and rest[0] == "-m":
        module, args = rest[1], tuple(rest[2:])
        if module == "pytest":
            return Command("pytest", "pytest", args, tuple(env))
        if module.startswith("experiments."):
            return Command("entry", module.removeprefix("experiments."), args, tuple(env))
        if module == "agents.workflows.research_loop":
            return Command("harness", args[0] if args else "", args[1:], tuple(env))
    return None


def option(args: Sequence[str], *names: str) -> str | None:
    """The value of the first of ``names`` in ``args`` (``--x v`` or ``--x=v``)."""
    for i, arg in enumerate(args):
        for name in names:
            if arg == name and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def split_entry_args(args: Sequence[str]) -> tuple[list[str], set[str]]:
    """An entry point's positionals (stage or action, block) and its option names."""
    positionals: list[str] = []
    options: set[str] = set()
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg.startswith("-"):
            options.add(arg.split("=", 1)[0])
            skip = "=" not in arg and arg in {"--workers", "--run", "--frozen-2025", "--keep"}
        else:
            positionals.append(arg)
    return positionals, options


#: What each role may run beyond read-only git: CLI commands and entry-point
#: actions (``None`` is every one), pytest, and the harness's helpers.
CLI_BY_ROLE: dict[str, frozenset[str] | None] = {
    "researcher": frozenset({"describe-schema"}),
    "registrar": frozenset({"describe-schema"}),
    "executor": None,
    "skeptic": frozenset({"describe-schema", "review", "validate-data"}),
    "scribe": frozenset({"describe-schema"}),
}
ENTRY_BY_ROLE: dict[str, frozenset[str] | None] = {
    "researcher": frozenset({"status"}),
    "registrar": frozenset({"status"}),
    "executor": None,
    "skeptic": frozenset({"status", "--dry-run"}),
    "scribe": frozenset({"status"}),
}
PYTEST_ROLES = frozenset({"executor", "skeptic"})
HARNESS_HELPERS = frozenset({"calendar", "check", "null-hash"})
#: The subcommands of ``src/trading_research/cli.py`` (a test keeps this in step).
CLI_COMMANDS = frozenset(
    {
        "generate-demo-data", "validate-data", "download", "prepare", "select", "train",
        "predict", "backtest", "retrain-search", "breaks", "exit-search", "screen", "review",
        "discover", "download-book", "describe-schema", "reproduce", "mm-backtest",
    }
)  # fmt: skip


#: ``paper`` subcommands that read the live stream, refused to every role.
PAPER_LIVE = frozenset({"run", "record"})
#: ``paper`` subcommands that read a stored day, allowed only outside protected days.
PAPER_STORED = frozenset({"replay", "parity"})
#: ``paper`` options whose value is a path.
PAPER_PATHS = ("--day", "--root", "--prints", "--run-dir", "--model", "--output", "-o")


@dataclass
class ToolPolicy:
    """One role's policy for one run of one study."""

    repo: Path
    role: str
    study: str
    grants: frozenset[str] = frozenset()
    web_hosts: frozenset[str] = frozenset()
    python: tuple[str, ...] = ()
    cli: tuple[str, ...] = ()
    calendar: tuple[Protected, ...] | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; the roles are {ROLES}")
        if not self.study or not all(c.isalnum() or c == "_" for c in self.study):
            raise ValueError(f"a study is named by letters, digits and underscores: {self.study!r}")
        self.repo = self.repo.resolve()
        if self.calendar is None:
            self.calendar = protected_calendar(self.repo)

    def decide(self, tool: str, data: dict[str, Any]) -> Decision:
        if tool == "Bash":
            if data.get("run_in_background") or data.get("dangerouslyDisableSandbox"):
                return deny("a command runs in the foreground and in the sandbox")
            return self.bash(str(data.get("command", "")))
        if tool in {"Read", "Glob", "Grep", "NotebookRead"}:
            return self.read(tool, data)
        if tool in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
            return self.write(tool, data)
        if tool == "WebFetch":
            return self.web(str(data.get("url", "")))
        if tool == "WebSearch":
            return deny("no web search: a results page can show data of a block nobody opened")
        if tool == "TodoWrite":
            return allow("bookkeeping")
        return deny(f"{tool} is not a tool of the research loop")

    # -- commands ----------------------------------------------------------------

    def bash(self, text: str) -> Decision:
        command = parse_command(text, self.python, self.cli)
        if command is None:
            return deny(
                "one command of the repository's, unchained: trading-research, a study entry "
                "point, pytest, read-only git, or the harness's helpers"
            )
        for key, value in command.env:
            if key != "PYTHONPATH" or not all(self.inside(p) for p in value.split(":")):
                return deny(f"only PYTHONPATH inside the repository may be set, not {key}")
        if command.kind == "git":
            if command.name not in GIT_READ:
                return deny(f"git {command.name} is not read-only; only log, diff, status, show")
            if any(a.startswith(GIT_WRITES_FILES) for a in command.args):
                return deny("git writes nothing: no --output, --ext-diff or --textconv")
            return allow(f"git {command.name} is read-only")
        if command.kind == "pytest":
            if self.role not in PYTEST_ROLES:
                return deny(f"the {self.role} does not run the tests")
            if any(a.startswith(("--junitxml", "--basetemp")) for a in command.args):
                return deny("pytest writes no file outside its caches")
            return allow("the tests read synthetic data and fixtures")
        if command.kind == "harness":
            if command.name not in HARNESS_HELPERS:
                return deny("an agent never starts a loop; only calendar, check, null-hash")
            return allow(f"the harness's {command.name}")
        if command.kind == "entry":
            return self.entry(command)
        return self.cli_command(command)

    def entry(self, command: Command) -> Decision:
        own = (self.repo / "experiments" / f"{self.study}.py").exists()
        known = {*ENTRY_POINTS, *([self.study] if own else [])}
        if command.name not in known:
            return deny(f"experiments.{command.name} is not a study entry point of this run")
        positionals, options = split_entry_args(command.args)
        action = "--dry-run" if "--dry-run" in options else (positionals[:1] or [""])[0]
        allowed = ENTRY_BY_ROLE[self.role]
        if allowed is not None and action not in allowed:
            return deny(f"the {self.role} runs only {sorted(allowed)} of an entry point")
        if command.name != self.study and action != "status":
            return deny(f"another study's entry point runs only status in a {self.study} run")
        if "--keep" in options:
            return deny("--keep writes the dry run's files outside the run")
        if action in {"open", "read"}:
            block = positionals[1] if len(positionals) > 1 else ""
            key = f"{command.name}:{block}"
            if key not in self.grants:
                return deny(f"{action} needs the owner's grant {key} for this run")
            problem = self.open_entry_problem(command.name, block) if action == "read" else None
            if problem is not None:
                return deny(problem)
        return allow(f"experiments.{command.name} {action}")

    def open_entry_problem(self, study: str, block: str) -> str | None:
        """Why ``read <block>`` may not run: its open entry is not in HEAD's
        ledger, or the read it opened has finished (a read is never re-run)."""
        path = LEDGER.format(study=study)
        done = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=self.repo,
                              capture_output=True, text=True, check=False)  # fmt: skip
        try:
            entries = json.loads(done.stdout).get(block) if done.returncode == 0 else None
        except (ValueError, AttributeError):
            entries = None
        if not isinstance(entries, list) or not entries:
            return f"read {block} needs its open entry committed in {path}; the owner commits it"
        if isinstance(entries[-1], dict) and entries[-1].get("results_sha256"):
            return f"{study}:{block} has been read (its results_sha256 is in {path}); never again"
        return None

    def cli_command(self, command: Command) -> Decision:
        name, args = command.name, command.args
        if name == "paper":
            return self.paper(args)
        if name not in CLI_COMMANDS:
            return deny(f"trading-research has no command {name!r}")
        allowed = CLI_BY_ROLE[self.role]
        if allowed is not None and name not in allowed:
            return deny(f"the {self.role} does not run trading-research {name}")
        if "--second-read" in args:
            return deny("a second read cannot change a status; it is never run by an agent")
        if name == "reproduce" and "--no-fetch" not in args:
            return deny("reproduce fetches what is missing; only --no-fetch runs")
        if name == "breaks" and option(args, "--prices") is None:
            return deny("breaks fetches bars into the shared data/bars; give it --prices")
        problem = self.path_problem(name, args)
        if problem is not None:
            return deny(problem)
        days = [option(args, o) for o in DATE_OPTIONS]
        named = [d for d in days if d is not None]
        if name in FETCHES and option(args, "--prices") is None and len(named) < 2:
            return deny(f"{name} names the days it reads with --start and --end")
        try:
            parsed = sorted(date.fromisoformat(d) for d in named)
        except ValueError:
            return deny(f"dates are YYYY-MM-DD: {named}")
        if not parsed:
            return allow(f"trading-research {name}")
        first, last = parsed[0], parsed[-1]
        if option(args, "--kind", "-k") in MONTHLY_KINDS:
            first, last = whole_months(first, last)
        grants = frozenset() if name in FREE_DAYS_ONLY else self.grants
        return self.days(first, last, f"trading-research {name}", grants)

    def paper(self, args: Sequence[str]) -> Decision:
        """``trading-research paper``: never live, and a stored day only outside
        every protected range.

        ``run`` and ``record`` read the venue's live public stream: today, which
        inside a registered prospective block is that block's held-out data, so
        no role runs them and no grant lifts it. ``replay`` and ``parity`` read a
        stored day, which must name its date (``--day`` a date, or a file whose
        name carries it) and lie outside every protected range of every
        registration, grants included: a protected day is read only through its
        study's entry point and ledger. ``status`` reads a run directory.
        """
        sub = args[0] if args else ""
        rest = args[1:]
        if sub in PAPER_LIVE:
            return deny(
                f"paper {sub} reads the live stream, which inside a registered prospective "
                "block is that block's held-out data; no role runs it"
            )
        if sub not in PAPER_STORED and sub != "status":
            return deny(f"trading-research paper has no subcommand {sub!r} a role runs")
        allowed = CLI_BY_ROLE[self.role]
        if allowed is not None and "paper" not in allowed:
            return deny(f"the {self.role} does not run trading-research paper")
        for flag in PAPER_PATHS:
            value = option(rest, flag)
            if value is not None and not self.inside(value):
                return deny(f"{flag} {value} is outside the repository")
        if sub == "status":
            target = next((a for a in rest if not a.startswith("-")), None)
            if target is not None and not self.inside(target):
                return deny(f"{target} is outside the repository")
            return allow("paper status reads a run directory")
        day = option(rest, "--day")
        found = re.findall(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", day or "")
        if not found:
            return deny(
                f"paper {sub} names its day: --day a date, or a file whose name carries the date"
            )
        try:
            days = sorted(date.fromisoformat(d) for d in found)
        except ValueError:
            return deny(f"dates are YYYY-MM-DD: {found}")
        return self.days(days[0], days[-1], f"trading-research paper {sub}", frozenset())

    def path_problem(self, name: str, args: Sequence[str]) -> str | None:
        """Paths stay inside the repository, and a write under ``data/`` stays
        under this study's root, ``data/<study>/``."""
        for flag in PATH_OPTIONS:
            value = option(args, flag)
            if value is not None and not self.inside(value):
                return f"{flag} {value} is outside the repository"
        target = option(args, "--output", "-o") or DATA_OUTPUTS.get(name)
        rel = self.relative(target) if target else None
        if rel is None or not (rel == "data" or rel.startswith("data/")):
            return None
        roots = [f"data/{self.study}", *(["data/demo"] if name == "generate-demo-data" else [])]
        if any(rel == r or rel.startswith(r + "/") for r in roots):
            return None
        return f"{name} writes {rel}; under data/ this run writes only data/{self.study}/"

    def days(
        self, first: date, last: date, what: str, grants: frozenset[str] | None = None
    ) -> Decision:
        hits = [p for p in self.calendar or () if p.first <= last and first <= p.last]
        closed = sorted({p.key for p in hits} - (self.grants if grants is None else grants))
        if closed:
            return deny(f"{what} reads {first} to {last}, which touches {', '.join(closed)}")
        return allow(f"{what} reads {first} to {last}, outside every protected range")

    # -- files -------------------------------------------------------------------

    def inside(self, path: str) -> bool:
        resolved = (self.repo / path).resolve()
        return resolved == self.repo or self.repo in resolved.parents

    def relative(self, path: str) -> str | None:
        resolved = (self.repo / path).resolve()
        if resolved != self.repo and self.repo not in resolved.parents:
            return None
        return resolved.relative_to(self.repo).as_posix()

    def read(self, tool: str, data: dict[str, Any]) -> Decision:
        path = str(data.get("file_path") or data.get("notebook_path") or data.get("path") or ".")
        rel = self.relative(path)
        if rel is None:
            return deny(f"{path} is outside the repository")
        if rel.split("/")[0] in UNREAD or rel.split("/")[-1].startswith(".env"):
            return deny(f"{rel}: data is read only by the commands, through their access")
        pattern = data.get("pattern") if tool == "Glob" else data.get("glob")
        if tool in {"Glob", "Grep"} and pattern:
            problem = pattern_problem(tool, rel, str(pattern))
            if problem is not None:
                return deny(problem)
        elif tool == "Grep" and rel == "." and not self.unread_ignored():
            return deny("a search from the root needs data/, models/ and artifacts/ ignored by git")
        return allow(f"read {rel}")

    def unread_ignored(self) -> bool:
        """Whether git ignores every unread directory, so a search from the root,
        which honours the ignore rules, skips them."""
        names = sorted(f"{name}/" for name in UNREAD - {".git"})
        done = subprocess.run(["git", "check-ignore", *names], cwd=self.repo,
                              capture_output=True, text=True, check=False)  # fmt: skip
        return set(done.stdout.split()) == set(names)

    def writable(self) -> dict[str, str]:
        """The files this role may write for this study: ``any`` or ``append``."""
        s = self.study
        doc = self.repo / "docs" / "preregistration" / f"{s}.md"
        frozen = (self.repo / "docs" / f"{s}_code_freeze.md").exists() or (
            doc.exists() and "Frozen configuration sha256:" in doc.read_text()
        )
        if self.role == "registrar":
            if frozen:
                return {f"docs/preregistration/{s}.md": "append"}
            return {f"docs/preregistration/{s}.md": "any", f"configs/{s}.yaml": "any"}
        if self.role == "scribe":
            return {"README.md": "any", "docs/findings.md": "any",
                    f"docs/{s}_run_log.md": "append", f"docs/{s}_defects.md": "append"}  # fmt: skip
        return {}

    def write(self, tool: str, data: dict[str, Any]) -> Decision:
        path = str(data.get("file_path") or data.get("notebook_path") or "")
        rel = self.relative(path) if path else None
        mode = self.writable().get(rel or "")
        if rel is None or mode is None:
            return deny(f"the {self.role} does not write {rel or path}")
        if mode == "append" and not self.appends(tool, rel, data):
            return deny(f"{rel} is only appended to")
        return allow(f"{self.role} writes {rel}")

    def appends(self, tool: str, rel: str, data: dict[str, Any]) -> bool:
        target = self.repo / rel
        before = target.read_text() if target.exists() else ""
        if tool == "Write":
            return str(data.get("content", "")).startswith(before)
        old, new = str(data.get("old_string", "")), str(data.get("new_string", ""))
        end = before.rstrip("\n").endswith(old.rstrip("\n"))
        return tool == "Edit" and bool(old) and end and new.startswith(old)

    def web(self, url: str) -> Decision:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        if any(host == h or host.endswith("." + h) for h in DATA_HOSTS):
            return deny(f"{host} serves market data, which is read only through the ledger")
        if self.role != "registrar" or host not in self.web_hosts:
            return deny(f"{host or url} is not a source the owner listed for the registrar")
        return allow(f"the registrar reads {host} for its fees")


def pattern_problem(tool: str, rel: str, pattern: str) -> str | None:
    """Why a search's glob could reach an unread directory, or ``None``.

    A glob is read from the search's path; one without a slash in a search's
    ``glob`` matches at any depth. From the repository's root a glob must start
    with a named directory or file, never a wildcard, so that no match can fall
    under ``data/``, ``models/``, ``artifacts/`` or ``.git/``."""
    if pattern.startswith("!"):
        return None  # an exclusion narrows a search
    if pattern.startswith("/") or ".." in pattern.split("/"):
        return "a pattern stays inside the repository"
    if tool == "Grep" and "/" not in pattern:
        pattern = "**/" + pattern
    full = pattern if rel == "." else f"{rel}/{pattern}"
    first = full.split("/")[0]
    if first in UNREAD or any(c in first for c in "*?[{"):
        return (
            f"{full} could reach data/, models/, artifacts/ or .git/; give the search a "
            "path, or start its glob with a directory"
        )
    return None


CODE = re.compile(r"```[^\n]*\n(.*?)```|`([^`\n]+)`", re.DOTALL)
NAMES_A_COMMAND = re.compile(
    r"(PYTHONPATH=\S+ )?(uv run (--no-sync )?)?(\.venv/bin/)?"
    r"(trading-research(\s|$)|python3? -m (experiments\.|agents\.|pytest(\s|$))|pytest(\s|$))"
)


def named_commands(markdown: str) -> list[str]:
    """Every command a text names in code (fenced lines, joined at a trailing
    backslash, and inline spans), each placeholder ``<...>`` read as ``X``."""
    found: list[str] = []
    for block, span in CODE.findall(markdown):
        text = block.replace("\\\n", " ") if block else span
        for line in text.splitlines():
            line = re.sub(r"<[^<>\s]+>", "X", line.strip().removeprefix("$ ").strip())
            if NAMES_A_COMMAND.match(line):
                found.append(line)
    return found
