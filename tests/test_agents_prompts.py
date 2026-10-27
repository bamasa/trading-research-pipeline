"""The harness's documents: CLAUDE.md states every binding rule, and every
command a role prompt, the protocol or CLAUDE.md names exists, with its options."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import typer.main
import yaml

from agents.workflows.policy import Command, named_commands, parse_command
from agents.workflows.prompts import STATUSES, system_prompt
from agents.workflows.research_loop import parser as harness_parser
from trading_research.cli import app

REPO = Path(__file__).resolve().parent.parent
ROLE_FILES = sorted((REPO / "agents/roles").glob("*.md"))
DOCUMENTS = [
    REPO / "CLAUDE.md",
    REPO / "agents/protocol.md",
    REPO / "agents/README.md",
    *ROLE_FILES,
]


def cli_options() -> dict[str, set[str]]:
    group = typer.main.get_command(app)
    commands = getattr(group, "commands", {})
    return {
        name: {o for p in cmd.params for o in (*p.opts, *p.secondary_opts) if o.startswith("-")}
        for name, cmd in commands.items()
    }


CLI = cli_options()
HARNESS_COMMANDS = {"run", "calendar", "null-hash", "check"}


def problem(command: Command) -> str | None:
    """Why ``command`` does not exist as written, or ``None``."""
    if command.kind == "cli":
        if not command.name:
            return None
        if command.name not in CLI:
            return f"trading-research has no {command.name}"
        unknown = [a.split("=")[0] for a in command.args if a.startswith("-")]
        unknown = [a for a in unknown if a not in CLI[command.name]]
        return f"{command.name} has no {unknown}" if unknown else None
    if command.kind == "entry":
        if not (REPO / "experiments" / f"{command.name}.py").exists():
            return f"experiments/{command.name}.py does not exist"
        module = importlib.import_module(f"experiments.{command.name}")
        try:
            module.parser().parse_args(list(command.args))
        except SystemExit:
            return f"experiments.{command.name} does not accept {command.args}"
        return None
    if command.kind == "harness":
        ok = command.name in HARNESS_COMMANDS or not command.name
        return None if ok else f"the harness has no {command.name}"
    return None


def test_the_cli_is_read_from_the_application_itself() -> None:
    assert {"download", "mm-backtest", "reproduce", "describe-schema"} <= set(CLI)
    assert "--allow-heldout" in CLI["mm-backtest"] and "--no-fetch" in CLI["reproduce"]


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: p.name)
def test_every_command_a_document_names_exists(path: Path) -> None:
    named = named_commands(path.read_text())
    assert named, f"{path.name} names no command"
    for text in named:
        command = parse_command(text)
        assert command is not None, f"{path.name}: {text!r} is not one command"
        assert problem(command) is None, f"{path.name}: {text!r}: {problem(command)}"


def test_the_checker_catches_a_command_that_does_not_exist() -> None:
    for text in (
        "trading-research fetch-everything",
        "trading-research download --since 2024-01-01",
        "python -m experiments.no_such_study status",
        "python -m experiments.spot_2026 publish",
        "python -m agents.workflows.research_loop deploy",
    ):
        command = parse_command(text)
        assert command is not None and problem(command) is not None, text
    help_text = harness_parser().format_help()
    assert all(name in help_text for name in HARNESS_COMMANDS)


#: Each binding rule and protocol rule of CLAUDE.md, by the words that state it.
RULES = {
    "nothing is published": ["No `git push`", "no `git remote`", "no pull request", "`main` is never"],
    "worktrees": ["another worktree is never checked out or edited"],
    "frozen code": ["Frozen study code is never edited", "`src/`", "`experiments/`",
                    "`configs/`", "`docs/preregistration/`", "`pyproject.toml`", "`uv.lock`"],
    "held-out days": ["only through the ledger", "the owner", "granted that block",
                      "open entry is committed", "A month archive"],
    "after a freeze": ["Nothing that decides a status changes after a freeze", "Fees, thresholds"],
    "provenance": ["\"the author's 2025 research\"", "another organisation"],
    "language": ["English", "No emoji", "No co-author lines", "credits an assistant"],
    "secrets": ["only from environment variables", "Nothing secret is committed"],
    "size and checks": ["about 120 lines", "`ruff format`", "`ruff check`", "`mypy`", "five minutes"],
    "registration first": ["Registration before data", "null-placeholder sha256"],
    "kill first": ["Hypothesis and kill condition first", "committed, before the run"],
    "logged": ["Every run is logged"],
    "amendments": ["Amendments are dated and appended"],
    "no re-ranking": ["No re-ranking after a read", "can only lower a status"],
    "full report": ["kills, placebos and neighbourhoods", "never quoted as a result"],
    "negative results": ["Negative results are results"],
    "never": ["`--second-read`", "Push, add or change a remote", "Chain commands"],
    "hand back": ["Handing back", "sha256", "`handback`", "registered sentence"],
}  # fmt: skip


@pytest.mark.parametrize("rule", sorted(RULES))
def test_claude_md_states_each_rule(rule: str) -> None:
    text = " ".join((REPO / "CLAUDE.md").read_text().split())
    missing = [words for words in RULES[rule] if " ".join(words.split()) not in text]
    assert not missing, f"CLAUDE.md does not state {rule!r}: {missing}"


def test_claude_md_is_short_and_lists_every_cli_command() -> None:
    text = (REPO / "CLAUDE.md").read_text()
    assert len(text.splitlines()) < 150
    for name in CLI:
        assert re.search(rf"`(trading-research )?{re.escape(name)}[` ]", text), name


@pytest.mark.parametrize("path", ROLE_FILES, ids=lambda p: p.stem)
def test_each_role_states_its_inputs_outputs_reads_limits_and_hand_back(path: Path) -> None:
    text = path.read_text()
    for heading in ("## Inputs", "## What you produce", "## Never", "## How you hand back"):
        assert heading in text, f"{path.name} has no {heading}"
    assert re.search(r"^## What you may (read|run)", text, re.MULTILINE), path.name
    example = re.findall(r"```handback\n(.*?)```", text, re.DOTALL)
    assert len(example) == 1, path.name
    data = yaml.safe_load(example[0])
    assert data["role"] == path.stem and data["status"] in STATUSES[path.stem]


def test_a_system_prompt_is_the_rules_the_loop_and_the_role_in_that_order() -> None:
    prompt = system_prompt(REPO, "skeptic")
    order = [prompt.index(f"<!-- {name} -->") for name in
             ("CLAUDE.md", "agents/protocol.md", "agents/roles/skeptic.md")]  # fmt: skip
    assert order == sorted(order)
    assert set(STATUSES) == {p.stem for p in ROLE_FILES}
