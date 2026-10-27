"""The harness's documents: CLAUDE.md states every binding rule, and every
command a role prompt, the protocol or CLAUDE.md names exists, with its options."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from agents.workflows.policy import Command, named_commands, parse_command
from agents.workflows.prompts import (
    HARNESS_COMMANDS,
    STATUSES,
    cli_options,
    command_problem,
    prompt_problems,
    read_handback,
    system_prompt,
)
from agents.workflows.research_loop import parser as harness_parser

REPO = Path(__file__).resolve().parent.parent
ROLE_FILES = sorted((REPO / "agents/roles").glob("*.md"))
DOCUMENTS = [
    REPO / "CLAUDE.md",
    REPO / "agents/protocol.md",
    REPO / "agents/README.md",
    *ROLE_FILES,
]
CLI = cli_options()


def problem(command: Command) -> str | None:
    return command_problem(REPO, command)


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


def test_the_loop_checks_its_own_prompts_before_any_model_is_called() -> None:
    assert prompt_problems(REPO) == []


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
    "nothing is published": ["No `git push`", "no `git remote`", "no pull request", "`main` is never",
                             "A role of the harness never commits"],
    "worktrees": ["another worktree is never checked out or edited"],
    "frozen code": ["Frozen study code is never edited", "`src/`", "`experiments/`",
                    "`configs/`", "`docs/preregistration/`", "`pyproject.toml`", "`uv.lock`"],
    "held-out days": ["only through the ledger", "the owner", "granted that block",
                      "open entry is committed", "A month archive"],
    "after a freeze": ["Nothing that decides a status changes after a freeze", "Fees, thresholds"],
    "provenance": ["\"the author's 2025 research\"", "another organisation"],
    "language": ["English", "No emoji", "No `Co-Authored-By`", "trailer line",
                 "credits an assistant, a model or a tool"],
    "secrets": ["only from environment variables", "read at run time", "Nothing secret is committed",
                "logged or written to a run directory"],
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


STEP_OF = {"researcher": "propose", "registrar": "register", "executor": "run",
           "skeptic": "review", "scribe": "record"}  # fmt: skip


@pytest.mark.parametrize("path", ROLE_FILES, ids=lambda p: p.stem)
def test_each_roles_handback_example_passes_the_orchestrators_check(path: Path) -> None:
    example = re.findall(r"```handback\n(.*?)```", path.read_text(), re.DOTALL)[0]
    reply = "```handback\n" + example.replace("<64 hex>", "a" * 64) + "```"
    assert read_handback(path.stem, STEP_OF[path.stem], reply)["role"] == path.stem


#: What each role's prompt must say, in its own words, beside the gate that enforces it.
DUTIES = {
    "researcher": ["the condition that would refute it and the outcome you expect, before anything "
                   "is computed", "You read no market data", "Propose a hypothesis without a kill "
                   "condition and an expected outcome", "is declined, with the reason",
                   "**The information set**"],
    "registrar": ["Fill a gap in the plan yourself: refuse", "null-placeholder sha256",
                  "only append a dated amendment"],
    "executor": ["never work around a refusal: a refusal is a result", "`status: waiting`",
                 "Re-run a read, a stage or a placebo because of what its result was"],
    "skeptic": ["Your job is to refute", "Work through all three, every time",
                "the refutation you tried", "**Leakage and look-ahead.**",
                "**Fidelity to the registration.**", "**Execution realism and money.**",
                "Soften a finding"],
    "scribe": ["the only headline you write is a sentence the registration fixed",
               "Write a sentence about a result that is not one of the verdict's headline "
               "sentences", "stands verbatim in a file you changed", "Commit or push"],
}  # fmt: skip


@pytest.mark.parametrize("role", sorted(DUTIES))
def test_each_role_prompt_states_the_duties_its_gate_enforces(role: str) -> None:
    text = " ".join((REPO / f"agents/roles/{role}.md").read_text().split())
    missing = [d for d in DUTIES[role] if " ".join(d.split()) not in text]
    assert not missing, f"{role}.md does not say {missing}"


def test_no_commit_and_no_harness_file_credits_a_co_author_or_a_tool() -> None:
    trailer = re.compile(r"(?im)^(co-authored-by|generated with|signed-off-by):")
    log = subprocess.run(["git", "log", "--format=%B"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout  # fmt: skip
    assert not trailer.search(log)
    harness = [*REPO.glob("agents/**/*.*"), *REPO.glob("tests/test_agents_*.py"),
               *REPO.glob("docs/agents*.md"), REPO / "CLAUDE.md"]  # fmt: skip
    for path in harness:
        if "runs" not in path.relative_to(REPO).parts and path.suffix in {".md", ".py", ".yaml"}:
            assert not trailer.search(path.read_text()), path


AGENTS_DOC = REPO / "docs" / "agents.md"


def test_the_harness_page_names_only_commands_that_exist() -> None:
    named = named_commands(AGENTS_DOC.read_text())
    assert len(named) >= 6
    for text in named:
        command = parse_command(text)
        assert command is not None and problem(command) is None, text


def test_every_test_the_rules_table_names_exists() -> None:
    text = AGENTS_DOC.read_text()
    table = text[text.index("## Each rule, how it is enforced") : text.index("## What is logged")]
    named = set(re.findall(r"`(test_[a-z0-9_]+)`", table))
    defined = {
        name
        for path in REPO.glob("tests/test_agents_*.py")
        for name in re.findall(r"^def (test_[a-z0-9_]+)\(", path.read_text(), re.MULTILINE)
    }
    assert len(named) >= 25 and not named - defined, sorted(named - defined)
    rules = re.findall(r"^\d\. \*\*([^*]+)\*\*", (REPO / "CLAUDE.md").read_text(), re.MULTILINE)
    first = [
        row.split("|")[1].strip().lower() for row in table.splitlines() if row.startswith("| ")
    ]
    for rule in rules:
        assert any(rule.rstrip(".").lower()[:20] in r for r in first), rule
