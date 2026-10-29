"""The README's claims about the command line: every ``trp`` command it names exists with
the options it shows, its first screen fits and carries a fee in every row, and none of
a short list of sentences a research repository should not say appears in it."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import typer.main
from typer.testing import CliRunner

from trading_research.cli import app

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")
ROOT = typer.main.get_command(app)

#: Phrases the README never uses in its own voice.
BANNED = (
    "profitable strategy",
    "beats the market",
    "guaranteed",
    "risk-free",
    "proven edge",
    "consistently profitable",
    "money machine",
    "honest",
    "rigorous",
)

FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
INLINE = re.compile(r"`([^`\n]+)`")


def named_commands(text: str) -> list[str]:
    """Every ``trp ...`` or ``trading-research ...`` command in code, one line each."""
    found: list[str] = []
    lines: list[str] = []
    for block in FENCED.findall(text):
        lines += block.replace("\\\n", " ").splitlines()
    stripped = FENCED.sub("", text)
    lines += INLINE.findall(stripped)
    for line in lines:
        line = line.split("#", 1)[0].strip()
        line = re.sub(r"^uv run (--no-sync )?", "", line)
        if re.match(r"^(trp|trading-research)(\s|$)", line):
            found.append(line)
    return found


def resolve(words: list[str]) -> tuple[Any, list[str], list[str]]:
    """The command the words reach, the path taken, and the words left after it."""
    command: Any = ROOT
    path: list[str] = []
    rest = list(words)
    while (
        rest
        and isinstance(getattr(command, "commands", None), dict)
        and rest[0] in command.commands
    ):
        command = command.commands[rest.pop(0)]
        path.append(command.name or "")
    return command, path, rest


COMMANDS = named_commands(README)


def test_the_readme_names_commands() -> None:
    assert len(COMMANDS) >= 40
    tops = {shlex.split(c)[1] for c in COMMANDS if len(shlex.split(c)) > 1}
    assert {"data", "search", "strategies", "verify", "paper", "agent", "study", "demo"} <= tops


@pytest.mark.parametrize("line", COMMANDS)
def test_every_command_the_readme_names_exists_with_its_options(line: str) -> None:
    words = shlex.split(line.replace("<", "X").replace(">", "X"))[1:]
    if not words:
        return
    command, path, rest = resolve(words)
    assert path or words[0].startswith("-"), f"{line!r}: {words[0]!r} is not a command of trp"
    options = {o for p in command.params for o in (*p.opts, *p.secondary_opts)} | {"--help"}
    for word in rest:
        if word.startswith("-"):
            assert word.split("=", 1)[0] in options, f"{line!r}: {word} is not an option of {path}"


def test_every_group_the_readme_names_is_in_the_help() -> None:
    shown = CliRunner().invoke(app, ["--help"], terminal_width=200).output
    for line in COMMANDS:
        words = shlex.split(line)[1:]
        if words and not words[0].startswith("-"):
            name = words[0]
            hidden = ROOT.commands[name].hidden  # type: ignore[attr-defined]
            assert hidden or f" {name} " in shown, name


def test_the_first_screen_fits_and_every_result_row_states_a_fee() -> None:
    lines = README.splitlines()
    first = lines[: lines.index("---")]
    while first and not first[-1].strip():
        first.pop()
    assert len(first) <= 25
    screen = "\n".join(first)
    for command in ("trp demo --execution all --verify", "trp verify all", "trp agent demo"):
        assert command in screen
    section = README.split("\n## Where it made money\n", 1)[1].split("\n## ", 1)[0]
    rows = [
        r for r in section.splitlines()
        if r.startswith("| ") and not r.startswith(("| Study", "| Direction"))
    ]  # fmt: skip
    assert len(rows) >= 12
    for row in rows:
        fee = row.split("|")[3 if row.count("|") == 6 else 2]
        assert re.search(r"\d", fee), row


@pytest.mark.parametrize("phrase", BANNED)
def test_no_banned_phrase(phrase: str) -> None:
    assert phrase not in README.lower()


def test_the_readme_has_the_sections_a_reader_walks() -> None:
    for heading in (
        "## Walk the pipeline",
        "## Add a strategy",
        "## Check it is not a fiction",
        "## Registered studies",
        "## Deploy: paper trading",
        "## Agents",
        "## Methodology",
        "## Data",
        "## Limitations",
        "## Disclaimer",
        "## License",
    ):
        assert f"\n{heading}\n" in README, heading
    for gone in ("closed-source", "disclosure", "prior work", "commercial setting"):
        assert gone not in README.lower()
