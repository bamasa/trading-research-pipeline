"""The demonstration plan for the funding-rate carry study: every command it
names exists (or is marked as the entry point to come), each command's outcome
under the tool policy is the one it states, and its blocks avoid every range
the registrations protect."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from agents.workflows.policy import (
    Command,
    Protected,
    ToolPolicy,
    named_commands,
    parse_command,
    protected_calendar,
    whole_months,
)
from agents.workflows.prompts import command_problem

REPO = Path(__file__).resolve().parent.parent
PLAN = (REPO / "docs/agents_demo_funding_carry.md").read_text()
STUDY = "funding_carry_2026"
CALENDAR = protected_calendar(REPO)
SPAN = r"(\d{4}-\d\d-\d\d) to (\d{4}-\d\d-\d\d)"


def problem(command: Command) -> str | None:
    return command_problem(REPO, command)


def section(heading: str) -> str:
    start = PLAN.index(heading)
    rest = PLAN[start + len(heading) :]
    end = re.search(r"^#{2,3} ", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


TO_COME = section("### The entry point to come")


def blocks() -> dict[str, list[tuple[date, date]]]:
    found: dict[str, list[tuple[date, date]]] = {}
    for name, dates in re.findall(r"^\| (D|E|H|E2|P) \| ([^|]+) \|", PLAN, re.MULTILINE):
        spans = re.findall(SPAN, dates)
        found[name] = [(date.fromisoformat(a), date.fromisoformat(b)) for a, b in spans]
    return found


def registered() -> tuple[Protected, ...]:
    held = [Protected(STUDY, n, a, b) for n in ("E", "H", "E2", "P") for a, b in blocks()[n]]
    return (*CALENDAR, *held)


def test_the_plan_says_nothing_was_run_and_holds_blocks_in_2026_kills_and_placebos() -> None:
    flat = " ".join(PLAN.split())
    assert "Nothing here has been run, fetched or registered" in flat
    assert set(blocks()) == {"D", "E", "H", "E2", "P"}
    assert all(a.year == b.year == 2026 for a, b in blocks()["H"])
    for words in ("K1", "K-fund", "K-basis", "K-placebo", "K-nbhd", "seeds 0 to 199", "m = 3"):
        assert words in flat, words


def test_every_command_outside_the_entry_point_to_come_exists() -> None:
    outside = PLAN.replace(TO_COME, "")
    named = named_commands(outside)
    assert len(named) >= 10
    for text in named:
        command = parse_command(text)
        assert command is not None and problem(command) is None, text


def test_the_entry_point_to_come_is_marked_as_not_yet_written() -> None:
    named = [parse_command(text) for text in named_commands(TO_COME)]
    assert named and all(c is not None and c.kind == "entry" and c.name == STUDY for c in named)
    if not (REPO / "experiments" / f"{STUDY}.py").exists():
        assert f"`experiments/{STUDY}.py` does not exist" in TO_COME
    else:
        assert all(c is not None and problem(c) is None for c in named)


def test_each_commands_outcome_under_the_policy_is_the_one_the_plan_states() -> None:
    rows = re.findall(r"^\| `([^`]+)` \| (allowed|refused) \| (allowed|refused) \|$", PLAN, re.M)
    assert len(rows) >= 6
    today = ToolPolicy(REPO, "executor", STUDY, calendar=CALENDAR)
    after = ToolPolicy(REPO, "executor", STUDY, calendar=registered())
    for command, now, later in rows:
        assert today.bash(command).allowed == (now == "allowed"), (command, today.bash(command))
        assert after.bash(command).allowed == (later == "allowed"), (command, after.bash(command))


def test_the_blocks_are_whole_months_clear_of_every_protected_range() -> None:
    today = ToolPolicy(REPO, "executor", STUDY, calendar=CALENDAR)
    for name in ("D", "H", "P"):
        for first, last in blocks()[name]:
            assert (first, last) == whole_months(first, last), name
            assert today.days(first, last, name).allowed, name
