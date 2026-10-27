"""The agent harness's tool policy: what each role may run, read and write.

Run against this repository's own registrations, so the protected calendar the
tests read is the one the studies registered.
"""

from __future__ import annotations

import ast
import re
from datetime import date
from pathlib import Path

import pytest

from agents.workflows.policy import (
    CLI_COMMANDS,
    DATA_HOSTS,
    ROLES,
    ToolPolicy,
    protected_calendar,
)

REPO = Path(__file__).resolve().parent.parent
CALENDAR = protected_calendar(REPO)


def policy(
    role: str = "executor",
    study: str = "funding_carry_2026",
    grants: frozenset[str] = frozenset(),
    web_hosts: frozenset[str] = frozenset(),
) -> ToolPolicy:
    return ToolPolicy(REPO, role, study, grants=grants, web_hosts=web_hosts, calendar=CALENDAR)


def bash(p: ToolPolicy, command: str) -> bool:
    return p.decide("Bash", {"command": command}).allowed


PUSHES = [
    "git push",
    "git push origin harness/agents",
    "git push --force origin main",
    "git remote add origin x",
    "git remote -v",
    "git -C .. push",
    "git fetch",
    "git checkout main",
    "git commit -m x",
    "uv run git push",
]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("command", PUSHES)
def test_no_role_pushes_touches_a_remote_or_moves_a_branch(role: str, command: str) -> None:
    assert not bash(policy(role), command)


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest -q; git push",
        "python -m pytest -q && git push",
        "python -m pytest -q || git push",
        "python -m pytest -q | tee out.txt",
        "python -m pytest -q > out.txt",
        "python -m pytest -q 2>&1",
        "echo $(git push)",
        "python -m pytest `git push`",
        "python -m pytest -q\ngit push",
        "(git push)",
        "python -c 'import os'",
        "rm -rf data",
        "curl https://data.binance.vision",
        "FOO=1 python -m experiments.spot_2026 status",
        "PYTHONPATH=/tmp python -m experiments.spot_2026 status",
    ],
)
def test_one_known_command_at_a_time_and_nothing_else(command: str) -> None:
    assert not bash(policy("executor"), command)


@pytest.mark.parametrize(
    "command",
    [
        "trading-research describe-schema book",
        "uv run --no-sync trading-research describe-schema trades",
        "python -m experiments.spot_2026 status",
        "PYTHONPATH=src .venv/bin/python -m experiments.short_horizon_2025 status",
        "python -m experiments.spot_2026 status",
        "python -m experiments.short_horizon_2025 status",
        "python -m pytest tests/test_agents_policy.py -q",
        "git log --oneline -5",
        "git diff --name-only HEAD~1..HEAD",
        "git status --porcelain",
        "git show spot/prereg:docs/spot_2026_run_log.md",
        "trading-research reproduce --no-fetch",
        "trading-research review --trades artifacts/trades.csv",
        "python -m agents.workflows.research_loop calendar",
        "python -m agents.workflows.research_loop null-hash configs/spot_2026.yaml",
    ],
)
def test_the_executor_runs_the_repository_commands(command: str) -> None:
    decision = policy("executor").decide("Bash", {"command": command})
    assert decision.allowed, decision.reason


def test_the_executor_runs_the_stages_of_its_own_study_only() -> None:
    assert bash(
        policy("executor", "short_horizon_2025"), "python -m experiments.short_horizon_2025 1"
    )
    assert bash(policy("executor", "spot_2026"), "python -m experiments.spot_2026 --dry-run")
    assert not bash(policy("executor"), "python -m experiments.spot_2026 1")
    assert not bash(policy("executor"), "python -m experiments.spot_2026 pages")
    assert not bash(policy("executor", "spot_2026"), "python -m experiments.short_horizon_2025 E")


def test_the_policy_knows_every_cli_command_and_no_other() -> None:
    tree = ast.parse((REPO / "src/trading_research/cli.py").read_text())
    names = {
        d.args[0].value
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        for d in node.decorator_list
        if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "command"
        if isinstance(d.args[0], ast.Constant)
    }
    assert names == CLI_COMMANDS


def test_the_downloaders_hosts_are_all_closed_to_the_web_tool() -> None:
    sources = [
        *REPO.glob("src/trading_research/data/*.py"),
        *REPO.glob("src/trading_research/spot/*.py"),
    ]
    hosts = {h for s in sources for h in re.findall(r"https://([a-z0-9.-]+)", s.read_text())}
    assert hosts - {"github.com"} <= set(DATA_HOSTS)


def test_the_calendar_holds_every_unread_or_spent_block_and_no_development_block() -> None:
    found = {p.key: (p.first, p.last) for p in protected_calendar(REPO)}
    assert found["spot_2026:H"] == (date(2026, 8, 8), date(2026, 10, 21))
    assert found["short_horizon_2025:H"] == (date(2025, 7, 8), date(2025, 8, 20))
    assert found["mm_prereg:F"] == (date(2024, 3, 12), date(2024, 4, 7))
    assert found["spot_2026:never_in_a_block"] == (date(2026, 1, 5), date(2026, 1, 7))
    assert not [k for k in found if k.endswith((":D", ":W", ":O", ":reader_test"))]


def test_a_fetch_that_touches_a_protected_day_is_refused() -> None:
    ex = policy("executor")
    out = "--output data/funding_carry_2026/raw"
    assert bash(ex, f"trading-research download --start 2024-01-01 --end 2024-01-31 {out}")
    assert not bash(ex, f"trading-research download --start 2025-08-01 --end 2025-08-02 {out}")
    assert not bash(ex, f"trading-research download --start 2026-10-01 --end 2026-10-02 {out}")
    assert not bash(ex, f"trading-research download-book --start 2026-01-06 --end 2026-01-06 {out}")
    # A month archive holds the whole month: February 2024 holds the market-making H.
    daily = f"trading-research download --start 2024-02-01 --end 2024-02-25 {out}"
    monthly = daily + " --kind fundingRate --market futures-um"
    assert bash(ex, daily)
    assert not bash(ex, monthly)
    assert not bash(ex, f"trading-research download --start 2024-01-01 {out}")


def test_a_fetch_writes_under_data_only_into_the_study_root() -> None:
    ex = policy("executor")
    days = "--start 2024-01-01 --end 2024-01-02"
    assert not bash(ex, f"trading-research download {days}")  # data/raw by default
    assert not bash(ex, f"trading-research download-book {days} --output data/book")
    assert bash(ex, f"trading-research download-book {days} --output data/funding_carry_2026/book")
    assert not bash(ex, f"trading-research validate-data --input {REPO.parent}")
    assert not bash(ex, "trading-research breaks --start 2020-01-01 --end 2020-12-31")


def test_open_read_and_allow_heldout_need_the_owners_grant_for_that_block() -> None:
    assert not bash(policy("executor", "spot_2026"), "python -m experiments.spot_2026 open H")
    sh = policy("executor", "short_horizon_2025")
    assert not bash(sh, "python -m experiments.short_horizon_2025 read H")
    granted = policy("executor", "spot_2026", grants=frozenset({"spot_2026:H"}))
    assert bash(granted, "python -m experiments.spot_2026 open H")
    assert not bash(granted, "python -m experiments.spot_2026 read P")
    assert not bash(granted, "python -m experiments.short_horizon_2025 open H")
    mm = "trading-research mm-backtest --start 2024-02-26 --end 2024-03-09 --frozen --allow-heldout"
    assert not bash(policy("executor", grants=frozenset({"mm_prereg:H"})), mm)
    both = frozenset({"mm_prereg:H", "mm_prereg_round2:H"})
    assert bash(policy("executor", grants=both), mm)
    assert not bash(policy("executor", grants=both), mm + " --second-read")


def test_each_role_runs_only_its_own_commands() -> None:
    assert not bash(policy("researcher"), "python -m pytest -q")
    assert not bash(policy("researcher"), "python -m experiments.spot_2026 1")
    assert not bash(policy("skeptic"), "trading-research download-book --start 2024-01-01")
    assert bash(policy("skeptic"), "trading-research review --trades artifacts/t.csv")
    assert bash(policy("skeptic", "spot_2026"), "python -m experiments.spot_2026 --dry-run")
    assert not bash(policy("scribe", "spot_2026"), "python -m experiments.spot_2026 pages")
    assert bash(policy("scribe"), "python -m experiments.spot_2026 status")
    assert not bash(policy("executor"), "python -m agents.workflows.research_loop run")
    assert not bash(policy("executor"), "python -m experiments.grand_search")
    assert (
        not policy("executor")
        .decide("Bash", {"command": "git status", "run_in_background": True})
        .allowed
    )


NEVER_WRITTEN = [
    "src/trading_research/cli.py",
    "src/trading_research/spot/new_module.py",
    "experiments/spot_2026.py",
    "experiments/new_script.py",
    "configs/spot_2026.yaml",
    "configs/short_horizon_2025.yaml",
    "docs/preregistration/short_horizon_2025.md",
    "pyproject.toml",
    "uv.lock",
    "CLAUDE.md",
    "agents/workflows/policy.py",
    "agents/roles/skeptic.md",
    "tests/test_agents_policy.py",
    "experiments/results/spot_2026_H.json",
    "experiments/results/spot_2026_ledger.json",
    "data/funding_carry_2026/raw/x.csv",
    "models/spot_2026/fitted_values.json",
    "../outside.txt",
]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("path", NEVER_WRITTEN)
def test_no_role_writes_study_code_results_data_or_the_harness(role: str, path: str) -> None:
    p = policy(role)
    assert not p.decide("Write", {"file_path": path, "content": "x"}).allowed
    assert not p.decide("Edit", {"file_path": path, "old_string": "a", "new_string": "b"}).allowed


def test_the_registrar_writes_its_own_registration_until_the_freeze_then_only_appends() -> None:
    new = policy("registrar")
    for path in ("configs/funding_carry_2026.yaml", "docs/preregistration/funding_carry_2026.md"):
        assert new.decide("Write", {"file_path": path, "content": "draft"}).allowed
    frozen = policy("registrar", study="spot_2026")
    doc = "docs/preregistration/spot_2026.md"
    assert not frozen.decide(
        "Write", {"file_path": "configs/spot_2026.yaml", "content": "x"}
    ).allowed
    assert not frozen.decide("Write", {"file_path": doc, "content": "rewritten"}).allowed
    tail = (REPO / doc).read_text().rstrip("\n").splitlines()[-1]
    append = {
        "file_path": doc,
        "old_string": tail,
        "new_string": tail + "\n\n### A dated amendment",
    }
    assert frozen.decide("Edit", append).allowed
    middle = {"file_path": doc, "old_string": "## Blocks", "new_string": "## Blocks\n\nmoved"}
    assert not frozen.decide("Edit", middle).allowed


def test_the_scribe_appends_to_the_run_log_and_edits_only_the_registers() -> None:
    scribe = policy("scribe", study="spot_2026")
    log = "docs/spot_2026_run_log.md"
    tail = (REPO / log).read_text().rstrip("\n").splitlines()[-1]
    assert scribe.decide(
        "Edit", {"file_path": log, "old_string": tail, "new_string": tail + "\n- x"}
    ).allowed
    first = (REPO / log).read_text().splitlines()[0]
    assert not scribe.decide(
        "Edit", {"file_path": log, "old_string": first, "new_string": "x"}
    ).allowed
    assert scribe.decide(
        "Edit", {"file_path": "docs/findings.md", "old_string": "a", "new_string": "b"}
    ).allowed
    assert scribe.decide("Write", {"file_path": "README.md", "content": "x"}).allowed
    assert not scribe.decide(
        "Write", {"file_path": "docs/spot_2026_code_freeze.md", "content": "x"}
    ).allowed


def test_reads_stay_inside_the_repository_and_out_of_the_data() -> None:
    p = policy("skeptic")
    assert p.decide("Read", {"file_path": "README.md"}).allowed
    assert p.decide("Grep", {"pattern": "K1", "path": "docs"}).allowed
    for path in (
        "data/spot/x.parquet",
        "models/x.pkl",
        ".git/config",
        ".env",
        "../x",
        "/etc/hosts",
    ):
        assert not p.decide("Read", {"file_path": path}).allowed, path
    assert not p.decide("Glob", {"pattern": "../**/*.md"}).allowed
    assert not p.decide("Grep", {"pattern": "x", "path": "data"}).allowed


def test_the_web_is_closed_but_for_the_owners_fee_pages_to_the_registrar() -> None:
    fees = "https://www.binance.com/en/fee/schedule"
    hosts = frozenset({"www.binance.com", "data.binance.vision"})
    assert policy("registrar", web_hosts=hosts).decide("WebFetch", {"url": fees}).allowed
    assert not policy("registrar").decide("WebFetch", {"url": fees}).allowed
    assert not policy("researcher", web_hosts=hosts).decide("WebFetch", {"url": fees}).allowed
    archive = "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
    assert not policy("registrar", web_hosts=hosts).decide("WebFetch", {"url": archive}).allowed
    for role in ROLES:
        assert not policy(role).decide("WebSearch", {"query": "BTC funding"}).allowed
        assert not policy(role).decide("Task", {"prompt": "x"}).allowed
