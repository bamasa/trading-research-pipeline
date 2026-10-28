"""The policy on paper trading: never live, a stored day only outside every protected range."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agents.workflows.policy import ROLES, Protected, ToolPolicy

REPO = Path(__file__).resolve().parent.parent
#: A prospective block running now, and a development block that is free.
CALENDAR = (Protected("spot_2026", "P", date(2026, 10, 23), date(2027, 1, 20)),)


def policy(role: str = "executor", grants: frozenset[str] = frozenset()) -> ToolPolicy:
    return ToolPolicy(REPO, role, "spot_2026", grants=grants, calendar=CALENDAR)


def decide(p: ToolPolicy, command: str) -> tuple[bool, str]:
    d = p.decide("Bash", {"command": command})
    return d.allowed, d.reason


LIVE = [
    "trading-research paper run --venue bybit --market linear --symbol BTCUSDT --minutes 2",
    "trp paper run --venue binance --market spot --symbol BTCUSDT --minutes 1",
    "trading-research paper record --venue bybit --market spot --symbol BTCUSDT --minutes 1",
    "PYTHONPATH=src python -m trading_research paper run --venue bybit --market linear "
    "--symbol BTCUSDT --minutes 1",
]


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("command", LIVE[:3])
def test_no_role_runs_or_records_live_even_with_every_grant(role: str, command: str) -> None:
    allowed, reason = decide(policy(role, grants=frozenset({"spot_2026:P"})), command)
    assert not allowed
    if role == "executor":
        assert "live stream" in reason and "held-out" in reason


def test_a_module_invocation_is_not_a_way_around() -> None:
    assert not decide(policy(), LIVE[3])[0]


@pytest.mark.parametrize(
    "command",
    [
        "trading-research paper replay --day 2026-10-20 --venue bybit --market linear --symbol BTCUSDT",
        "trp paper replay --day tests/fixtures/paper/bybit-linear-BTCUSDT-2025-06-03.jsonl",
        "trading-research paper parity --day 2026-09-30 --venue bybit --market linear --symbol BTCUSDT",
        "trading-research paper status runs/paper/x",
    ],
)
def test_a_stored_day_outside_every_block_replays(command: str) -> None:
    allowed, reason = decide(policy(), command)
    assert allowed, reason


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize(
    "command",
    [
        "trading-research paper replay --day 2026-10-27 --venue bybit --market linear --symbol BTCUSDT",
        "trp paper replay --day tests/fixtures/paper/bybit-linear-BTCUSDT-2026-10-27.jsonl",
        "trading-research paper parity --day 2027-01-20 --venue bybit --market linear --symbol BTCUSDT",
    ],
)
def test_a_protected_day_never_replays_whatever_the_grant(role: str, command: str) -> None:
    allowed, reason = decide(policy(role, grants=frozenset({"spot_2026:P"})), command)
    assert not allowed
    if role == "executor":
        assert "spot_2026:P" in reason


def test_a_replay_names_its_day_and_stays_inside_the_repository() -> None:
    assert not decide(policy(), "trading-research paper replay --day data/paper/latest")[0]
    assert not decide(policy(), "trading-research paper replay --day /tmp/2026-01-02.jsonl")[0]
    assert not decide(policy(), "trading-research paper status /tmp/run")[0]
    assert not decide(policy(), "trading-research paper launch")[0]


@pytest.mark.parametrize("role", [r for r in ROLES if r != "executor"])
def test_only_the_executor_replays(role: str) -> None:
    command = "trading-research paper replay --day 2026-10-20"
    assert not decide(policy(role), command)[0]
