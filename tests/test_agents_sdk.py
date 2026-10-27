"""The SDK runner's options against the installed agent SDK, without a model.

Skipped unless the optional ``claude-agent-sdk`` is installed. It never starts
a session: it builds a turn's options and calls the policy's hook and tool
callback as the SDK would, so a renamed field or class in the SDK fails here
rather than in a live run. The package is kept out of the study's environment;
``docs/agents.md`` shows the small environment this test runs in.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from agents.workflows.policy import ROLES, ToolPolicy
from agents.workflows.research_loop import DISALLOWED, TOOLS_BY_ROLE, RoleCall, SdkRunner

sdk = pytest.importorskip("claude_agent_sdk")
REPO = Path(__file__).resolve().parent.parent


def turn(role: str, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict[str, Any]]]:
    monkeypatch.setenv("TRP_AGENT_MODEL", "a-model-name")
    policy = ToolPolicy(REPO, role, "funding_carry_2026", calendar=())
    call = RoleCall(role, "step", "the system prompt", "the step prompt", policy)
    events: list[dict[str, Any]] = []
    return SdkRunner(REPO).options(call, events, sdk), events


@pytest.mark.parametrize("role", ROLES)
def test_a_turns_options_load_no_settings_and_give_the_role_only_its_tools(
    role: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    options, _ = turn(role, monkeypatch)
    assert isinstance(options, sdk.ClaudeAgentOptions)
    assert options.setting_sources == [] and options.allowed_tools == []
    assert options.tools == TOOLS_BY_ROLE[role] and options.disallowed_tools == DISALLOWED
    assert options.model == "a-model-name" and options.permission_mode == "default"
    assert options.system_prompt == "the system prompt" and options.cwd == str(REPO)
    assert set(options.env) == {"PYTHONPATH"}


def test_the_policy_answers_as_the_hook_and_as_the_tool_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, events = turn("executor", monkeypatch)
    hook = options.hooks["PreToolUse"][0].hooks[0]

    def ask(command: str) -> dict[str, Any]:
        data = {"tool_name": "Bash", "tool_input": {"command": command}}
        found: dict[str, Any] = asyncio.run(hook(data, "id", None))
        return found

    denied = ask("git push")["hookSpecificOutput"]
    assert (
        denied["permissionDecision"] == "deny"
        and "not read-only" in denied["permissionDecisionReason"]
    )
    assert ask("git status") == {}
    assert [e["allowed"] for e in events] == [False, True]
    callback = options.can_use_tool
    refused = asyncio.run(callback("Write", {"file_path": "src/x.py", "content": "x"}, None))
    assert isinstance(refused, sdk.PermissionResultDeny)
    allowed = asyncio.run(callback("Read", {"file_path": "README.md"}, None))
    assert isinstance(allowed, sdk.PermissionResultAllow)


def test_the_result_message_carries_what_the_runner_logs() -> None:
    import dataclasses

    fields = {f.name for f in dataclasses.fields(sdk.ResultMessage)}
    assert {"subtype", "num_turns", "total_cost_usd"} <= fields
    assert {"text"} <= {f.name for f in dataclasses.fields(sdk.TextBlock)}
