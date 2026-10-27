"""The research loop's orchestrator: propose, register, run, review, verdict, record.

Runs the five roles of ``agents/roles/`` and the code's verdict in the order
``agents/protocol.md`` fixes, each role through an LLM agent whose every tool
call passes :class:`~agents.workflows.policy.ToolPolicy`. The orchestrator
checks the gates it can check mechanically (a handback per role; the
registration committed with its null-placeholder sha256 before the run; no
blocking finding; the results file the ledger recorded; only registered
sentences recorded) and stops at every gate the owner holds. Every step is
logged under ``agents/runs/<UTC time>-<study>/``, which git ignores::

    python -m agents.workflows.research_loop run --study funding_carry_2026 --brief-file b.md
    python -m agents.workflows.research_loop run --resume agents/runs/<dir>
    python -m agents.workflows.research_loop calendar
    python -m agents.workflows.research_loop null-hash configs/spot_2026.yaml
    python -m agents.workflows.research_loop check --role executor "git push"

The model comes from ``TRP_AGENT_MODEL`` (``TRP_AGENT_MODEL_<ROLE>`` overrides
it per role); the SDK reads its credentials from the environment itself, and
nothing here reads, logs or stores a key. The SDK, ``claude-agent-sdk``, is an
optional dependency imported only when a run starts: everything else, and the
tests, work without it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agents.workflows.policy import ROLES, Protected, ToolPolicy, protected_calendar
from agents.workflows.prompts import (
    STEPS,
    HandbackError,
    StepContext,
    blocking,
    prompt_problems,
    read_handback,
    step_prompt,
    system_prompt,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("agents/runs")
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RoleCall:
    """One role's turn: who, which step, its prompts and its tool policy."""

    role: str
    step: str
    system: str
    prompt: str
    policy: ToolPolicy


@dataclass
class RoleReply:
    text: str
    events: list[dict[str, Any]] = field(default_factory=list)


#: Anything that answers a role's turn: the SDK in a run, a fake in the tests.
Runner = Callable[[RoleCall], Awaitable[RoleReply]]


class RunLog:
    """The run's directory: one file per artefact, nothing outside it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def new(cls, repo: Path, study: str, now: datetime | None = None) -> RunLog:
        stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
        return cls(repo / RUNS / f"{stamp}-{study}")

    def write(self, name: str, text: str) -> Path:
        target = self.path / name
        target.write_text(text)
        return target

    def write_json(self, name: str, value: Any) -> Path:
        return self.write(name, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")

    def handbacks(self) -> dict[str, dict[str, Any]]:
        """The handbacks already written, by step: what a resumed run starts from."""
        found: dict[str, dict[str, Any]] = {}
        for path in sorted(self.path.glob("*.handback.yaml")):
            data = yaml.safe_load(path.read_text())
            found[str(data["step"])] = data
        return found


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return done.stdout


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def null_sha256(config_text: str) -> str:
    """The YAML's sha256 with every placeholder put back to ``null``, by the
    repository's own helpers (``market_making/prereg.py``)."""
    from trading_research.market_making.prereg import sha256_bytes, unfrozen_text

    return sha256_bytes(unfrozen_text(config_text).encode())


def registration_gate(repo: Path, config: str, document: str) -> str | None:
    """Why the run may not start yet, or ``None``: both files exist, are
    committed and unchanged, and the document holds the YAML's null hash."""
    for path in (config, document):
        if not (repo / path).exists():
            return f"{path} does not exist: the registration comes first"
        if not git(repo, "log", "-1", "--format=%H", "--", path).strip():
            return f"{path} is not committed: the owner commits the registration"
    if git(repo, "status", "--porcelain", "--", config, document).strip():
        return f"{config} or {document} changed since its commit"
    expected = null_sha256((repo / config).read_text())
    if expected not in SHA256.findall((repo / document).read_text()):
        return f"{document} does not hold the null-placeholder sha256 {expected}"
    return None


def headline_sentences(page: str, block: str) -> list[str]:
    """The bullets under ``### Headline sentences`` in the page's section for ``block``."""
    found: list[str] = []
    in_block = in_list = False
    for line in page.splitlines():
        if line.startswith("## "):
            in_block, in_list = line[3:].split(" ")[0] == block, False
        elif line.startswith("### "):
            in_list = in_block and line[4:].strip().lower() == "headline sentences"
        elif in_list and line.startswith("- "):
            found.append(line[2:].strip())
    return found


def verdict(repo: Path, run: dict[str, Any]) -> dict[str, Any]:
    """The verdict as the code computed it: statuses and first kills from the
    results file, its sha256 against the ledger's record, the page's sentences."""
    results = run.get("results")
    if not results:
        reason = "no held-out block was read in this run; a development number is not a result"
        return {"read": False, "reason": reason, "sentences": []}
    block, path = str(run.get("block") or ""), repo / str(results)
    data = json.loads(path.read_text())
    statuses = {
        slot: {"status": s.get("status"), "first_kill": s.get("first_kill")}
        for slot, s in dict(data.get("statuses", {})).items()
    }
    found: dict[str, Any] = {"read": True, "block": block, "results": str(results)}
    found.update(results_sha256=file_sha256(path), statuses=statuses, ledger_match=False)
    if run.get("ledger"):
        entries = json.loads((repo / str(run["ledger"])).read_text()).get(block, [])
        recorded = [e.get("results_sha256") for e in entries if e.get("read") == "first read"]
        found["ledger_match"] = found["results_sha256"] in recorded
    page = run.get("page")
    found["sentences"] = headline_sentences((repo / str(page)).read_text(), block) if page else []
    return found


def stop_reason(step: str, handback: dict[str, Any], done: dict[str, dict[str, Any]]) -> str | None:
    """Why the loop stops after ``step``, or ``None`` to go on."""
    status = handback.get("status")
    if step == "propose" and status == "declined":
        return f"the researcher declined: {handback.get('reason', 'no reason given')}"
    if step == "register" and status == "refused":
        return f"the registrar refused; missing: {handback.get('missing', [])}"
    if step == "run" and status in {"refused", "failed"}:
        return f"the run was {status}; the owner reads the records before anything else"
    if step == "review":
        found = blocking(handback.get("findings") or [])
        if found or status == "block":
            return f"blocking findings: {[f.get('where') for f in found]}"
    if step == "verdict" and handback.get("read") and not handback.get("ledger_match"):
        return "the results file is not the one the ledger recorded for the block's first read"
    if step == "record":
        registered = set(done.get("verdict", {}).get("sentences", []))
        extra = [s for s in handback.get("sentences", []) if s not in registered]
        if extra:
            return f"sentences that are not registered headline sentences: {extra}"
    return None


@dataclass
class ResearchLoop:
    """One run of the loop for one study, resumable from its run directory."""

    repo: Path
    study: str
    brief: str
    runner: Runner
    log: RunLog
    grants: frozenset[str] = frozenset()
    web_hosts: frozenset[str] = frozenset()
    python: str = "python"
    cli: str = "trading-research"
    config: str = ""
    document: str = ""
    gate: Callable[[Path, str, str], str | None] = registration_gate
    calendar: tuple[Protected, ...] | None = None

    def __post_init__(self) -> None:
        self.config = self.config or f"configs/{self.study}.yaml"
        self.document = self.document or f"docs/preregistration/{self.study}.md"
        if self.calendar is None:
            self.calendar = protected_calendar(self.repo)

    async def run(self) -> dict[str, Any]:
        problems = prompt_problems(self.repo)
        if problems:
            return self.stop("start", f"the prompts name commands that do not exist: {problems}")
        done = self.log.handbacks()
        for number, (step, role) in enumerate(STEPS, start=1):
            if step in done:
                continue
            if step == "run":
                problem = self.gate(self.repo, self.config, self.document)
                if problem is not None:
                    return self.stop(step, f"waiting on the owner: {problem}")
            if role == "code":
                handback = {"role": "code", "step": step, "status": "computed"}
                try:
                    handback.update(verdict(self.repo, done.get("run", {})))
                except (OSError, ValueError) as exc:
                    return self.stop(step, f"the results named by the run cannot be read: {exc}")
                self.log.write_json("verdict.json", handback)
                self.log.write(f"{number:02d}-{step}.handback.yaml", yaml.safe_dump(handback))
            else:
                found = await self.turn(number, step, role, done)
                if isinstance(found, str):
                    return self.stop(step, found)
                handback = found
            done[step] = handback
            reason = stop_reason(step, handback, done)
            if reason is not None:
                return self.stop(step, reason)
        return {"status": "complete", "run": str(self.log.path)}

    def policy(self, role: str) -> ToolPolicy:
        return ToolPolicy(
            self.repo,
            role,
            self.study,
            grants=self.grants,
            web_hosts=self.web_hosts,
            python=(self.python,),
            cli=(self.cli,),
            calendar=self.calendar,
        )

    async def turn(
        self, number: int, step: str, role: str, done: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | str:
        """One role's turn: its prompts, its reply, its handback (or why it has none)."""
        system = system_prompt(self.repo, role)
        context = StepContext(
            self.study,
            self.brief,
            self.config,
            self.document,
            grants=self.grants,
            prefix=self.python,
            cli=self.cli,
            earlier=tuple(done.items()),
        )
        prompt = step_prompt(step, role, context)
        stem = f"{number:02d}-{step}"
        self.log.write(f"system-{role}.md", system)
        self.log.write(f"{stem}.prompt.md", prompt)
        reply = await self.runner(RoleCall(role, step, system, prompt, self.policy(role)))
        self.log.write(f"{stem}.reply.md", reply.text)
        events = "".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in reply.events)
        self.log.write(f"{stem}.events.jsonl", events)
        try:
            handback = read_handback(role, step, reply.text)
        except HandbackError as exc:
            self.log.write(f"{stem}.invalid.txt", str(exc) + "\n")
            return f"the {role}'s handback is not valid: {exc}"
        self.log.write(f"{stem}.handback.yaml", yaml.safe_dump(handback, sort_keys=False))
        return handback

    def stop(self, step: str, reason: str) -> dict[str, Any]:
        record = {"status": "stopped", "step": step, "reason": reason, "run": str(self.log.path)}
        line = json.dumps({**record, "utc": datetime.now(UTC).isoformat()}, sort_keys=True)
        with (self.log.path / "stops.jsonl").open("a") as stops:
            stops.write(line + "\n")
        return record


class SdkRunner:
    """A role's turn through the agent SDK, every tool call decided by the policy.

    The policy runs as a hook before every tool call, which no allow rule or mode
    can bypass, and again as the SDK's tool callback. Project and user settings
    are not loaded (``setting_sources=[]``), so nothing on the machine widens it.
    """

    def __init__(self, repo: Path, max_turns: int | None = None) -> None:
        self.repo = repo
        self.max_turns = max_turns or int(os.environ.get("TRP_AGENT_MAX_TURNS", "40"))

    @staticmethod
    def model(role: str) -> str:
        found = os.environ.get(f"TRP_AGENT_MODEL_{role.upper()}") or os.environ.get(
            "TRP_AGENT_MODEL"
        )
        if not found:
            raise RuntimeError("set TRP_AGENT_MODEL (or TRP_AGENT_MODEL_<ROLE>) to a model name")
        return found

    async def __call__(self, call: RoleCall) -> RoleReply:
        import importlib

        sdk = importlib.import_module("claude_agent_sdk")
        events: list[dict[str, Any]] = []

        async def hook(
            data: dict[str, Any], tool_use_id: str | None, _context: Any
        ) -> dict[str, Any]:
            tool, given = str(data.get("tool_name")), dict(data.get("tool_input") or {})
            decision = call.policy.decide(tool, given)
            events.append({"tool": tool, "input": given, "allowed": decision.allowed,
                           "reason": decision.reason, "id": tool_use_id})  # fmt: skip
            if decision.allowed:
                return {}
            output = {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                      "permissionDecisionReason": decision.reason}  # fmt: skip
            return {"hookSpecificOutput": output}

        async def can_use(tool: str, given: dict[str, Any], _context: Any) -> Any:
            decision = call.policy.decide(tool, given)
            if decision.allowed:
                return sdk.PermissionResultAllow(updated_input=given)
            return sdk.PermissionResultDeny(message=decision.reason)

        denied = ["WebSearch", "Task", "Bash(git push)", "Bash(git push *)", "Bash(git remote *)"]
        options = sdk.ClaudeAgentOptions(
            system_prompt=call.system,
            cwd=str(self.repo),
            model=self.model(call.role),
            max_turns=self.max_turns,
            setting_sources=[],
            allowed_tools=[],
            disallowed_tools=denied,
            permission_mode="default",
            can_use_tool=can_use,
            hooks={"PreToolUse": [sdk.HookMatcher(matcher=None, hooks=[hook])]},
            env={"PYTHONPATH": str(self.repo / "src")},
        )
        texts: list[str] = []
        async with sdk.ClaudeSDKClient(options=options) as client:
            await client.query(call.prompt)
            async for message in client.receive_response():
                if isinstance(message, sdk.AssistantMessage):
                    texts += [b.text for b in message.content if isinstance(b, sdk.TextBlock)]
                elif isinstance(message, sdk.ResultMessage):
                    events.append({"result": message.subtype, "turns": message.num_turns})
        return RoleReply("\n".join(texts), events)


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(
        prog="python -m agents.workflows.research_loop",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = found.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the loop for one study, or resume a run")
    run.add_argument("--study", help="the study's name: configs/<study>.yaml and its document")
    run.add_argument("--brief", help="the owner's brief, in a sentence or two")
    run.add_argument("--brief-file", type=Path, help="the owner's brief, from a file")
    run.add_argument("--grant", action="append", default=[], help="<config>:<block> to open")
    run.add_argument("--web-host", action="append", default=[], help="a fee page's host")
    run.add_argument("--resume", type=Path, help="a run directory to continue")
    run.add_argument("--config", default="", help="the YAML, when not configs/<study>.yaml")
    run.add_argument("--document", default="", help="the document, when not by the study name")
    sub.add_parser("calendar", help="every range a registration protects")
    null_hash = sub.add_parser("null-hash", help="a YAML's null-placeholder sha256")
    null_hash.add_argument("config", type=Path)
    check = sub.add_parser("check", help="would the policy allow this command for this role")
    check.add_argument("--role", required=True, choices=ROLES)
    check.add_argument("--study", default="adhoc")
    check.add_argument("--grant", action="append", default=[])
    check.add_argument("text", help="the command, quoted")
    return found


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "calendar":
        for p in protected_calendar(REPO):
            print(f"{p.first} {p.last} {p.key}")
        return 0
    if args.command == "null-hash":
        print(null_sha256(args.config.read_text()))
        return 0
    if args.command == "check":
        policy = ToolPolicy(REPO, args.role, args.study, grants=frozenset(args.grant))
        decision = policy.bash(args.text)
        print(f"{'allowed' if decision.allowed else 'refused'}: {decision.reason}")
        return 0 if decision.allowed else 1
    if args.resume is not None:
        log = RunLog(args.resume)
        meta = json.loads((log.path / "run.json").read_text())
        meta["grants"] = sorted({*meta.get("grants", []), *args.grant})
    else:
        brief = args.brief_file.read_text() if args.brief_file else args.brief
        if not args.study or not brief:
            parser().error("a new run needs --study and --brief or --brief-file")
        log = RunLog.new(REPO, args.study)
        meta = {"study": args.study, "brief": brief, "grants": sorted(set(args.grant)),
                "web_hosts": sorted(set(args.web_host)), "config": args.config,
                "document": args.document, "model": os.environ.get("TRP_AGENT_MODEL"),
                "commit": git(REPO, "rev-parse", "HEAD").strip(),
                "started": datetime.now(UTC).isoformat()}  # fmt: skip
    log.write_json("run.json", meta)
    python = os.environ.get("TRP_AGENT_PYTHON", "python")
    cli = os.environ.get("TRP_AGENT_CLI", "trading-research")
    loop = ResearchLoop(REPO, meta["study"], meta["brief"], SdkRunner(REPO), log,
                        grants=frozenset(meta["grants"]), web_hosts=frozenset(meta.get("web_hosts", [])),
                        python=python, cli=cli, config=meta.get("config", ""),
                        document=meta.get("document", ""))  # fmt: skip
    result = asyncio.run(loop.run())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
