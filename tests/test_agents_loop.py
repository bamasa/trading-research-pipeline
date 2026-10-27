"""The research loop's orchestrator, with a fake client in place of the agent SDK.

No test reaches a model or the network: each role's turn is a canned reply, and
the verdict reads the spot study's committed results, ledger and page.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agents.workflows.policy import protected_calendar
from agents.workflows.prompts import STEPS, HandbackError, read_handback
from agents.workflows.research_loop import ResearchLoop, RoleCall, RoleReply, RunLog

REPO = Path(__file__).resolve().parent.parent
CALENDAR = protected_calendar(REPO)
H_SENTENCE = "S25-BB on H, Bybit spot BTCUSDT: killed; first kill K1 at -19.43 bp against 0.00 bp."

HANDBACKS = {
    "researcher": """status: proposed
hypotheses:
  - {id: C1, claim: carry pays, metric: daily net bp, kills: [K1], placebos: [], expected: killed}
""",
    "registrar": f"status: registered\nnull_sha256: '{'a' * 64}'\nfiles: []\n",
    "executor": """status: ran
commands: [python -m experiments.spot_2026 status]
block: H
results: experiments/results/spot_2026_H.json
ledger: experiments/results/spot_2026_ledger.json
page: experiments/results/spot_2026_page.md
""",
    "skeptic": "status: pass\nfindings:\n  - {lens: execution, severity: minor, where: 'x.py:1', finding: f}\n",
    "scribe": f"status: recorded\nsentences: ['{H_SENTENCE}']\n",
}


class FakeClient:
    """Answers each role with a canned handback and records what it was given."""

    def __init__(self, **overrides: str) -> None:
        self.replies = {**HANDBACKS, **overrides}
        self.calls: list[RoleCall] = []

    async def __call__(self, call: RoleCall) -> RoleReply:
        self.calls.append(call)
        tried = call.policy.decide("Bash", {"command": "git push"})
        event = {"tool": "Bash", "allowed": tried.allowed, "reason": tried.reason}
        reply = self.replies[call.role]
        if reply.startswith("RAW:"):
            return RoleReply(reply.removeprefix("RAW:"), [event])
        return RoleReply(f"Done.\n\n```handback\n{reply}```\n", [event])


def open_gate(_repo: Path, _config: str, _document: str) -> str | None:
    return None


def closed_gate(_repo: Path, _config: str, _document: str) -> str | None:
    return "not committed"


def make_loop(tmp_path: Path, client: FakeClient, gate: Any = open_gate, **kw: Any) -> ResearchLoop:
    brief = "Test the loop on the spot study's committed read."
    log = RunLog(tmp_path / "run")
    return ResearchLoop(REPO, "spot_2026", brief, client, log, gate=gate, calendar=CALENDAR, **kw)


def roles(client: FakeClient) -> list[str]:
    return [c.role for c in client.calls]


def test_the_steps_run_in_the_registered_order_and_the_verdict_is_the_codes(tmp_path: Path) -> None:
    client = FakeClient()
    result = asyncio.run(make_loop(tmp_path, client).run())
    assert result["status"] == "complete", result
    assert [s for s, _ in STEPS] == ["propose", "register", "run", "review", "verdict", "record"]
    assert roles(client) == ["researcher", "registrar", "executor", "skeptic", "scribe"]
    run = tmp_path / "run"
    verdict = (run / "verdict.json").read_text()
    assert '"ledger_match": true' in verdict and '"status": "killed"' in verdict
    for stem in ("01-propose", "02-register", "03-run", "04-review", "06-record"):
        for suffix in (".prompt.md", ".reply.md", ".handback.yaml", ".events.jsonl"):
            assert (run / f"{stem}{suffix}").exists(), stem + suffix
    assert '"allowed": false' in (run / "03-run.events.jsonl").read_text()


def test_every_role_gets_the_rules_the_loop_its_file_and_the_earlier_handbacks(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    asyncio.run(make_loop(tmp_path, client, grants=frozenset({"spot_2026:H"})).run())
    rules = (REPO / "CLAUDE.md").read_text()
    loop = (REPO / "agents/protocol.md").read_text()
    for call in client.calls:
        assert rules in call.system and loop in call.system
        assert (REPO / f"agents/roles/{call.role}.md").read_text() in call.system
        assert call.policy.role == call.role and call.policy.study == "spot_2026"
        assert "Test the loop on the spot study's committed read." in call.prompt
        assert "spot_2026:H" in call.prompt and "handback" in call.prompt
    skeptic = next(c for c in client.calls if c.role == "skeptic")
    assert "### propose" in skeptic.prompt and "id: C1" in skeptic.prompt
    assert "### run" in skeptic.prompt and "spot_2026_H.json" in skeptic.prompt
    assert "A blocking finding stops the loop" in skeptic.prompt


def test_a_blocking_finding_stops_the_loop_before_the_verdict(tmp_path: Path) -> None:
    blocking = "status: block\nfindings:\n  - {lens: leakage, severity: blocking, where: 'a.py:3', finding: f}\n"
    client = FakeClient(skeptic=blocking)
    result = asyncio.run(make_loop(tmp_path, client).run())
    assert result["status"] == "stopped" and result["step"] == "review"
    assert "scribe" not in roles(client)
    assert not (tmp_path / "run" / "verdict.json").exists()
    assert "a.py:3" in (tmp_path / "run" / "stops.jsonl").read_text()


def test_a_blocking_finding_stops_the_loop_even_under_a_pass(tmp_path: Path) -> None:
    mislabelled = (
        "status: pass\nfindings:\n  - {lens: fidelity, severity: blocking, where: 'b:1'}\n"
    )
    result = asyncio.run(make_loop(tmp_path, FakeClient(skeptic=mislabelled)).run())
    assert result["status"] == "stopped" and result["step"] == "review"


def test_the_run_waits_for_the_owner_to_commit_the_registration_and_resumes(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    waiting = asyncio.run(make_loop(tmp_path, client, gate=closed_gate).run())
    assert waiting["status"] == "stopped" and waiting["step"] == "run"
    assert roles(client) == ["researcher", "registrar"]
    resumed = FakeClient()
    result = asyncio.run(make_loop(tmp_path, resumed).run())
    assert result["status"] == "complete"
    assert roles(resumed) == ["executor", "skeptic", "scribe"]


@pytest.mark.parametrize(
    ("role", "reply", "step"),
    [
        ("researcher", "status: declined\nreason: no kill condition can be stated\n", "propose"),
        ("registrar", "status: refused\nmissing: [the fee table's sources]\n", "register"),
        ("executor", "status: refused\ncommands: []\n", "run"),
        ("scribe", "status: recorded\nsentences: ['The carry pays.']\n", "record"),
        ("skeptic", "RAW:Nothing to report.\n```yaml\nstatus: pass\n```\n", "review"),
    ],
)
def test_a_refusal_an_unregistered_sentence_or_no_handback_stops_the_loop(
    tmp_path: Path, role: str, reply: str, step: str
) -> None:
    client = FakeClient(**{role: reply})
    result = asyncio.run(make_loop(tmp_path, client).run())
    assert result["status"] == "stopped" and result["step"] == step, result


def test_a_handback_must_carry_its_roles_keys() -> None:
    def reply(body: str) -> str:
        return f"```handback\n{body}```"

    with pytest.raises(HandbackError):
        read_handback("researcher", "propose", reply("status: proposed\nhypotheses: [{id: C1}]\n"))
    with pytest.raises(HandbackError):
        read_handback("registrar", "register", reply("status: registered\nnull_sha256: x\n"))
    with pytest.raises(HandbackError):
        bad = "status: pass\nfindings: [{lens: taste, severity: blocking, where: a}]\n"
        read_handback("skeptic", "review", reply(bad))
    with pytest.raises(HandbackError):
        read_handback("scribe", "record", reply("status: done\n"))
    last = reply("status: declined\n") + "\n" + reply("status: proposed\nhypotheses: []\n")
    with pytest.raises(HandbackError):
        read_handback("researcher", "propose", last)


def test_the_sdk_is_optional_and_the_model_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents.workflows.research_loop import SdkRunner

    assert "claude_agent_sdk" not in sys.modules
    monkeypatch.delenv("TRP_AGENT_MODEL", raising=False)
    monkeypatch.delenv("TRP_AGENT_MODEL_SKEPTIC", raising=False)
    with pytest.raises(RuntimeError):
        SdkRunner.model("skeptic")
    monkeypatch.setenv("TRP_AGENT_MODEL", "model-a")
    monkeypatch.setenv("TRP_AGENT_MODEL_SKEPTIC", "model-b")
    assert SdkRunner.model("researcher") == "model-a"
    assert SdkRunner.model("skeptic") == "model-b"


def test_the_helpers_answer_without_a_model(capsys: pytest.CaptureFixture[str]) -> None:
    from agents.workflows.research_loop import main

    assert main(["check", "--role", "executor", "git push"]) == 1
    assert main(["null-hash", str(REPO / "configs/spot_2026.yaml")]) == 0
    assert main(["calendar"]) == 0
    printed = capsys.readouterr().out
    assert "refused" in printed and "4e75c7addf549d8d" in printed
    assert "2026-08-08 2026-10-21 spot_2026:H" in printed
    assert len(CALENDAR) == len([line for line in printed.splitlines() if ":" in line[22:]])


def test_a_run_does_not_start_without_a_model_or_the_sdk(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agents.workflows.research_loop import main

    monkeypatch.delenv("TRP_AGENT_MODEL", raising=False)
    before = sorted((REPO / "agents" / "runs").glob("*"))
    assert main(["run", "--study", "funding_carry_2026", "--brief", "b"]) == 2
    assert "TRP_AGENT_MODEL" in capsys.readouterr().err
    assert sorted((REPO / "agents" / "runs").glob("*")) == before


def test_the_run_directories_are_ignored_by_git() -> None:
    found = subprocess.run(["git", "check-ignore", "agents/runs/x/run.json"], cwd=REPO, check=False)
    assert found.returncode == 0
