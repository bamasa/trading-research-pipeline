"""The research loop's orchestrator, with a fake client in place of the agent SDK.

No test reaches a model or the network. The loop runs in a scratch git
repository holding the synthetic carry study of ``agents/demo/`` (its
registration committed, its D amendment and H's open entry committed, H read
by its entry point on synthetic draws), and each role's turn is a canned
reply; the scribe's also writes what it hands back. The verdict's own test
reads the spot study's committed results, ledger and page.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agents.workflows import demo
from agents.workflows.prompts import STEPS, HandbackError, read_handback
from agents.workflows.research_loop import (
    ResearchLoop,
    RoleCall,
    RoleReply,
    RunLog,
    headline_sentences,
    null_sha256,
    tree_state,
    verdict,
)

REPO = Path(__file__).resolve().parent.parent
LENSES = """lenses:
  leakage: {checks: [the open entry precedes the read], refutation: none found}
  fidelity: {checks: [the null hash], refutation: none found}
  execution: {checks: [the fee], refutation: none found}
"""
HANDBACKS = {
    "researcher": """status: proposed
hypotheses:
  - {id: C1, claim: carry pays, metric: daily net bp, kills: [K1], placebos: [], expected: killed}
known: [nothing]
blocks: {D: [2031-01-01, 2031-06-30], H: [2031-08-01, 2031-10-31]}
""",
    "registrar": f"status: registered\nnull_sha256: '{'a' * 64}'\nfiles: []\n",
    "executor": f"""status: ran
commands: [{demo.ENTRY} read H]
block: H
results: {demo.RESULTS}
ledger: {demo.LEDGER}
page: {demo.PAGE}
""",
    "skeptic": "status: pass\n"
    + LENSES
    + "findings:\n  - {lens: execution, severity: minor, where: 'x.py:1', finding: f, "
    "refutation: r}\n",
    "scribe": "status: recorded\nsentences: SENTENCES\n",
}
Act = Callable[[RoleCall], None]


def sentences(repo: Path) -> list[str]:
    page = repo / demo.PAGE
    return headline_sentences(page.read_text(), "H") if page.exists() else []


def scribe_writes(call: RoleCall) -> None:
    """The scribe appends the page's sentences to the run log, through its policy."""
    repo = call.policy.repo
    text = "# run log\n" + "".join(f"- {s}\n" for s in sentences(repo))
    assert call.policy.decide("Write", {"file_path": demo.RUN_LOG, "content": text}).allowed
    (repo / demo.RUN_LOG).write_text(text)


class FakeClient:
    """Answers each role with a canned handback, after an optional act of its
    own, and records what it was given."""

    def __init__(self, acts: dict[str, Act] | None = None, **overrides: str) -> None:
        self.replies = {**HANDBACKS, **overrides}
        self.acts: dict[str, Act] = {"scribe": scribe_writes, **(acts or {})}
        self.calls: list[RoleCall] = []

    async def __call__(self, call: RoleCall) -> RoleReply:
        self.calls.append(call)
        tried = call.policy.decide("Bash", {"command": "git push"})
        event = {"tool": "Bash", "allowed": tried.allowed, "reason": tried.reason}
        self.acts.get(call.role, lambda _call: None)(call)
        reply = self.replies[call.role]
        if "SENTENCES" in reply:
            reply = reply.replace("SENTENCES", json.dumps(sentences(call.policy.repo)))
        if reply.startswith("RAW:"):
            return RoleReply(reply.removeprefix("RAW:"), [event])
        return RoleReply(f"Done.\n\n```handback\n{reply}```\n", [event])


def scratch(root: Path, *, committed: bool = True) -> Path:
    """The scratch repository with the synthetic study registered and, unless
    told otherwise, committed, amended, opened and read on its synthetic draws."""
    repo = demo.build_repo(root)
    text = (demo.SOURCE / "synth_carry.yaml").read_text()
    (repo / demo.CONFIG).write_text(text)
    (repo / demo.DOCUMENT).write_text(demo.document_text(null_sha256(text)))
    if not committed:
        return repo
    demo.commit_code(repo)
    demo.run_entry(repo, "D")
    demo.commit("the D amendment")(repo)
    demo.run_entry(repo, "open", "H")
    demo.commit("H's open entry")(repo)
    assert '"read": "H"' in demo.run_entry(repo, "read", "H")
    return repo


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return scratch(tmp_path_factory.mktemp("template") / "repo")


@pytest.fixture
def repo(template: Path, tmp_path: Path) -> Path:
    """A fresh copy of the read scratch repository, its git history included."""
    return Path(shutil.copytree(template, tmp_path / "repo", symlinks=True))


def make_loop(repo: Path, client: FakeClient, **kw: Any) -> ResearchLoop:
    brief = "Test the loop on the synthetic study's read."
    log = RunLog(run_dir(repo))
    return ResearchLoop(repo, demo.STUDY, brief, client, log, rules=REPO, **kw)


def run_dir(repo: Path) -> Path:
    return repo / "agents" / "runs" / "test"


def roles(client: FakeClient) -> list[str]:
    return [c.role for c in client.calls]


def finding(severity: str, where: str, lens: str = "leakage") -> str:
    return (
        f"findings:\n  - {{lens: {lens}, severity: {severity}, where: '{where}', "
        "finding: f, refutation: r}\n"
    )


def test_the_steps_run_in_the_registered_order_and_the_verdict_is_the_codes(repo: Path) -> None:
    client = FakeClient()
    result = asyncio.run(make_loop(repo, client).run())
    assert result["status"] == "complete", result
    assert [s for s, _ in STEPS] == ["propose", "register", "run", "review", "verdict", "record"]
    assert roles(client) == ["researcher", "registrar", "executor", "skeptic", "scribe"]
    run = run_dir(repo)
    found = json.loads((run / "verdict.json").read_text())
    results = json.loads((repo / demo.RESULTS).read_text())["statuses"]["C1"]
    assert found["ledger_match"] and found["statuses"]["C1"]["status"] == results["status"]
    assert found["sentences"] == sentences(repo) and len(found["sentences"]) == 1
    for stem in ("01-propose", "02-register", "03-run", "04-review", "06-record"):
        for suffix in (
            ".prompt.md",
            ".reply.md",
            ".handback.yaml",
            ".events.jsonl",
            ".changed.json",
        ):
            assert (run / f"{stem}{suffix}").exists(), stem + suffix
    assert '"allowed": false' in (run / "03-run.events.jsonl").read_text()
    assert json.loads((run / "06-record.changed.json").read_text()) == [demo.RUN_LOG]


def test_every_role_gets_the_rules_the_loop_its_file_and_the_earlier_handbacks(
    repo: Path,
) -> None:
    client = FakeClient()
    asyncio.run(make_loop(repo, client, grants=frozenset({demo.GRANT})).run())
    rules = (REPO / "CLAUDE.md").read_text()
    loop = (REPO / "agents/protocol.md").read_text()
    for call in client.calls:
        assert rules in call.system and loop in call.system
        assert (REPO / f"agents/roles/{call.role}.md").read_text() in call.system
        assert call.policy.role == call.role and call.policy.study == demo.STUDY
        assert "Test the loop on the synthetic study's read." in call.prompt
        assert demo.GRANT in call.prompt and "handback" in call.prompt
    skeptic = next(c for c in client.calls if c.role == "skeptic")
    assert "### propose" in skeptic.prompt and "id: C1" in skeptic.prompt
    assert "### run" in skeptic.prompt and demo.RESULTS in skeptic.prompt
    assert "All three lenses" in skeptic.prompt and "a blocking finding stops" in skeptic.prompt


def test_a_blocking_finding_stops_the_loop_before_the_verdict(repo: Path) -> None:
    client = FakeClient(skeptic="status: block\n" + LENSES + finding("blocking", "a.py:3"))
    result = asyncio.run(make_loop(repo, client).run())
    assert result["status"] == "stopped" and result["step"] == "review"
    assert "scribe" not in roles(client)
    assert not (run_dir(repo) / "verdict.json").exists()
    assert "a.py:3" in (run_dir(repo) / "stops.jsonl").read_text()


def test_a_blocking_finding_stops_the_loop_even_under_a_pass(repo: Path) -> None:
    mislabelled = "status: pass\n" + LENSES + finding("blocking", "b:1", "fidelity")
    result = asyncio.run(make_loop(repo, FakeClient(skeptic=mislabelled)).run())
    assert result["status"] == "stopped" and result["step"] == "review"


def test_the_run_waits_for_the_owner_to_commit_the_registration_and_resumes(
    tmp_path: Path,
) -> None:
    repo = scratch(tmp_path / "repo", committed=False)
    client = FakeClient()
    waiting = asyncio.run(make_loop(repo, client).run())
    assert waiting["status"] == "stopped" and waiting["step"] == "run"
    assert "is not committed" in waiting["reason"]
    assert roles(client) == ["researcher", "registrar"]
    demo.commit_code(repo)
    resumed = FakeClient(executor="status: ran\ncommands: []\n")
    result = asyncio.run(make_loop(repo, resumed).run())
    assert result["status"] == "complete", result
    assert roles(resumed) == ["executor", "skeptic", "scribe"]
    assert json.loads((run_dir(repo) / "verdict.json").read_text())["read"] is False


def test_a_registration_changed_since_its_commit_holds_the_run(repo: Path) -> None:
    with (repo / demo.DOCUMENT).open("a") as document:
        document.write("\nan edit after the commit\n")
    result = asyncio.run(make_loop(repo, FakeClient()).run())
    assert result["step"] == "run" and "changed since its commit" in result["reason"]


def test_a_resumed_run_takes_again_the_step_that_stopped_it_and_keeps_the_old_one(
    repo: Path,
) -> None:
    blocking = "status: block\n" + LENSES + finding("blocking", "a:1")
    first = asyncio.run(make_loop(repo, FakeClient(skeptic=blocking)).run())
    assert first["step"] == "review"
    again = FakeClient()
    result = asyncio.run(make_loop(repo, again).run())
    assert result["status"] == "complete"
    assert roles(again) == ["skeptic", "scribe"]
    aside = run_dir(repo) / "superseded"
    kept = list(aside.glob("*-04-review/04-review.handback.yaml"))
    assert len(kept) == 1 and "blocking" in kept[0].read_text()
    assert len(list(aside.glob("*-04-review/04-review.reply.md"))) == 1
    prompt = again.calls[0].prompt
    assert "## This step again" in prompt and "blocking findings: ['a:1']" in prompt


def test_an_executor_waiting_on_the_owner_stops_the_loop_and_is_taken_again(repo: Path) -> None:
    wait = "status: waiting\ncommands: []\nwaiting_on: the D amendment, committed\n"
    first = asyncio.run(make_loop(repo, FakeClient(executor=wait)).run())
    assert first["step"] == "run" and "waiting on the owner: the D amendment" in first["reason"]
    again = FakeClient()
    assert asyncio.run(make_loop(repo, again).run())["status"] == "complete"
    assert roles(again) == ["executor", "skeptic", "scribe"]


def write(path: str, text: str = "x\n") -> Act:
    def act(call: RoleCall) -> None:
        target = call.policy.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    return act


@pytest.mark.parametrize(
    ("role", "path", "step"),
    [
        ("researcher", "docs/proposal.md", "propose"),
        ("registrar", "configs/spot_2026.yaml", "register"),
        ("executor", "src/trading_research/spot/ledger.py", "run"),
        ("executor", "src/trading_research/pipeline/short_horizon/statuses.py", "run"),
        ("executor", "experiments/synth_carry.py", "run"),
        ("executor", "tests/test_new.py", "run"),
        ("executor", "CLAUDE.md", "run"),
        ("executor", "docs/findings.md", "run"),
        ("executor", "configs/spot_2026.yaml", "run"),
        ("skeptic", demo.RESULTS, "review"),
    ],
)
def test_a_turn_that_changes_what_its_role_may_not_stops_the_loop(
    repo: Path, role: str, path: str, step: str
) -> None:
    result = asyncio.run(make_loop(repo, FakeClient({role: write(path)})).run())
    assert result["status"] == "stopped" and result["step"] == step, result
    assert "changed what it may not" in result["reason"] and path in result["reason"]


def test_a_scribe_that_also_rewrites_the_page_stops_the_loop(repo: Path) -> None:
    def both(call: RoleCall) -> None:
        scribe_writes(call)
        write(demo.PAGE, "# a better page\n")(call)

    result = asyncio.run(make_loop(repo, FakeClient({"scribe": both})).run())
    assert result["step"] == "record" and demo.PAGE in result["reason"]


def test_the_executor_may_change_what_its_entry_point_writes(repo: Path) -> None:
    acts = {"executor": write("experiments/results/synth_carry_extra.json", "{}\n")}
    assert asyncio.run(make_loop(repo, FakeClient(acts)).run())["status"] == "complete"


@pytest.mark.parametrize(
    ("role", "reply", "step"),
    [
        ("researcher", "status: declined\nreason: no kill condition can be stated\n", "propose"),
        ("registrar", "status: refused\nmissing: [the fee table's sources]\n", "register"),
        ("executor", "status: refused\ncommands: []\n", "run"),
        ("scribe", "status: recorded\nsentences: ['The carry pays.']\n", "record"),
        ("scribe", "status: recorded\nsentences: []\n", "record"),
        ("skeptic", "RAW:Nothing to report.\n```yaml\nstatus: pass\n```\n", "review"),
        ("skeptic", "status: pass\nfindings: []\n", "review"),
    ],
)
def test_a_refusal_an_unregistered_sentence_or_no_handback_stops_the_loop(
    repo: Path, role: str, reply: str, step: str
) -> None:
    client = FakeClient()
    client.replies[role] = reply
    result = asyncio.run(make_loop(repo, client).run())
    assert result["status"] == "stopped" and result["step"] == step, result


def test_a_sentence_handed_back_but_not_written_stops_the_loop(repo: Path) -> None:
    result = asyncio.run(make_loop(repo, FakeClient({"scribe": lambda _call: None})).run())
    assert result["step"] == "record" and "did not write" in result["reason"]


def test_a_handback_must_carry_its_roles_keys() -> None:
    def reply(body: str) -> str:
        return f"```handback\n{body}```"

    proposal = "hypotheses: [{id: C1, claim: c, metric: m, kills: [K1], placebos: [], expected: e}]"
    full = f"status: proposed\n{proposal}\nknown: []\nblocks: {{H: [a, b]}}\n"
    assert read_handback("researcher", "propose", reply(full))["status"] == "proposed"
    unfounded = "findings: [{lens: leakage, severity: minor, where: a}]\n"
    bad = [
        ("researcher", "status: proposed\nhypotheses: [{id: C1}]\n"),
        ("researcher", f"status: proposed\n{proposal}\nblocks: {{H: [a, b]}}\n"),
        ("researcher", f"status: proposed\n{proposal}\nknown: []\n"),
        ("researcher", full.replace("metric: m, ", "")),
        ("researcher", full.replace("placebos: [], ", "")),
        ("researcher", "status: declined\n"),
        ("registrar", "status: registered\nnull_sha256: x\n"),
        ("executor", "status: waiting\ncommands: []\n"),
        ("skeptic", "status: pass\nfindings: [{lens: taste, severity: blocking, where: a}]\n"),
        ("skeptic", "status: pass\n" + LENSES.replace("  execution:", "  money:")),
        ("skeptic", "status: pass\n" + LENSES.replace("refutation: none found}", "}")),
        ("skeptic", "status: pass\n" + LENSES + unfounded),
        ("scribe", "status: done\n"),
    ]
    for role, body in bad:
        with pytest.raises(HandbackError):
            read_handback(role, "x", reply(body))
    last = (
        reply("status: declined\nreason: r\n") + "\n" + reply("status: proposed\nhypotheses: []\n")
    )
    with pytest.raises(HandbackError):
        read_handback("researcher", "propose", last)


def test_the_verdict_on_the_spot_studys_committed_read_is_the_codes() -> None:
    run = {
        "block": "H",
        "results": "experiments/results/spot_2026_H.json",
        "ledger": "experiments/results/spot_2026_ledger.json",
        "page": "experiments/results/spot_2026_page.md",
    }
    found = verdict(REPO, run)
    assert found["ledger_match"] and found["statuses"]["S25-BB"]["status"] == "killed"
    sentence = (
        "S25-BB on H, Bybit spot BTCUSDT: killed; first kill K1 at -19.43 bp against 0.00 bp."
    )
    assert sentence in found["sentences"]
    assert verdict(REPO, {})["read"] is False


def test_what_a_turn_changed_is_read_from_git(tmp_path: Path) -> None:
    repo = demo.build_repo(tmp_path / "repo")
    before = tree_state(repo)
    (repo / "README.md").write_text("changed\n")
    (repo / "docs" / "new.md").write_text("new\n")
    (repo / "agents" / "runs").mkdir(parents=True)
    (repo / "agents" / "runs" / "ignored.txt").write_text("ignored\n")
    assert sorted(set(tree_state(repo)) - set(before)) == ["README.md", "docs/new.md"]


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


def test_the_helpers_answer_without_a_model(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from agents.workflows.policy import protected_calendar
    from agents.workflows.research_loop import main

    monkeypatch.chdir(REPO)
    assert main(["check", "--role", "executor", "git push"]) == 1
    assert main(["null-hash", str(REPO / "configs/spot_2026.yaml")]) == 0
    assert main(["calendar"]) == 0
    printed = capsys.readouterr().out
    assert "refused" in printed and "4e75c7addf549d8d" in printed
    assert "2026-08-08 2026-10-21 spot_2026:H" in printed
    calendar = protected_calendar(REPO)
    assert len(calendar) == len([line for line in printed.splitlines() if ":" in line[22:]])


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
