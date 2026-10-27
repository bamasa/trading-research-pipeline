"""The research loop end to end on the synthetic study, through the command a
reviewer runs: ``python -m agents.workflows.research_loop demo``.

The scripted client acts each role through its policy, the entry point runs
for real on seeded synthetic draws in a scratch git repository, and the
demonstration acts as the owner at each stop. No model, no network, no data.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from agents.workflows import demo
from agents.workflows.research_loop import headline_sentences, main

REPO = Path(__file__).resolve().parent.parent
STEPS = ("01-propose", "02-register", "03-run", "04-review", "06-record")


@pytest.fixture(scope="module")
def summary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    before = sorted((REPO / "agents" / "runs").glob("*"))
    into = tmp_path_factory.mktemp("demo") / "scratch"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = main(["demo", "--into", str(into)])
    found: dict[str, Any] = json.loads(printed.getvalue())
    found["exit"] = code
    assert sorted((REPO / "agents" / "runs").glob("*")) == before, "the demo wrote into REPO"
    return found


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    ).stdout


def test_the_loop_runs_propose_to_record_and_stops_only_for_the_owner(
    summary: dict[str, Any],
) -> None:
    assert summary["exit"] == 0 and summary["status"] == "complete", summary["reason"]
    stops = summary["stops"]
    assert len(stops) == 3 and all(s.startswith("waiting on the owner") for s in stops)
    assert "configs/synth_carry.yaml is not committed" in stops[0]
    assert "the D amendment" in stops[1] and "open entry" in stops[2]
    run = Path(summary["run"])
    assert run.parent == Path(summary["repository"]) / "agents" / "runs"
    order = [yaml.safe_load((run / f"{s}.handback.yaml").read_text())["step"] for s in STEPS]
    assert order == ["propose", "register", "run", "review", "record"]
    assert yaml.safe_load((run / "05-verdict.handback.yaml").read_text())["step"] == "verdict"


def test_the_verdict_is_the_entry_points_and_the_scribe_wrote_its_sentence(
    summary: dict[str, Any],
) -> None:
    repo = Path(summary["repository"])
    verdict = summary["verdict"]
    results = json.loads((repo / demo.RESULTS).read_text())
    assert verdict["read"] and verdict["ledger_match"] and verdict["block"] == "H"
    assert verdict["statuses"]["C1"] == {
        "status": results["statuses"]["C1"]["status"],
        "first_kill": results["statuses"]["C1"]["first_kill"],
    }
    sentences = headline_sentences((repo / demo.PAGE).read_text(), "H")
    assert verdict["sentences"] == sentences and len(sentences) == 1
    assert sentences[0] in (repo / demo.RUN_LOG).read_text()
    assert sentences[0] in (repo / "docs" / "findings.md").read_text()


#: Each refusal the owner named, by role, by what was attempted and by why.
REFUSALS = [
    ("executor", "git push origin demo", "git push is not read-only"),
    ("executor", "git remote add origin", "git remote is not read-only"),
    ("executor", "src/trading_research/spot/ledger.py", "does not write"),
    ("executor", "src/trading_research/pipeline/short_horizon/statuses.py", "does not write"),
    ("executor", "--start 2026-09-01", "touches spot_2026:H"),
    ("executor", "--start 2031-08-01", "touches synth_carry:H"),
    ("executor", f"{demo.ENTRY} open H", "needs the owner's grant"),
    ("executor", f"{demo.ENTRY} read H", "needs its open entry committed"),
    ("registrar", "git commit", "not read-only"),
    ("registrar", f"experiments/{demo.STUDY}.py", "does not write"),
    ("researcher", "trading-research download", "does not run"),
    ("researcher", "data/spot_2026/", "data is read only by the commands"),
    ("skeptic", f"{demo.ENTRY} read H", "runs only"),
    ("skeptic", demo.RESULTS, "does not write"),
    ("scribe", demo.PAGE, "does not write"),
]


@pytest.mark.parametrize(("role", "attempted", "why"), REFUSALS)
def test_the_policy_refused_each_attempt_and_said_why(
    summary: dict[str, Any], role: str, attempted: str, why: str
) -> None:
    found = [
        r
        for r in summary["refused"]
        if r["role"] == role and attempted in json.dumps(r["input"]) and why in r["reason"]
    ]
    assert found, (role, attempted, why)
    assert summary["allowed_but_never"] == []


def test_the_run_directory_is_readable_and_logs_every_decision(summary: dict[str, Any]) -> None:
    run = Path(summary["run"])
    for stem in STEPS:
        for suffix in (".prompt.md", ".reply.md", ".handback.yaml", ".changed.json"):
            assert (run / f"{stem}{suffix}").read_text().strip(), stem + suffix
        for line in (run / f"{stem}.events.jsonl").read_text().splitlines():
            event = json.loads(line)
            assert {"tool", "input", "allowed", "reason"} <= set(event)
    meta = json.loads((run / "run.json").read_text())
    assert meta["study"] == demo.STUDY and meta["grants"] == [demo.GRANT]
    assert len(meta["resumes"]) == 3 and "key" not in json.dumps(meta).lower()
    assert len(list((run / "superseded").glob("*-03-run/03-run.handback.yaml"))) == 2
    stops = [json.loads(line) for line in (run / "stops.jsonl").read_text().splitlines()]
    assert [s["step"] for s in stops] == ["run", "run", "run"]
    changed = {s: json.loads((run / f"{s}.changed.json").read_text()) for s in STEPS}
    assert changed["01-propose"] == [] and changed["04-review"] == []
    assert changed["02-register"] == [demo.CONFIG, demo.DOCUMENT]
    assert set(changed["03-run"]) == {demo.LEDGER, demo.PAGE, demo.RESULTS}
    assert changed["06-record"] == ["docs/findings.md", demo.RUN_LOG]


def test_the_scratch_repository_has_only_the_owners_commits_and_no_remote(
    summary: dict[str, Any],
) -> None:
    repo = summary["repository"]
    assert set(git(repo, "log", "--format=%an").split("\n")) - {""} == {"demo owner"}
    assert len(git(repo, "log", "--format=%H").split()) == 4
    assert git(repo, "remote").strip() == ""
    assert "Co-Authored-By" not in git(repo, "log", "--format=%B")
    assert not (Path(repo) / "src").exists() and not (Path(repo) / "data").exists()
