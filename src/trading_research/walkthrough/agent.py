"""The agent harness under ``agents/``, run from the command line it serves.

The harness is not part of the installed package: it lives in the
repository, beside the registrations it reads. Everything here runs its own
entry point, ``python -m agents.workflows.research_loop``, from the
repository's root with the package on the path, as its README says, and reads
back what it wrote under ``agents/runs/``.

A model is used only when the environment holds a key for it
(``ANTHROPIC_API_KEY``), a model name (``TRP_AGENT_MODEL``) and the optional
SDK; the key is never read here, only whether it is set, and never printed or
written. Without them the demonstration runs the harness's scripted client,
which acts every role through the same tool policy.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

MODULE = "agents.workflows.research_loop"


def harness(repo: Path | None = None) -> Path:
    """The repository holding the harness: ``repo``, the working directory, or this checkout."""
    for candidate in (repo, Path.cwd(), Path(__file__).resolve().parents[3]):
        if (
            candidate is not None
            and (candidate / "agents" / "workflows" / "research_loop.py").exists()
        ):
            return candidate.resolve()
    raise FileNotFoundError("no agents/workflows/research_loop.py here: run from the repository")


def environment(repo: Path) -> dict[str, str]:
    """The caller's environment with the harness and the package importable."""
    path = os.pathsep.join(
        p for p in (str(repo), str(repo / "src"), os.environ.get("PYTHONPATH")) if p
    )
    return {**os.environ, "PYTHONPATH": path}


def loop(
    args: list[str], repo: Path | None = None, *, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    """``python -m agents.workflows.research_loop <args>`` from the repository's root."""
    root = harness(repo)
    return subprocess.run(
        [sys.executable, "-m", MODULE, *args],
        cwd=root,
        env=environment(root),
        capture_output=capture,
        text=True,
        check=False,
    )


def live_missing() -> str | None:
    """Why a model cannot act the roles here, or ``None``. Reads only whether a key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set"
    if not os.environ.get("TRP_AGENT_MODEL"):
        return "TRP_AGENT_MODEL (the model's name) is not set"
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return "the optional SDK package, claude-agent-sdk, is not installed"
    return None


def demo(into: Path, repo: Path | None = None) -> dict[str, Any]:
    """The research loop end to end on the harness's synthetic study, in a scratch repository.

    With a model available (:func:`live_missing` is ``None``) the roles are the
    model's; otherwise the harness's scripted client acts them. Either way the
    owner's three commits are made by the demonstration, as in the harness's own.
    """
    missing = live_missing()
    if missing is None:
        live = live_demo(into, harness(repo))
        live["client"] = "a live model, through the agent SDK"
        return live
    done = loop(["demo", "--into", str(into)], repo)
    if done.returncode not in (0, 3):
        raise RuntimeError(f"the harness's demonstration failed: {done.stderr[-2000:]}")
    summary: dict[str, Any] = json.loads(done.stdout)
    summary["client"] = f"the harness's scripted client ({missing})"
    return summary


def live_demo(into: Path, repo: Path) -> dict[str, Any]:  # pragma: no cover - needs a model
    """The harness's demonstration with the SDK's runner in place of the scripted client."""
    import asyncio

    sys.path.insert(0, str(repo))
    try:
        from agents.workflows import demo as scripted
        from agents.workflows.research_loop import ResearchLoop, RunLog, SdkRunner
    finally:
        sys.path.remove(str(repo))
    scratch = scripted.build_repo(into)
    log = RunLog.new(scratch, scripted.STUDY)
    log.write_json("run.json", {"study": scripted.STUDY, "brief": scripted.BRIEF,
                                "model": os.environ.get("TRP_AGENT_MODEL"), "started": datetime.now(UTC).isoformat()})  # fmt: skip
    grants: frozenset[str] = frozenset()
    stops, owner = [], []
    result: dict[str, Any] = {}
    for expected, act, granted in (*scripted.OWNER, (None, None, frozenset())):
        loop_ = ResearchLoop(scratch, scripted.STUDY, scripted.BRIEF, SdkRunner(scratch), log,
                             grants=grants, rules=repo)  # fmt: skip
        result = asyncio.run(loop_.run())
        if result["status"] == "complete" or expected is None or expected not in result["reason"]:
            break
        stops.append(result["reason"])
        act(scratch)  # type: ignore[misc]
        owner.append(f"{expected}: done")
        grants = granted
    verdict = log.path / "verdict.json"
    return {"status": result.get("status"), "reason": result.get("reason"), "repository": str(scratch),
            "run": str(log.path), "stops": stops, "owner": owner,
            "verdict": json.loads(verdict.read_text()) if verdict.exists() else None}  # fmt: skip


def runs(repo: Path | None = None) -> list[Path]:
    """The run directories under ``agents/runs``, oldest first."""
    root = harness(repo) / "agents" / "runs"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def status(run: Path) -> dict[str, Any]:
    """A run directory as it stands: the study, each step's handback, the last stop, the verdict."""
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    steps = []
    for path in sorted(run.glob("*.handback.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        steps.append({"file": path.name, "step": data.get("step"), "role": data.get("role"),
                      "status": data.get("status")})  # fmt: skip
    stops = (run / "stops.jsonl").read_text().splitlines() if (run / "stops.jsonl").exists() else []
    verdict = (
        json.loads((run / "verdict.json").read_text()) if (run / "verdict.json").exists() else None
    )
    return {
        "run": str(run),
        "study": meta.get("study"),
        "grants": meta.get("grants", []),
        "resumes": len(meta.get("resumes", [])),
        "steps": steps,
        "last_stop": json.loads(stops[-1]) if stops else None,
        "verdict": verdict,
    }
