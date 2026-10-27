"""Each role's prompts, and the handback that ends its reply.

A role's system prompt is the repository's rules (``CLAUDE.md``), the loop
(``agents/protocol.md``) and the role's own file, in that order, read from the
working tree the run starts on. The step prompt carries the owner's brief, the
study and its registration's paths, the grants, the environment's command
prefix, the earlier steps' handbacks and the gate the step must pass. A reply
counts only through its last fenced block tagged ``handback``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: The loop's steps and the role of each; the verdict belongs to the code.
STEPS: tuple[tuple[str, str], ...] = (
    ("propose", "researcher"),
    ("register", "registrar"),
    ("run", "executor"),
    ("review", "skeptic"),
    ("verdict", "code"),
    ("record", "scribe"),
)
STATUSES: dict[str, frozenset[str]] = {
    "researcher": frozenset({"proposed", "declined"}),
    "registrar": frozenset({"registered", "refused"}),
    "executor": frozenset({"ran", "refused", "failed"}),
    "skeptic": frozenset({"pass", "block"}),
    "scribe": frozenset({"recorded"}),
}
SEVERITIES = ("blocking", "major", "minor")
LENSES = ("leakage", "fidelity", "execution")
GATES: dict[str, str] = {
    "propose": "Every hypothesis has a kill condition and an expected outcome, stated now.",
    "register": "The YAML and the document agree, nothing is left open but the placeholders, "
    "and the document holds the YAML's null-placeholder sha256. The owner commits it.",
    "run": "Only the registered commands, in order; a refusal is reported and ends the step.",
    "review": "A blocking finding stops the loop; every finding has a file:line.",
    "record": "Every sentence you write is one of the verdict's headline sentences, verbatim.",
}
HANDBACK = re.compile(r"```handback[ \t]*\n(.*?)```", re.DOTALL)
SHA256 = re.compile(r"[0-9a-f]{64}")


class HandbackError(ValueError):
    """A reply without a valid handback block for its role."""


def system_prompt(repo: Path, role: str) -> str:
    parts = ("CLAUDE.md", "agents/protocol.md", f"agents/roles/{role}.md")
    return "\n\n".join(f"<!-- {name} -->\n{(repo / name).read_text()}" for name in parts)


@dataclass(frozen=True)
class StepContext:
    study: str
    brief: str
    config: str
    document: str
    grants: frozenset[str] = frozenset()
    prefix: str = "python"
    cli: str = "trading-research"
    earlier: tuple[tuple[str, dict[str, Any]], ...] = ()


def step_prompt(step: str, role: str, ctx: StepContext) -> str:
    grants = ", ".join(sorted(ctx.grants)) or "none"
    lines = [
        f"# Step: {step} (you are the {role})",
        "",
        f"Study: `{ctx.study}`. Registration: `{ctx.document}` with `{ctx.config}`.",
        f"The owner's grants for this run: {grants}.",
        f"Run Python as `{ctx.prefix}` and the CLI as `{ctx.cli}`, one command per call.",
        "",
        "## The owner's brief",
        "",
        ctx.brief.strip(),
        "",
        "## The earlier steps' handbacks",
        "",
    ]
    for name, handback in ctx.earlier:
        dumped = yaml.safe_dump(handback, sort_keys=False, allow_unicode=True)
        lines += [f"### {name}", "", "```yaml", dumped.rstrip(), "```", ""]
    if not ctx.earlier:
        lines += ["None: this is the first step.", ""]
    lines += [
        "## Your gate",
        "",
        GATES[step],
        "",
        "End your reply with one fenced block tagged `handback`, in YAML, with the keys "
        f"`agents/protocol.md` gives for the {role}.",
    ]
    return "\n".join(lines)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise HandbackError(message)


def read_handback(role: str, step: str, reply: str) -> dict[str, Any]:
    """The last handback block of ``reply``, checked against the role's keys."""
    blocks = HANDBACK.findall(reply)
    _require(bool(blocks), f"the {role}'s reply has no handback block")
    try:
        data = yaml.safe_load(blocks[-1])
    except yaml.YAMLError as exc:
        raise HandbackError(f"the {role}'s handback is not YAML: {exc}") from exc
    _require(isinstance(data, dict), f"the {role}'s handback is not a mapping")
    _require(data.get("status") in STATUSES[role], f"{role} status {data.get('status')!r}")
    data["role"], data["step"] = role, step
    status = data["status"]
    if role == "researcher" and status == "proposed":
        hypotheses = data.get("hypotheses")
        _require(isinstance(hypotheses, list) and bool(hypotheses), "a proposal without hypotheses")
        for h in hypotheses:
            _require(isinstance(h, dict) and bool(h.get("kills")), f"no kill condition: {h}")
            _require(bool(h.get("expected")), f"no expected outcome: {h}")
    if role == "registrar" and status == "registered":
        _require(bool(SHA256.fullmatch(str(data.get("null_sha256", "")))), "no null_sha256")
    if role == "skeptic":
        findings = data.setdefault("findings", []) or []
        for f in findings:
            _require(isinstance(f, dict) and f.get("severity") in SEVERITIES, f"severity: {f}")
            _require(f.get("lens") in LENSES and bool(f.get("where")), f"lens or where: {f}")
    if role == "scribe":
        _require(isinstance(data.get("sentences", []), list), "sentences is a list")
    return data


def blocking(findings: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in findings if f.get("severity") == "blocking"]
