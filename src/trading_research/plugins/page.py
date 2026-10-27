"""The conformance page of one plugin, and the index of a run over many."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.plugins.conformance import Conformance

CAVEAT = (
    "Measured on synthetic data from `data.synthetic.generate`: the numbers say the "
    "code ran as stated, under the fee tier named, and nothing about any market."
)


def _source(conformance: Conformance) -> str:
    path = conformance.spec.source
    if path is None:
        return conformance.spec.module
    parts = path.parts
    if "trading_research" in parts:
        return "src/" + "/".join(parts[parts.index("trading_research") :])
    return path.name


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4g}" if abs(value) >= 1e-3 or value == 0 else f"{value:.3e}"
    return str(value).replace("|", "/").replace("\n", " ")


def _table(frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in columns) + " |")
    return lines


def render(conformance: Conformance) -> str:
    spec = conformance.spec
    verdict = "passes every check" if conformance.passed else "FAILS at least one check"
    lines = [
        f"# `{spec.name}`: conformance of a {spec.kind} plugin",
        "",
        f"**{verdict}.** {spec.summary}",
        "",
        f"- origin: {spec.origin}",
        f"- defined in: `{_source(conformance)}` (module `{spec.module}`)",
        f"- sha256 of that file: `{spec.fingerprint() or 'unavailable'}`",
        f"- built with: `{conformance.params or 'its defaults'}`",
        f"- data: {conformance.market}",
        f"- fee tier: {conformance.fee_tier}",
        f"- checked: {conformance.when}",
        "",
        "## Checks",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for result in conformance.results:
        lines.append(f"| {result.check} | {result.status} | {_cell(result.detail)} |")
    numbered = [r for r in conformance.results if r.numbers]
    if numbered:
        lines += ["", "## Numbers", ""]
        for result in numbered:
            pairs = ", ".join(f"{k} = {_cell(v)}" for k, v in result.numbers.items())
            lines.append(f"- {result.check}: {pairs}")
    if conformance.table is not None and not conformance.table.empty:
        lines += ["", "## Execution", "", *_table(conformance.table)]
        lines += [
            "",
            "Per attempt, misses included: gross is the market's move in the position's "
            "favour, cost the fees, spread and slippage paid, net their difference, all in "
            "bp of notional; `net_bp` is the sum over attempts.",
        ]
    lines += ["", CAVEAT, ""]
    return "\n".join(lines)


def write(conformance: Conformance, root: Path) -> Path:
    """``root/<name>/page.md``, and the execution table beside it as CSV."""
    folder = root / conformance.spec.name
    folder.mkdir(parents=True, exist_ok=True)
    page = folder / "page.md"
    page.write_text(render(conformance), encoding="utf-8")
    if conformance.table is not None:
        conformance.table.to_csv(folder / "execution.csv", index=False)
    return page


def summary_row(item: Conformance, page: Path | None = None) -> dict[str, Any]:
    """One line of the index: each check's status, and whether the plugin passed."""
    return {
        "kind": item.spec.kind,
        "name": item.spec.name,
        "origin": item.spec.origin,
        "a. protocol": item.status("a."),
        "b. no look-ahead": item.status("b. no look-ahead"),
        "b. declared look-back": item.status("b. declared look-back"),
        "c. execution": item.status("c."),
        "d. synthetic": item.status("d."),
        "passed": "yes" if item.passed else "no",
        "page": str(page) if page else "",
    }


def unfinished_row(kind: str, name: str, origin: str, reason: str) -> dict[str, Any]:
    """The index line of a plugin whose check did not finish: every check failed."""
    statuses = dict.fromkeys(
        (
            "a. protocol",
            "b. no look-ahead",
            "b. declared look-back",
            "c. execution",
            "d. synthetic",
        ),
        "fail",
    )
    return {
        "kind": kind,
        "name": name,
        "origin": origin,
        **statuses,
        "passed": "no",
        "page": "",
        "reason": reason,
    }


def write_unfinished(row: dict[str, Any], root: Path) -> Path:
    folder = root / str(row["name"])
    folder.mkdir(parents=True, exist_ok=True)
    page = folder / "page.md"
    page.write_text(
        f"# `{row['name']}`: conformance of a {row['kind']} plugin\n\n"
        f"**FAILS: the check did not finish.** {row.get('reason', '')}\n",
        encoding="utf-8",
    )
    return page


def write_index(rows: list[dict[str, Any]], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).drop(columns=["page", "reason"], errors="ignore")
    failing = table[table["passed"] == "no"]
    lines = [
        "# Plugin conformance",
        "",
        f"{len(table)} plugins checked, {len(failing)} with at least one failed check. "
        "Each name links to its page.",
        "",
        *_table(table.assign(name=[f"[{n}]({n}/page.md)" for n in table["name"]])),
        "",
        CAVEAT,
        "",
    ]
    path = root / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    table.to_csv(root / "index.csv", index=False)
    return path
