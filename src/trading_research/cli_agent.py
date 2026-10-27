"""``trp agent``: the research loop through the agent harness under ``agents/``.

Each command runs the harness's own entry point
(``python -m agents.workflows.research_loop``) from the repository's root; see
:mod:`trading_research.walkthrough.agent`. A model acts the roles only when the
environment names one (``TRP_AGENT_MODEL``) and holds its key; the harness
refuses to start a run otherwise, and says why. No key is read, printed or
written here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from trading_research.cli_common import STUDIES_PANEL

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="agent",
    help="9 Agent: the research loop through the agent harness, every tool call under its policy.",
    no_args_is_help=True,
)
app.info.rich_help_panel = STUDIES_PANEL

Grant = Annotated[
    list[str] | None, typer.Option("--grant", help="<config>:<block> the owner grants.")
]


def _relay(args: list[str]) -> None:
    from trading_research.walkthrough.agent import loop

    try:
        done = loop(args)
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    if done.returncode:
        raise typer.Exit(code=done.returncode)


@app.command("run")
def run_cmd(
    study: Annotated[str, typer.Option("--study", help="configs/<study>.yaml and its document.")],
    brief: Annotated[str | None, typer.Option("--brief", help="The owner's brief.")] = None,
    brief_file: Annotated[
        Path | None, typer.Option("--brief-file", help="The brief, from a file.")
    ] = None,
    grant: Grant = None,
) -> None:
    """Start the loop for one study; it stops at every gate the owner holds."""
    args = ["run", "--study", study]
    args += ["--brief", brief] if brief else []
    args += ["--brief-file", str(brief_file)] if brief_file else []
    for item in grant or []:
        args += ["--grant", item]
    _relay(args)


@app.command("resume")
def resume_cmd(
    run: Annotated[Path, typer.Argument(help="agents/runs/<run>")], grant: Grant = None
) -> None:
    """Take a stopped run up again at the step that stopped it, with any new grant."""
    args = ["run", "--resume", str(run)]
    for item in grant or []:
        args += ["--grant", item]
    _relay(args)


@app.command("status")
def status_cmd(
    run: Annotated[
        Path | None, typer.Argument(help="A run directory; every run when omitted.")
    ] = None,
) -> None:
    """A run's steps, handbacks, last stop and verdict; or the list of runs."""
    from trading_research.walkthrough.agent import runs, status

    if run is None:
        found = runs()
        if not found:
            console.print("no runs under agents/runs")
            return
        table = Table(title="agents/runs", show_edge=False)
        for column in ("run", "study", "steps", "last stop"):
            table.add_column(column, overflow="fold")
        for path in found:
            state = status(path)
            stop = state["last_stop"]
            table.add_row(path.name, str(state["study"]), str(len(state["steps"])),
                          "" if stop is None else f"{stop.get('step')}: {stop.get('reason')}")  # fmt: skip
        console.print(table)
        return
    if not run.is_dir():
        err_console.print(f"[red]no run directory {run}[/red]")
        raise typer.Exit(code=2)
    console.print_json(json.dumps(status(run), default=str))


@app.command("demo")
def demo_cmd(
    into: Annotated[
        Path | None, typer.Option("--into", help="An empty directory for the scratch repository.")
    ] = None,
) -> None:
    """The whole loop on the harness's synthetic study: a model if one is configured, else its scripted client."""
    from datetime import UTC, datetime

    from trading_research.walkthrough.agent import demo, harness

    target = into or harness() / "agents" / "runs" / f"demo-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    summary = demo(target)
    console.print(f"client: {summary['client']}")
    console.print(f"status: {summary['status']}; owner's stops: {len(summary.get('stops', []))}; "
                  f"refusals made and explained: {len(summary.get('refused', []))}")  # fmt: skip
    verdict = summary.get("verdict") or {}
    console.print(
        f"verdict: read {verdict.get('read')}, ledger match {verdict.get('ledger_match')}"
    )
    console.print(f"[dim]-> {summary['run']}[/dim]")
    if summary["status"] != "complete":
        raise typer.Exit(code=3)


@app.command("calendar")
def calendar_cmd() -> None:
    """Every range a registration protects: no command of the loop touches them without a grant."""
    _relay(["calendar"])
