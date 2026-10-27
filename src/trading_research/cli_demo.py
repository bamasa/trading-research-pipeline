"""``trp demo``: the walkthrough end to end on a synthetic market, in a few minutes.

See :mod:`trading_research.walkthrough.demo`. Writes ``<out>/page.md`` and
``<out>/manifest.json`` beside every step's files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from trading_research.cli_common import RUN_PANEL, single

console = Console()
err_console = Console(stderr=True)


def demo(
    execution: Annotated[
        str,
        typer.Option(
            "--execution", help="taker, passive, maker or all: the modes the verdict reads."
        ),
    ] = "all",
    out: Annotated[
        Path, typer.Option("--out", help="Where every step's files and the page go.")
    ] = Path("artifacts/demo"),
    seed: Annotated[
        int, typer.Option("--seed", help="Seeds the generated market and the placebos.")
    ] = 20260101,
    agent: Annotated[
        bool, typer.Option("--agent", help="Also run the research loop through the agent harness.")
    ] = False,
    small: Annotated[
        bool, typer.Option("--small", help="Two instruments, six days, two holds: the tests' set.")
    ] = False,
) -> None:
    """data → regimes → instruments → signal → execution → verdict on generated data, and the page."""
    from trading_research.walkthrough.demo import DemoConfig, run

    if execution not in ("taker", "passive", "maker", "all"):
        err_console.print("[red]--execution must be taker, passive, maker or all[/red]")
        raise typer.Exit(code=2)
    changes = {"out": out, "execution": execution, "seed": seed, "agent": agent}
    config = DemoConfig.small(**changes) if small else DemoConfig(**changes)  # type: ignore[arg-type]
    try:
        result = run(config, report=lambda line: console.print(line, highlight=False))
    except FileExistsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    console.print(f"\n[bold]{result.page}[/bold]  statuses: {result.statuses}")


app = single(
    "demo",
    "The whole walkthrough on a synthetic market in minutes, a page per execution mode.",
    demo,
    RUN_PANEL,
)
