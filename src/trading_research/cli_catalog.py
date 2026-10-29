"""``trp catalog``: everything the repository can run, as one page.

Reads the registries, the command tree and the fee and study tables through
:mod:`trading_research.catalog` and writes ``docs/catalog.md``; ``--check``
only compares, for a test or a hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from trading_research.cli_common import RUN_PANEL, single

console = Console()


def catalog(
    write: Annotated[
        Path | None,
        typer.Option("--write", help="Write the page here, e.g. docs/catalog.md."),
    ] = None,
    check: Annotated[
        Path | None,
        typer.Option(
            "--check", help="Exit 1 when this file differs from the page as it would be written."
        ),
    ] = None,
) -> None:
    """The catalogue: data sources, synthetic markets, strategies, models, features, filters,
    labels, executions, backtests, evaluation, detectors, fees, studies, test cases."""
    from trading_research import catalog as cat
    from trading_research.studies.registry import repository

    text = cat.page(repository())
    if check is not None:
        if not check.exists() or check.read_text(encoding="utf-8") != text:
            console.print(
                f"[red]{check} is not the catalogue as written now: run trp catalog --write {check}[/red]"
            )
            raise typer.Exit(code=1)
        console.print(f"{check} is current")
        return
    if write is not None:
        write.parent.mkdir(parents=True, exist_ok=True)
        write.write_text(text, encoding="utf-8")
        console.print(f"-> {write}")
        return
    console.print(text, markup=False, highlight=False)


app = single(
    "catalog",
    "Everything the repository can run, by kind: data, strategies, models, features, "
    "backtests, evaluation; --write docs/catalog.md.",
    catalog,
    RUN_PANEL,
)
