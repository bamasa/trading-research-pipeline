"""``trp fees``: every fee table the repository holds, as one table.

Reads what :mod:`trading_research.fees` reads (the cost modules and the
registrations' YAML files) and prints it; changes nothing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from trading_research.cli_common import RUN_PANEL, single

console = Console()


def fees(
    venue: Annotated[
        str | None,
        typer.Option("--venue", help="Only this venue's rows, e.g. bybit_linear_perpetual."),
    ] = None,
    csv: Annotated[Path | None, typer.Option("--csv", help="Also write the table here.")] = None,
) -> None:
    """Every fee table the repository holds: venue, tier, period, maker, taker, source."""
    from trading_research import fees as fee_tables

    frame = fee_tables.table(venue=venue)
    if frame.empty:
        console.print(f"[red]no fee rows for {venue!r}[/red]")
        raise typer.Exit(code=1)
    rendered = Table(title=f"{len(frame)} fee rows, in bp of notional per side", show_edge=False)
    for column in ("venue", "tier", "period", "maker", "taker", "source"):
        right = column in ("maker", "taker")
        rendered.add_column(column, justify="right" if right else "left", overflow="fold")
    for row in frame.itertuples(index=False):
        source = str(row.source).split("; http")[0]  # the file; the page is in --csv
        cells = [
            "—"
            if isinstance(v, float) and math.isnan(v)
            else (f"{v:g}" if isinstance(v, float) else str(v))
            for v in (row.venue, row.tier, row.period, row.maker_bp, row.taker_bp, source)
        ]
        rendered.add_row(*cells)
    console.print(rendered)
    console.print(
        "[dim]A negative maker fee is a rebate. Each source stays the authority for its rows.[/dim]"
    )
    if csv is not None:
        csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv, index=False)
        console.print(f"-> {csv}")


app = single(
    "fees",
    "Every fee table the repository holds: venue, tier, period, maker, taker, source.",
    fees,
    RUN_PANEL,
)
