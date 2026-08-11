"""Command line interface.

Each stage of the pipeline is a command, and every command reads and writes
files on disk rather than passing objects in memory. That is slower than a
single in-process run and it is the right trade: a stage can be inspected,
re-run or replaced without re-running what came before, and a reviewer can look
at what actually went into a model instead of taking the pipeline's word for it.

Commands available so far::

    lobml generate-demo-data --output data/demo
    lobml validate-data --input data/demo
    lobml describe-schema

The remaining commands from the specification — ``download``, ``build-features``,
``train``, ``backtest``, ``report`` and ``run`` — arrive with the stages they
drive. A command that exists but does nothing is worse than one that is
absent, because it implies a capability the project does not have yet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from lobml import __version__
from lobml.data import store, synthetic
from lobml.data.schema import BOOK_SCHEMA, TRADE_SCHEMA
from lobml.data.validate import Severity, ValidationReport, validate_book, validate_trades

app = typer.Typer(
    name="lobml",
    help="Leakage-aware ML research and backtesting for limit order books.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"lobml {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Leakage-aware ML research and backtesting for limit order books."""


@app.command("generate-demo-data")
def generate_demo_data(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory to write the dataset into.")
    ] = Path("data/demo"),
    steps: Annotated[
        int, typer.Option("--steps", "-n", min=2, help="Number of book snapshots to generate.")
    ] = 20_000,
    seed: Annotated[
        int, typer.Option("--seed", help="Seed. The same seed always yields the same dataset.")
    ] = 20260101,
    depth: Annotated[
        int, typer.Option("--depth", min=1, max=50, help="Order-book levels per side.")
    ] = 10,
    symbol: Annotated[
        str, typer.Option("--symbol", help="Instrument name to stamp on the rows.")
    ] = "DEMOUSDT",
    signal_strength: Annotated[
        float,
        typer.Option(
            "--signal-strength",
            min=0.0,
            help="Strength of the injected predictable component. Pass 0 for a dataset with no edge, "
            "which is what the leakage tests use.",
        ),
    ] = 0.15,
) -> None:
    """Generate a synthetic dataset for tests, CI and the quickstart.

    Synthetic data exists so the pipeline can run without downloading anything.
    It is not evidence about markets: the generator puts a signal in and the
    pipeline is expected to find it, which tests the pipeline and nothing else.
    """
    config = synthetic.SyntheticConfig(
        symbol=symbol,
        n_steps=steps,
        depth=depth,
        seed=seed,
        signal_strength=signal_strength,
    )
    with console.status(f"Generating {steps:,} snapshots…"):
        dataset = synthetic.generate(config)
        path = store.write_dataset(
            output,
            trades=dataset.trades,
            book=dataset.book,
            latent=dataset.latent,
            manifests=dataset.manifests,
            extra={
                "warning": "SYNTHETIC DATA. Generated, not observed. "
                "Any result computed on it says nothing about real markets.",
                "config": {
                    field: getattr(config, field)
                    for field in synthetic.SyntheticConfig.__dataclass_fields__
                },
            },
        )

    table = Table(title="Synthetic dataset", title_justify="left", show_edge=False)
    table.add_column("plane", style="bold")
    table.add_column("rows", justify="right")
    table.add_column("span")
    table.add_row("book", f"{len(dataset.book):,}", _span(dataset.book))
    table.add_row("trades", f"{len(dataset.trades):,}", _span(dataset.trades))
    console.print(table)
    console.print(f"\nWritten to [bold]{path}[/bold]  (seed {seed}, depth {depth})")
    console.print(
        "[yellow]Synthetic data — not a market. Results on it are not evidence of edge.[/yellow]"
    )


@app.command("validate-data")
def validate_data(
    input_dir: Annotated[
        Path, typer.Option("--input", "-i", help="Dataset directory to check.")
    ] = Path("data/demo"),
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero on warnings as well as errors.")
    ] = False,
) -> None:
    """Check a dataset against its contract and report quality problems.

    Errors mean the data violates an invariant the pipeline relies on. Warnings
    — sequence gaps, locked books, long pauses — are usually survivable but
    change what a rolling window measures, so they are always shown.
    """
    if not input_dir.exists():
        err_console.print(f"[red]No such directory:[/red] {input_dir}")
        raise typer.Exit(code=2)

    reports: list[ValidationReport] = []
    if store.has_plane(input_dir, "trades"):
        reports.append(validate_trades(store.read_trades(input_dir)))
    if store.has_plane(input_dir, "book"):
        reports.append(validate_book(store.read_book(input_dir)))

    if not reports:
        err_console.print(f"[red]No recognised data planes in[/red] {input_dir}")
        raise typer.Exit(code=2)

    failed = False
    warned = False
    for report in reports:
        console.print(f"\n[bold]{report.plane}[/bold]  {report.rows:,} rows")
        if not report.findings:
            console.print("  [green]all checks passed[/green]")
        for finding in report:
            style = {
                Severity.ERROR: "red",
                Severity.WARNING: "yellow",
                Severity.INFO: "dim",
            }[finding.severity]
            console.print(f"  [{style}]{finding}[/{style}]")
        failed |= not report.ok
        warned |= bool(report.warnings)

    console.print()
    if failed:
        err_console.print("[red]Validation failed.[/red]")
        raise typer.Exit(code=1)
    if warned and strict:
        err_console.print("[yellow]Warnings present and --strict was given.[/yellow]")
        raise typer.Exit(code=1)
    console.print("[green]Validation passed.[/green]")


@app.command("describe-schema")
def describe_schema(
    plane: Annotated[
        str, typer.Argument(help="Which contract to print: 'trades' or 'book'.")
    ] = "trades",
) -> None:
    """Print a data contract, so the expected columns and units are never guesswork."""
    schema = {"trades": TRADE_SCHEMA, "book": BOOK_SCHEMA}.get(plane)
    if schema is None:
        err_console.print(f"[red]Unknown plane[/red] {plane!r}; expected 'trades' or 'book'")
        raise typer.Exit(code=2)

    table = Table(title=f"{schema.name} (schema version {schema.version})", title_justify="left")
    table.add_column("column", style="bold")
    table.add_column("dtype")
    table.add_column("unit")
    table.add_column("description")
    for column in schema.columns:
        table.add_row(column.name, column.dtype, column.unit or "—", column.description)
    console.print(table)


def _span(df: object) -> str:
    """Human-readable time span of a frame, for the summary table."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame) or df.empty:
        return "—"
    ts = df["timestamp"]
    return f"{ts.min():%Y-%m-%d %H:%M:%S} → {ts.max():%Y-%m-%d %H:%M:%S}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
