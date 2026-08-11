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

    try:
        datasets = store.discover(input_dir)
    except store.DatasetNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    failed = False
    warned = False

    for dataset in datasets:
        label = dataset.manifest.get("symbol", "") or dataset.manifest.get("source", "")
        header = f"\n[bold]{dataset.plane}[/bold]"
        if label:
            header += f"  {label}"
        if dataset.sampled:
            header += f"  [dim](sampled to {dataset.manifest['resample_grid']})[/dim]"
        console.print(header)

        total_rows = 0
        previous_end = None
        # Files are validated one at a time. Two months of book data is tens of
        # millions of rows, and holding all of it just to check it would defeat
        # the point of partitioning by day in the first place.
        for file in dataset.files:
            frame = store.read_plane(dataset, file)
            report: ValidationReport = (
                validate_trades(frame)
                if dataset.plane == "trades"
                else validate_book(frame, depth=dataset.depth, sampled=dataset.sampled)
            )
            total_rows += report.rows

            if len(dataset.files) > 1:
                _print_findings(report, prefix=f"  {file.stem}: ", quiet=True)
            else:
                _print_findings(report, prefix="  ")

            failed |= not report.ok
            warned |= bool(report.warnings)

            # Partitioning by day only works if the days actually join up. A
            # single file can be perfectly ordered while the sequence of files
            # is not, and nothing inside one file can reveal that.
            start = frame["timestamp"].iloc[0]
            if previous_end is not None and start < previous_end:
                err_console.print(
                    f"  [red]{file.stem}: starts at {start}, before the previous file ended "
                    f"at {previous_end}; the partitions overlap or are out of order[/red]"
                )
                failed = True
            previous_end = frame["timestamp"].iloc[-1]
            del frame

        if len(dataset.files) > 1:
            console.print(f"  {len(dataset.files)} file(s), {total_rows:,} rows total")

    console.print()
    if failed:
        err_console.print("[red]Validation failed.[/red]")
        raise typer.Exit(code=1)
    if warned and strict:
        err_console.print("[yellow]Warnings present and --strict was given.[/yellow]")
        raise typer.Exit(code=1)
    console.print("[green]Validation passed.[/green]")


@app.command("download")
def download(
    start: Annotated[str, typer.Option("--start", help="First day, YYYY-MM-DD.")],
    end: Annotated[str, typer.Option("--end", help="Last day, inclusive, YYYY-MM-DD.")],
    symbol: Annotated[
        str, typer.Option("--symbol", "-s", help="Instrument, e.g. BTCUSDT.")
    ] = "BTCUSDT",
    kind: Annotated[
        str,
        typer.Option(
            "--kind", "-k", help="Dataset: 'aggTrades' (trades) or 'bookTicker' (best bid/ask)."
        ),
    ] = "aggTrades",
    market: Annotated[
        str,
        typer.Option(
            "--market",
            "-m",
            help="'spot', 'futures-um' or 'futures-cm'. Only futures publish a book.",
        ),
    ] = "futures-um",
    output: Annotated[Path, typer.Option("--output", "-o", help="Destination directory.")] = Path(
        "data/raw"
    ),
    grid: Annotated[
        str | None,
        typer.Option(
            "--grid",
            help="Resample the book to this interval, e.g. '100ms'. Ignored for trades. "
            "Without it the raw update stream is kept, which is very large.",
        ),
    ] = None,
    keep_raw: Annotated[
        bool,
        typer.Option(
            "--keep-raw/--no-keep-raw",
            help="Keep the downloaded archives so a rerun costs nothing.",
        ),
    ] = True,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Reconvert days that are already present.")
    ] = False,
    no_verify: Annotated[
        bool, typer.Option("--no-verify", help="Skip SHA-256 verification. Not recommended.")
    ] = False,
) -> None:
    """Download public Binance market data and convert it to the project contract.

    No credentials are involved: these are static files on a public endpoint.

    Only the futures markets publish order-book data, and only the best bid and
    ask. Depth beyond the touch is not published by any exchange for free and
    needs the collector.
    """
    from datetime import date

    from lobml.data.binance import ArchiveSpec, BinanceArchiveError, DayResult, download_range

    try:
        spec = ArchiveSpec(market=market, kind=kind, symbol=symbol.upper())  # type: ignore[arg-type]
        first = date.fromisoformat(start)
        last = date.fromisoformat(end)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if kind == "bookTicker" and market == "spot":
        err_console.print("[red]Spot publishes no order-book data.[/red] Use --market futures-um.")
        raise typer.Exit(code=2)

    if kind == "bookTicker" and not grid:
        console.print(
            "[yellow]No --grid given: keeping the raw update stream "
            "(~13.5M rows and ~110 MB per day for BTCUSDT).[/yellow]"
        )

    n_days = (last - first).days + 1
    console.print(
        f"[bold]{spec.symbol} {spec.kind}[/bold] ({spec.market}), {first} → {last}, {n_days} day(s)"
    )
    if grid:
        console.print(
            f"Book resampled to [bold]{grid}[/bold]; intra-interval updates are discarded."
        )

    downloaded = 0
    rows = 0
    failures: list[DayResult] = []

    def report(result: DayResult) -> None:
        nonlocal downloaded, rows
        downloaded += result.bytes_downloaded
        rows += result.rows
        if result.output is None:
            failures.append(result)
            console.print(f"  [yellow]{result.day}  skipped: {result.skipped}[/yellow]")
        elif result.skipped:
            console.print(f"  [dim]{result.day}  {result.skipped}[/dim]")
        else:
            console.print(
                f"  {result.day}  {result.rows:>10,} rows  "
                f"[dim]{result.bytes_downloaded / 1e6:6.0f} MB  "
                f"(total {downloaded / 1e9:.2f} GB)[/dim]"
            )

    try:
        results = download_range(
            spec,
            first,
            last,
            output,
            grid=grid,
            keep_raw=keep_raw,
            verify=not no_verify,
            overwrite=overwrite,
            on_day=report,
        )
    except BinanceArchiveError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    converted = [r for r in results if r.output and not r.skipped]
    console.print(
        f"\n[green]{len(converted)} day(s) converted[/green], {rows:,} rows, "
        f"{downloaded / 1e9:.2f} GB downloaded → [bold]{Path(output) / spec.symbol / spec.kind}[/bold]"
    )
    if failures:
        console.print(
            f"[yellow]{len(failures)} day(s) unavailable — see the manifest for the list.[/yellow]"
        )


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


def _print_findings(report: ValidationReport, *, prefix: str = "", quiet: bool = False) -> None:
    """Render one report's findings.

    ``quiet`` suppresses informational lines, which is what makes a sixty-file
    dataset readable: repeating the same coverage note per day buries the one
    line that differs.
    """
    styles = {Severity.ERROR: "red", Severity.WARNING: "yellow", Severity.INFO: "dim"}
    shown = [f for f in report if not (quiet and f.severity is Severity.INFO)]
    if not shown:
        if not quiet:
            console.print(f"{prefix}[green]all checks passed[/green]")
        return
    for finding in shown:
        console.print(f"{prefix}[{styles[finding.severity]}]{finding}[/{styles[finding.severity]}]")


def _span(df: object) -> str:
    """Human-readable time span of a frame, for the summary table."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame) or df.empty:
        return "—"
    ts = df["timestamp"]
    return f"{ts.min():%Y-%m-%d %H:%M:%S} → {ts.max():%Y-%m-%d %H:%M:%S}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
