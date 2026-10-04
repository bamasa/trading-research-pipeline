"""Command line interface.

Each stage of the pipeline is a command, and every command reads and writes
files on disk rather than passing objects in memory. That is slower than a
single in-process run and it is the right trade: a stage can be inspected,
re-run or replaced without re-running what came before, and a reviewer can look
at what actually went into a model instead of taking the pipeline's word for it.

Commands available so far::

    trading_research generate-demo-data --output data/demo
    trading_research validate-data --input data/demo
    trading_research describe-schema

The remaining commands from the specification — ``download``, ``build-features``,
``train``, ``backtest``, ``report`` and ``run`` — arrive with the stages they
drive. A command that exists but does nothing is worse than one that is
absent, because it implies a capability the project does not have yet.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from trading_research import __version__
from trading_research.data import store, synthetic
from trading_research.data.schema import BARS_SCHEMA, BOOK_SCHEMA, TRADE_SCHEMA
from trading_research.data.validate import (
    Severity,
    ValidationReport,
    validate_book,
    validate_trades,
)

app = typer.Typer(
    name="trading-research",
    help="Leakage-aware research pipeline for systematic trading.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"trading-research {__version__}")
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
    """Leakage-aware research pipeline for systematic trading."""


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
            "--kind",
            "-k",
            help="Dataset: 'aggTrades' (trades), 'bookTicker' (best bid/ask) or 'klines' (bars).",
        ),
    ] = "aggTrades",
    interval: Annotated[
        str | None,
        typer.Option(
            "--interval",
            help="Bar length for --kind klines, e.g. '1d' or '1h'. Fetched a month at a time.",
        ),
    ] = None,
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

    from trading_research.data.binance import (
        ArchiveSpec,
        BinanceArchiveError,
        DayResult,
        download_range,
    )

    try:
        spec = ArchiveSpec(market=market, kind=kind, symbol=symbol.upper(), interval=interval)  # type: ignore[arg-type]
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
        f"[bold]{spec.symbol} {spec.directory}[/bold] ({spec.market}), "
        f"{first} → {last}, {n_days} day(s)"
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
        f"{downloaded / 1e9:.2f} GB downloaded → [bold]{Path(output) / spec.symbol / spec.directory}[/bold]"
    )
    if failures:
        console.print(
            f"[yellow]{len(failures)} day(s) unavailable — see the manifest for the list.[/yellow]"
        )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
#
# Each stage is its own command, reading and writing files. That is slower than
# one in-process run and it is the point: a stage re-runs without repeating the
# ones before it, and every intermediate can be looked at. A pipeline that only
# emits a final number is one whose middle nobody checks.


def _as_date(value: str, field: str) -> date:
    from datetime import date as _date

    try:
        return _date.fromisoformat(value)
    except ValueError as exc:
        err_console.print(f"[red]{field} must be YYYY-MM-DD, got {value!r}[/red]")
        raise typer.Exit(code=2) from exc


@app.command("prepare")
def prepare_cmd(
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Instrument.")] = "BTCUSDT",
    raw: Annotated[Path, typer.Option("--raw", help="Downloaded data directory.")] = Path(
        "data/raw"
    ),
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write features.")] = Path(
        "artifacts/prepared"
    ),
    horizon: Annotated[
        int, typer.Option("--horizon", min=1, help="Label horizon in observations.")
    ] = 1200,
    subsample: Annotated[int, typer.Option("--subsample", min=1, help="Keep every Nth row.")] = 50,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Rebuild days already present.")
    ] = False,
) -> None:
    """Build the feature matrix from raw data, one file per day.

    Processed a day at a time because the full series does not fit in memory,
    with rolling windows warmed from the previous day rather than restarted.
    """
    from trading_research.pipeline.stages import StageError, prepare

    try:
        out = prepare(
            raw, output, symbol, horizon=horizon, subsample=subsample, overwrite=overwrite
        )
    except StageError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]prepared[/green] -> {out}")


@app.command("select")
def select_cmd(
    train_start: Annotated[str, typer.Option("--train-start", help="YYYY-MM-DD.")],
    train_end: Annotated[str, typer.Option("--train-end", help="YYYY-MM-DD, inclusive.")],
    prepared: Annotated[Path, typer.Option("--prepared", help="Prepared features.")] = Path(
        "artifacts/prepared"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/features"),
    threshold_bp: Annotated[float, typer.Option("--threshold-bp", help="Label threshold.")] = 11.02,
    max_features: Annotated[int, typer.Option("--max-features", min=1)] = 40,
    score_target: Annotated[
        str, typer.Option("--score", help="Ranking criterion: 'ic', 'direction' or 'label'.")
    ] = "ic",
) -> None:
    """Choose features on a training window and write the list.

    Reads only the training window, so later data cannot influence which
    features are chosen — selecting over the whole sample is one of the most
    effective ways to manufacture an edge that is not there.
    """
    from trading_research.pipeline.stages import StageError, select

    try:
        out = select(
            prepared,
            output,
            train_start=_as_date(train_start, "--train-start"),
            train_end=_as_date(train_end, "--train-end"),
            threshold_bp=threshold_bp,
            max_features=max_features,
            score_target=score_target,
        )
    except StageError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    from trading_research.pipeline.stages import StageManifest

    report = StageManifest.read(out)["outputs"]
    console.print(f"[green]selected[/green] {report['n_selected']} of {report['n_input']} -> {out}")


@app.command("train")
def train_cmd(
    train_start: Annotated[str, typer.Option("--train-start", help="YYYY-MM-DD.")],
    train_end: Annotated[str, typer.Option("--train-end", help="YYYY-MM-DD, inclusive.")],
    prepared: Annotated[Path, typer.Option("--prepared")] = Path("artifacts/prepared"),
    features: Annotated[Path, typer.Option("--features")] = Path("artifacts/features"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/model"),
    model: Annotated[
        str, typer.Option("--model", "-m", help="always_hold, class_prior, logistic, xgboost, tcn.")
    ] = "logistic",
    threshold_bp: Annotated[float, typer.Option("--threshold-bp")] = 11.02,
    calibrate: Annotated[
        str | None,
        typer.Option(
            "--calibrate",
            help="Calibrate probabilities: 'isotonic' or 'sigmoid'. Both are monotonic, "
            "so a swept-threshold strategy trades identically either way; calibration "
            "matters when the number is used as a number.",
        ),
    ] = None,
) -> None:
    """Fit one model on the training window."""
    from trading_research.pipeline.stages import StageError, train

    try:
        out = train(
            prepared,
            features,
            output,
            train_start=_as_date(train_start, "--train-start"),
            train_end=_as_date(train_end, "--train-end"),
            threshold_bp=threshold_bp,
            model=model,
            calibrate=calibrate,
        )
    except (StageError, ImportError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]trained[/green] {model} -> {out}")


@app.command("predict")
def predict_cmd(
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD.")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD, inclusive.")],
    prepared: Annotated[Path, typer.Option("--prepared")] = Path("artifacts/prepared"),
    model: Annotated[Path, typer.Option("--model")] = Path("artifacts/model"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/predictions"),
) -> None:
    """Score a fitted model over a period. Fits nothing."""
    from trading_research.pipeline.stages import StageError, predict

    try:
        out = predict(
            prepared,
            model,
            output,
            start=_as_date(start, "--start"),
            end=_as_date(end, "--end"),
        )
    except StageError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]predicted[/green] -> {out}")


@app.command("backtest")
def backtest_cmd(
    predictions: Annotated[Path, typer.Option("--predictions")] = Path("artifacts/predictions"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/backtest"),
    min_confidence: Annotated[
        float, typer.Option("--min-confidence", help="Chosen on validation, not here.")
    ] = 0.5,
    hold_periods: Annotated[
        int, typer.Option("--hold", min=1, help="Rows to hold a position.")
    ] = 24,
    cooldown_periods: Annotated[int, typer.Option("--cooldown", min=0)] = 0,
    fee_bp_per_side: Annotated[float, typer.Option("--fee-bp", min=0.0)] = 5.0,
    slippage_bp: Annotated[float, typer.Option("--slippage-bp", min=0.0)] = 0.5,
    take_profit_bp: Annotated[
        float | None,
        typer.Option("--take-profit-bp", help="Close once this far in front. Caps winners."),
    ] = None,
    stop_loss_bp: Annotated[
        float | None,
        typer.Option("--stop-loss-bp", help="Close once this far behind. Realises dips."),
    ] = None,
) -> None:
    """Turn probabilities into trades and profit.

    Chooses nothing: the confidence threshold arrives from whoever selected it
    on validation. Keeping that decision outside this command is what stops it
    from quietly being tuned on whatever period the backtest is run over.
    """
    from trading_research.pipeline.stages import StageError, StageManifest, backtest

    try:
        out = backtest(
            predictions,
            output,
            min_confidence=min_confidence,
            hold_periods=hold_periods,
            cooldown_periods=cooldown_periods,
            fee_bp_per_side=fee_bp_per_side,
            slippage_bp=slippage_bp,
            take_profit_bp=take_profit_bp,
            stop_loss_bp=stop_loss_bp,
        )
    except StageError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    results = StageManifest.read(out)["outputs"]
    table = Table(title="Backtest", title_justify="left", show_edge=False)
    table.add_column("")
    table.add_column("thinned", justify="right")
    table.add_column("every signal", justify="right")
    for key in ("trades", "gross_per_trade_bp", "net_per_trade_bp", "net_bp", "hit_rate"):
        table.add_row(
            key,
            f"{results['thinned'][key]:,.2f}",
            f"{results['every_signal'][key]:,.2f}",
        )
    console.print(table)
    reasons = results.get("exit_reasons") or {}
    if reasons:
        console.print("exits: " + ", ".join(f"{k} {v}" for k, v in reasons.items()))
    console.print(f"\n-> {out}")


@app.command("retrain-search")
def retrain_search_cmd(
    prepared: Annotated[Path, typer.Option("--prepared")] = Path("artifacts/prepared"),
    features: Annotated[Path, typer.Option("--features")] = Path("artifacts/features"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/retrain"),
    model: Annotated[str, typer.Option("--model", help="One model per run.")] = "logistic",
    threshold_bp: Annotated[float, typer.Option("--threshold-bp", min=0.0)] = 11.0,
    hold_periods: Annotated[int, typer.Option("--hold", min=1)] = 24,
    cooldown_periods: Annotated[int, typer.Option("--cooldown", min=0)] = 0,
    fee_bp_per_side: Annotated[float, typer.Option("--fee-bp", min=0.0)] = 5.0,
    slippage_bp: Annotated[float, typer.Option("--slippage-bp", min=0.0)] = 0.5,
    train_days: Annotated[
        str, typer.Option("--train-days", help="Comma-separated window lengths to try.")
    ] = "3,7,14,21",
    apply_days: Annotated[str, typer.Option("--apply-days")] = "1,3,7",
    minimum_windows: Annotated[int, typer.Option("--min-windows", min=1)] = 3,
    validation_fraction: Annotated[float, typer.Option("--validation-fraction")] = 0.5,
    objective: Annotated[str, typer.Option("--objective")] = "net_per_trade_bp",
    start: Annotated[
        str | None,
        typer.Option(
            "--start", help="First day of the span. Days used to select features go before."
        ),
    ] = None,
    end: Annotated[
        str | None, typer.Option("--end", help="Last day of the span, inclusive.")
    ] = None,
    calibrate: Annotated[str | None, typer.Option("--calibrate")] = None,
    max_train_rows: Annotated[
        int | None, typer.Option("--max-train-rows", help="Keep only the tail. For the network.")
    ] = None,
) -> None:
    """Search how often to retrain: window length, apply length, step.

    The schedule is chosen on the first part of the span and applied once to the
    rest. One model per run, because fitting a network in the same process as a
    boosted tree deadlocks on macOS.
    """
    import json

    import pandas as pd

    from trading_research.backtest.costs import TakerCosts
    from trading_research.pipeline.retraining import (
        DayCache,
        positive_window_share,
        prepared_evaluator,
        summarise,
        trade_weighted,
    )
    from trading_research.pipeline.stages import StageError, StageManifest, load_selected
    from trading_research.validation.retrain import ScheduleError, expand_grid, search, split_span

    def numbers(text: str, flag: str) -> list[int]:
        try:
            return [int(part) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise typer.BadParameter(f"{flag} must be comma-separated integers") from exc

    try:
        columns = load_selected(features)
        cache = DayCache(prepared)
        available = cache.available()
        # The days that chose the features must not be scored, but they are in
        # the past and a schedule may train on them — which is what lets a
        # 21-day window trade from the first day of the span rather than the
        # twenty-second.
        span = available
        if start is not None:
            span = [d for d in span if d >= _as_date(start, "--start")]
        if end is not None:
            span = [d for d in span if d <= _as_date(end, "--end")]
        history = [d for d in available if d < min(span)] if span else []
        validation_days, test_days = split_span(span, validation_fraction=validation_fraction)
        costs = TakerCosts(fee_bp_per_side=fee_bp_per_side, slippage_bp=slippage_bp)
        evaluate = prepared_evaluator(
            prepared,
            columns,
            threshold_bp=threshold_bp,
            hold_periods=hold_periods,
            cooldown_periods=cooldown_periods,
            costs=costs,
            model=model,
            calibrate=calibrate,
            max_train_rows=max_train_rows,
        )
        grid = expand_grid(numbers(train_days, "--train-days"), numbers(apply_days, "--apply-days"))
        console.print(
            f"{model}: {len(grid)} schedules, "
            f"{len(validation_days)} validation days, {len(test_days)} test days"
        )
        outcome = search(
            validation_days,
            evaluate,
            grid=grid,
            objective=objective,
            minimum_windows=minimum_windows,
            test_days=test_days,
            history=history,
            aggregate=trade_weighted,
            on_result=lambda r: console.print(
                f"  {r.schedule.label}: {r.metrics['windows']:.0f} windows, "
                f"{objective} {r.metrics.get(objective, float('nan')):.2f}"
            ),
        )
    except (StageError, ScheduleError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Retraining schedules on validation ({model})", show_edge=False)
    ordered = summarise(outcome.validation, objective)
    for column in ordered.columns:
        table.add_column(column, justify="left" if column == "schedule" else "right")
    for _, row in ordered.iterrows():
        table.add_row(*[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)
    console.print(f"\n[green]{outcome.summary()}[/green]")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(out / "validation.csv", index=False)
    test_outputs: dict[str, object] = {}
    if outcome.test is not None:
        pd.DataFrame(outcome.test.per_window).to_csv(out / "test_windows.csv", index=False)
        test_outputs = {
            "metrics": outcome.test.metrics,
            "positive_window_share": positive_window_share(outcome.test),
        }
        console.print(
            f"test: {outcome.test.metrics['windows']:.0f} windows, "
            f"{positive_window_share(outcome.test):.0%} of them positive"
        )
    (out / "chosen.json").write_text(
        json.dumps(
            {
                "chosen": outcome.chosen.label,
                "objective": objective,
                "train_days": outcome.chosen.train_days,
                "apply_days": outcome.chosen.apply_days,
                "step_days": outcome.chosen.step,
                "test": test_outputs,
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    StageManifest(
        stage="retrain-search",
        inputs={"prepared_dir": str(prepared), "features_dir": str(features)},
        params={
            "model": model,
            "threshold_bp": threshold_bp,
            "hold_periods": hold_periods,
            "cooldown_periods": cooldown_periods,
            "objective": objective,
            "grid": [s.label for s in grid],
        },
        outputs={
            "chosen": outcome.chosen.label,
            **({"test": test_outputs} if test_outputs else {}),
        },
    ).write(out)
    console.print(f"\n-> {out}")


@app.command("breaks")
def breaks_cmd(
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Instrument.")] = "BTCUSDT",
    interval: Annotated[str, typer.Option("--interval", help="Bar length, e.g. '1d'.")] = "1d",
    market: Annotated[str, typer.Option("--market", "-m")] = "futures-um",
    start: Annotated[
        str | None, typer.Option("--start", help="First bar, YYYY-MM-DD. Not needed with --prices.")
    ] = None,
    end: Annotated[
        str | None, typer.Option("--end", help="Last bar, inclusive. Not needed with --prices.")
    ] = None,
    prices: Annotated[
        Path | None,
        typer.Option(
            "--prices",
            help="A CSV with 'timestamp' and 'close' columns, instead of fetching bars. "
            "Needs no network.",
        ),
    ] = None,
    history: Annotated[
        int, typer.Option("--history", min=50, help="Returns the monitor is fitted on.")
    ] = 365,
    online: Annotated[
        int, typer.Option("--online", min=1, help="Returns streamed before a refit.")
    ] = 90,
    statistics: Annotated[
        str, typer.Option("--statistics", help="Comma-separated: scale, dependence, mean.")
    ] = "scale,dependence",
    threshold: Annotated[
        str | None,
        typer.Option(
            "--threshold",
            help="Log-odds threshold: one number, or per family as 'scale=9,dependence=8'. "
            "Default: the recorded null calibration.",
        ),
    ] = None,
    calibrate_now: Annotated[
        bool,
        typer.Option(
            "--calibrate", help="Calibrate the thresholds on Gaussian nulls for this setting."
        ),
    ] = False,
    plot: Annotated[
        bool, typer.Option("--plot", help="Render the figure for both themes (needs matplotlib).")
    ] = False,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/breaks"),
) -> None:
    """Where a return series stopped behaving as before, and what changed.

    The second regime detector: the series is whitened by its own history
    (an AR(p) fit, a conditional scale, the innovation distribution) and the
    Shiryaev-Roberts odds of a change are read per family on the whitened
    stream — scale, dependence, mean. A break is the first step a family's
    odds clear a threshold calibrated against a null, and the history is
    re-anchored there.

    Bars come from Binance's public klines archives, fetched once and kept
    under data/, or from any CSV of closes with --prices.
    """
    import json

    from trading_research.data.ensure import DataUnavailable, ensure_bars, load_bars
    from trading_research.pipeline.stages import StageManifest
    from trading_research.validation.changepoint import ChangepointError
    from trading_research.validation.structural_breaks import (
        MonitorSpec,
        calibrate,
        detect_breaks,
        log_returns,
        read_prices,
    )

    families = tuple(part.strip() for part in statistics.split(",") if part.strip())

    def parse_threshold(text: str) -> float | dict[str, float]:
        if "=" not in text:
            try:
                return float(text)
            except ValueError as exc:
                raise typer.BadParameter("--threshold must be a number or name=number,…") from exc
        out: dict[str, float] = {}
        for part in text.split(","):
            name, _, value = part.partition("=")
            try:
                out[name.strip()] = float(value)
            except ValueError as exc:
                raise typer.BadParameter(f"--threshold: {part!r} is not name=number") from exc
        return out

    try:
        if prices is not None:
            close = read_prices(prices)
            source = str(prices)
        else:
            if start is None or end is None:
                err_console.print(
                    "[red]--start and --end are needed unless --prices is given[/red]"
                )
                raise typer.Exit(code=2)
            first, last = _as_date(start, "--start"), _as_date(end, "--end")
            ensure_bars(
                symbol,
                interval,
                first,
                last,
                market=market,  # type: ignore[arg-type]
                on_progress=lambda line: console.print(f"  [dim]{line}[/dim]"),
            )
            bars = load_bars(symbol, interval, start=first, end=last)
            close = bars.set_index("timestamp")["close"]
            source = f"binance-{market}-klines-{interval}"
        returns = log_returns(close)

        level: float | dict[str, float] | None = None
        if calibrate_now:
            console.print(f"calibrating on Gaussian nulls for history {history}, online {online}")
            level = calibrate(history, online, statistics=families)
        elif threshold is not None:
            level = parse_threshold(threshold)
        spec = MonitorSpec(
            history_len=history, online_len=online, statistics=families, threshold=level
        )
        console.print(
            f"{symbol} {interval}: {len(returns)} returns, {returns.index[0]:%Y-%m-%d} → "
            f"{returns.index[-1]:%Y-%m-%d}; history {history}, online {online}, "
            f"families {', '.join(families)}"
        )
        result = detect_breaks(
            returns, spec, on_progress=lambda line: console.print(f"  [dim]{line}[/dim]")
        )
    except (ChangepointError, DataUnavailable, ImportError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = result.table()
    rendered = Table(title=f"Breaks flagged on {symbol} {interval}", show_edge=False)
    for column in ("date", "statistic", "direction", "log_odds", "threshold", "excursion"):
        rendered.add_column(column, justify="left" if column in ("date", "statistic") else "right")
    rendered.add_column("history", justify="right")
    rendered.add_column("step", justify="right")
    for _, row in table.iterrows():
        rendered.add_row(
            f"{row['timestamp']:%Y-%m-%d}",
            str(row["statistic"]),
            "up" if row["direction"] > 0 else "down",
            f"{row['log_odds']:.2f}",
            f"{row['threshold']:.2f}",
            f"{row['excursion']:.2f}",
            str(row["history_len"]),
            str(row["step_in_window"]),
        )
    console.print(rendered)
    if table.empty:
        console.print("[green]no break flagged[/green]")
    console.print(
        f"\n{len(result.windows)} window(s) walked; thresholds "
        + ", ".join(f"{k} {v:.2f}" for k, v in result.thresholds.items())
    )

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    annotated = result.annotate(returns)
    annotated.to_csv(out / "steps.csv", index=False)
    table.to_csv(out / "breaks.csv", index=False)
    (out / "thresholds.json").write_text(
        json.dumps(result.thresholds, indent=2) + "\n", encoding="utf-8"
    )
    StageManifest(
        stage="breaks",
        inputs={"source": source, "symbol": symbol, "interval": interval, "returns": len(returns)},
        params={
            "history_len": history,
            "online_len": online,
            "statistics": list(families),
            "thresholds": result.thresholds,
        },
        outputs={
            "breaks": len(result.breaks),
            "windows": len(result.windows),
            "flagged": [f"{t:%Y-%m-%d}" for t in table.get("timestamp", [])],
        },
    ).write(out)
    if plot:
        try:
            from trading_research.reporting import plots

            written = plots.both_themes(
                lambda path: plots.structural_breaks(
                    annotated,
                    path,
                    thresholds=result.thresholds,
                    title=f"{symbol} {interval}: breaks on the whitened stream",
                    history_len=history,
                ),
                out / "breaks.png",
            )
        except ImportError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        console.print(f"figure -> {', '.join(str(p) for p in written)}")
    console.print(f"\n-> {out}")


@app.command("exit-search")
def exit_search_cmd(
    prepared: Annotated[Path, typer.Option("--prepared")] = Path("artifacts/prepared"),
    features: Annotated[Path, typer.Option("--features")] = Path("artifacts/features"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/exits"),
    model: Annotated[str, typer.Option("--model", help="One model per run.")] = "logistic",
    threshold_bp: Annotated[float, typer.Option("--threshold-bp", min=0.0)] = 11.0,
    hold_periods: Annotated[int, typer.Option("--hold", min=1)] = 24,
    cooldown_periods: Annotated[int, typer.Option("--cooldown", min=0)] = 0,
    fee_bp_per_side: Annotated[float, typer.Option("--fee-bp", min=0.0)] = 5.0,
    slippage_bp: Annotated[float, typer.Option("--slippage-bp", min=0.0)] = 0.5,
    train_days: Annotated[int, typer.Option("--train-days", min=1)] = 14,
    validation_days: Annotated[int, typer.Option("--validation-days", min=1)] = 7,
    test_days: Annotated[int, typer.Option("--test-days", min=1)] = 7,
    folds: Annotated[int, typer.Option("--folds", min=1)] = 7,
    minimum_trades: Annotated[int, typer.Option("--min-trades", min=1)] = 50,
    objective: Annotated[str, typer.Option("--objective")] = "net_per_trade_bp",
    calibrate: Annotated[str | None, typer.Option("--calibrate")] = None,
) -> None:
    """Search how to leave a trade: clock, take-profit, stop, trail, signal.

    The model is fitted once per fold and every policy is scored against the
    same predictions, so the table compares exits rather than fits. The winner
    is chosen on validation and applied once to test.
    """
    import json

    import pandas as pd

    from trading_research.backtest.costs import TakerCosts
    from trading_research.pipeline.exits import policy_evaluator, walk_forward_predictions
    from trading_research.pipeline.stages import (
        StageError,
        StageManifest,
        load_prepared,
        load_selected,
    )
    from trading_research.validation.exits import ExitSearchError, default_grid, search
    from trading_research.validation.splits import WalkForwardSpec, walk_forward

    try:
        columns = load_selected(features)
        frame = load_prepared(prepared)
        costs = TakerCosts(fee_bp_per_side=fee_bp_per_side, slippage_bp=slippage_bp)

        spec = WalkForwardSpec(
            train_days=train_days,
            validation_days=validation_days,
            test_days=test_days,
            step_days=1,
            n_folds=folds,
        )
        days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
        frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)
        schedule = walk_forward(days, spec)

        console.print(f"{model}: fitting {len(schedule)} folds over {len(days)} days")
        blocks, min_confidence = walk_forward_predictions(
            frame,
            columns,
            schedule,
            threshold_bp=threshold_bp,
            costs=costs,
            model=model,
            calibrate=calibrate,
            purge=hold_periods,
            on_fold=lambda fold: console.print(f"  fold {fold.index} fitted"),
        )
        console.print(f"entry threshold from validation: {min_confidence:.3f}")

        grid = default_grid(hold_periods, cooldown_periods=cooldown_periods)
        outcome = search(
            policy_evaluator(blocks["validation"], costs, min_confidence=min_confidence),
            grid=grid,
            objective=objective,
            minimum_trades=minimum_trades,
            baseline=grid[0],
            on_test=policy_evaluator(blocks["test"], costs, min_confidence=min_confidence),
            on_result=lambda policy, metrics: console.print(
                f"  {policy.label}: {metrics.get('trades', 0):.0f} trades, "
                f"{objective} {metrics.get(objective, float('nan')):.2f}"
            ),
        )
    except (StageError, ExitSearchError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    columns_wanted = [
        c
        for c in (
            "policy",
            "family",
            "trades",
            "hit_rate",
            "gross_per_trade_bp",
            objective,
            "net_bp",
            "skipped",
        )
        if c in outcome.validation.columns
    ]
    ordered = outcome.validation[columns_wanted].sort_values(
        objective, ascending=False, na_position="last"
    )

    table = Table(title=f"Exit policies on validation ({model})", show_edge=False)
    for column in ordered.columns:
        table.add_column(column, justify="left" if column in ("policy", "family") else "right")
    for _, row in ordered.iterrows():
        table.add_row(*[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row])
    console.print(table)
    console.print(f"\n[green]{outcome.summary()}[/green]")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(out / "validation.csv", index=False)
    if outcome.test is not None:
        pd.Series(outcome.test).to_frame("value").to_csv(out / "test.csv")
    (out / "chosen.json").write_text(
        json.dumps(
            {
                "chosen": outcome.chosen.label,
                "family": outcome.chosen.family,
                "objective": objective,
                "min_confidence": min_confidence,
                "policy": outcome.chosen.__dict__,
                "test": outcome.test,
                "test_baseline": outcome.test_baseline,
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    StageManifest(
        stage="exit-search",
        inputs={"prepared_dir": str(prepared), "features_dir": str(features)},
        params={
            "model": model,
            "threshold_bp": threshold_bp,
            "hold_periods": hold_periods,
            "objective": objective,
            "grid": [p.label for p in grid],
        },
        outputs={"chosen": outcome.chosen.label, "test": outcome.test},
    ).write(out)
    console.print(f"\n-> {out}")


@app.command("screen")
def screen_cmd(
    raw: Annotated[Path, typer.Option("--raw", help="Downloaded data directory.")] = Path(
        "data/raw"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/screen"),
    horizon_s: Annotated[float, typer.Option("--horizon-s", min=1.0)] = 120.0,
    max_days: Annotated[int | None, typer.Option("--max-days", min=1)] = None,
    fee_bp_per_side: Annotated[float, typer.Option("--fee-bp", min=0.0)] = 5.0,
    slippage_bp: Annotated[float, typer.Option("--slippage-bp", min=0.0)] = 0.5,
) -> None:
    """Rank instruments by how much room there is to trade them.

    Step 0, and the only step that runs before anything is fitted. Measures the
    share of moments whose move over the horizon clears the cost of trading it
    — an upper bound that assumes perfect prediction, so no model can beat it
    and none is needed to compute it.

    A screen eliminates rather than selects: an instrument with 10% headroom is
    not necessarily tradeable, and one with 0.1% definitely is not.
    """
    from trading_research.backtest.costs import TakerCosts
    from trading_research.data.screen import screen_directory

    costs = TakerCosts(fee_bp_per_side=fee_bp_per_side, slippage_bp=slippage_bp)
    table = screen_directory(raw, horizon_s=horizon_s, max_days=max_days, costs=costs)
    if table.empty:
        err_console.print(f"[red]no instruments found under {raw}[/red]")
        raise typer.Exit(code=1)

    rendered = Table(title=f"Headroom at {horizon_s:g}s", show_edge=False)
    for column in table.columns:
        rendered.add_column(column, justify="left" if column == "symbol" else "right")
    for _, row in table.iterrows():
        rendered.add_row(
            *["—" if pd.isna(v) else (f"{v:,.4g}" if isinstance(v, float) else str(v)) for v in row]
        )
    console.print(rendered)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "screen.csv", index=False)
    console.print(f"\n-> {out / 'screen.csv'}")


@app.command("review")
def review_cmd(
    trades: Annotated[Path, typer.Option("--trades", help="CSV of trades with a net_bp column.")],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/review"),
    name: Annotated[str, typer.Option("--name")] = "strategy",
    days: Annotated[float, typer.Option("--days", min=0.1)] = 1.0,
    cost_bp: Annotated[float, typer.Option("--cost-bp", min=0.0)] = 11.02,
    configurations: Annotated[
        int | None,
        typer.Option(
            "--configurations",
            help="How many were tried in total, not how many are shown. Omitted "
            "means the selection dimension cannot be scored above 1.",
        ),
    ] = None,
) -> None:
    """Assemble the evidence a strategy review needs.

    Computes the arithmetic half of ``docs/evaluation/rubric.md`` — the result,
    the dispersion, the drawdown, the sample size needed to establish the edge
    — and writes it as a document a reviewer or an agent can read.

    It scores nothing. A tool that produced the evidence and graded it would be
    marking its own work, which is the failure the rubric exists to catch.
    """
    from trading_research.evaluation import build_scorecard

    if not trades.exists():
        err_console.print(f"[red]no such file: {trades}[/red]")
        raise typer.Exit(code=1)

    frame = pd.read_csv(trades)
    if "net_bp" not in frame.columns:
        err_console.print(f"[red]{trades} has no net_bp column[/red]")
        raise typer.Exit(code=1)

    per_period = pd.DataFrame()
    if "fold" in frame.columns:
        per_period = frame.groupby("fold", as_index=False).agg(net_bp=("net_bp", "sum"))
    card = build_scorecard(
        name,
        frame["net_bp"].to_numpy(),
        days=days,
        gross_per_trade_bp=frame["gross_bp"].to_numpy() if "gross_bp" in frame else None,
        per_period=per_period,
        context={"cost_bp_per_trade": cost_bp, "source": str(trades)},
        configurations_tried=configurations,
    )
    written = card.write(Path(output) / f"{name}.md")
    console.print(card.to_markdown())
    console.print(f"-> {written}")
    console.print(
        "\nReview it against [bold]docs/evaluation/rubric.md[/bold] using "
        "[bold]docs/evaluation/prompt.md[/bold]."
    )


@app.command("discover")
def discover_cmd(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/screen"),
    listed_before: Annotated[
        str | None,
        typer.Option("--listed-before", help="Drop contracts listed on or after this date."),
    ] = None,
    top: Annotated[int, typer.Option("--top", min=1)] = 40,
    minimum_volume_usd: Annotated[float, typer.Option("--min-volume", min=0.0)] = 3e6,
) -> None:
    """List the contracts worth screening, ranked by traded volume.

    The step before the screen, and until now the one done by hand. A candidate
    nobody thought of is one the screen never sees, so this enumerates what the
    venue lists rather than relying on anyone's recall.

    ``--listed-before`` matters more than it looks. Volume comes from today and
    the data being screened does not: without the filter the list fills with
    recent listings whose archives return nothing, displacing contracts that
    have history.

    Volume ranks what gets *measured*. The screen decides what is *good*, and
    the two are different questions — the highest-volume contract came second
    from bottom on headroom.
    """
    from trading_research.data.discover import candidate_table

    cutoff = _as_date(listed_before, "--listed-before") if listed_before else None
    try:
        table = candidate_table(
            listed_before=cutoff, top=top, minimum_volume_usd=minimum_volume_usd
        )
    except OSError as exc:
        err_console.print(f"[red]could not reach the exchange listing: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if table.empty:
        err_console.print("[red]no contracts matched[/red]")
        raise typer.Exit(code=1)

    rendered = Table(title=f"{len(table)} candidates by volume", show_edge=False)
    for column in table.columns:
        rendered.add_column(column, justify="left" if column == "symbol" else "right")
    for _, row in table.head(20).iterrows():
        rendered.add_row(*[f"{v:,.4g}" if isinstance(v, float) else str(v) for v in row])
    console.print(rendered)
    if len(table) > 20:
        console.print(f"[dim]… and {len(table) - 20} more[/dim]")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "candidates.csv", index=False)
    console.print(f"\n-> {out / 'candidates.csv'}")
    console.print(
        "\nDownload a few days of each, then rank them with [bold]trading-research screen[/bold]."
    )


@app.command("download-book")
def download_book_cmd(
    start: Annotated[str, typer.Option("--start", help="YYYY-MM-DD.")],
    end: Annotated[str, typer.Option("--end", help="YYYY-MM-DD, inclusive.")],
    symbol: Annotated[str, typer.Option("--symbol", "-s")] = "BTCUSDT",
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("data/book"),
    depth: Annotated[int, typer.Option("--depth", min=1, max=500)] = 10,
    grid_ms: Annotated[int, typer.Option("--grid-ms", min=1)] = 100,
    keep_archive: Annotated[bool, typer.Option("--keep-archive/--no-keep-archive")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Days replayed in parallel.")
    ] = 1,
) -> None:
    """Fetch multi-level order books from Bybit's free archives.

    The one thing Binance does not publish. Bybit gives five hundred levels per
    side at tick granularity, back to 2023, with no registration and no fee —
    which is what the depth-dependent features in the registry have been
    waiting for.

    The archive is a stream: one snapshot a day and then deltas, where a size of
    zero removes a level. This replays it and photographs the book on a grid.
    Sequence gaps are counted and reported rather than smoothed over, because a
    missed update leaves the book wrong until the next snapshot and nothing
    downstream can tell.

    Expect about 200 MB of download and two minutes of replay per instrument-day,
    producing roughly 34 MB at ten levels.
    """
    from trading_research.data.bybit import download_range

    result = download_range(
        symbol,
        _as_date(start, "--start"),
        _as_date(end, "--end"),
        Path(output) / symbol,
        depth=depth,
        grid_ms=grid_ms,
        keep_archive=keep_archive,
        overwrite=overwrite,
        workers=workers,
        on_day=lambda row: console.print(
            f"  {row['day']}: "
            + (
                f"[yellow]{row['skipped']}[/yellow]"
                if row.get("skipped")
                else f"{row['rows']:,} rows, {row.get('gaps', 0)} sequence gap(s)"
            )
        ),
    )

    done = result[result.get("rows", 0) > 0] if "rows" in result else result
    console.print(
        f"\n[green]{len(done)} day(s)[/green], {int(done['rows'].sum()):,} rows "
        f"-> {Path(output) / symbol}"
    )
    if "gaps" in done and done["gaps"].sum():
        console.print(
            f"[yellow]{int(done['gaps'].sum())} sequence gap(s) across the range; "
            f"the book is stale from each until the next snapshot.[/yellow]"
        )


@app.command("describe-schema")
def describe_schema(
    plane: Annotated[
        str, typer.Argument(help="Which contract to print: 'trades', 'book' or 'bars'.")
    ] = "trades",
) -> None:
    """Print a data contract, so the expected columns and units are never guesswork."""
    schema = {"trades": TRADE_SCHEMA, "book": BOOK_SCHEMA, "bars": BARS_SCHEMA}.get(plane)
    if schema is None:
        err_console.print(
            f"[red]Unknown plane[/red] {plane!r}; expected 'trades', 'book' or 'bars'"
        )
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


@app.command("reproduce")
def reproduce_cmd(
    book: Annotated[Path, typer.Option("--book", help="Where the panel lives.")] = Path(
        "data/universe"
    ),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/discovery"),
    fetch: Annotated[
        bool, typer.Option("--fetch/--no-fetch", help="Download what is missing.")
    ] = True,
) -> None:
    """Run the whole pipeline on a clean checkout and report what it finds.

    Every other command here does one step. This runs all of them in order --
    fetch, screen the sources, search a configuration on the search block, read
    the held-out block once, and measure the market state the answer depends on
    -- so the result in the documents can be reproduced rather than taken on
    trust.

    It takes a while the first time, because the data is not in the repository
    and never will be: a day of one instrument's book is hundreds of megabytes.
    Afterwards it is instant, since only missing days are fetched.

    The held-out block is read once, with everything already chosen. If the
    numbers disappoint, that is the answer -- the run does not go back and try
    again, because a pipeline that reruns until it likes the result is a search
    with extra steps.
    """
    from trading_research.pipeline.discovery import run

    console.print(f"[bold]Running the pipeline end to end[/bold] (data under {book})\n")
    result = run(book_root=book, fetch=fetch, report=lambda line: console.print(f"  {line}"))

    if not result.held_out:
        console.print("\n[yellow]The run stopped before a result.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="What the pipeline found")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    held = result.held_out
    for label, value in (
        ("instruments positive", f"{held['positive_instruments']}/{held['instruments']}"),
        ("median instrument, net", f"{held['median_instrument_bp']:+.2f} bp"),
        ("gross per trade", f"{held['gross_per_trade_bp']:+.2f} bp"),
        ("net per trade", f"{held['net_per_trade_bp']:+.2f} bp"),
        ("trades", f"{held['trades']:,.0f}"),
        ("t, counting trades", f"{held['trade_t']:+.2f}"),
        ("t, counting days", f"{held['cluster_t']:+.2f}"),
        ("days", f"{held['clusters']:.0f}"),
    ):
        table.add_row(label, value)
    console.print()
    console.print(table)

    console.print(
        f"\n  [dim]{held['verdict']}[/dim]"
        f"\n  [dim]index autocorrelation: {result.regime['search']:+.4f} on the search "
        f"block, {result.regime['held_out']:+.4f} held out — the condition the result "
        f"carries[/dim]"
    )

    output.mkdir(parents=True, exist_ok=True)
    result.summary().to_csv(output / "stages.csv", index=False)
    pd.DataFrame([{**held, **{f"regime_{k}": v for k, v in result.regime.items()}}]).to_csv(
        output / "result.csv", index=False
    )
    for stage in result.stages:
        if stage.table is not None:
            stage.table.to_csv(output / f"{stage.name.replace(' ', '_')}.csv", index=False)
    console.print(f"\n[dim]-> {output}[/dim]")


@app.command("mm-backtest")
def mm_backtest_cmd(
    start: Annotated[str, typer.Option("--start", help="First day, YYYY-MM-DD.")],
    end: Annotated[str, typer.Option("--end", help="Last day, inclusive, YYYY-MM-DD.")],
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Instrument.")] = "BICOUSDT",
    strategy: Annotated[str, typer.Option("--strategy", help="s0, s1, s2 or s3.")] = "s1",
    frozen: Annotated[
        bool, typer.Option("--frozen", help="Read every parameter from the frozen registration.")
    ] = False,
    params: Annotated[
        Path, typer.Option("--params", help="The registration --frozen reads.")
    ] = Path("configs/mm_prereg.yaml"),
    clip_notional: Annotated[float | None, typer.Option("--clip-notional")] = None,
    sigma_ref: Annotated[
        float | None, typer.Option("--sigma-ref", help="S1-S3, bp a minute.")
    ] = None,
    skew_bp: Annotated[float, typer.Option("--skew-bp")] = 2.0,
    k: Annotated[float, typer.Option("--k")] = 0.5,
    min_edge_bp: Annotated[float, typer.Option("--min-edge-bp")] = 4.0,
    m_ticks: Annotated[int, typer.Option("--m-ticks", help="S2.")] = 3,
    soft_limit_clips: Annotated[float, typer.Option("--soft-limit-clips")] = 6.0,
    guards: Annotated[str, typer.Option("--guards", help="S3: S1 or S2.")] = "S1",
    action: Annotated[str, typer.Option("--action", help="S3: pull or widen.")] = "widen",
    guard_window_min: Annotated[float, typer.Option("--guard-window-min")] = 15.0,
    regime_policy: Annotated[
        str, typer.Option("--regime-policy", help="none, guard_pull or guard_widen (S0-S2).")
    ] = "none",
    fee_tier: Annotated[str, typer.Option("--fee-tier", help="A transcribed Bybit tier.")] = "base",
    latency_ms: Annotated[float, typer.Option("--latency-ms", min=0.0)] = 10.0,
    clip_touch_share: Annotated[
        float, typer.Option("--clip-touch-share", help="Clip cap as a share of the touch.")
    ] = 0.10,
    cancel_model: Annotated[
        str, typer.Option("--cancel-model", help="pessimistic, proportional or optimistic.")
    ] = "proportional",
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    book: Annotated[list[Path] | None, typer.Option("--book", help="Book roots, in order.")] = None,
    trades: Annotated[Path, typer.Option("--trades")] = Path("data/trades"),
    funding: Annotated[Path | None, typer.Option("--funding")] = Path("data/funding"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/mm"),
    allow_heldout: Annotated[
        bool, typer.Option("--allow-heldout", help="Open held-out days through the ledger.")
    ] = False,
    second_read: Annotated[
        bool, typer.Option("--second-read", help="With --allow-heldout: a recorded second read.")
    ] = False,
) -> None:
    """Quote both sides of one instrument in event time, day by day.

    The market maker of the pre-registered study, S0 to S3, on the event-time
    simulator: fills only from prints, a queue per order, latency, inventory
    limits, fees and funding. Prints the daily table and the decomposition of
    the net, and writes both with a manifest.

    Days of the held-out and boundary blocks are refused unless
    ``--allow-heldout`` is given, and then they are read only with the frozen
    values and only through the ledger, which checks the committed frozen
    configuration and records the read. Days outside every registered block —
    synthetic ones included — are read freely.
    """
    from dataclasses import asdict, replace
    from datetime import timedelta

    from trading_research.backtest.costs import fee_tier as tier_named
    from trading_research.market_making import prereg
    from trading_research.market_making.queue import CancelAttribution
    from trading_research.market_making.simulator import SimConfig
    from trading_research.pipeline import market_making as mm
    from trading_research.pipeline.execution import execute

    first, last = _as_date(start, "start"), _as_date(end, "end")
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    name = strategy.upper()
    try:
        if not days:
            raise ValueError("--end is before --start")
        held = {prereg.block_of(d) for d in days} - {None, "D"}
        if held and allow_heldout and not frozen:
            raise prereg.HeldOutLocked("a held-out block is read with the frozen values: --frozen")
        changes = {
            "fees": tier_named(fee_tier),
            "order_latency_ns": int(latency_ms * 1e6),
            "cancel_latency_ns": int(latency_ms * 1e6),
            "cancel_attribution": CancelAttribution(cancel_model),
            "clip_touch_share": clip_touch_share,
        }
        if frozen:
            plan = mm.frozen_plan(name, symbol, path=params, **changes)
        else:
            if clip_notional is None or (name != "S0" and sigma_ref is None):
                raise ValueError(
                    "without --frozen, give --clip-notional (and --sigma-ref for S1-S3)"
                )
            values: dict[str, object] = {
                "skew_bp": skew_bp,
                "k": k,
                "min_edge_bp": min_edge_bp,
                "m_ticks": m_ticks,
            }
            values |= {
                "guards": guards.upper(),
                "action": action,
                "guard_window_min": guard_window_min,
            }
            config = SimConfig(clip_notional=clip_notional, soft_limit_clips=soft_limit_clips)
            chosen = {p: values[p] for p in mm.PARAMETERS.get(name, ())}
            plan = mm.MakerPlan(name, config.with_(**changes), chosen, sigma_ref or 1.0)
        if regime_policy != "none":
            plan = replace(plan, regime_policy=regime_policy, guard_window_min=guard_window_min)
        access = mm.access_for(days, allow_heldout=allow_heldout, second_read=second_read)
    except (prereg.HeldOutLocked, prereg.PreRegistrationError, ValueError, KeyError) as exc:
        err_console.print(f"[red]refused:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    roots = tuple(book) if book else (Path("data/book"), Path("data/book_fresh"))
    sources = mm.MakerSources(roots, trades, funding)
    console.print(
        f"[bold]{plan.label}[/bold] on {symbol}, {first} to {last} ({access.stamp}), "
        f"{plan.config.fees.name} fees, {latency_ms:g} ms, {cancel_model} cancels"
    )
    result = execute(
        None, "market_maker", symbol=symbol, days=days, plan=plan, sources=sources,
        access=access, workers=workers,
    )  # fmt: skip
    assert result.days is not None
    _print_mm_days(result.days)
    _print_mm_decomposition(result.summary)

    out = output / f"{symbol}_{plan.label}_{first}_{last}"
    out.mkdir(parents=True, exist_ok=True)
    result.days.to_csv(out / "days.csv", index=False)
    result.attempts.to_csv(out / "attempts.csv", index=False)
    manifest = {
        "symbol": symbol,
        "days": [d.isoformat() for d in days],
        "plan": plan.label,
        "params": dict(plan.params),
        "sigma_ref": plan.sigma_ref,
        "config": asdict(plan.config),
        "access": access.stamp,
        "frozen_from": str(params) if frozen else None,
        "summary": result.summary,
        "version": __version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    console.print(f"\n[dim]-> {out}[/dim]")


#: The daily table's columns: the decomposition, then what the day did.
_MM_DAY_COLUMNS = (
    "day",
    "status",
    "net",
    "spread",
    "adverse",
    "inventory",
    "fees",
    "funding",
    "fills",
    "flattens",
    "max_abs_position",
    "flags",
)


def _print_mm_days(days: pd.DataFrame) -> None:
    table = Table(title="Day by day (quote currency)", show_edge=False)
    for column in _MM_DAY_COLUMNS:
        left = column in ("day", "status", "flags")
        table.add_column(column, justify="left" if left else "right", no_wrap=column == "day")
    for _, row in days.iterrows():
        cells = []
        for column in _MM_DAY_COLUMNS:
            value = row.get(column)
            if isinstance(value, float):
                cells.append("—" if pd.isna(value) else f"{value:,.4g}")
            else:
                cells.append("" if value is None else str(value))
        table.add_row(*cells)
    console.print(table)


def _print_mm_decomposition(summary: dict[str, float]) -> None:
    """Where the net came from: the daily mean of each term, and the net per turnover."""
    if not summary.get("days"):
        console.print("[yellow]No day could enter the result.[/yellow]")
        return
    table = Table(title="Decomposition, mean per usable day", show_edge=False)
    table.add_column("term")
    table.add_column("value", justify="right")
    rows = [
        ("spread captured", "spread"),
        ("adverse selection (5 s)", "adverse"),
        ("inventory", "inventory"),
        ("fees (paid)", "fees"),
        ("funding", "funding"),
        ("net", "net_per_day"),
        ("net, bp of turnover", "net_bp_of_turnover"),
        ("fills a day", "fills_per_day"),
        ("days used / excluded", None),
    ]
    for label, key in rows:
        if key is None:
            value = f"{summary['days']:.0f} / {summary.get('excluded_days', 0.0):.0f}"
        elif key == "fills_per_day":
            value = f"{summary[key]:,.1f}"
        else:
            value = f"{summary.get(key, float('nan')):+,.4f}"
        table.add_row(label, value)
    console.print(table)
    console.print("  [dim]net = spread + adverse + inventory - fees + funding[/dim]")
