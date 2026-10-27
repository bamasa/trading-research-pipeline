"""Commands for synthetic data with planted answers, and the checks that read it.

Two groups, registered on the main application:

``trading-research synthetic <generator>``
    Write a dataset with a planted effect (or its null, with ``--null``) and
    its ``truth.json`` under ``data/synthetic/``: ``book-days``, ``break-days``,
    ``cross-section``, ``order-flow``, ``funding``.
``trading-research verify <what>``
    Run the repository's own components on planted data and its null and judge
    them against the truth: ``edge``, ``detector``, ``simulator``,
    ``reversion``, ``all``. Each writes ``artifacts/verify/<what>/page.md`` and
    exits non-zero when a binding criterion fails.

Nothing here reads market data, and nothing written is evidence about a market.
Heavy imports wait for the command that needs them, so the main command line
starts as fast as it did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

synthetic_app = typer.Typer(
    name="synthetic",
    help="Write synthetic datasets with a planted, known answer (and their nulls).",
    no_args_is_help=True,
    add_completion=False,
)
verify_app = typer.Typer(
    name="verify",
    help="Check the repository's components recover planted effects and find nothing on nulls.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

SEED_HELP = "Seed. The same seed and options always give the same data."
NULL_HELP = "Write the null: the same data with nothing planted."
OUT_HELP = "Directory to write into."
VERIFY_OUT = Path("artifacts/verify")

Seed = Annotated[int, typer.Option("--seed", help=SEED_HELP)]
Null = Annotated[bool, typer.Option("--null", help=NULL_HELP)]


def _written(path: Path, truth: Any, rows: list[tuple[str, str]]) -> None:
    table = Table(title="Synthetic dataset", title_justify="left", show_edge=False)
    table.add_column("item", style="bold")
    table.add_column("value")
    for name, value in rows:
        table.add_row(name, value)
    for name, value in truth.planted.items():
        table.add_row(f"planted {name}", str(value))
    console.print(table)
    console.print(f"\nWritten to [bold]{path}[/bold], truth in [bold]{path / 'truth.json'}[/bold]")
    console.print("[yellow]Synthetic data with a planted answer: not a market.[/yellow]")


@synthetic_app.command("book-days")
def book_days_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help=OUT_HELP)] = Path(
        "data/synthetic/book_days"
    ),
    days: Annotated[int, typer.Option("--days", min=1, help="Sessions to generate.")] = 16,
    session_s: Annotated[
        int, typer.Option("--session-s", min=300, help="Seconds a session.")
    ] = 3600,
    feature: Annotated[str, typer.Option("--feature", help="imbalance or flow.")] = "imbalance",
    size_bp: Annotated[
        float, typer.Option("--size-bp", help="Planted move over the horizon.")
    ] = 16.0,
    horizon_s: Annotated[float, typer.Option("--horizon-s", help="Planted horizon.")] = 30.0,
    decay_s: Annotated[float, typer.Option("--decay-s", help="Half-life of the drift.")] = 10.0,
    noise_bp: Annotated[float, typer.Option("--noise-bp", help="Noise, bp per sqrt s.")] = 1.0,
    null: Null = False,
    seed: Seed = 20200106,
) -> None:
    """Order-book days (10 levels, 100 ms) and prints with a planted short-horizon edge."""
    from trading_research.synthetic.book_days import BookDaysSpec, generate

    spec = BookDaysSpec(
        days=days, session_s=session_s, feature=feature, size_bp=size_bp,  # type: ignore[arg-type]
        horizon_s=horizon_s, decay_s=decay_s, noise_bp=noise_bp, null=null, seed=seed,
    )  # fmt: skip
    dataset = generate(spec)
    path = dataset.write(out)
    rows = sum(len(f) for f in dataset.book.values())
    prints = sum(len(f) for f in dataset.trades.values())
    _written(path, dataset.truth, [("book rows", f"{rows:,}"), ("prints", f"{prints:,}")])
    console.print(f"Check it: trading-research validate-data --input {path}")


@synthetic_app.command("break-days")
def break_days_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help=OUT_HELP)] = Path(
        "data/synthetic/break_days"
    ),
    days: Annotated[int, typer.Option("--days", min=2, help="Whole days to generate.")] = 2,
    break_at_h: Annotated[
        float, typer.Option("--break-at-h", help="Hours after the start.")
    ] = 30.0,
    vol_ratio: Annotated[float, typer.Option("--vol-ratio", help="Return scale after.")] = 1.6,
    spread_after: Annotated[int, typer.Option("--spread-after", help="Spread ticks after.")] = 4,
    edge_after: Annotated[float, typer.Option("--edge-after", help="Edge sign after.")] = -1.0,
    step_ms: Annotated[int, typer.Option("--step-ms", help="Snapshot interval.")] = 100,
    null: Null = False,
    seed: Seed = 20200106,
) -> None:
    """Whole days of book with one regime break at a known time (or none, with --null)."""
    from trading_research.synthetic.break_days import BreakDaysSpec, generate_breaks

    spec = BreakDaysSpec(
        break_at_s=break_at_h * 3600, vol_ratio=vol_ratio, spread_ticks_after=spread_after,
        edge_after=edge_after, null=null,
    ).with_book(days=days, step_ms=step_ms, seed=seed)  # fmt: skip
    dataset = generate_breaks(spec)
    path = dataset.write(out)
    rows = sum(len(f) for f in dataset.book.values())
    _written(path, dataset.truth, [("book rows", f"{rows:,}")])
    console.print(f"Check it: trading-research validate-data --input {path}")


@synthetic_app.command("cross-section")
def cross_section_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help=OUT_HELP)] = Path(
        "data/synthetic/cross_section"
    ),
    days: Annotated[int, typer.Option("--days", min=2, help="Days of bars.")] = 16,
    instruments: Annotated[int, typer.Option("--instruments", min=6)] = 20,
    size_bp: Annotated[float, typer.Option("--size-bp", help="Idiosyncratic transient sd.")] = 8.0,
    market_size_bp: Annotated[float, typer.Option("--market-size-bp")] = 20.0,
    half_life_bars: Annotated[float, typer.Option("--half-life-bars")] = 60.0,
    variant: Annotated[str, typer.Option("--variant", help="reversal or momentum.")] = "reversal",
    null: Null = False,
    seed: Seed = 20200106,
) -> None:
    """A universe on 5 s bars with a planted reversion or momentum."""
    from trading_research.synthetic.cross_section import CrossSectionSpec, generate

    spec = CrossSectionSpec(
        n_instruments=instruments, days=days, size_bp=size_bp, market_size_bp=market_size_bp,
        half_life_bars=half_life_bars, variant=variant, null=null, seed=seed,  # type: ignore[arg-type]
    )  # fmt: skip
    dataset = generate(spec)
    path = dataset.write(out)
    _written(path, dataset.truth, [("bars", f"{dataset.log_prices.size:,}")])
    console.print("Bars are checked with data.validate.validate_bars before they are written.")


@synthetic_app.command("order-flow")
def order_flow_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help=OUT_HELP)] = Path(
        "data/synthetic/order_flow"
    ),
    days: Annotated[int, typer.Option("--days", min=1, help="Sessions to generate.")] = 4,
    adverse_bp: Annotated[
        float, typer.Option("--adverse-bp", help="Planted adverse selection.")
    ] = 0.6,
    arrival_rate: Annotated[
        float, typer.Option("--arrival-rate", help="Per level, a second.")
    ] = 0.8,
    cancel_rate: Annotated[
        float, typer.Option("--cancel-rate", help="Per order, a second.")
    ] = 0.08,
    spread_ticks: Annotated[int, typer.Option("--spread-ticks", min=2)] = 2,
    cancel_position: Annotated[
        str, typer.Option("--cancel-position", help="uniform, back or front.")
    ] = "uniform",
    null: Null = False,
    seed: Seed = 20200106,
) -> None:
    """An order-level book for the market-making simulator, with the queue's truth."""
    from trading_research.synthetic.order_flow import OrderFlowSpec, generate

    spec = OrderFlowSpec(
        days=days, adverse_bp=adverse_bp, arrival_rate=arrival_rate, cancel_rate=cancel_rate,
        spread_ticks=spread_ticks, cancel_position=cancel_position, null=null, seed=seed,  # type: ignore[arg-type]
    )  # fmt: skip
    dataset = generate(spec)
    path = dataset.write(out)
    snapshots = sum(len(d.book_ts) for d in dataset.days)
    prints = sum(len(d.trade_ts) for d in dataset.days)
    _written(path, dataset.truth, [("snapshots", f"{snapshots:,}"), ("prints", f"{prints:,}")])
    console.print(f"Check it: trading-research validate-data --input {path}")


@synthetic_app.command("funding")
def funding_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help=OUT_HELP)] = Path(
        "data/synthetic/funding"
    ),
    days: Annotated[int, typer.Option("--days", min=1, help="Days of settlements.")] = 365,
    persistence: Annotated[
        float, typer.Option("--persistence", help="AR(1) per settlement.")
    ] = 0.9,
    mean_bp: Annotated[float, typer.Option("--mean-bp")] = 1.0,
    sd_bp: Annotated[float, typer.Option("--sd-bp")] = 1.5,
    null: Null = False,
    seed: Seed = 20200106,
) -> None:
    """A funding-rate series with a planted persistence, in the venue's file layout."""
    from trading_research.synthetic.funding import FundingSpec, estimate_persistence, generate

    spec = FundingSpec(
        days=days, persistence=persistence, mean_bp=mean_bp, sd_bp=sd_bp, null=null, seed=seed
    )
    dataset = generate(spec)
    path = dataset.write(out)
    measured = estimate_persistence(dataset.series["rate"].to_numpy())
    _written(
        path,
        dataset.truth,
        [("settlements", f"{len(dataset.series):,}"), ("measured persistence", f"{measured:.3f}")],
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _report(result: Any, out: Path) -> bool:
    page = result.write(out)
    table = Table(
        title=f"verify {result.what}: {'PASS' if result.passed else 'FAIL'}",
        title_justify="left",
        show_edge=False,
    )
    for column in ("check", "planted", "recovered", "tolerance", "result"):
        table.add_column(column, overflow="fold")
    for c in result.criteria:
        style = "green" if c.result == "pass" else ("red" if c.result == "FAIL" else "dim")
        table.add_row(
            c.check, c.planted, c.recovered, c.tolerance, f"[{style}]{c.result}[/{style}]"
        )
    console.print(table)
    console.print(f"Page: [bold]{page}[/bold]  ({result.seconds:.0f} s)\n")
    return bool(result.passed)


def _progress(message: str) -> None:
    err_console.print(f"[dim]{message}[/dim]")


@verify_app.command("edge")
def verify_edge_cmd(
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy",
            "-s",
            help="A model or rule name (order_flow, logistic, order_flow:flow_imbalance_24) or "
            "package.module:attribute. Repeat for several. Default: the repository's rules "
            "and two learned models.",
        ),
    ] = None,
    feature: Annotated[
        list[str] | None, typer.Option("--feature", help="imbalance and/or flow; default both.")
    ] = None,
    size_bp: Annotated[float, typer.Option("--size-bp", help="Planted size, bp.")] = 16.0,
    fee_bp: Annotated[float, typer.Option("--fee-bp", help="Stated taker fee a side, bp.")] = 5.5,
    days: Annotated[int, typer.Option("--days", min=6)] = 16,
    seed: Seed = 20200106,
    out: Annotated[Path, typer.Option("--out", help=OUT_HELP)] = VERIFY_OUT,
) -> None:
    """A planted edge through features, labels, model or rule and the taker executor."""
    from trading_research.synthetic.edge import load_strategy
    from trading_research.synthetic.verify import verify_edge

    strategies = [load_strategy(s) for s in strategy] if strategy else None
    result = verify_edge(
        strategies=strategies,
        features=tuple(feature or ("imbalance", "flow")),
        size_bp=size_bp,
        fee_bp=fee_bp,
        days=days,
        seed=seed,
        progress=_progress,
    )
    if not _report(result, out):
        raise typer.Exit(code=1)


@verify_app.command("detector")
def verify_detector_cmd(
    paths: Annotated[int, typer.Option("--paths", min=1, help="Planted and null paths.")] = 3,
    step_ms: Annotated[int, typer.Option("--step-ms", help="Snapshot interval.")] = 100,
    vol_ratio: Annotated[float, typer.Option("--vol-ratio")] = 1.6,
    seed: Seed = 20200106,
    out: Annotated[Path, typer.Option("--out", help=OUT_HELP)] = VERIFY_OUT,
) -> None:
    """The regime detectors on a planted break and on the same days without it."""
    from trading_research.synthetic.verify import verify_detector

    result = verify_detector(
        paths=paths, seed=seed, step_ms=step_ms, vol_ratio=vol_ratio, progress=_progress
    )
    if not _report(result, out):
        raise typer.Exit(code=1)


@verify_app.command("simulator")
def verify_simulator_cmd(
    days: Annotated[int, typer.Option("--days", min=1)] = 4,
    adverse_bp: Annotated[float, typer.Option("--adverse-bp")] = 0.6,
    seed: Seed = 20200106,
    out: Annotated[Path, typer.Option("--out", help=OUT_HELP)] = VERIFY_OUT,
) -> None:
    """The market-making simulator's markouts and fills against the planted truth."""
    from trading_research.synthetic.verify import verify_simulator

    result = verify_simulator(days=days, seed=seed, adverse_bp=adverse_bp, progress=_progress)
    if not _report(result, out):
        raise typer.Exit(code=1)


@verify_app.command("reversion")
def verify_reversion_cmd(
    paths: Annotated[int, typer.Option("--paths", min=1)] = 3,
    days: Annotated[int, typer.Option("--days", min=6)] = 16,
    cost_bp: Annotated[
        float, typer.Option("--cost-bp", help="Stated cost a name, round trip.")
    ] = 12.0,
    seed: Seed = 20200106,
    out: Annotated[Path, typer.Option("--out", help=OUT_HELP)] = VERIFY_OUT,
) -> None:
    """The cross-sectional and index reversion rules on a planted cross-section."""
    from trading_research.synthetic.verify import verify_reversion

    result = verify_reversion(
        paths=paths, days=days, seed=seed, cost_bp=cost_bp, progress=_progress
    )
    if not _report(result, out):
        raise typer.Exit(code=1)


@verify_app.command("all")
def verify_all_cmd(
    seed: Seed = 20200106,
    out: Annotated[Path, typer.Option("--out", help=OUT_HELP)] = VERIFY_OUT,
) -> None:
    """Every check with its defaults; one summary page beside each check's own."""
    from trading_research.synthetic.verify import CHECKS, write_all

    results = []
    for what, check in CHECKS.items():
        console.print(f"[bold]verify {what}[/bold]")
        result = check(seed=seed, progress=_progress)
        _report(result, out)
        results.append(result)
    page = write_all(results, out)
    passed = all(r.passed for r in results)
    console.print(
        f"[{'green' if passed else 'red'}]verify all: "
        + ", ".join(f"{r.what} {'pass' if r.passed else 'FAIL'}" for r in results)
        + f"[/{'green' if passed else 'red'}]\nSummary: [bold]{page}[/bold]"
    )
    if not passed:
        raise typer.Exit(code=1)
