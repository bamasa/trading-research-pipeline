"""``trp search``: step 5, the pipeline chooses instrument, frequency, method, model and execution, then reads once.

Each subcommand is one step of :mod:`trading_research.walkthrough.steps`; each
reads the previous step's ``chosen.json`` under ``--run`` and writes its own
directory there, so the search can be stopped and looked at after any step::

    trp search instruments --data data/walkthrough --run artifacts/search
    trp search frequency --run artifacts/search
    trp search method --run artifacts/search
    trp search model --run artifacts/search
    trp search execution --run artifacts/search
    trp search read --run artifacts/search
    trp search status --run artifacts/search

``trp search run`` runs them in order.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from trading_research.cli_common import STEPS_PANEL

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="search",
    help="5 Search: the pipeline chooses instrument, hold, method, model, execution; reads once.",
    no_args_is_help=True,
)
app.info.rich_help_panel = STEPS_PANEL

Run = Annotated[Path, typer.Option("--run", help="The search's directory: one folder per step.")]
DEFAULT_RUN = Path("artifacts/search")


def _report(line: str) -> None:
    console.print(line, highlight=False)


def _guard(step: str, call) -> None:  # type: ignore[no-untyped-def]
    from trading_research.walkthrough.record import ReadRefused, SearchError

    try:
        out = call()
    except ReadRefused as exc:
        err_console.print(f"[red]refused:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except (SearchError, ValueError, KeyError, FileNotFoundError) as exc:
        err_console.print(f"[red]{step}: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[dim]-> {out}[/dim]")


@app.command("instruments")
def instruments_cmd(
    data: Annotated[
        Path | None,
        typer.Option(
            "--data", help="A dataset root: raw/, and book/ trades/ funding/ for a maker."
        ),
    ] = None,
    run: Run = DEFAULT_RUN,
    symbols: Annotated[
        str | None, typer.Option("--symbols", help="Comma-separated; default all.")
    ] = None,
    horizon_s: Annotated[float, typer.Option("--horizon-s", help="The screen's horizon.")] = 60.0,
    search_share: Annotated[
        float, typer.Option("--search-share", help="Days the search sees.")
    ] = 0.65,
    fee: Annotated[str, typer.Option("--fee", help="A Bybit linear tier of trp fees.")] = "base",
    slippage_bp: Annotated[float, typer.Option("--slippage-bp")] = 0.5,
    execution: Annotated[
        str, typer.Option("--execution", help="The family: taker, passive, maker or all.")
    ] = "all",
    holds: Annotated[
        str, typer.Option("--holds", help="Holds to search, seconds.")
    ] = "30,60,120,300",
    minimum_trades: Annotated[int, typer.Option("--minimum-trades")] = 30,
    min_days: Annotated[
        int, typer.Option("--min-days", help="Days with an attempt a verdict needs.")
    ] = 2,
    placebos: Annotated[int, typer.Option("--placebos", help="Placebo draws per grid mode.")] = 40,
    seed: Annotated[int, typer.Option("--seed")] = 20260101,
    discover: Annotated[
        bool, typer.Option("--discover", help="Also list what the venue trades today (network).")
    ] = False,
) -> None:
    """Screen every instrument on the search block; registers the search first unless it is."""
    from trading_research.walkthrough import steps
    from trading_research.walkthrough.layout import Dataset
    from trading_research.walkthrough.record import SearchRun

    family = steps.FAMILY if execution == "all" else tuple(execution.split(","))
    _guard(
        "instruments",
        lambda: steps.instruments(
            SearchRun(run), Dataset.under(data) if data else None,
            symbols=symbols.split(",") if symbols else None, horizon_s=horizon_s,
            search_share=search_share, fee=fee, slippage_bp=slippage_bp, family=family,
            holds_s=tuple(int(h) for h in holds.split(",")), minimum_trades=minimum_trades,
            min_days=min_days, placebo_draws=placebos, seed=seed, discover=discover, report=_report,
        ),
    )  # fmt: skip


def register_cmd(**options: object) -> None:
    """Write the search's registration from the dataset's file names, before any day is read."""
    from trading_research.walkthrough import steps
    from trading_research.walkthrough.layout import Dataset
    from trading_research.walkthrough.record import SearchRun

    execution = str(options.pop("execution"))
    holds = str(options.pop("holds"))
    symbols = options.pop("symbols")
    data, run = options.pop("data"), Path(str(options.pop("run")))
    if data is None:
        err_console.print("[red]register needs --data[/red]")
        raise typer.Exit(code=2)
    placebos = options.pop("placebos")
    family = steps.FAMILY if execution == "all" else tuple(execution.split(","))

    def call() -> Path:
        steps.register(
            SearchRun(run), Dataset.under(Path(str(data))),
            symbols=str(symbols).split(",") if symbols else None,
            family=family, holds_s=tuple(int(h) for h in holds.split(",")),
            placebo_draws=int(str(placebos)), report=_report, **options,  # type: ignore[arg-type]
        )  # fmt: skip
        return SearchRun(run).registration_path

    _guard("register", call)


_shared = inspect.signature(instruments_cmd, eval_str=True)
register_cmd.__signature__ = _shared.replace(  # type: ignore[attr-defined]
    parameters=[q for q in _shared.parameters.values() if q.name != "discover"]
)
register_cmd.__annotations__ = {
    q.name: q.annotation for q in _shared.parameters.values() if q.name != "discover"
}
app.command("register")(register_cmd)
app.registered_commands.insert(0, app.registered_commands.pop())  # the first step, listed first


def _step(name: str, help_text: str) -> None:
    def command(run: Run = DEFAULT_RUN) -> None:
        from trading_research.walkthrough import steps
        from trading_research.walkthrough.record import SearchRun

        _guard(name, lambda: steps.STEP_FUNCTIONS[name](SearchRun(run), report=_report))

    command.__doc__ = help_text
    app.command(name)(command)


_step(
    "frequency",
    "The hold and the label threshold, by their neighbourhood on validation; the peak reported.",
)
_step("method", "Every rule against every learned model at that frequency.")
_step("model", "The chosen family's candidates: the rules, or a model's parameters.")
_step(
    "execution",
    "The signal as a taker, a passive entry and a market maker on the search block; all frozen.",
)


@app.command("read")
def read_cmd(
    run: Run = DEFAULT_RUN,
    second_read: Annotated[
        bool, typer.Option("--second-read", help="Read again, stamped 'second read'. Not a result.")
    ] = False,
) -> None:
    """The held-out block, once, through the run's ledger; the verdict as registered."""
    from trading_research.walkthrough import steps
    from trading_research.walkthrough.record import SearchRun

    _guard("read", lambda: steps.read(SearchRun(run), second_read=second_read, report=_report))


@app.command("run")
def run_cmd(
    data: Annotated[Path, typer.Option("--data", help="A dataset root.")],
    run: Run = DEFAULT_RUN,
    execution: Annotated[
        str, typer.Option("--execution", help="taker, passive, maker or all.")
    ] = "all",
    fee: Annotated[str, typer.Option("--fee")] = "base",
) -> None:
    """Every step in order, the read last."""
    from trading_research.walkthrough import steps
    from trading_research.walkthrough.layout import Dataset
    from trading_research.walkthrough.record import SearchRun

    family = steps.FAMILY if execution == "all" else tuple(execution.split(","))
    _guard(
        "run",
        lambda: steps.run_all(
            SearchRun(run),
            Dataset.under(data),
            report=_report,
            family=family,
            fee=fee,
        ),
    )


@app.command("status")
def status_cmd(run: Run = DEFAULT_RUN) -> None:
    """Which steps are complete, what each chose, and the ledger's reads."""
    from trading_research.walkthrough.record import STEPS, SearchRun

    search = SearchRun(run)
    table = Table(title=f"Search {run}", show_edge=False)
    table.add_column("step")
    table.add_column("state")
    table.add_column("chose", overflow="fold")
    for step in STEPS:
        path = search.step_dir(step) / "chosen.json"
        chose = ""
        if path.exists():
            found = json.loads(path.read_text())
            chose = {
                "instruments": found.get("symbol"),
                "frequency": f"hold {found.get('hold_s')}s, label > {found.get('threshold_bp')} bp",
                "method": found.get("method"),
                "model": f"{found.get('model')} {found.get('params')}",
                "execution": found.get("execution"),
                "read": json.dumps(found.get("statuses")),
            }[step]
        table.add_row(step, "complete" if path.exists() else "not run", str(chose))
    console.print(table)
    reads = search.status().get("held-out reads")
    if reads:
        console.print(f"held-out reads recorded in the ledger: {reads}")
