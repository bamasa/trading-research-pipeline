"""``trp pairs``: screen pairs for cointegration, backtest one walk-forward, write a synthetic pair.

Reads bars in the data contract from ``<root>/<SYMBOL>/klines-<interval>/``,
where ``trp data download --kind klines`` writes them (``data/raw`` by default)
and where ``trp pairs synthetic`` writes its generated pair. Each command
writes a page, csv files and a summary under ``--out``. See
:mod:`trading_research.pairs` and ``docs/pairs.md``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from trading_research.cli_common import STEPS_PANEL

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="pairs",
    help="Pairs: screen every pair for cointegration, backtest one walk-forward, write a "
    "synthetic pair.",
    no_args_is_help=True,
)
app.info.rich_help_panel = STEPS_PANEL

Symbols = Annotated[
    list[str],
    typer.Option("--symbol", "-s", help="Instrument, e.g. ETHUSDT. Repeat for each leg."),
]
Root = Annotated[
    Path,
    typer.Option("--root", help="Where the bars live: <root>/<SYMBOL>/klines-<interval>/."),
]
Interval = Annotated[str, typer.Option("--interval", help="Bar length, e.g. 1h.")]
FeeBp = Annotated[
    float | None,
    typer.Option(
        "--fee-bp",
        min=0.0,
        help="Taker fee a fill, bp. Default: Binance USD-M's standard taker fee.",
    ),
]
Tier = Annotated[
    str | None,
    typer.Option("--tier", help="A Bybit linear tier by name instead (trp fees lists them)."),
]
Slippage = Annotated[
    float,
    typer.Option("--slippage-bp", min=0.0, help="Slippage a fill, bp, the half-spread included."),
]
Lags = Annotated[
    int, typer.Option("--lags", min=0, help="Augmentation lags of the Engle-Granger test.")
]


def _day(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{name}: {value!r} is not YYYY-MM-DD") from exc


def _costs(fee_bp: float | None, tier: str | None, slippage_bp: float):  # type: ignore[no-untyped-def]
    from trading_research.pairs.costs import leg_costs

    try:
        return leg_costs(fee_bp=fee_bp, tier=tier, slippage_bp=slippage_bp)
    except (KeyError, ValueError) as exc:
        err_console.print(f"[red]{exc.args[0] if exc.args else exc}[/red]")
        raise typer.Exit(code=2) from exc


def _bars(root: Path, symbols: list[str], interval: str, start: date | None, end: date | None):  # type: ignore[no-untyped-def]
    from trading_research.pairs.data import load
    from trading_research.pairs.stats import PairsError

    try:
        return load(root, symbols, interval, start=start, end=end)
    except PairsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command("screen")
def screen_cmd(
    symbol: Symbols,
    start: Annotated[str, typer.Option("--start", help="First day of the formation window.")],
    end: Annotated[str, typer.Option("--end", help="Last day of the formation window.")],
    root: Root = Path("data/raw"),
    interval: Interval = "1h",
    alpha: Annotated[
        float, typer.Option("--alpha", help="Family-wise level of Holm's correction.")
    ] = 0.05,
    lags: Lags = 1,
    fee_bp: FeeBp = None,
    tier: Tier = None,
    slippage_bp: Slippage = 0.5,
    out: Annotated[Path, typer.Option("--out", help="Directory to write into.")] = Path(
        "artifacts/pairs/screen"
    ),
) -> None:
    """Every pair among the symbols on one formation window: cointegration, half-life, cost; Holm-corrected."""
    from trading_research.pairs.data import log_closes
    from trading_research.pairs.report import write_screen
    from trading_research.pairs.screen import screen
    from trading_research.pairs.stats import PairsError

    if len(symbol) < 2:
        raise typer.BadParameter("name two symbols or more with --symbol")
    costs = _costs(fee_bp, tier, slippage_bp)
    first, last = _day(start, "--start"), _day(end, "--end")
    bars = _bars(root, symbol, interval, first, last)
    try:
        table = screen(log_closes(bars), costs, alpha=alpha, lags=lags)
    except PairsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    window = (
        f"{bars.index[0]:%Y-%m-%d %H:%M} to {bars.index[-1]:%Y-%m-%d %H:%M} UTC ({len(bars)} bars)"
    )
    page = write_screen(table, out, source=str(root), window=window, costs=costs, alpha=alpha)
    shown = Table(title=f"pairs screen, {window}", title_justify="left", show_edge=False)
    columns = (
        "rank",
        "a",
        "b",
        "beta",
        "eg_t",
        "p_holm",
        "passes",
        "half_life_bars",
        "spread_sd_bp",
        "round_trip_cost_bp",
    )
    for column in columns:
        shown.add_column(column, justify="left" if column in ("a", "b") else "right")
    for row in table.itertuples(index=False):
        shown.add_row(*[_fmt(getattr(row, c)) for c in columns])
    console.print(shown)
    console.print(f"Page: [bold]{page}[/bold]")


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


@app.command("backtest")
def backtest_cmd(
    symbol: Symbols,
    start: Annotated[
        str | None, typer.Option("--start", help="First day read; the first formation starts here.")
    ] = None,
    end: Annotated[str | None, typer.Option("--end", help="Last day read.")] = None,
    root: Root = Path("data/raw"),
    interval: Interval = "1h",
    formation: Annotated[
        int, typer.Option("--formation", min=100, help="Bars the hedge and bands are fitted on.")
    ] = 720,
    trading: Annotated[
        int, typer.Option("--trading", min=2, help="Bars traded on each fit; the walk's step.")
    ] = 168,
    entry: Annotated[float, typer.Option("--entry", help="Enter at |z| >= this.")] = 2.0,
    exit_at: Annotated[float, typer.Option("--exit", help="Leave at |z| <= this.")] = 0.5,
    stop: Annotated[float, typer.Option("--stop", help="Stop at |z| >= this, against.")] = 4.0,
    max_hold: Annotated[
        int, typer.Option("--max-hold", min=1, help="Leave after this many bars.")
    ] = 72,
    gate_p: Annotated[
        float | None,
        typer.Option(
            "--gate-p", help="Trade a window only when its formation p-value is at most this."
        ),
    ] = None,
    lags: Lags = 1,
    fee_bp: FeeBp = None,
    tier: Tier = None,
    slippage_bp: Slippage = 0.5,
    out: Annotated[Path, typer.Option("--out", help="Directory to write into.")] = Path(
        "artifacts/pairs/backtest"
    ),
) -> None:
    """One pair walk-forward: fit on formation, trade the next window, both legs charged per fill."""
    from trading_research.pairs.backtest import PairRule, backtest
    from trading_research.pairs.report import summary_table, write_backtest
    from trading_research.pairs.stats import PairsError

    if len(symbol) != 2:
        raise typer.BadParameter(
            "name exactly two symbols with --symbol; the first is a, regressed on b"
        )
    costs = _costs(fee_bp, tier, slippage_bp)
    bars = _bars(root, symbol, interval, _day(start, "--start"), _day(end, "--end"))
    try:
        rule = PairRule(
            formation=formation,
            trading=trading,
            entry_z=entry,
            exit_z=exit_at,
            stop_z=stop,
            max_hold=max_hold,
            gate_p=gate_p,
            lags=lags,
        )
        result = backtest(bars, symbol[0], symbol[1], rule, costs)
    except PairsError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    page = write_backtest(result, out, source=str(root))
    shown = Table(
        title=f"pairs backtest, {symbol[0]} on {symbol[1]}", title_justify="left", show_edge=False
    )
    shown.add_column("measure")
    shown.add_column("value", justify="right")
    for row in summary_table(result.summary).itertuples(index=False):
        shown.add_row(str(row.measure), _fmt(row.value))
    console.print(shown)
    console.print(f"Page: [bold]{page}[/bold]")


@app.command("synthetic")
def synthetic_cmd(
    out: Annotated[Path, typer.Option("--out", help="Directory to write into.")] = Path(
        "data/synthetic/pairs"
    ),
    variant: Annotated[
        str,
        typer.Option(
            "--variant",
            help="planted (a cointegrated pair), independent (two random walks) or break "
            "(cointegrated, then not).",
        ),
    ] = "planted",
    symbol: Annotated[
        list[str] | None,
        typer.Option(
            "--symbol",
            "-s",
            help="Names of the two legs, a then b (default SYNTHAUSDT, SYNTHBUSDT). Different "
            "names let several pairs share one --out.",
        ),
    ] = None,
    days: Annotated[int, typer.Option("--days", min=40, help="Days of hourly bars.")] = 120,
    beta: Annotated[float, typer.Option("--beta", help="Planted hedge ratio.")] = 0.8,
    half_life: Annotated[
        float, typer.Option("--half-life", help="Planted half-life of the residual, bars.")
    ] = 6.0,
    spread_sd_bp: Annotated[
        float, typer.Option("--spread-sd-bp", help="Planted residual sd, bp.")
    ] = 60.0,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Seed. The same seed and options always give the same data."),
    ] = 20200106,
) -> None:
    """A synthetic pair of hourly bars with a planted cointegration, or a null, and its truth.json.

    truth.json describes the last pair written to --out; write each variant to its own
    directory to keep each one's truth.
    """
    from trading_research.synthetic.pairs import PairSpec, generate

    names = {}
    if symbol:
        if len(symbol) != 2:
            raise typer.BadParameter("name exactly two symbols with --symbol, or none")
        names = {"symbol_a": symbol[0], "symbol_b": symbol[1]}
    try:
        spec = PairSpec(
            **names,  # type: ignore[arg-type]
            days=days,
            beta=beta,
            half_life_bars=half_life,
            spread_sd_bp=spread_sd_bp,
            variant=variant,  # type: ignore[arg-type]
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    dataset = generate(spec)
    path = dataset.write(out)
    shown = Table(title="Synthetic pair", title_justify="left", show_edge=False)
    shown.add_column("item", style="bold")
    shown.add_column("value")
    shown.add_row("written to", str(path))
    shown.add_row("symbols", ", ".join(dataset.bars))
    for name, value in dataset.truth.planted.items():
        shown.add_row(name, str(value))
    console.print(shown)
    a, b = list(dataset.bars)
    first = date.fromisoformat(spec.start)
    console.print(
        f"[dim]Next: trp pairs screen --root {path} -s {a} -s {b} --start {first} "
        f"--end {first + timedelta(days=29)}\n"
        f"      trp pairs backtest --root {path} -s {a} -s {b}[/dim]"
    )
