"""``trp prod``: frozen strategies run as one supervised portfolio.

Commands::

    prod check  <deployment.yaml>
    prod run    <deployment.yaml> --day NAME=<stored day> [--day ...] | --minutes N
    prod status <run dir>

The broker is the paper ledger: no command takes, reads or sends a key, and no
order leaves the process. ``check`` and ``run`` refuse a deployment that names a
credential or a private endpoint, or whose model files are not the ones it pins.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

prod_app = typer.Typer(
    name="prod",
    help="Production: frozen strategies supervised, balanced and observed; the broker is paper.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

DeploymentArg = Annotated[Path, typer.Argument(help="The deployment's YAML file.")]


def _load(path: Path) -> Any:
    from trading_research.production.spec import DeploymentError, load

    try:
        return load(path)
    except (OSError, DeploymentError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


@prod_app.command("check")
def check_cmd(deployment: DeploymentArg) -> None:
    """Check a deployment: every field, the key guard, each model against its pin; the plan."""
    from trading_research.production.allocator import allocate
    from trading_research.production.spec import check_pins

    spec = _load(deployment)
    problems = check_pins(spec)
    portfolio = spec.portfolio
    plan = allocate(
        {s.name: s.expected.sd_bp for s in spec.strategies},
        {},
        capital=spec.capital,
        max_gross=portfolio.max_gross,
        max_weight=portfolio.max_weight,
    )
    table = Table(title=f"Deployment {spec.name}", title_justify="left", show_edge=False)
    for column in ("strategy", "venue", "symbol", "execution", "model", "expected bp",
                   "weight", "capital"):  # fmt: skip
        table.add_column(column, justify="right" if column in ("weight", "capital") else "left")
    for s in spec.strategies:
        allocation = plan[s.name]
        table.add_row(
            s.name, f"{s.venue} {s.market}", s.symbol, s.execution, Path(s.model).name,
            f"{s.expected.net_bp:+.2f} ± {s.expected.sd_bp:.1f}",
            f"{allocation.weight:.2f}", f"{allocation.capital:,.2f}",
        )  # fmt: skip
    console.print(table)
    console.print(
        f"capital {spec.capital:,.2f} · daily loss {portfolio.daily_loss:,.2f} · "
        f"drawdown {portfolio.max_drawdown:,.2f} · health on {spec.host}:{spec.port}"
    )
    if problems:
        for problem in problems:
            err_console.print(f"[red]{problem}[/red]")
        raise typer.Exit(code=1)
    console.print("checked: fields, key guard and model pins hold")


def _days(given: list[str]) -> dict[str, str]:
    days = {}
    for item in given:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            err_console.print(f"[red]--day takes NAME=PATH, got {item!r}[/red]")
            raise typer.Exit(code=2)
        days[name] = path
    return days


@prod_app.command("run")
def run_cmd(
    deployment: DeploymentArg,
    day: Annotated[
        list[str] | None,
        typer.Option("--day", help="NAME=PATH: replay a stored day for that strategy (repeat)."),
    ] = None,
    minutes: Annotated[
        float | None, typer.Option("--minutes", min=0.05, help="Run live on public streams.")
    ] = None,
    speed: Annotated[float, typer.Option("--speed", help="Replay pace; 0 is at once.")] = 0.0,
    port: Annotated[int | None, typer.Option("--port", help="Health port; 0 picks one.")] = None,
    serve: Annotated[bool, typer.Option("--serve/--no-serve", help="Serve health.")] = True,
    root: Annotated[Path, typer.Option("--root", help="Where recordings live.")] = Path(
        "data/paper"
    ),
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Run a deployment on replayed days or live public streams, supervised."""
    from trading_research.production.runner import run
    from trading_research.production.spec import DeploymentError

    spec = _load(deployment)
    if (day is None) == (minutes is None):
        err_console.print("[red]give --day NAME=PATH for each strategy, or --minutes[/red]")
        raise typer.Exit(code=2)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    target = run_dir or Path("runs/prod") / f"{stamp}-{spec.name}"
    try:
        result = run(
            spec, target, days=_days(day) if day else None, minutes=minutes, speed=speed,
            root=root, serve=serve, port=port, command=sys.argv[1:],
        )  # fmt: skip
    except DeploymentError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    _report(result.snapshot)
    console.print(f"supervisor -> {target / 'supervisor.json'}")


def _report(snapshot: dict[str, Any]) -> None:
    table = Table(title=f"Deployment {snapshot['deployment']}", title_justify="left",
                  show_edge=False)  # fmt: skip
    for column in ("strategy", "state", "reason", "fills", "round trips", "equity", "weight"):
        table.add_column(column)
    for name, s in snapshot["strategies"].items():
        table.add_row(
            name, s["state"], s["reason"] or "", str(s["fills"]), str(s["round_trips"]),
            f"{s['equity']:+,.4f}", f"{s['weight']:.2f}",
        )  # fmt: skip
    console.print(table)
    console.print(
        f"portfolio equity {snapshot['equity']:+,.4f} · drawdown {snapshot['drawdown']:,.4f} · "
        f"halt: {snapshot['portfolio_halt'] or 'none'}"
    )


@prod_app.command("status")
def status_cmd(
    run_dir: Annotated[Path, typer.Argument(help="A run directory of prod run.")],
) -> None:
    """Show a run's supervisor state: strategies, transitions, portfolio."""
    path = run_dir / "supervisor.json"
    if not path.exists():
        err_console.print(f"[red]{path} does not exist[/red]")
        raise typer.Exit(code=2)
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    _report(snapshot)
    for item in snapshot.get("transitions", []):
        console.print(f"{item['strategy']}: {item['from']} -> {item['to']} ({item['reason']})")
