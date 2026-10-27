"""``trading-research strategies``: list, show, write, check and backtest plugins.

Every kind of part (strategy, model, feature set, label, execution) is listed
here, not only strategies: the group is named for what a fork adds most often.
The registry and the checks live in :mod:`trading_research.plugins`; this
module only parses arguments and prints. Nothing here reads the network or a
study's data: ``check`` and ``backtest`` run on the synthetic demo market
unless ``backtest`` is pointed at a dataset with ``--input``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

plugins_app = typer.Typer(
    name="strategies",
    help="Plugins: list, show, write a new one from a template, check, backtest.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

KindOption = Annotated[
    str | None,
    typer.Option("--kind", "-k", help="strategy, model, features, label or execution."),
]


def _kind(value: str | None) -> str | None:
    from trading_research.plugins import KINDS

    if value is not None and value not in KINDS:
        err_console.print(f"[red]--kind must be one of {', '.join(KINDS)}[/red]")
        raise typer.Exit(code=2)
    return value


def _spec(name: str, kind: str | None) -> Any:
    from trading_research.plugins import PluginError, get

    try:
        return get(name, _kind(kind))
    except PluginError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _params(values: list[str] | None) -> dict[str, Any]:
    """``--param key=value`` pairs, the value read as JSON when it parses."""
    out: dict[str, Any] = {}
    for item in values or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            err_console.print(f"[red]--param takes key=value, got {item!r}[/red]")
            raise typer.Exit(code=2)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


ParamOption = Annotated[
    list[str] | None,
    typer.Option("--param", "-p", help="A constructor argument, key=value; repeatable."),
]


@plugins_app.command("list")
def list_cmd(kind: KindOption = None) -> None:
    """Every registered plugin: built-ins, contrib files and installed entry points."""
    from trading_research.plugins import load_errors, specs

    table = Table(show_edge=False)
    for column in ("kind", "name", "origin", "summary"):
        table.add_column(column, overflow="fold")
    for spec in specs(_kind(kind)):
        table.add_row(spec.kind, spec.name, spec.origin, spec.summary)
    console.print(table)
    for module, reason in load_errors():
        err_console.print(f"[yellow]not loaded:[/yellow] {module}: {reason}")


@plugins_app.command("show")
def show_cmd(name: str, kind: KindOption = None, param: ParamOption = None) -> None:
    """One plugin: where it is defined, what it declares, and its documentation."""
    import inspect

    from trading_research.plugins.protocols import DECLARED

    spec = _spec(name, kind)
    console.print(f"[bold]{spec.name}[/bold] ({spec.kind}, {spec.origin})")
    console.print(f"  module       {spec.module}")
    console.print(f"  file         {spec.source.name if spec.source else 'unknown'}")
    console.print(f"  sha256       {spec.fingerprint() or 'unavailable'}")
    if spec.check_params:
        console.print(f"  checked with {spec.check_params}")
    try:
        instance = spec.build(**_params(param))
    except Exception as exc:
        err_console.print(f"[red]could not be built: {type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    for attribute in (*DECLARED[spec.kind], "target", "stateful"):
        if hasattr(instance, attribute):
            console.print(f"  {attribute:<12} {getattr(instance, attribute)!r}")
    doc = inspect.getdoc(spec.defined_by or spec.factory) or ""
    if doc:
        console.print("\n" + doc)


@plugins_app.command("new")
def new_cmd(
    name: str,
    kind: Annotated[
        str, typer.Option("--kind", "-k", help="strategy, model, features, label or execution.")
    ] = "strategy",
    tests: Annotated[
        Path | None,
        typer.Option("--tests", help="Where the test goes; the checkout's tests/ by default."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
) -> None:
    """Write a working plugin from the template, and a test for it, under contrib/."""
    import importlib

    from trading_research.plugins import names
    from trading_research.plugins.registry import CONTRIB
    from trading_research.plugins.templates import render

    kind = _kind(kind) or "strategy"
    try:
        module, test = render(kind, name)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if name in names(kind):
        err_console.print(f"[red]a {kind} called {name!r} is already registered[/red]")
        raise typer.Exit(code=2)
    folder = Path(importlib.import_module(CONTRIB[kind]).__path__[0])
    target = folder / f"{name}.py"
    tests_dir = tests or _checkout_tests(folder)
    test_path = tests_dir / f"test_plugins_contrib_{name}.py"
    existing = [p for p in (target, test_path) if p.exists()]
    if existing and not force:
        err_console.print(
            f"[red]already there: {', '.join(p.name for p in existing)} (--force)[/red]"
        )
        raise typer.Exit(code=2)
    target.write_text(module, encoding="utf-8")
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_path.write_text(test, encoding="utf-8")
    console.print(f"[green]wrote[/green] {target}")
    console.print(f"[green]wrote[/green] {test_path}")
    console.print(
        f"\nNext: edit it, keep its declarations true, then\n"
        f"  trading-research strategies check {name} --kind {kind}\n"
        f"  python -m pytest {test_path.name}"
    )


def _checkout_tests(contrib: Path) -> Path:
    """``tests/`` of the source checkout the package lives in."""
    for parent in contrib.parents:
        if (parent / "pyproject.toml").exists() and (parent / "tests").is_dir():
            return parent / "tests"
    return Path("tests")


@plugins_app.command("check")
def check_cmd(
    name: Annotated[str | None, typer.Argument(help="The plugin; omit with --all.")] = None,
    kind: KindOption = None,
    all_: Annotated[
        bool, typer.Option("--all", help="Check every registered plugin (of --kind, if given).")
    ] = False,
    execution: Annotated[
        str,
        typer.Option(
            "--execution",
            "-e",
            help="Execution plugins to run decisions through: names separated by commas "
            "(taker, passive, maker, event_taker, or any registered) or all.",
        ),
    ] = "all",
    fee_tier: Annotated[str, typer.Option("--fee-tier", help="A row of the fee table.")] = "base",
    steps: Annotated[
        int, typer.Option("--steps", min=12_000, help="Synthetic snapshots.")
    ] = 72_000,
    seed: Annotated[int, typer.Option("--seed")] = 20260101,
    param: ParamOption = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/plugins"),
) -> None:
    """The conformance suite: protocol, no look-ahead, every execution, synthetic checks.

    Writes a page per plugin under OUTPUT/<name>/page.md, and with --all an
    index. Exits 1 when any check failed.
    """
    _one_omp_thread()
    from trading_research.backtest.costs import fee_tier as tier_named
    from trading_research.plugins import page as pages
    from trading_research.plugins import specs
    from trading_research.plugins.conformance import check, make_setup

    if (name is None) == (not all_):
        err_console.print("[red]give a plugin's name, or --all[/red]")
        raise typer.Exit(code=2)
    try:
        tier_named(fee_tier)
        _executions(execution)
    except (KeyError, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if all_:
        from trading_research.plugins import worker

        rows = []
        for spec in specs(_kind(kind)):
            with console.status(f"checking {spec.kind} {spec.name}"):
                row = worker.run(
                    spec.kind, spec.name, spec.origin, output,
                    execution=execution, fee_tier=fee_tier, steps=steps, seed=seed,
                )  # fmt: skip
            rows.append(row)
            _print_row(row)
        console.print(f"\nindex -> {pages.write_index(rows, output)}")
        if any(row["passed"] != "yes" for row in rows):
            raise typer.Exit(code=1)
        return
    chosen = _executions(execution)
    setup = make_setup(executions=chosen, fee_tier=fee_tier, steps=steps, seed=seed)
    result = check(_spec(name or "", kind), setup, _params(param))
    _print_result(result, pages.write(result, output))
    if not result.passed:
        raise typer.Exit(code=1)


def _print_row(row: dict[str, Any]) -> None:
    checks = [k for k in row if k[:2] in ("a.", "b.", "c.", "d.")]
    marks = "  ".join(f"{k.split(' ')[0]} {row[k]}" for k in checks)
    verdict = "[green]pass[/green]" if row["passed"] == "yes" else "[red]FAIL[/red]"
    console.print(f"{verdict} {row['kind']} [bold]{row['name']}[/bold]: {marks}")
    if row.get("reason"):
        console.print(f"    [red]{row['reason']}[/red]")


def _print_result(result: Any, path: Path) -> None:
    colour = {"pass": "green", "fail": "red"}
    marks = "  ".join(
        f"{r.check.split(':')[0]} [{colour.get(r.status, 'dim')}]{r.status}[/]"
        for r in result.results
    )
    verdict = "[green]pass[/green]" if result.passed else "[red]FAIL[/red]"
    console.print(f"{verdict} {result.spec.kind} [bold]{result.spec.name}[/bold]: {marks}")
    for failed in (r for r in result.results if r.status == "fail"):
        console.print(f"    [red]{failed.check}[/red]: {failed.detail}")
    console.print(f"    [dim]-> {path}[/dim]")


@plugins_app.command("backtest")
def backtest_cmd(
    name: str,
    kind: KindOption = None,
    execution: Annotated[
        str,
        typer.Option("--execution", "-e", help="Names separated by commas, or all."),
    ] = "taker,passive",
    fee_tier: Annotated[str, typer.Option("--fee-tier", help="A row of the fee table.")] = "base",
    input_dir: Annotated[
        Path | None,
        typer.Option(
            "--input", "-i", help="A dataset in the store's layout; synthetic if omitted."
        ),
    ] = None,
    train_share: Annotated[
        float,
        typer.Option("--train-share", min=0.1, max=0.9, help="Rows fitted on before trading."),
    ] = 0.4,
    steps: Annotated[
        int, typer.Option("--steps", min=12_000, help="Synthetic snapshots.")
    ] = 72_000,
    seed: Annotated[int, typer.Option("--seed")] = 20260101,
    param: ParamOption = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("artifacts/plugins"),
) -> None:
    """Fit a strategy (or a model, made into one) on the front, trade the rest.

    The strategy is fitted on the first --train-share of the rows and its
    decisions after that are run through each execution named, with the fee
    tier's row applied. Writes the table and a manifest under
    OUTPUT/<name>/backtest/.
    """
    _one_omp_thread()
    import numpy as np
    import pandas as pd

    from trading_research import __version__
    from trading_research.backtest.costs import fee_tier as tier_named
    from trading_research.plugins.conformance import Setup, run_executions
    from trading_research.plugins.market import demo_market, market_from_dataset
    from trading_research.plugins.model_strategy import ModelStrategy

    spec = _spec(name, kind)
    if spec.kind not in ("strategy", "model"):
        err_console.print(
            f"[red]{name} is a {spec.kind}; backtest takes a strategy or a model[/red]"
        )
        raise typer.Exit(code=2)
    try:
        tier = tier_named(fee_tier)
        market = (
            market_from_dataset(input_dir) if input_dir else demo_market(steps=steps, seed=seed)
        )
        chosen = _executions(execution)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    built = spec.build(**{**spec.check_params, **_params(param)})
    strategy = built if spec.kind == "strategy" else ModelStrategy(built, fees=tier)
    setup = Setup(market, fee_tier, tuple(chosen), train_share)
    strategy.fit(setup.history)
    decisions = np.asarray(strategy.decide(setup.frame)).copy()
    decisions[: setup.train_end] = 0
    result, table = run_executions(decisions, int(strategy.horizon), setup, chosen)

    console.print(
        f"[bold]{spec.name}[/bold] on {market.source}\n"
        f"fitted on {setup.train_end:,} rows, traded on {len(setup.frame) - setup.train_end:,}; "
        f"{int((decisions != 0).sum()):,} decision rows, hold {int(strategy.horizon)} rows, "
        f"{fee_tier} fees ({tier.maker_bp:g} bp maker, {tier.taker_bp:g} bp taker)"
    )
    shown = Table(show_edge=False)
    columns = [
        "execution", "attempts", "fills", "fill_rate", "gross_per_attempt_bp",
        "cost_per_attempt_bp", "net_per_attempt_bp", "net_bp",
    ]  # fmt: skip
    for column in columns:
        shown.add_column(column, justify="left" if column == "execution" else "right")
    for _, row in table.iterrows():
        shown.add_row(*[_number(row.get(c)) for c in columns])
    console.print(shown)
    if result.status == "fail":
        err_console.print(f"[red]{result.detail}[/red]")

    out = output / spec.name / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results.csv", index=False)
    manifest = {
        "plugin": {"kind": spec.kind, "name": spec.name, "origin": spec.origin,
                   "module": spec.module, "sha256": spec.fingerprint()},
        "params": _params(param),
        "data": market.source,
        "fee_tier": fee_tier,
        "train_rows": setup.train_end,
        "trade_rows": int(len(setup.frame) - setup.train_end),
        "decision_rows": int((decisions != 0).sum()),
        "hold_rows": int(strategy.horizon),
        "results": json.loads(pd.DataFrame(table).to_json(orient="records")),
        "version": __version__,
    }  # fmt: skip
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    console.print(f"[dim]-> {out}[/dim]")
    if result.status == "fail":
        raise typer.Exit(code=1)


def _one_omp_thread() -> None:
    """One OpenMP thread unless the caller chose otherwise, before any model library loads.

    The network and a booster in one process can deadlock (``stages.ENSEMBLES``
    keeps the network out of the default ensemble for that reason):
    ``ensemble_all`` hung past 90 s on its fit with the default threads and
    finished in seconds with one.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _executions(value: str) -> list[str]:
    from trading_research.plugins import names

    known = names("execution")
    if value == "all":
        return known
    chosen = [e.strip() for e in value.split(",") if e.strip()]
    unknown = [e for e in chosen if e not in known]
    if unknown:
        raise ValueError(f"unknown execution(s) {unknown}; registered: {known}")
    return chosen


def _number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN: a rate with nothing to divide by
            return "-"
        if value.is_integer() and abs(value) < 1e12:
            return f"{int(value):,}"
        return f"{value:,.3f}" if abs(value) < 1e6 else f"{value:,.0f}"
    return str(value)
