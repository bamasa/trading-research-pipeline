"""``trp bench``: score a strategy on fixed tracks, and the board of every built-in one.

The tracks and the rule are in :mod:`trading_research.bench`; this module only
parses arguments, prints and writes. ``run`` takes any registered strategy or
model, so a plugin of your own (``trp strategies new``) is scored the same way
as the repository's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

bench_app = typer.Typer(
    name="bench",
    help="Benchmark: score a strategy on fixed tracks with a known answer, and the board.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

TracksOption = Annotated[
    str, typer.Option("--tracks", "-t", help="Track names separated by commas, or all.")
]
DataOption = Annotated[
    Path,
    typer.Option("--data", help="Where the recorded tracks' archives are (data/universe ...)."),
]
OutputOption = Annotated[Path, typer.Option("--output", "-o")]


def _tracks(value: str) -> list[Any]:
    from trading_research.bench import iter_tracks

    try:
        return list(iter_tracks(value))
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _print_score(score: Any) -> None:
    from trading_research.bench import FEE_LADDER

    if score.status != "scored":
        console.print(f"  [dim]{score.track:<18} {score.status}: {score.reason[:160]}[/dim]")
        return
    colour = {True: "green", False: "red", None: "yellow"}[score.passed]
    if not score.net_at:
        console.print(f"  {score.track:<18} [{colour}]{score.verdict}[/{colour}]")
        return
    ladder = "  ".join(f"{name.split(' ')[0]} {score.net_at[name]:+.1f}" for name, _ in FEE_LADDER)
    console.print(
        f"  {score.track:<18} [{colour}]{score.verdict}[/{colour}]\n"
        f"  {'':<18} {score.trades:,} trades, gross {score.gross_bp:+.2f}, net "
        f"{score.net_bp:+.2f} bp/trade, t {score.net_t:+.2f}, placebo {score.placebo:.0%}, "
        f"break-even fee {score.breakeven_taker_bp:+.2f} bp/side\n"
        f"  {'':<18} [dim]net at fee: {ladder}[/dim]"
    )


@bench_app.command("list")
def list_cmd(data: DataOption = Path("data")) -> None:
    """The tracks: what each asks, what a good strategy does there, and whether it is here."""
    from trading_research.bench import TRACKS
    from trading_research.bench.tracks import recorded_present, today

    table = Table(show_edge=False)
    table.add_column("track", no_wrap=True)
    table.add_column("kind", no_wrap=True)
    for column in ("question", "a good strategy", "here"):
        table.add_column(column, overflow="fold")
    for track in TRACKS.values():
        if track.kind == "synthetic":
            here = "generated"
        elif track.opens is not None and today() < track.opens:
            here = f"[cyan]sealed until {track.opens.isoformat()}[/cyan]"
        else:
            count = len(recorded_present(track.name, data))
            here = f"{count} instruments" if count >= 6 else "[yellow]not on disk[/yellow]"
        table.add_row(track.name, track.kind, track.question, track.expected, here)
    console.print(table)


@bench_app.command("run")
def run_cmd(
    name: Annotated[str, typer.Argument(help="A registered strategy or model.")],
    kind: Annotated[
        str | None, typer.Option("--kind", "-k", help="strategy or model, if the name is both.")
    ] = None,
    tracks: TracksOption = "all",
    param: Annotated[
        list[str] | None,
        typer.Option("--param", "-p", help="A constructor argument, key=value; repeatable."),
    ] = None,
    data: DataOption = Path("data"),
    output: OutputOption = Path("artifacts/bench"),
) -> None:
    """Score one strategy or model on the tracks; writes OUTPUT/<name>/card.md and card.json."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from trading_research.bench import MarketCache, run, write_card
    from trading_research.cli_plugins import _params, _spec

    spec = _spec(name, kind)
    if spec.kind not in ("strategy", "model"):
        err_console.print(f"[red]{name} is a {spec.kind}; bench takes a strategy or a model[/red]")
        raise typer.Exit(code=2)
    chosen = _tracks(tracks)
    console.print(f"[bold]{spec.name}[/bold] ({spec.kind}, {spec.origin})")
    card = run(spec, _params(param), chosen, MarketCache(data), on_score=_print_score)
    console.print(f"marks [bold]{card.marks}[/bold]  -> {write_card(card, output)}")


@bench_app.command("board")
def board_cmd(
    kind: Annotated[
        str, typer.Option("--kind", "-k", help="strategy, model, or both.")
    ] = "strategy",
    names: Annotated[
        str | None, typer.Option("--names", help="Only these plugins, separated by commas.")
    ] = None,
    tracks: TracksOption = "all",
    data: DataOption = Path("data"),
    output: OutputOption = Path("artifacts/bench"),
    workers: Annotated[
        int, typer.Option("--workers", "-w", min=1, help="Plugins scored at once, a process each.")
    ] = 1,
) -> None:
    """Score every registered plugin of KIND on the tracks; writes OUTPUT/board.json and a card each.

    With --workers above one, each plugin is scored in its own process, which
    builds its own copy of the recorded tracks (about 4 GB each).
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from trading_research.bench import MarketCache, board_rows, run, write_card
    from trading_research.plugins import specs

    kinds = ("strategy", "model") if kind == "both" else (kind,)
    if any(k not in ("strategy", "model") for k in kinds):
        err_console.print("[red]--kind is strategy, model or both[/red]")
        raise typer.Exit(code=2)
    wanted = {n.strip() for n in names.split(",")} if names else None
    chosen = _tracks(tracks)
    todo = [s for k in kinds for s in specs(k) if wanted is None or s.name in wanted]
    cards = []
    if workers == 1:
        cache = MarketCache(data)
        for spec in todo:
            console.print(f"[bold]{spec.name}[/bold] ({spec.kind}, {spec.origin})")
            card = run(spec, {}, chosen, cache, on_score=_print_score)
            write_card(card, output)
            cards.append(card)
            console.print(f"  marks [bold]{card.marks}[/bold]")
    else:
        from concurrent.futures import ProcessPoolExecutor

        names_only = [t.name for t in chosen]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one_card, s.kind, s.name, names_only, str(data)) for s in todo]
            for future in futures:
                card = future.result()
                console.print(f"[bold]{card.name}[/bold] ({card.kind}, {card.origin})")
                for score in card.scores:
                    _print_score(score)
                write_card(card, output)
                cards.append(card)
                console.print(f"  marks [bold]{card.marks}[/bold]")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "board.json"
    payload = {"tracks": [t.name for t in chosen], "rows": board_rows(cards)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"-> {path}")


def _one_card(kind: str, name: str, tracks: list[str], data: str) -> Any:
    """One plugin on the tracks, in a worker process of its own."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from trading_research.bench import TRACKS, MarketCache, run
    from trading_research.plugins import get

    return run(get(name, kind), {}, [TRACKS[t] for t in tracks], MarketCache(Path(data)))


@bench_app.command("fetch")
def fetch_cmd(
    data: DataOption = Path("data"),
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Say what would be fetched, and fetch nothing.")
    ] = False,
    workers: Annotated[int, typer.Option("--workers", "-w", min=1)] = 6,
) -> None:
    """Download the recorded tracks' top-of-book from Bybit's free archives; days on disk are kept."""
    from trading_research.bench import TRACKS
    from trading_research.bench.tracks import (
        RECORDED,
        fitted_days,
        recorded_days,
        today,
        track_symbols,
    )
    from trading_research.data.ensure import DataUnavailable, ensure_universe

    for name, (folder, *_rest) in RECORDED.items():
        days = recorded_days(name)
        opens = TRACKS[name].opens
        if opens is not None and today() < opens:
            # The fitted days ended before the track was registered; only the
            # scored ones are sealed.
            days = fitted_days(name)
            console.print(
                f"[bold]{name}[/bold]: sealed until {opens}; its fitted days only, "
                f"{days[0]} to {days[-1]} -> {data / folder}"
            )
        else:
            console.print(f"[bold]{name}[/bold]: {days[0]} to {days[-1]} -> {data / folder}")
        try:
            ensure_universe(
                track_symbols(name), days[0], days[-1], root=data / folder, workers=workers,
                dry_run=dry_run, on_progress=lambda line: console.print(f"  {line}"),
            )  # fmt: skip
        except DataUnavailable as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    if not dry_run:
        console.print("next: trading-research bench verify")


@bench_app.command("state")
def state_cmd(tracks: TracksOption = "all", data: DataOption = Path("data")) -> None:
    """The index's ten-minute autocorrelation on each panel track's scored span.

    The state the reversion rule needs (about -0.10 where it paid, zero where it
    did not); the forward track's registration is read against it.
    """
    from trading_research.bench.state import index_autocorrelation

    for track in _tracks(tracks):
        if track.kind == "synthetic" and "reversion" not in track.name:
            continue
        try:
            value = index_autocorrelation(track.markets(data))
        except FileNotFoundError as exc:
            console.print(f"  {track.name:<18} [yellow]not here[/yellow]: {str(exc)[:120]}")
            continue
        console.print(f"  {track.name:<18} index ten-minute autocorrelation {value:+.4f}")


@bench_app.command("regimes")
def regimes_cmd(
    kind: Annotated[str, typer.Option("--kind", "-k", help="strategy, model, or both.")] = "both",
    names: Annotated[
        str | None, typer.Option("--names", help="Only these plugins, separated by commas.")
    ] = None,
    data: DataOption = Path("data"),
    output: OutputOption = Path("artifacts/bench"),
    fetch: Annotated[
        bool, typer.Option("--fetch", help="Download the episodes' bars first.")
    ] = False,
) -> None:
    """The regime suite: each strategy across ten dated breaks, and whether the kill switch caught them.

    Writes OUTPUT/regimes.json and OUTPUT/regimes.md. Outside version 1's frozen
    rule; see bench/regimes.py.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from trading_research.bench.regimes import (
        BARS_ROOT_NAME,
        EPISODES,
        PANEL,
        page,
        run_suite,
        summary,
    )
    from trading_research.plugins import specs

    if fetch:
        from trading_research.data.ensure import ensure_bars

        for ep in EPISODES:
            for symbol in PANEL:
                ensure_bars(symbol, "1m", ep.fit_start, ep.last_day, root=data / BARS_ROOT_NAME,
                            market="futures-um", on_progress=None)  # fmt: skip
            console.print(f"  {ep.name}: bars on disk")
    kinds = ("strategy", "model") if kind == "both" else (kind,)
    wanted = {n.strip() for n in names.split(",")} if names else None
    todo = [s for k in kinds for s in specs(k) if wanted is None or s.name in wanted]

    def show(name: str, score: Any) -> None:
        if score.status in ("n/a", "no trades"):
            console.print(
                f"  [dim]{score.episode:<16} {name:<16} {score.status} {score.reason[:80]}[/dim]"
            )
            return
        colour = {"holds": "green", "breaks": "red"}.get(score.status, "yellow")
        stop = (
            "no stop"
            if score.stop_hours_after_break is None
            else f"stop {score.stop_hours_after_break:+.0f} h"
        )
        console.print(
            f"  {score.episode:<16} {name:<16} [{colour}]{score.status:<10}[/{colour}] "
            f"net {score.net_bp_before:+.2f} -> {score.net_bp_after:+.2f} bp, "
            f"{score.equity_left_running_pct:+.1f}% left running, "
            f"{score.equity_kill_switch_pct:+.1f}% with the switch ({stop})"
        )

    results = run_suite(todo, data, on_score=show)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": [{"name": e.name, "breaks": e.breaks.isoformat(), "what": e.what} for e in EPISODES],
        "rows": [{"name": n, "summary": summary(s), "episodes": [x.as_dict() for x in s]}
                 for n, s in results.items()],
    }  # fmt: skip
    (output / "regimes.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "regimes.md").write_text(page(results), encoding="utf-8")
    console.print(f"-> {output / 'regimes.md'}")


FORWARD_OUTPUT = Path("experiments/results/bench_forward_2026_11.json")


@bench_app.command("forward")
def forward_cmd(
    data: DataOption = Path("data"),
    output: OutputOption = FORWARD_OUTPUT,
) -> None:
    """Read the forward track once: the twelve registered strategies, F1 and P1 as registered.

    Refuses before the track opens, when OUTPUT already exists (the read is
    done once, and its result is committed as it comes), and when a strategy's
    source is not the one registered.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from trading_research.bench import TRACKS, MarketCache
    from trading_research.bench.forward import read
    from trading_research.bench.manifest import FORWARD_TRACK
    from trading_research.bench.tracks import today

    track = TRACKS[FORWARD_TRACK]
    if track.opens is not None and today() < track.opens:
        err_console.print(
            f"[cyan]{track.name} is sealed until {track.opens.isoformat()}[/cyan]; nothing was read"
        )
        raise typer.Exit(code=2)
    if output.exists():
        err_console.print(
            f"[red]{output} exists: the forward track is read once[/red]. A defect found "
            "after the read goes in a dated amendment that reports both numbers."
        )
        raise typer.Exit(code=2)
    try:
        payload = read(MarketCache(data), on_score=_print_score)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"data {payload['data_sha256'][:16]}, {payload['instruments']} instruments")
    fitted = payload["fitted_matches_registration"]
    console.print(
        "fitted days " + ("[green]match the registration[/green]" if fitted
                          else "[red]differ from the registration[/red]")
    )  # fmt: skip
    console.print("[bold]F1[/bold]")
    for name, d in payload["f1"].items():
        colour = "green" if d["status"] == "passes" else "red"
        holm = "-" if d["holm_p"] is None else f"{d['holm_p']:.3f}"
        console.print(
            f"  {name:<16} [{colour}]{d['status']}[/{colour}]  Holm p {holm}  {d['verdict']}"
        )
    p1 = payload["p1"]
    net = "-" if p1["net_bp"] is None else f"{p1['net_bp']:+.2f} bp"
    console.print(
        f"[bold]P1[/bold] state {p1['state']:+.4f}, predicted {p1['predicted'] or 'nothing'}, "
        f"reversion net {net}: [bold]{p1['status']}[/bold] {p1['why']}"
    )
    console.print(f"-> {output}; add data_sha256 to the manifest's FINGERPRINTS")


@bench_app.command("verify")
def verify_cmd(tracks: TracksOption = "all", data: DataOption = Path("data")) -> None:
    """Fingerprint each track's data and compare it with version 1's; exits 1 on a difference."""
    from trading_research.bench import FINGERPRINTS, VERSION
    from trading_research.bench.manifest import FORWARD_FIT_FINGERPRINT
    from trading_research.bench.tracks import Sealed, fitted_window

    differs = False
    for track in _tracks(tracks):
        try:
            found = track.markets(data).fingerprint()
        except Sealed:
            try:
                fitted = fitted_window(track.name, data).fingerprint()
            except FileNotFoundError:
                console.print(f"  {track.name:<18} [cyan]sealed[/cyan], fitted days not on disk")
                continue
            if not FORWARD_FIT_FINGERPRINT:
                console.print(
                    f"  {track.name:<18} [cyan]sealed[/cyan]; fitted days not fingerprinted "
                    f"yet: {fitted}"
                )
            elif fitted == FORWARD_FIT_FINGERPRINT:
                console.print(
                    f"  {track.name:<18} [cyan]sealed[/cyan]; fitted days [green]match the "
                    f"registration[/green]  {fitted[:16]}"
                )
            else:
                differs = True
                console.print(
                    f"  {track.name:<18} [cyan]sealed[/cyan]; fitted days [red]differ[/red]  "
                    f"found {fitted[:16]}, expected {FORWARD_FIT_FINGERPRINT[:16]}"
                )
            continue
        except FileNotFoundError as exc:
            console.print(f"  {track.name:<18} [yellow]not on disk[/yellow]: {str(exc)[:120]}")
            continue
        expected = FINGERPRINTS.get(track.name, "")
        if not expected:
            console.print(f"  {track.name:<18} read for the first time; its fingerprint is {found}")
            continue
        if found == expected:
            console.print(
                f"  {track.name:<18} [green]matches version {VERSION}[/green]  {found[:16]}"
            )
        else:
            differs = True
            console.print(
                f"  {track.name:<18} [red]differs from version {VERSION}[/red]  "
                f"found {found[:16]}, expected {expected[:16] or 'none'}"
            )
    if differs:
        raise typer.Exit(code=1)
