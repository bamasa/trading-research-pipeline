"""``trading-research paper``: the deployment step as paper trading.

Commands::

    paper replay --day <file|dir|date> --model <path|always_hold> --execution taker|passive|maker
    paper run    --venue bybit|binance --market spot|linear --symbol BTCUSDT --model ... --minutes N
    paper record --venue ... --market ... --symbol ... --minutes N
    paper status <run dir>
    paper parity --day <file|dir|date>

Public market data only: no command takes, reads or sends a key, and no order
leaves the process. ``run`` refuses to start if its configuration references
anything that looks like an exchange key.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

paper_app = typer.Typer(
    name="paper",
    help="Paper trading on a public stream or a replayed day. No order is ever sent.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

ModelOption = Annotated[
    str,
    typer.Option(
        "--model",
        "-m",
        help="train's output directory (model.pkl), a .pkl file, or 'always_hold'.",
    ),
]
ExecutionOption = Annotated[str, typer.Option("--execution", "-x", help="taker, passive or maker.")]
FeeTierOption = Annotated[
    str | None, typer.Option("--fee-tier", help="Tier of the instrument's dated fee table.")
]
FeeBpOption = Annotated[
    str | None,
    typer.Option("--fee-bp", help="MAKER,TAKER in bp: a fee you state, recorded as not sourced."),
]
FeeAsOfOption = Annotated[
    str | None,
    typer.Option(
        "--fee-asof",
        help="Look every fill's fee up at this UTC date instead of its own instant "
        "(a synthetic or older day the table does not reach).",
    ),
]


def _settings(**kwargs: Any) -> Any:
    from trading_research.live.decide import ExecutionSettings

    return ExecutionSettings(**kwargs)


def _schedule(
    venue: str, market: str, symbol: str, tier: str | None, fee_bp: str | None, fee_asof: str | None
) -> Any:
    import pandas as pd

    from trading_research.live.fees import FeeNotSourced, schedule_for

    given = None
    if fee_bp:
        try:
            maker, taker = (float(x) for x in fee_bp.split(","))
        except ValueError as exc:
            err_console.print("[red]--fee-bp is MAKER,TAKER in basis points, e.g. 2,5.5[/red]")
            raise typer.Exit(code=2) from exc
        given = (maker, taker)
    as_of = int(pd.Timestamp(fee_asof, tz="UTC").value) if fee_asof else None
    try:
        return schedule_for(venue, market, symbol, tier, given=given, as_of_ns=as_of)
    except FeeNotSourced as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


def _run_dir(
    given: Path | None, venue: str, market: str, symbol: str, mode: str, kind: str
) -> Path:
    if given is not None:
        return given
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs/paper") / f"{stamp}-{kind}-{venue}-{market}-{symbol}-{mode}"


def _session(
    *,
    venue: str,
    market: str,
    symbol: str,
    live: bool,
    model: str,
    execution: str,
    schedule: Any,
    clip: float,
    latency_ms: float,
    hold_s: float,
    min_confidence: float,
    every: int,
    max_position_clips: float,
    max_loss: float,
    stale_after_s: float,
    kill_on_gap: bool,
    quoter: str,
    run_dir: Path,
) -> Any:
    from trading_research.live.decide import load_model
    from trading_research.live.ledger import Limits, PaperLedger
    from trading_research.live.session import Session, SessionConfig

    try:
        loaded = load_model(model)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    maker_bp = 0.0
    if execution == "maker" and quoter == "skew":
        maker_bp = schedule.quote(schedule.as_of_ns or time.time_ns()).maker_bp
    settings = _settings(
        mode=execution,
        clip=clip,
        latency_ns=int(latency_ms * 1e6),
        cancel_latency_ns=int(latency_ms * 1e6),
        hold_ns=int(hold_s * 1e9),
        quoter=quoter,
        maker_bp=maker_bp,
    )
    limits = Limits(
        max_position=clip * max_position_clips,
        max_loss=max_loss,
        stale_after_s=stale_after_s,
        kill_on_gap=kill_on_gap,
    )
    config = SessionConfig(
        venue=venue,
        market=market,
        symbol=symbol,
        live=live,
        execution=settings,
        limits=limits,
        min_confidence=min_confidence,
        every=every,
        run_dir=run_dir,
    )
    return Session(config, loaded, PaperLedger(schedule, limits))


def _report(page: Path, summary: dict[str, Any]) -> None:
    pnl = summary["pnl"]
    table = Table(title="Paper run", title_justify="left", show_edge=False)
    table.add_column("")
    table.add_column("value", justify="right")
    for label, value in (
        ("rows / prints", f"{summary['rows']:,} / {summary['prints']:,}"),
        ("boundaries", f"{summary['boundaries']:,}"),
        ("decisions buy / sell", f"{summary['decisions_buy']} / {summary['decisions_sell']}"),
        ("fills", f"{pnl['fills']}"),
        ("PnL gross", f"{pnl['gross']:+,.4f}"),
        ("fees", f"{pnl['fees']:,.4f}"),
        ("PnL net", f"{pnl['net']:+,.4f}"),
        ("gaps", f"{summary['gaps']}"),
        ("kill switch", f"{pnl['killed'] or 'not tripped'}"),
    ):
        table.add_row(label, value)
    console.print(table)
    console.print(f"page -> {page}")


@paper_app.command("replay")
def replay_cmd(
    day: Annotated[
        str,
        typer.Option(
            "--day",
            help="A raw log, a Bybit archive (.zip/.data), a dataset directory, a staged "
            "day parquet, or a date of a recording (with --venue/--market/--symbol).",
        ),
    ],
    model: ModelOption = "always_hold",
    execution: ExecutionOption = "taker",
    fee_tier: FeeTierOption = None,
    fee_bp: FeeBpOption = None,
    fee_asof: FeeAsOfOption = None,
    venue: Annotated[str, typer.Option("--venue", help="Venue whose fee table applies.")] = "bybit",
    market: Annotated[str, typer.Option("--market", help="spot or linear.")] = "linear",
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    prints: Annotated[
        Path | None, typer.Option("--prints", help="Prints for a Bybit archive day.")
    ] = None,
    root: Annotated[Path, typer.Option("--root", help="Where recordings live.")] = Path(
        "data/paper"
    ),
    speed: Annotated[
        float, typer.Option("--speed", help="0: as fast as possible; 1: wall clock; 10: 10x.")
    ] = 0.0,
    clip: Annotated[float, typer.Option("--clip", help="Order size, base units.")] = 0.001,
    latency_ms: Annotated[float, typer.Option("--latency-ms")] = 10.0,
    hold_s: Annotated[float, typer.Option("--hold-s")] = 120.0,
    min_confidence: Annotated[float, typer.Option("--min-confidence")] = 0.5,
    every: Annotated[int, typer.Option("--every", min=1, help="prepare's subsample.")] = 50,
    max_position_clips: Annotated[float, typer.Option("--max-position-clips")] = 10.0,
    max_loss: Annotated[float, typer.Option("--max-loss", help="Quote currency.")] = 100.0,
    stale_after_s: Annotated[float, typer.Option("--stale-after-s")] = 5.0,
    kill_on_gap: Annotated[bool, typer.Option("--kill-on-gap/--no-kill-on-gap")] = True,
    quoter: Annotated[str, typer.Option("--quoter", help="Maker only: touch or skew.")] = "touch",
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Replay a stored day through the paper loop."""
    from trading_research.live.page import write_run
    from trading_research.live.replay import ReplayError, open_day, paced

    try:
        stored = open_day(day, venue=venue, market=market, symbol=symbol, root=root, prints=prints)
    except ReplayError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    name = (symbol or stored.symbol or "INSTRUMENT").upper()
    schedule = _schedule(venue, market, name, fee_tier, fee_bp, fee_asof)
    if stored.events:
        from trading_research.live.fees import FeeNotSourced

        try:
            schedule.quote(stored.events[0].ts_ns)
        except FeeNotSourced as exc:
            err_console.print(
                f"[red]{exc}[/red]\nFor a day the table does not reach, pass --fee-asof "
                "<date the table covers> or --fee-bp MAKER,TAKER."
            )
            raise typer.Exit(code=2) from exc
    target = _run_dir(run_dir, venue, market, name, execution, "replay")
    session = _session(
        venue=venue, market=market, symbol=name, live=False, model=model, execution=execution,
        schedule=schedule, clip=clip, latency_ms=latency_ms, hold_s=hold_s,
        min_confidence=min_confidence, every=every, max_position_clips=max_position_clips,
        max_loss=max_loss, stale_after_s=stale_after_s, kill_on_gap=kill_on_gap,
        quoter=quoter, run_dir=target,
    )  # fmt: skip
    summary = session.run(paced(stored.events, speed))
    source: dict[str, Any] = {
        "kind": stored.kind,
        "path": str(stored.path),
        "stamps": stored.stamps,
    }
    manifest_path = stored.path / "manifest.json" if stored.path.is_dir() else None
    if manifest_path is not None and manifest_path.exists():
        dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
        source["seed"] = (dataset.get("config") or {}).get("seed")
        source["warning"] = dataset.get("warning")
    page = write_run(session, summary, command=sys.argv[1:], source=source)
    _report(page, summary)


@paper_app.command("run")
def run_cmd(
    venue: Annotated[str, typer.Option("--venue", help="bybit or binance.")],
    market: Annotated[str, typer.Option("--market", help="spot or linear.")],
    symbol: Annotated[str, typer.Option("--symbol")],
    minutes: Annotated[float, typer.Option("--minutes", min=0.05, help="How long to run.")],
    model: ModelOption = "always_hold",
    execution: ExecutionOption = "taker",
    fee_tier: FeeTierOption = None,
    fee_bp: FeeBpOption = None,
    clip: Annotated[float, typer.Option("--clip", help="Order size, base units.")] = 0.001,
    latency_ms: Annotated[float, typer.Option("--latency-ms")] = 10.0,
    hold_s: Annotated[float, typer.Option("--hold-s")] = 120.0,
    min_confidence: Annotated[float, typer.Option("--min-confidence")] = 0.5,
    every: Annotated[int, typer.Option("--every", min=1, help="prepare's subsample.")] = 50,
    max_position_clips: Annotated[float, typer.Option("--max-position-clips")] = 10.0,
    max_loss: Annotated[float, typer.Option("--max-loss", help="Quote currency.")] = 100.0,
    stale_after_s: Annotated[float, typer.Option("--stale-after-s")] = 5.0,
    kill_on_gap: Annotated[bool, typer.Option("--kill-on-gap/--no-kill-on-gap")] = True,
    quoter: Annotated[str, typer.Option("--quoter", help="Maker only: touch or skew.")] = "touch",
    record: Annotated[
        bool, typer.Option("--record/--no-record", help="Keep the raw log for replay.")
    ] = True,
    root: Annotated[Path, typer.Option("--root", help="Where recordings go.")] = Path("data/paper"),
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
) -> None:
    """Paper-trade a public stream for N minutes. No key, no order."""
    from trading_research.live.guard import KeyGuardRefusal, refuse_keys
    from trading_research.live.page import write_run
    from trading_research.live.record import RawRecorder
    from trading_research.live.stream import LiveStream, connections

    symbol = symbol.upper()
    configuration = {
        "venue": venue, "market": market, "symbol": symbol, "model": model,
        "execution": execution, "fee_tier": fee_tier, "fee_bp": fee_bp, "root": str(root),
        "run_dir": str(run_dir) if run_dir else None, "quoter": quoter,
    }  # fmt: skip
    try:
        plan = connections(venue, market, symbol)
        refuse_keys(configuration, [c.url for c in plan])
    except (KeyGuardRefusal, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    schedule = _schedule(venue, market, symbol, fee_tier, fee_bp, None)
    target = _run_dir(run_dir, venue, market, symbol, execution, "live")
    session = _session(
        venue=venue, market=market, symbol=symbol, live=True, model=model, execution=execution,
        schedule=schedule, clip=clip, latency_ms=latency_ms, hold_s=hold_s,
        min_confidence=min_confidence, every=every, max_position_clips=max_position_clips,
        max_loss=max_loss, stale_after_s=stale_after_s, kill_on_gap=kill_on_gap,
        quoter=quoter, run_dir=target,
    )  # fmt: skip
    recorder = RawRecorder(root, venue, market, symbol) if record else None
    stream = LiveStream(venue, market, symbol, on_raw=recorder)
    until = time.time_ns() + int(minutes * 60e9)
    console.print(f"paper run on {venue} {market} {symbol} for {minutes:g} min -> {target}")
    try:
        summary = session.run(stream.events(until_ns=until))
    finally:
        if recorder is not None:
            for gap in stream.gaps:
                if gap.kind == "reconnect":
                    recorder.note(gap.recv_ns, "reconnect", gap.detail)
            recorder.close()
    summary["stream"] = {
        "messages": stream.stats.messages,
        "reconnects": stream.stats.reconnects,
        "resyncs": stream.stats.resyncs,
        "bytes": stream.stats.bytes_received,
        "messages_per_s": stream.stats.messages
        / max((stream.stats.last_message_ns - stream.stats.first_message_ns) / 1e9, 1e-9),
    }
    source: dict[str, Any] = {"kind": "live", "urls": [c.url for c in plan]}
    if recorder is not None:
        source["raw_logs"] = sorted(
            str(Path(root) / f"{venue}-{market}" / symbol / "raw" / f"{d}.jsonl.gz")
            for d in recorder.written
        )
    page = write_run(session, summary, command=sys.argv[1:], source=source)
    _report(page, summary)


@paper_app.command("record")
def record_cmd(
    venue: Annotated[str, typer.Option("--venue", help="bybit or binance.")],
    market: Annotated[str, typer.Option("--market", help="spot or linear.")],
    symbol: Annotated[str, typer.Option("--symbol")],
    minutes: Annotated[float, typer.Option("--minutes", min=0.01)],
    root: Annotated[Path, typer.Option("--root", help="Where recordings go.")] = Path("data/paper"),
    export: Annotated[
        bool, typer.Option("--export/--no-export", help="Also write the archive layouts.")
    ] = True,
) -> None:
    """Record a public stream to day files for replay and the batch pipeline."""
    from trading_research.live.events import BookRow, Gap, Print
    from trading_research.live.guard import KeyGuardRefusal, refuse_keys
    from trading_research.live.record import RawRecorder, export_day, raw_log_path
    from trading_research.live.stream import LiveStream, connections

    symbol = symbol.upper()
    try:
        refuse_keys(
            {"venue": venue, "market": market, "symbol": symbol, "root": str(root)},
            [c.url for c in connections(venue, market, symbol)],
        )
    except (KeyGuardRefusal, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    recorder = RawRecorder(root, venue, market, symbol)
    stream = LiveStream(venue, market, symbol, on_raw=recorder)
    counts = {"rows": 0, "prints": 0, "gaps": 0}
    until = time.time_ns() + int(minutes * 60e9)
    try:
        for event in stream.events(until_ns=until):
            if isinstance(event, BookRow):
                counts["rows"] += 1
            elif isinstance(event, Print):
                counts["prints"] += 1
            elif isinstance(event, Gap):
                counts["gaps"] += 1
                if event.kind == "reconnect":
                    recorder.note(event.recv_ns, "reconnect", event.detail)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
    console.print(
        f"recorded {stream.stats.messages:,} messages: {counts['rows']:,} rows, "
        f"{counts['prints']:,} prints, {counts['gaps']} gaps, {stream.stats.reconnects} reconnects"
    )
    for day in recorder.days():
        path = raw_log_path(root, venue, market, symbol, day)
        console.print(f"raw log -> {path}")
        if export:
            for what, written in export_day(path, venue, market, symbol, root).items():
                console.print(f"{what} -> {written}")


@paper_app.command("status")
def status_cmd(
    run_dir: Annotated[Path, typer.Argument(help="A run directory under runs/paper.")],
) -> None:
    """Where a run stands: running or finished, its PnL, position, gaps and kill switch."""
    status_path = run_dir / "status.json"
    summary_path = run_dir / "summary.json"
    if not status_path.exists() and not summary_path.exists():
        err_console.print(f"[red]{run_dir} is not a paper run directory[/red]")
        raise typer.Exit(code=2)
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    table = Table(title=f"{run_dir.name}", title_justify="left", show_edge=False)
    table.add_column("")
    table.add_column("value", justify="right")
    for key in (
        "state",
        "rows",
        "prints",
        "boundaries",
        "gaps",
        "position",
        "gross",
        "fees",
        "net",
        "killed",
    ):
        if key in status:
            value = status[key]
            table.add_row(key, f"{value:+,.4f}" if isinstance(value, float) else str(value))
    if "updated_ns" in status:
        age = (time.time_ns() - int(status["updated_ns"])) / 1e9
        table.add_row("updated", f"{age:,.0f} s ago")
    console.print(table)
    if (run_dir / "page.md").exists():
        console.print(f"page -> {run_dir / 'page.md'}")


@paper_app.command("parity")
def parity_cmd(
    day: Annotated[str, typer.Option("--day", help="A stored day, as for replay.")],
    venue: Annotated[str | None, typer.Option("--venue")] = None,
    market: Annotated[str | None, typer.Option("--market")] = None,
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    root: Annotated[Path, typer.Option("--root")] = Path("data/paper"),
    subsample: Annotated[int, typer.Option("--subsample", min=1, help="prepare's.")] = 50,
    every: Annotated[int, typer.Option("--every", min=1, help="Registry/timed boundaries.")] = 1,
    sets: Annotated[str, typer.Option("--sets")] = "registry,timed,prepared",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="CSV of every feature.")
    ] = None,
) -> None:
    """Streaming against batch features on a stored day: the largest deviation per feature."""
    from trading_research.live.parity import parity
    from trading_research.live.replay import ReplayError, open_day

    try:
        stored = open_day(day, venue=venue, market=market, symbol=symbol, root=root)
    except ReplayError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    report = parity(stored, tuple(sets.split(",")), subsample=subsample, every=every)
    frame = report.table
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    table = Table(title="Parity", title_justify="left", show_edge=False)
    for column in ("set", "features", "boundaries", "max |dev|", "max rel dev", "outside tol"):
        table.add_column(column, justify="right")
    for name, group in frame.groupby("set", sort=False):
        table.add_row(
            str(name),
            str(len(group)),
            f"{int(group['boundaries'].max()):,}",
            f"{group['max_abs_dev'].max():.2e}",
            f"{group['max_rel_dev'].max():.2e}",
            ", ".join(group.loc[~group["within_tolerance"], "feature"]) or "none",
        )
    console.print(table)
    if not report.ok:
        raise typer.Exit(code=1)
