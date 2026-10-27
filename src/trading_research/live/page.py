"""The run's page: what was decided, what filled, what it made, and how the loop behaved.

A run directory holds, when a session ends:

* ``page.md``: the page a reader opens first;
* ``summary.json``: every number on the page;
* ``manifest.json``: what produced it: the model file's sha256, the fee table
  and the quote statuses the fills paid, the code version, the command, and
  the seeds (a paper session draws no random numbers; a replayed synthetic
  day's generator seed is copied from its dataset manifest);
* ``decisions.csv``, ``orders.csv``, ``fills.csv``, ``gaps.csv``, and
  ``marks.parquet`` (a mark at every row) and, for the maker, ``quotes.parquet``.

Nothing in a run directory is committed: ``runs/`` is ignored by git.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research import __version__
from trading_research.live.session import Session

WARNING = (
    "Paper trading. No order was sent anywhere: every order on this page is a row in a "
    "table, filled against public prints by the repository's rules."
)


def code_version() -> dict[str, Any]:
    """The commit the code ran from and whether the tree differed from it."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _iso(ns: int | None) -> str:
    return "" if ns is None else pd.Timestamp(int(ns), tz="UTC").isoformat()


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def write_run(
    session: Session,
    summary: dict[str, Any],
    *,
    command: list[str],
    source: dict[str, Any],
) -> Path:
    """Write the tables, the manifest, the summary and the page; returns the page's path."""
    run_dir = session.config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ex = session.execution
    session.decisions_frame().to_csv(run_dir / "decisions.csv", index=False)
    session.gaps_frame().to_csv(run_dir / "gaps.csv", index=False)
    session.ledger.fills_frame().to_csv(run_dir / "fills.csv", index=False)
    session.ledger.marks_frame().to_parquet(run_dir / "marks.parquet", index=False)
    if ex is not None:
        ex.orders_frame().to_csv(run_dir / "orders.csv", index=False)
        if ex.quotes:
            ex.quotes_frame().to_parquet(run_dir / "quotes.parquet", index=False)
    config = session.config
    model = session.model
    manifest = {
        "warning": WARNING,
        "trading_research_version": __version__,
        "code": code_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "command": command,
        "venue": config.venue,
        "market": config.market,
        "symbol": config.symbol,
        "live": config.live,
        "source": source,
        "model": {
            "name": model.name,
            "path": model.path,
            "sha256": model.sha256,
            "features": model.features,
            "feature_set": session.feature_set.name if session.feature_set else None,
            "min_confidence": config.min_confidence,
            "every_rows": config.every,
        },
        "execution": {
            k: getattr(config.execution, k) for k in config.execution.__dataclass_fields__
        },
        "limits": {k: getattr(config.limits, k) for k in config.limits.__dataclass_fields__},
        "fees": session.ledger.fees.describe(),
        "fee_statuses_paid": summary["pnl"]["fee_statuses"],
        "seeds": {
            "session": None,
            "note": "a paper session draws no random numbers",
            "dataset_seed": source.get("seed"),
        },
        "files": {},
    }
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name not in ("manifest.json", "page.md", "status.json"):
            manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    page = render(session, summary, manifest)
    target = run_dir / "page.md"
    target.write_text(page, encoding="utf-8")
    session.write_status("finished")
    return target


def render(session: Session, summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    config = session.config
    pnl = summary["pnl"]
    fees = manifest["fees"]
    lines = [
        f"# Paper run: {config.venue} {config.market} {config.symbol}, {config.execution.mode}",
        "",
        f"> {WARNING}",
        "",
        f"- source: {manifest['source'].get('kind')} "
        f"({'live public stream' if config.live else 'replayed stored day'})",
        f"- span: {_iso(session.first_ns)} to {_iso(session.last_ns)} "
        f"({summary['span_s']:,.1f} s of data, {summary['wall_s']:,.1f} s of wall clock)",
        f"- model: `{manifest['model']['name']}`, sha256 `{manifest['model']['sha256'][:16]}...`, "
        f"{len(manifest['model']['features'])} features ({manifest['model']['feature_set']}), "
        f"min confidence {config.min_confidence}, a boundary every {config.every} rows",
        f"- fees: tier `{fees['tier']}`; "
        + (
            "fills paid at " + ", ".join(f"{k} x{v}" for k, v in pnl["fee_statuses"].items())
            if pnl["fee_statuses"]
            else "no fill"
        ),
        f"- code: {manifest['code'].get('commit')} (dirty: {manifest['code'].get('dirty')})",
        "",
        "## Money",
        "",
        "| | value |",
        "|---|---:|",
        f"| PnL gross (before fees), quote | {_fmt(pnl['gross'])} |",
        f"| fees paid, quote | {_fmt(pnl['fees'])} |",
        f"| PnL net at the fee used, quote | {_fmt(pnl['net'])} |",
        f"| gross / net per turnover, bp | {_fmt(pnl['gross_bp_turnover'], 2)} / "
        f"{_fmt(pnl['net_bp_turnover'], 2)} |",
        f"| turnover, quote | {_fmt(pnl['turnover'], 2)} |",
        f"| fills | {pnl['fills']} |",
        f"| position at the end, base | {_fmt(pnl['position'], 6)} |",
        f"| max drawdown (net), quote | {_fmt(pnl['max_drawdown'])} |",
        f"| orders refused by the position limit | {pnl['refused_by_limit']} |",
        f"| kill switch | {pnl['killed'] or 'not tripped'} "
        f"{'at ' + _iso(pnl['killed_ns']) if pnl['killed_ns'] else ''} |",
        "",
        "## Decisions and fills",
        "",
        f"- rows {summary['rows']:,}, prints {summary['prints']:,}, boundaries {summary['boundaries']:,}",
        f"- decisions: buy {summary['decisions_buy']}, sell {summary['decisions_sell']}; outcomes "
        + ", ".join(f"{k} {v}" for k, v in summary["decision_outcomes"].items()),
        "- fill rate: "
        + (
            ", ".join(f"{k} {_fmt(v, 3)}" for k, v in summary["fill_rate"].items())
            or "no order sent"
        ),
        "- execution counters: "
        + (", ".join(f"{k} {v}" for k, v in summary["execution_counts"].items()) or "none"),
    ]
    fills = session.ledger.fills_frame()
    if len(fills):
        lines += [
            "",
            "First fills:",
            "",
            "| time | side | price | size | maker | fee bp | reason | path |",
        ]
        lines += ["|---|---:|---:|---:|---|---:|---|---|"]
        for record in fills.head(20).to_dict("records"):
            lines.append(
                f"| {_iso(int(record['ts_ns']))} | {int(record['side']):+d} | "
                f"{record['price']:,.6g} | {record['size']:g} | {record['maker']} | "
                f"{record['fee_bp']:g} | {record['reason']} | {record['path']} |"
            )
    lines += ["", "## The loop", ""]
    for name, stats in summary["latency"].items():
        if stats:
            lines.append(
                f"- {name}: p50 {stats['p50']:,.2f}, p90 {stats['p90']:,.2f}, "
                f"p99 {stats['p99']:,.2f}, max {stats['max']:,.2f} (n {stats['n']:,})"
            )
    if summary["span_s"] > 0:
        lines.append(
            f"- events a second: {(summary['rows'] + summary['prints']) / summary['span_s']:,.1f} "
            f"(rows {summary['rows'] / summary['span_s']:,.1f}, prints "
            f"{summary['prints'] / summary['span_s']:,.1f})"
        )
    lines.append(
        f"- gaps: {summary['gaps']}"
        + (
            " (" + ", ".join(f"{k} {v}" for k, v in summary["gap_kinds"].items()) + ")"
            if summary["gaps"]
            else ""
        )
    )
    lines.append(
        "- other counters: "
        + (", ".join(f"{k} {v}" for k, v in summary["counts"].items()) or "none")
    )
    lines += [
        "",
        "## Manifest",
        "",
        "`manifest.json` holds the model's sha256, the fee table with its sources, the seeds, "
        "the command and the sha256 of every table in this directory.",
        "",
    ]
    return "\n".join(lines)
