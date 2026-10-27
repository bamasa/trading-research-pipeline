"""The walkthrough end to end on a generated market, in a few minutes: ``trp demo``.

Every step is the one a reader runs by hand, in the same order, on files a
reader can open afterwards:

1. **data**: a synthetic market written in the downloaders' layout
   (:func:`~trading_research.walkthrough.layout.write_synthetic`, the
   generator of ``trp data generate-demo-data``), then checked against its
   contract (``trp data validate-data``);
2. **regimes**: the break detector on each instrument's closes over the
   search block (``trp regimes breaks``);
3. **search**: instruments, frequency, method, model and execution chosen on
   the search block, the search registered before it reads anything
   (``trp search ...``), every ``prepare``, ``select``, ``train`` and
   ``predict`` of the signal inside it;
4. **verdict**: the held-out block read once through the run's ledger, in
   each requested execution mode: the clustered statistic, the kill
   conditions written before the read, the placebos, Holm across the modes and
   the fee ladder; a second read is then asked for and refused;
5. **page**: ``page.md`` with one table per execution mode and what was
   checked, and ``manifest.json`` with every seed and every file's sha256.

``--agent`` adds the research loop through the agent harness (its scripted
client unless a model is configured). Nothing here is evidence about a market:
the generator plants an order-flow signal and the pipeline is expected to find
whatever it planted.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from trading_research import __version__
from trading_research import fees as fee_tables
from trading_research.walkthrough.layout import SyntheticInstrument, write_synthetic
from trading_research.walkthrough.record import ReadRefused, head_commit, sha256_file, write_json

MARKER = "demo_manifest.json"
MODE_TITLES = {"taker": "Taker", "passive": "Passive entry", "maker": "Market maker"}


@dataclass(frozen=True)
class DemoConfig:
    """What a demo generates and how it reads it. ``small`` is the test's set: two
    instruments, six days, two holds."""

    out: Path = Path("artifacts/demo")
    execution: str = "all"
    seed: int = 20260101
    days: int = 10
    first_day: date = date(2031, 3, 3)
    instruments: int = 3
    holds_s: tuple[int, ...] = (30, 60, 120, 300)
    fee: str = "base"
    slippage_bp: float = 0.5
    placebos: int = 40
    agent: bool = False

    @classmethod
    def small(cls, **changes: Any) -> DemoConfig:
        return cls(**{"days": 6, "instruments": 2, "holds_s": (60, 300), "placebos": 20, **changes})

    @property
    def synthetic(self) -> list[SyntheticInstrument]:
        """The generated instruments: the generator's signal strong, weak, absent."""
        strengths = (0.30, 0.15, 0.0)
        return [
            SyntheticInstrument(f"SYN{chr(65 + i)}USDT", self.seed + 1 + i, strengths[i % 3])
            for i in range(self.instruments)
        ]


@dataclass
class DemoResult:
    out: Path
    page: Path
    manifest: Path
    seconds: dict[str, float] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)


def _prepare_out(out: Path) -> None:
    """A fresh output directory; a previous demo's is replaced, anything else refused."""
    if out.exists() and any(out.iterdir()):
        if not (out / MARKER).exists():
            raise FileExistsError(f"{out} holds files that are not a demo's; choose another --out")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / MARKER, {"status": "running"})


def run(config: DemoConfig, report: Callable[[str], None] = print) -> DemoResult:
    """The demo: data, regimes, search, verdict, page."""
    from trading_research.api import Pipeline, PipelineConfig
    from trading_research.walkthrough import steps

    started = time.perf_counter()
    out = config.out
    _prepare_out(out)
    seconds: dict[str, float] = {}

    tick = time.perf_counter()
    report("== data: a synthetic market in the downloaders' layout")
    generated = write_synthetic(
        out / "data", config.synthetic, first_day=config.first_day, days=config.days
    )
    seconds["generate"] = time.perf_counter() - tick

    pipeline = Pipeline(
        PipelineConfig(
            data=out / "data", out=out, execution=config.execution, fee=config.fee,
            slippage_bp=config.slippage_bp, holds_s=config.holds_s, placebos=config.placebos,
            seed=config.seed,
        ),
        report=report,
    )  # fmt: skip
    result = pipeline.run()
    seconds.update(result.seconds)

    tick = time.perf_counter()
    try:
        steps.read(pipeline.search, report=report)
        second = "allowed (this should not happen)"
    except ReadRefused as exc:
        second = f"refused: {exc}"
    report(f"a second read of the held-out block: {second}")

    agent: dict[str, Any] | None = None
    if config.agent:
        from trading_research.walkthrough.agent import demo as agent_demo

        report("== agent: the research loop through the harness")
        tick = time.perf_counter()
        agent = agent_demo(out / "agent")
        seconds["agent"] = time.perf_counter() - tick
    seconds["total"] = time.perf_counter() - started

    page = out / "page.md"
    page.write_text(render(config, pipeline, seconds, second, agent))
    manifest = write_json(
        out / "manifest.json", manifest_of(config, pipeline, generated, seconds, page)
    )
    write_json(out / MARKER, {"status": "complete", "page": str(page), "manifest": str(manifest)})
    statuses = {m: r["status"] for m, r in result.verdict["modes"].items()}
    report(f"page -> {page}  ({seconds['total']:.0f} s)")
    return DemoResult(out, page, manifest, seconds, statuses)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def _bp(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(number) else f"{number:+.2f}"


def _num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(number) else f"{number:.{digits}f}"


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def fee_line(config: DemoConfig) -> str:
    """The sentence the page states the fee in, from the fee table itself."""
    row = fee_tables.tier(config.fee)
    return (
        f"Fee: {row.venue}, tier {row.tier}: maker {row.maker_bp:g} bp, taker {row.taker_bp:g} bp "
        f"a side, slippage {config.slippage_bp:g} bp a crossing leg ({row.source.split(';')[0]}, "
        f"{row.period}; `trp fees`)."
    )


def render(
    config: DemoConfig,
    pipeline: Any,
    seconds: dict[str, float],
    second_read: str,
    agent: dict[str, Any] | None,
) -> str:
    run = pipeline.search
    registration, registration_sha = run.registration()
    chosen = {
        s: _load(run.step_dir(s) / "chosen.json")
        for s in ("instruments", "frequency", "method", "model", "execution", "read")
    }
    results = _load(Path(chosen["read"]["results"]))
    ledger = _load(run.ledger_path)["H"][0]
    days = registration["days"]
    lines = [
        "# The pipeline end to end, on a synthetic market",
        "",
        "> **Synthetic data.** Generated by `trading_research.data.synthetic`, which plants an "
        "order-flow signal; nothing below is evidence about a market. It shows every step running "
        "on files a reader can open, and the discipline holding: the search registered before it "
        "read anything, the held-out block read once through a ledger.",
        "",
        f"Wall time: **{seconds['total']:.0f} s** for execution `{config.execution}` on this machine "
        f"({', '.join(f'{k} {v:.0f} s' for k, v in seconds.items() if k != 'total')}).",
        "",
        f"Data: {config.instruments} instruments by {config.days} days at one-second snapshots, "
        f"seeds {', '.join(str(i.seed) for i in config.synthetic)}; search block {days['search'][0]} "
        f"to {days['search'][-1]} ({len(days['search'])} days), held out {days['held_out'][0]} to "
        f"{days['held_out'][-1]} ({len(days['held_out'])} days).",
        "",
        fee_line(config),
        "",
        "## The commands, in order",
        "",
        "Each step is a command a reader can run alone on the previous step's files:",
        "",
        "```bash",
        f"trp search register --data {config.out}/data --run {config.out}/search --execution {config.execution}"
        "   # from file names, before any day is read",
        f"trp data validate-data --input {config.out}/data/raw",
        f"trp regimes breaks --prices {config.out}/regimes/<SYMBOL>/closes.csv --history 250 --online 60"
        "   # the search block's closes",
        f"trp search instruments --run {config.out}/search",
        f"trp search frequency --run {config.out}/search      # prepare, select, walk-forward fits",
        f"trp search method --run {config.out}/search",
        f"trp search model --run {config.out}/search",
        f"trp search execution --run {config.out}/search      # prepare at the hold, train, predict",
        f"trp search read --run {config.out}/search           # once, through the ledger",
        "```",
        "",
        "## What the search chose, on the search block",
        "",
    ]
    freq = chosen["frequency"]["frequency"]
    peak, pick = freq["peak"], freq["chosen"]
    best = chosen["method"]["method_best"]
    lines += [
        f"- **Instrument**: {chosen['instruments']['symbol']}, first of {len(chosen['instruments']['ranked'])} by edge over cost (`{run.step_dir('instruments').name}/screen.csv`).",
        f"- **Frequency**: hold {pick['hold_s']:.0f} s, label threshold {pick['threshold_bp']:.2f} bp, "
        f"{_bp(pick['neighbourhood_bp'])} bp a trade with its neighbours ({_bp(pick['net_per_trade_bp'])} alone); "
        f"the outright peak, {peak['hold_s']:.0f} s at {peak['threshold_bp']:.2f} bp ({_bp(peak['net_per_trade_bp'])} bp), "
        + ("was declined." if freq["peak_declined"] else "is the neighbourhood's choice too."),
        f"- **Method**: {chosen['method']['method']} (best rule "
        f"{(best.get('rule') or {}).get('model', '—')} {_bp((best.get('rule') or {}).get('net_per_trade_bp'))} bp, "
        f"best model {(best.get('model') or {}).get('model', '—')} {_bp((best.get('model') or {}).get('net_per_trade_bp'))} bp a trade on validation).",
        f"- **Model**: {chosen['model']['model']} {json.dumps(chosen['model']['params'])}.",
        f"- **Execution**: {chosen['execution']['execution']}, by net per attempt on the search block's last days "
        f"(`{run.step_dir('execution').name}/execution.csv`); every choice frozen in `frozen.json`, "
        f"sha256 `{chosen['execution']['frozen_sha256'][:16]}…`.",
        "",
        "## The held-out block, read once",
        "",
        f"The ledger recorded the {ledger['read']} at {ledger['utc']} before the block was opened: "
        f"registration `{ledger['registration_sha256'][:16]}…`, frozen choices `{ledger['frozen_sha256'][:16]}…`, "
        f"results `{ledger.get('results_sha256', '')[:16]}…`. The entry threshold was chosen on the last "
        f"search day ({_num(results['entry_threshold'], 3)}).",
        "",
    ]
    for mode in registration["family"]:
        if mode not in results["modes"]:
            continue
        row = results["modes"][mode]
        lines += [f"### {MODE_TITLES[mode]}", "", "| quantity | value |", "|---|---:|"]
        clustered = row.get("clustered", {})
        for label, value in (
            ("attempts", str(row["attempts"])),
            ("fill rate", _num(row["fill_rate"], 3)),
            ("gross per attempt, bp", _bp(row["gross_per_attempt_bp"])),
            ("cost per attempt, bp", _bp(row["cost_per_attempt_bp"])),
            ("net per attempt, bp", _bp(row["net_per_attempt_bp"])),
            ("days with an attempt", str(row["days"])),
            ("day-level t (clustered by day)", _bp(row["day_t"])),
            ("t counting attempts", _bp(clustered.get("trade_t"))),
            ("one-sided p", _num(row["p_one_sided"], 3)),
            (f"Holm across m = {registration['holm']['m']}", "passes" if row["holm"] else "fails"),
            (
                f"placebos' 95th percentile ({row['placebos']} draws), bp",
                _bp(row["placebo_p95_bp"]),
            ),
            (
                "status",
                f"**{row['status']}**"
                + (f" ({', '.join(row['decided_by'])})" if row["decided_by"] else ""),
            ),
        ):
            lines.append(f"| {label} | {value} |")
        lines += ["", "Fee ladder, the same attempts re-costed (decisions unchanged):", "",
                  "| tier | maker bp | taker bp | net per attempt, bp |", "|---|---:|---:|---:|"]  # fmt: skip
        for rung in row["fee_ladder"]:
            lines.append(
                f"| {rung['tier']} | {rung['maker_bp']:g} | {rung['taker_bp']:g} | {_bp(rung['net_per_attempt_bp'])} |"
            )
        lines.append("")
    for mode, why in results.get("skipped_modes", {}).items():
        lines += [f"### {MODE_TITLES[mode]}", "", f"Not run: {why}", ""]
    stage = results["taker_backtest_stage"]
    lines += [
        "## What was checked",
        "",
        f"- The search was registered before any data was read (`registration.json`, sha256 `{registration_sha[:16]}…`), and every step checked it unchanged.",
        "- No step before the read could open a held-out day: each read a view of the search block's files only (`search/views/search`).",
        "- The kill conditions were written into the registration before the read, in this order: "
        + "; ".join(
            f"{k['name']}: {k['test']} ({k['outcome']})" for k in registration["kill_conditions"]
        )
        + ".",
        f"- Placebos: {registration['placebos']['draws']} shuffled-state shifts of the decisions (seed {registration['placebos']['seed']}), "
        f"{registration['placebos']['maker_draws']} for the market maker, whose every placebo is a simulation.",
        f"- Holm's procedure across the family {registration['family']}, m = {registration['holm']['m']}, fixed before the read.",
        f"- A second read of the held-out block: {second_read}.",
        f"- The taker's file stage (`trp backtest --predictions …`) on the same held-out predictions: "
        f"{stage['trades']:.0f} trades, {_bp(stage['net_per_trade_bp'])} bp net a trade. The stage prices "
        "each trade at the prepared snapshots, the taker above at the five-second grid's prices: the "
        "same decisions, two price paths.",
        "- The data met its contract (`validation.txt`); the break detector flagged "
        + ", ".join(f"{s}: {r['breaks']}" for s, r in pipeline.result.regimes.items())
        + " break(s) on the search block's five-minute closes; the search used no regime guard.",
        f"- Every seed and every file's sha256: `manifest.json`. Package {__version__}.",
        "",
    ]
    if agent is not None:
        verdict = agent.get("verdict") or {}
        lines += [
            "## The research loop through the agent harness",
            "",
            f"Client: {agent['client']}. Status: **{agent['status']}**; the owner's stops: "
            f"{len(agent.get('stops', []))}; refusals the policy made and explained: {len(agent.get('refused', []))}. "
            f"The code's verdict read the held-out block: {verdict.get('read')}, its results the ledger's: "
            f"{verdict.get('ledger_match')}. Run directory: `{Path(agent['run']).relative_to(config.out) if str(agent['run']).startswith(str(config.out)) else agent['run']}`.",
            "",
        ]
    return "\n".join(lines)


def manifest_of(
    config: DemoConfig,
    pipeline: Any,
    generated: dict[str, Any],
    seconds: dict[str, float],
    page: Path,
) -> dict[str, Any]:
    run = pipeline.search
    registration, registration_sha = run.registration()
    read = _load(run.step_dir("read") / "chosen.json")
    files = {}
    for item in generated["instruments"]:
        for name, digest in item["files"].items():
            files[f"data/{name}"] = digest
    for path in (
        run.registration_path,
        run.ledger_path,
        Path(read["frozen"]),
        Path(read["results"]),
        page,
    ):
        files[str(path.relative_to(config.out))] = sha256_file(path)
    return {
        "config": {
            k: (str(v) if isinstance(v, Path | date) else v) for k, v in asdict(config).items()
        },
        "seeds": {
            "instruments": {i.symbol: i.seed for i in config.synthetic},
            "placebos": registration["placebos"]["seed"],
            "models": "each model's own default seed (42 for the learned ones)",
        },
        "fee": asdict(fee_tables.tier(config.fee)),
        "registration_sha256": registration_sha,
        "statuses": read["statuses"],
        "seconds": seconds,
        "files_sha256": files,
        "version": __version__,
        "commit": head_commit(),
    }
