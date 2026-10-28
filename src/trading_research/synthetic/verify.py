"""The checks: each component run on planted data and its null, judged against the truth.

Every check returns a :class:`CheckResult` of criteria, each with the planted
value, the recovered value, the tolerance stated before the run and whether it
passed, and writes ``page.md`` and ``result.json`` under
``artifacts/verify/<what>/``. A check passes only when every binding criterion
passes; rows marked ``info`` are measurements reported beside it.

Significance follows the repository's own bar
(:mod:`trading_research.evaluation.significance`): an effect is *found* when its
day-clustered t is at least 3, and *nothing is found* when its absolute t is
below 3. The residual (recovered minus planted, trade by trade) must have an
absolute day-clustered t below 3: a larger one means the measurement is biased,
or the strategy reads something it could not have known.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: The repository's bar for a day-clustered t (``significance.T_BAR``).
T_BAR = 3.0

#: A t on fewer trades or days than this is not read as significant either way:
#: with a handful of days the cluster dispersion is itself a guess.
MIN_TRADES = 30
MIN_DAYS = 5

DEFAULT_OUT = Path("artifacts/verify")


@dataclass(frozen=True)
class Criterion:
    """One row of a check: what was required, what was planted, what came back."""

    check: str
    planted: str
    recovered: str
    tolerance: str
    passed: bool
    binding: bool = True

    @property
    def result(self) -> str:
        if not self.binding:
            return "info"
        return "pass" if self.passed else "FAIL"


@dataclass
class CheckResult:
    """A check's criteria, the seeds and settings it ran with, and its tables."""

    what: str
    title: str
    planted: str
    proves: str
    does_not_prove: str
    command: str
    criteria: list[Criterion] = field(default_factory=list)
    seeds: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria if c.binding)

    @property
    def failures(self) -> list[Criterion]:
        return [c for c in self.criteria if c.binding and not c.passed]

    def add(
        self,
        check: str,
        planted: object,
        recovered: object,
        tolerance: str,
        passed: bool,
        *,
        binding: bool = True,
    ) -> None:
        self.criteria.append(
            Criterion(check, _text(planted), _text(recovered), tolerance, bool(passed), binding)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "title": self.title,
            "passed": self.passed,
            "seconds": round(self.seconds, 1),
            "command": self.command,
            "seeds": self.seeds,
            "settings": self.settings,
            "criteria": [
                {
                    "check": c.check,
                    "planted": c.planted,
                    "recovered": c.recovered,
                    "tolerance": c.tolerance,
                    "result": c.result,
                }
                for c in self.criteria
            ],
        }

    def page(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# verify {self.what}: {self.title}",
            "",
            f"**{status}**: {sum(c.passed for c in self.criteria if c.binding)} of "
            f"{sum(c.binding for c in self.criteria)} binding criteria passed, "
            f"in {self.seconds:.0f} s. Synthetic data: a statement about the code that "
            "read it, never about a market.",
            "",
            f"Reproduce: `{self.command}`",
            "",
            "## What was planted",
            "",
            self.planted,
            "",
            "## Criteria",
            "",
            "| check | planted | recovered | tolerance | result |",
            "|---|---|---|---|---|",
        ]
        for c in self.criteria:
            lines.append(
                f"| {c.check} | {c.planted} | {c.recovered} | {c.tolerance} | {c.result} |"
            )
        lines += ["", "## Seeds and settings", "", "```json"]
        lines.append(json.dumps({"seeds": self.seeds, "settings": self.settings}, indent=2))
        lines += ["```", "", "## What this proves", "", self.proves, ""]
        lines += ["## What it does not prove", "", self.does_not_prove, ""]
        for title, table in self.tables:
            lines += [f"## {title}", "", _markdown(table), ""]
        return "\n".join(lines).rstrip() + "\n"

    def write(self, out: Path | str = DEFAULT_OUT) -> Path:
        directory = Path(out) / self.what
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "page.md").write_text(self.page(), encoding="utf-8")
        payload = {**self.to_dict(), "written_utc": datetime.now(UTC).isoformat()}
        (directory / "result.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return directory / "page.md"


def _text(value: object) -> str:
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.3g}"
    return str(value)


def _markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "(no rows)"
    frame = table.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(_cell)
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _cell(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if float(value).is_integer() and abs(value) < 1e12:
        return str(int(value))
    return f"{value:.3g}"


def _t(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}"


def _bp(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f} bp"


def _finite(value: float) -> bool:
    return bool(math.isfinite(value))


def _enough(summary: dict[str, float]) -> bool:
    """Enough trades and days for a day-clustered t to be read at all."""
    return summary.get("trades", 0) >= MIN_TRADES and summary.get("days_traded", 0) >= MIN_DAYS


# ---------------------------------------------------------------------------
# edge
# ---------------------------------------------------------------------------

#: The repository's strategies run by default, per planted feature: the rules
#: that read the feature, and two learned models offered every feature.
DEFAULT_EDGE_STRATEGIES: dict[str, tuple[str, ...]] = {
    "imbalance": ("order_flow", "spread_capture", "logistic", "lightgbm"),
    "flow": ("order_flow:flow_imbalance_24", "logistic", "lightgbm"),
}


def verify_edge(
    *,
    strategies: Sequence[Any] | None = None,
    features: Sequence[str] = ("imbalance", "flow"),
    size_bp: float = 16.0,
    fee_bp: float = 5.5,
    days: int = 16,
    seed: int = 20200106,
    progress: Callable[[str], None] | None = None,
) -> CheckResult:
    """Each strategy on planted book days and their null, with placebo shifts."""
    from trading_research.backtest.costs import FeeTier
    from trading_research.synthetic.book_days import BookDaysSpec, generate
    from trading_research.synthetic.edge import (
        EdgeSettings,
        day_features,
        recover,
        strategy_name,
    )

    started = time.perf_counter()
    settings = EdgeSettings(fee=FeeTier("stated", min(2.0, fee_bp), fee_bp))
    base = BookDaysSpec()
    spec0 = BookDaysSpec(size_bp=size_bp, days=days, seed=seed)
    result = CheckResult(
        what="edge",
        title="a planted short-horizon edge through features, model or rule, and the taker executor",
        planted=(
            f"Book days (10 levels, {spec0.step_ms} ms, {days} sessions of "
            f"{spec0.session_s} s): about {spec0.episodes_per_hour:g} episodes an hour, "
            f"each moving the expected log mid {size_bp:g} bp over {spec0.horizon_s:g} s "
            f"(drift half-life {spec0.decay_s:g} s) in the direction its feature shows "
            f"for {spec0.signal_s:g} s; unpredictable noise {spec0.noise_bp:g} bp per "
            f"square-root second; spread {spec0.spread_ticks} ticks of about one bp. The "
            "null keeps every episode and draws each drift's direction independently of "
            f"the feature. Taker fee stated at {fee_bp:g} bp a side, latency 10 ms, one "
            "position at a time, hold = the planted horizon. Days split in order: "
            f"{settings.train_share:.0%} train, {settings.validation_share:.0%} choose the "
            "confidence threshold, the rest are traded."
        ),
        proves=(
            "The pipeline, from features to the taker executor, finds a planted drift of "
            "known size, measures what its trades were owed without bias (realised against "
            "the planted drift of the same trades, which also detects a strategy reading "
            "the future), finds nothing significant on the null, and loses the effect when "
            "the decisions are moved to other days."
        ),
        does_not_prove=(
            "That any market carries such a drift, that real features show it this "
            "clearly, or that a strategy found here would clear real costs: the fee is "
            "stated, the book has no queue, and the spread, latency and depth are the "
            "generator's."
        ),
        command=(
            f"trading-research verify edge --size-bp {size_bp:g} --fee-bp {fee_bp:g} "
            f"--days {days} --seed {seed}"
        ),
        seeds={"book_days": seed, "null": seed},
        settings={
            "size_bp": size_bp,
            "fee_taker_bp": fee_bp,
            "days": days,
            "features": list(features),
            "horizon_s": base.horizon_s,
            "decay_s": base.decay_s,
        },
    )
    rows = []
    for feature in features:
        chosen = list(strategies) if strategies else list(DEFAULT_EDGE_STRATEGIES[feature])
        reports = {}
        for null in (False, True):
            spec = BookDaysSpec(feature=feature, size_bp=size_bp, days=days, seed=seed, null=null)  # type: ignore[arg-type]
            if progress:
                progress(f"edge: {feature}, {'null' if null else 'planted'}: generating")
            dataset = generate(spec)
            frames = {day: day_features(dataset, day) for day in dataset.days}
            for strategy in chosen:
                if progress:
                    progress(
                        f"edge: {feature}, {'null' if null else 'planted'}: {strategy_name(strategy)}"
                    )
                reports[strategy_name(strategy), null] = recover(
                    strategy, dataset, settings, frames=frames
                )
        for strategy in chosen:
            name = strategy_name(strategy)
            planted, null_report = reports[name, False], reports[name, True]
            _edge_rows(result, feature, name, size_bp, planted, null_report)
            for report in (planted, null_report):
                rows.append({"feature": feature, **report.as_dict()})
    table = pd.DataFrame(rows)
    keep = [
        "feature", "strategy", "null", "trades", "confidence", "recovered_bp", "recovered_t",
        "planted_bp", "residual_bp", "residual_t", "net_bp", "net_t", "placebo_recovered_bp",
        "placebo_recovered_t", "placebo_max_shift_t",
    ]  # fmt: skip
    result.tables.append(("Per strategy and dataset", table[[c for c in keep if c in table]]))
    result.seconds = time.perf_counter() - started
    return result


def _edge_rows(
    result: CheckResult, feature: str, name: str, size_bp: float, planted: Any, null: Any
) -> None:
    s, n = planted.summary, null.summary
    tag = f"{feature} / {name}"
    t = s.get("recovered_t", float("nan"))
    result.add(
        f"{tag}: planted edge found",
        f"{size_bp:g} bp over the horizon",
        f"{_bp(s.get('recovered_bp', float('nan')))}, t {_t(t)}, {int(s.get('trades', 0))} trades",
        f"day-clustered t >= {T_BAR:g} on >= {MIN_TRADES} trades and {MIN_DAYS} days",
        _enough(s) and _finite(t) and t >= T_BAR,
    )
    owed = s.get("planted_bp", float("nan"))
    result.add(
        f"{tag}: its trades sit on the episodes",
        f"{size_bp:g} bp",
        f"{_bp(owed)} owed per trade ({owed / size_bp:.0%})" if _finite(owed) else "no trades",
        ">= 50% of the planted size owed per trade",
        _finite(owed) and owed >= 0.5 * size_bp,
    )
    residual, se, rt = (
        s.get("residual_bp", float("nan")),
        s.get("residual_se_bp", float("nan")),
        s.get("residual_t", float("nan")),
    )
    result.add(
        f"{tag}: recovered equals planted for the same trades",
        _bp(owed),
        f"{_bp(s.get('recovered_bp', float('nan')))} (residual {_bp(residual)}, t {_t(rt)})",
        f"|residual t| < {T_BAR:g} (3 SE = {_bp(3 * se)})",
        _finite(rt) and abs(rt) < T_BAR,
    )
    p = planted.placebo
    pt, worst = p.get("recovered_t", float("nan")), p.get("max_shift_t", float("nan"))
    shifted = p.get("recovered_bp", float("nan"))
    real = s.get("recovered_bp", float("nan"))
    readable = p.get("trades", 0) >= MIN_TRADES and s.get("test_days", 0) >= MIN_DAYS
    result.add(
        f"{tag}: placebo shifts kill it",
        f"0 bp on decisions moved to other days (owed {_bp(p.get('planted_bp', float('nan')))})",
        f"{_bp(shifted)}, pooled t {_t(pt)}, max shift t {_t(worst)}",
        f"pooled t < {T_BAR:g} (or too few to read), |mean| <= 10% of the real "
        "one, real t above every shift's",
        (not readable or not _finite(pt) or pt < T_BAR)
        and _finite(t)
        and _finite(real)
        and (not _finite(shifted) or abs(shifted) <= 0.1 * abs(real))
        and (not _finite(worst) or t > worst),
    )
    nt = n.get("recovered_t", float("nan"))
    result.add(
        f"{tag}: nothing on the null",
        "0 bp",
        f"{_bp(n.get('recovered_bp', float('nan')))}, t {_t(nt)}, {int(n.get('trades', 0))} trades",
        f"|day-clustered t| < {T_BAR:g}, or under {MIN_TRADES} trades or {MIN_DAYS} days",
        not _enough(n) or not _finite(nt) or abs(nt) < T_BAR,
    )
    nr = n.get("residual_t", float("nan"))
    result.add(
        f"{tag}: no leak on the null",
        _bp(n.get("planted_bp", float("nan"))),
        f"residual {_bp(n.get('residual_bp', float('nan')))}, t {_t(nr)}",
        f"|residual t| < {T_BAR:g}",
        not _finite(nr) or abs(nr) < T_BAR,
    )


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def verify_detector(
    *,
    paths: int = 3,
    seed: int = 20200106,
    step_ms: int = 100,
    vol_ratio: float = 1.6,
    max_false_per_day: float = 0.5,
    changepoint_steps: int = 4000,
    structural_steps: int = 60,
    progress: Callable[[str], None] | None = None,
) -> CheckResult:
    """Both registered detectors on a planted break and on the same days without it."""
    from trading_research.data.grid import to_grid
    from trading_research.market_making import flags
    from trading_research.synthetic.break_days import BreakDaysSpec, generate_breaks
    from trading_research.synthetic.common import day_start_ns

    started = time.perf_counter()
    spec0 = BreakDaysSpec(vol_ratio=vol_ratio).with_book(step_ms=step_ms, seed=seed)
    days = spec0.book.days
    result = CheckResult(
        what="detector",
        title="the regime detectors on a planted break and on its null",
        planted=(
            f"{paths} paths of {days} whole days of book (10 levels, {step_ms} ms) with "
            f"the planted edge of `book_days`; at {spec0.break_at_s / 3600:g} h every part "
            f"of the return becomes {vol_ratio:g} times larger, the spread goes from "
            f"{spec0.book.spread_ticks} to {spec0.spread_ticks_after} ticks and the edge "
            "flips sign. The null is the same draws without the break. The detectors are "
            "the registered ones of the market-making study (`market_making.flags`): "
            "`changepoint.detect` with its defaults on the right-labelled one-second grid "
            "(mid and spread in bp), and `structural_breaks.detect_breaks` on one-minute "
            "log mid returns, history 250, online 60, scale and dependence at 7.77 and 7.65."
        ),
        proves=(
            "How long each detector takes to flag a change of known size and time, and how "
            "often it flags one where none was planted, on data whose only change is the "
            "planted one."
        ),
        does_not_prove=(
            "How the detectors behave on a market, whose changes are gradual, several at "
            "once and of no known time, or that a flag helps a strategy."
        ),
        command=f"trading-research verify detector --paths {paths} --seed {seed} --step-ms {step_ms}",
        seeds={"paths": [seed + k for k in range(paths)]},
        settings={
            "days_per_path": days,
            "break_at_h": spec0.break_at_s / 3600,
            "vol_ratio": vol_ratio,
            "spread_ticks_after": spec0.spread_ticks_after,
            "edge_after": spec0.edge_after,
            "max_false_flags_per_day": max_false_per_day,
            "changepoint_within_rows": changepoint_steps,
            "structural_within_minutes": structural_steps,
        },
    )
    step_s = {"changepoint": 1.0, "structural_breaks": 60.0}
    within = {"changepoint": changepoint_steps, "structural_breaks": structural_steps}
    rows = []
    for k in range(paths):
        for null in (False, True):
            spec = BreakDaysSpec(vol_ratio=vol_ratio, null=null).with_book(
                step_ms=step_ms, seed=seed + k
            )
            if progress:
                progress(f"detector: path {k}, {'null' if null else 'break'}")
            dataset = generate_breaks(spec)
            seconds, minutes = [], []
            for day in dataset.days:
                book = dataset.book[day][["timestamp", "bid_price_0", "ask_price_0"]]
                seconds.append(to_grid(book, 1, label="right"))
                minutes.append(to_grid(book, 60, label="right"))
            span = (day_start_ns(dataset.days[0]), day_start_ns(dataset.days[-1]) + 86_400 * 10**9)
            timeline = flags.flags_from_grids(
                spec.book.symbol,
                pd.concat(seconds, ignore_index=True),
                pd.concat(minutes, ignore_index=True),
                span=span,
            )
            frame = timeline.frame()
            for detector in ("changepoint", "structural_breaks"):
                mine = frame[frame["detector"] == detector]
                # A flag is known at its effective time less the decision latency;
                # one known at or before the break read nothing after it.
                known = mine["effective_ns"].to_numpy(dtype=np.int64) - flags.DECISION_LATENCY_NS
                lag_s = (known - spec.break_ns) / 1e9
                before = int((lag_s <= 0).sum()) if not null else len(mine)
                after = lag_s[lag_s > 0] if not null else np.array([])
                rows.append(
                    {
                        "path": k,
                        "stream": "null" if null else "break",
                        "detector": detector,
                        "flags": len(mine),
                        "false_flags": before,
                        "watched_days": (spec.break_at_s / 86_400) if not null else days,
                        "first_lag_steps": float(after[0] / step_s[detector])
                        if len(after)
                        else float("nan"),
                        "flags_after_first": max(0, len(after) - 1),
                        "statistics": ",".join(sorted(set(mine["statistic"]))),
                    }
                )
    table = pd.DataFrame(rows)
    for detector in ("changepoint", "structural_breaks"):
        mine = table[table["detector"] == detector]
        planted = mine[mine["stream"] == "break"]
        lags = planted["first_lag_steps"].to_numpy(dtype=np.float64)
        unit = "s" if detector == "changepoint" else "min"
        found = np.isfinite(lags) & (lags <= within[detector])
        result.add(
            f"{detector}: finds the planted break",
            f"break at {spec0.break_at_s / 3600:g} h, x{vol_ratio:g} scale, spread x{spec0.spread_ticks_after / spec0.book.spread_ticks:g}",
            f"{int(found.sum())} of {paths} paths; first-flag lags "
            + ", ".join("none" if not np.isfinite(v) else f"{v:.0f} {unit}" for v in lags),
            f"every path within {within[detector]} {unit}",
            bool(found.all()),
        )
        false = int(mine["false_flags"].sum())
        watched = float(mine["watched_days"].sum())
        rate = false / watched if watched else float("nan")
        result.add(
            f"{detector}: false alarms",
            "0 flags without a break",
            f"{false} flags in {watched:.1f} watched days ({rate:.2f} a day)",
            f"<= {max_false_per_day:g} a day (null days and the days before the break)",
            rate <= max_false_per_day,
        )
        repeats = int(planted["flags_after_first"].sum())
        result.add(
            f"{detector}: flags after the first detection",
            "one change",
            f"{repeats} further flags over {paths} paths",
            "measured",
            True,
            binding=False,
        )
    result.tables.append(("Per path, stream and detector", table))
    result.seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------


def verify_simulator(
    *,
    days: int = 4,
    seed: int = 20200106,
    adverse_bp: float = 0.6,
    fill_tolerance: float = 0.02,
    progress: Callable[[str], None] | None = None,
) -> CheckResult:
    """The simulator's markouts and fill rates on planted order flow, against the truth."""
    from trading_research.synthetic.order_flow import OrderFlowSpec, generate
    from trading_research.synthetic.simulator_check import adverse_fills, probe_orders, share

    started = time.perf_counter()
    spec0 = OrderFlowSpec(days=days, seed=seed, adverse_bp=adverse_bp)
    queue_spec = OrderFlowSpec(days=days, seed=seed + 1, adverse_bp=0.0, noise_rate=0.02)
    result = CheckResult(
        what="simulator",
        title="the market-making simulator on order flow with a planted adverse selection",
        planted=(
            f"An order-level book ({days} sessions of {spec0.session_s} s, snapshots every "
            f"{spec0.snapshot_ms} ms at ten levels, spread {spec0.spread_ticks} ticks of "
            f"{spec0.tick_bp:g} bp): limit orders arrive at {spec0.arrival_rate:g} a second per "
            f"level and cancel at {spec0.cancel_rate:g} a second each, uniformly in the "
            f"queue; {spec0.market_rate:g} market orders a second per side. After a print the "
            f"book moves {spec0.move_ticks} tick against its maker with probability "
            f"{spec0.informed_share:.2f}: a planted adverse selection of {adverse_bp:g} bp; "
            f"{spec0.noise_rate:g} noise moves a second. The null plants none and adds the "
            "same number of moves as noise. The queue-stress flow has almost no moves "
            f"({queue_spec.noise_rate:g} a second), so fills come from the queue rather than "
            "from prices moving through a stale order. The truth of every fill is an "
            "order-level replay of the same orders on the order log (`order_flow.true_fills`)."
        ),
        proves=(
            "The simulator's post-fill markouts recover a planted adverse selection and find "
            "none on the null, and its fills, estimated from 100 ms snapshots, match an "
            "order-level replay of the same orders within a stated tolerance, with its "
            "cancellation brackets on either side of the truth."
        ),
        does_not_prove=(
            "That real queues, cancellations or informed flow behave like these Poisson "
            "flows, or that a quoter would earn anything on a market."
        ),
        command=f"trading-research verify simulator --days {days} --seed {seed} --adverse-bp {adverse_bp:g}",
        seeds={"order_flow": seed, "null": seed, "queue_stress": seed + 1},
        settings={
            "adverse_bp": adverse_bp,
            "fill_tolerance_share": fill_tolerance,
            "probe_hold_s": 3.0,
            "queue_probe_hold_s": 10.0,
            "quoter_for_markouts": "S0 (TouchQuoter), clip 1 lot, soft limit 6 clips",
        },
    )
    table_rows = []
    for null in (False, True):
        spec = OrderFlowSpec(days=days, seed=seed, adverse_bp=adverse_bp, null=null)
        if progress:
            progress(f"simulator: {'null' if null else 'planted'} flow")
        dataset = generate(spec)
        fills = pd.concat([adverse_fills(dataset, d) for d in dataset.days], ignore_index=True)
        weights = (fills["size"] * fills["price"]).to_numpy()
        recovered = float(np.average(fills["adverse_bp"], weights=weights))
        planted = float(np.average(fills["planted_bp"], weights=weights))
        split = float(np.average(fills["adverse_split_bp"], weights=weights))
        se = float(fills["adverse_bp"].std(ddof=1) / math.sqrt(len(fills)))
        tolerance = max(3 * se, 0.1)
        label = "null" if null else "planted"
        result.add(
            f"S0 adverse selection, {label} flow",
            _bp(planted),
            f"{_bp(recovered)} over {len(fills)} passive fills",
            f"within {tolerance:.2f} bp (3 SE, at least 0.1)",
            abs(recovered - planted) <= tolerance,
        )
        result.add(
            f"S0 adverse part of the simulator's own split, {label} flow",
            _bp(planted),
            f"{_bp(split)} (spread measured from the last snapshot's mid)",
            "measured: carries the snapshot's staleness, R18",
            True,
            binding=False,
        )
        table_rows.append(
            {
                "flow": label,
                "measure": "S0 fills",
                "fills": len(fills),
                "adverse_bp": recovered,
                "planted_bp": planted,
                "snapshot_split_adverse_bp": split,
                "markout_5s_bp": float(np.average(fills["markout_5s"], weights=weights)),
            }
        )
        if null:
            continue
        probes = pd.concat([probe_orders(dataset, d) for d in dataset.days], ignore_index=True)
        sim, _ = share(probes["sim_hit"])
        true, _ = share(probes["true_hit"])
        result.add(
            "fill rate of held touch orders, planted flow (proportional)",
            f"{true:.3f} filled (order-level replay)",
            f"{sim:.3f} filled, {len(probes)} orders, {float((probes['sim_hit'] == probes['true_hit']).mean()):.1%} agree",
            f"within {fill_tolerance:g}",
            abs(sim - true) <= fill_tolerance,
        )
    shares = {}
    if progress:
        progress("simulator: queue-stress flow")
    queue_data = generate(queue_spec)
    for attribution in ("pessimistic", "proportional", "optimistic"):
        probes = pd.concat(
            [
                probe_orders(queue_data, d, attribution=attribution, hold_s=10.0)
                for d in queue_data.days
            ],
            ignore_index=True,
        )
        shares[attribution] = (share(probes["sim_hit"])[0], *share(probes["true_hit"]), len(probes))
        table_rows.append(
            {
                "flow": "queue stress",
                "measure": f"probe fill share, {attribution}",
                "fills": int(probes["sim_hit"].sum()),
                "orders": len(probes),
                "sim_share": shares[attribution][0],
                "true_share": shares[attribution][1],
                "sim_volume": float(probes["filled"].sum()),
                "true_volume": float(probes["true_filled"].sum()),
            }
        )
    sim, true, _true_se, count = shares["proportional"]
    result.add(
        "fill rate of held touch orders, queue stress (proportional)",
        f"{true:.3f} filled (order-level replay)",
        f"{sim:.3f} filled, {count} orders",
        f"within {fill_tolerance:g}",
        abs(sim - true) <= fill_tolerance,
    )
    low, high = shares["pessimistic"], shares["optimistic"]
    truth_mid = (low[1] + high[1]) / 2
    margin = 2 * max(low[2], high[2])
    result.add(
        "cancellation brackets contain the truth, queue stress",
        f"{truth_mid:.3f} filled",
        f"pessimistic {low[0]:.3f}, optimistic {high[0]:.3f}",
        f"pessimistic <= truth <= optimistic, give or take 2 SE ({margin:.3f})",
        low[0] - margin <= low[1] and high[1] <= high[0] + margin,
    )
    result.tables.append(("Measurements", pd.DataFrame(table_rows)))
    result.seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# reversion
# ---------------------------------------------------------------------------


def verify_reversion(
    *,
    paths: int = 3,
    seed: int = 20200106,
    days: int = 16,
    cost_bp: float = 12.0,
    progress: Callable[[str], None] | None = None,
) -> CheckResult:
    """The relative rule and the frozen index rule on planted reversion, momentum and null."""
    from trading_research.synthetic.cross_section import CrossSectionSpec, Variant, generate
    from trading_research.synthetic.reversion_check import (
        ReversionSettings,
        index_trades,
        relative_trades,
        summary,
    )

    started = time.perf_counter()
    spec0 = CrossSectionSpec(days=days, seed=seed)
    settings = ReversionSettings(cost_bp=cost_bp)
    result = CheckResult(
        what="reversion",
        title="the cross-sectional and index reversion rules on a planted cross-section",
        planted=(
            f"{paths} paths of {spec0.n_instruments} instruments on {spec0.bar_s} s bars over "
            f"{days} days. Each log price is a market and an idiosyncratic random walk "
            f"({spec0.market_vol_bp:g} and {spec0.idio_vol_bp:g} bp a bar) plus transients "
            f"with half-life {spec0.half_life_bars:g} bars ({spec0.half_life_bars * spec0.bar_s / 60:g} "
            f"min): idiosyncratic sd {spec0.size_bp:g} bp, market sd {spec0.market_size_bp:g} bp. "
            "Reversal carries the transient (a move decays back); momentum carries its "
            "complement (a move continues); the null replaces the transient with a random "
            "walk of the same step variance. The rules: `strategies.relative.backtest` "
            "(demeaned, basket 3 a side, lookback and hold one half-life) and the frozen "
            "`strategies.reversion` configuration (index of the others over 120 bars, "
            f"threshold for 60 trades a day set on the first {settings.train_days} days, "
            f"hold and cooldown 120 bars). Costs stated at {cost_bp:g} bp a name and round trip."
        ),
        proves=(
            "The repository's two reversion rules recover a planted reversion of known size "
            "and half-life, lose what a planted momentum takes from them, and find nothing "
            "significant on a null with the same return variance; each measures what its "
            "trades were owed without bias."
        ),
        does_not_prove=(
            "That real instruments revert, at what half-life, or that the effect would "
            "survive costs: §27 found the index effect conditional on a market state, and "
            "nothing here models one."
        ),
        command=f"trading-research verify reversion --paths {paths} --seed {seed} --days {days}",
        seeds={"paths": [seed + k for k in range(paths)]},
        settings={"days": days, "cost_bp": cost_bp, "train_days_index_rule": settings.train_days},
    )
    collected: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for variant, null in (("reversal", False), ("momentum", False), ("reversal", True)):
        label = "null" if null else variant
        for k in range(paths):
            if progress:
                progress(f"reversion: {label}, path {k}")
            kind: Variant = "momentum" if variant == "momentum" else "reversal"
            dataset = generate(CrossSectionSpec(days=days, seed=seed + k, variant=kind, null=null))
            for rule, trades in (
                ("relative", relative_trades(dataset, settings)),
                ("index", index_trades(dataset, settings)),
            ):
                if trades.empty:
                    continue
                trades = trades.copy()
                trades["day"] = [f"{k}:{d}" for d in trades["day"]]
                collected.setdefault((rule, label), []).append(trades)
    rows = []
    for rule in ("relative", "index"):
        series = "symbol" if rule == "index" else None
        for label in ("reversal", "momentum", "null"):
            parts = collected.get((rule, label), [])
            trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            s = summary(trades, series)
            rows.append({"rule": rule, "dataset": label, **s})
            t = s.get("recovered_t", float("nan"))
            owed = s.get("planted_bp", float("nan"))
            recovered = f"{_bp(s.get('recovered_bp', float('nan')))}, t {_t(t)}, {int(s.get('trades', 0))} trades"
            if label == "reversal":
                result.add(f"{rule}: planted reversion found", _bp(owed), recovered,
                           f"day-clustered t >= {T_BAR:g}", _finite(t) and t >= T_BAR)  # fmt: skip
            elif label == "momentum":
                result.add(f"{rule}: planted momentum costs it", _bp(owed), recovered,
                           f"day-clustered t <= -{T_BAR:g}", _finite(t) and t <= -T_BAR)  # fmt: skip
            else:
                result.add(f"{rule}: nothing on the null", "0 bp", recovered,
                           f"|day-clustered t| < {T_BAR:g}", not _finite(t) or abs(t) < T_BAR)  # fmt: skip
            rt = s.get("residual_t", float("nan"))
            result.add(
                f"{rule}: recovered equals planted, {label}",
                _bp(owed),
                f"residual {_bp(s.get('residual_bp', float('nan')))}, t {_t(rt)}",
                f"|residual t| < {T_BAR:g}",
                not _finite(rt) or abs(rt) < T_BAR,
            )
    result.tables.append(
        ("Per rule and dataset (pooled over paths, clustered by path-day)", pd.DataFrame(rows))
    )
    result.seconds = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------

CHECKS: dict[str, Callable[..., CheckResult]] = {
    "edge": verify_edge,
    "detector": verify_detector,
    "simulator": verify_simulator,
    "reversion": verify_reversion,
}

#: ``verify all --quick``: the smallest sizes tried at which every check's verdict,
#: and the components that fail, matched the default run's on the default seed
#: (28 October 2026: edge at 10 and 12 days and the reversion at one or two paths
#: changed a verdict, so they keep 14 days and their defaults). The same checks and
#: tolerances; fewer days and paths, so wider intervals.
QUICK: dict[str, dict[str, Any]] = {
    "edge": {"days": 14},
    "detector": {"paths": 2},
    "simulator": {"days": 2},
    "reversion": {},
}


def summary_page(results: Sequence[CheckResult]) -> str:
    """The table of every check's criteria, for ``verify all``."""
    passed = all(r.passed for r in results)
    lines = [
        "# verify all",
        "",
        f"**{'PASS' if passed else 'FAIL'}**: "
        + ", ".join(f"{r.what} {'pass' if r.passed else 'FAIL'}" for r in results)
        + ". Synthetic data: a statement about the code that read it, never about a market.",
        "",
        "| check | criterion | planted | recovered | tolerance | result |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for c in r.criteria:
            lines.append(
                f"| {r.what} | {c.check} | {c.planted} | {c.recovered} | {c.tolerance} | {c.result} |"
            )
    lines += ["", "Each check's page: " + ", ".join(f"`{r.what}/page.md`" for r in results), ""]
    return "\n".join(lines)


def write_all(results: Sequence[CheckResult], out: Path | str = DEFAULT_OUT) -> Path:
    directory = Path(out) / "all"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "page.md").write_text(summary_page(results), encoding="utf-8")
    payload = {
        "passed": all(r.passed for r in results),
        "checks": [r.to_dict() for r in results],
        "written_utc": datetime.now(UTC).isoformat(),
    }
    (directory / "result.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", "utf-8"
    )
    return directory / "page.md"
