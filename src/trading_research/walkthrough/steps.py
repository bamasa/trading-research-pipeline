"""The search, one step at a time: instruments, frequency, method, model, execution, then the read.

Each step reads the previous step's ``chosen.json`` (and the data its view
allows), writes its own directory under the run, and can be run alone::

    trp search instruments --data data/demo --run artifacts/search
    trp search frequency --run artifacts/search
    trp search method --run artifacts/search
    trp search model --run artifacts/search
    trp search execution --run artifacts/search
    trp search read --run artifacts/search        # once, through the ledger

1. **instruments**: the search is registered (fee, grids, methods, execution
   family, kills, placebos, Holm's m) before a byte of data is read; then
   every instrument is screened on the search block alone
   (:func:`trading_research.data.screen.screen_frame`, ranked by edge over
   cost) and the first is taken. ``--discover`` also lists what the venue
   trades today (:func:`trading_research.data.discover.candidate_table`).
2. **frequency**: the hold and the label's threshold, a grid scored by a
   walk-forward on the search block's validation days with the baseline
   model, chosen by ``reproduce``'s neighbourhood rule; the outright peak is
   reported and, unless it is the neighbourhood's choice, declined.
3. **method**: every rule against every learned model at that frequency: does
   learning earn its place?
4. **model**: the chosen family's candidates, a learned model's parameters
   included.
5. **execution**: the chosen signal fitted once (``train``) and scored
   (``predict``) on the search block, executed as a taker, a passive entry and
   a market maker; the mode chosen by net per attempt; every choice frozen in
   ``frozen.json``.
6. **read**: the held-out block, opened once through the run's ledger, read
   with everything frozen, the verdict computed as the registration says.

A step that reads data reads a view (:func:`~trading_research.walkthrough.layout.view`):
until the read, no held-out day is in any directory the search can open.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research import __version__
from trading_research.backtest.costs import BYBIT_G1_EXAMPLES, fee_tier
from trading_research.walkthrough import execution as ex
from trading_research.walkthrough import signal as sg
from trading_research.walkthrough import verdict as vd
from trading_research.walkthrough.layout import Dataset, protected_overlap, split_days, view
from trading_research.walkthrough.record import (
    ReadRefused,
    SearchError,
    SearchRun,
    sha256_file,
    write_json,
)

Report = Callable[[str], None]
FAMILY: tuple[str, ...] = ("taker", "passive", "maker")
GRID_SECONDS = 5


def _quiet(_: str) -> None:
    return None


def _step_seconds(dataset: Dataset, symbol: str, day: date) -> float:
    stamps = pd.read_parquet(
        dataset.raw / symbol / "bookTicker" / f"{day}.parquet", columns=["timestamp"]
    )
    return float(stamps["timestamp"].diff().dt.total_seconds().median())


def _days(values: Sequence[str]) -> list[date]:
    return [date.fromisoformat(v) for v in values]


# ---------------------------------------------------------------------------
# 1. instruments
# ---------------------------------------------------------------------------


def register(
    run: SearchRun,
    dataset: Dataset,
    *,
    symbols: Sequence[str] | None = None,
    horizon_s: float = 60.0,
    search_share: float = 0.65,
    fee: str = "base",
    slippage_bp: float = 0.5,
    family: Sequence[str] = FAMILY,
    holds_s: Sequence[int] = (30, 60, 120, 300),
    thresholds: Sequence[float] = (0.0, 0.5, 1.0),
    minimum_trades: int = 30,
    min_days: int = 2,
    placebo_draws: int = 40,
    maker_placebo_draws: int = 3,
    seed: int = 20260101,
    report: Report = _quiet,
) -> str:
    """Write the search's registration from file names alone, before any day is read.

    The blocks come from the dataset's file names and ``search_share``; days a
    registration of this repository protects are refused here.
    """
    names = list(symbols or dataset.symbols())
    if not names:
        raise SearchError(f"no instrument with best bid and ask files under {dataset.raw}")
    unknown = sorted(set(family) - set(FAMILY))
    if unknown:
        raise SearchError(f"execution family members must be {FAMILY}, not {unknown}")
    common = sorted(set.intersection(*(set(dataset.days(s)) for s in names)))
    search_days, held_out = split_days(common, search_share)
    touched = protected_overlap(common)
    if touched:
        raise SearchError(
            f"the dataset's days touch ranges a registration protects: {touched}; a search here "
            "never computes on them (python -m agents.workflows.research_loop calendar)"
        )
    costs = ex.Costs.named(fee, slippage_bp)
    digest = run.register(
        {
            "written_before": "any day of the dataset is read: the blocks come from file names",
            "dataset": dataset.as_dict(),
            "candidates": names,
            "days": {"search": [str(d) for d in search_days], "held_out": [str(d) for d in held_out]},
            "search_share": search_share,
            "screen_horizon_s": horizon_s,
            "fee": costs.describe(),
            "frequency_grid": {"holds_s": list(holds_s), "label_threshold_share_of_round_trip": list(thresholds)},
            "methods": {"rules": list(sg.RULES), "models": list(sg.MODELS)},
            "parameters": {k: list(v) for k, v in sg.PARAMETERS.items()},
            "selection": "frequency by reproduce's neighbourhood rule, then the best net per trade on validation, at least minimum_trades",
            "minimum_trades": minimum_trades,
            "family": list(family),
            "holm": {"alpha": 0.05, "m": len(family)},
            "kill_conditions": list(vd.KILLS),
            "min_days_with_attempt": min_days,
            "placebos": {"kind": "shuffled-state circular shift of the decisions (market_making/verdicts.state_shift_rows)",
                         "draws": placebo_draws, "maker_draws": maker_placebo_draws, "seed": seed, "percentile": 95},
            "fee_ladder": "Bybit's linear tiers for the instrument's group (trp fees), the same attempts re-costed",
            "version": __version__,
        }
    )  # fmt: skip
    report(f"registered before reading data: {run.registration_path} ({digest[:12]})")
    return digest


def instruments(
    run: SearchRun,
    dataset: Dataset | None = None,
    *,
    discover: bool = False,
    report: Report = _quiet,
    **registration: Any,
) -> Path:
    """Screen every candidate on the search block alone; the search is registered first
    (by :func:`register`, with ``registration``'s options) unless it already is."""
    from trading_research.data.screen import screen_frame

    if not run.registration_path.exists():
        if dataset is None:
            raise SearchError("the first step needs --data: no registration exists yet")
        register(run, dataset, report=report, **registration)
    elif registration:
        report("the search is registered already; its registration's values hold")
    values, _ = run.registration()
    dataset = Dataset.from_dict(values["dataset"])
    names = list(values["candidates"])
    search_days = _days(values["days"]["search"])
    held_out = _days(values["days"]["held_out"])
    horizon_s = float(values["screen_horizon_s"])
    costs = ex.Costs.named(values["fee"]["tier"], values["fee"]["slippage_bp"])
    seen = view(dataset, names, search_days, run.root / "views" / "search")
    rows = []
    for symbol in names:
        try:
            book = pd.concat(
                [
                    pd.read_parquet(seen.raw / symbol / "bookTicker" / f"{d}.parquet")
                    for d in search_days
                ],
                ignore_index=True,
            )
            rows.append(screen_frame(book, symbol, horizon_s=horizon_s, costs=costs.taker).as_row())
        except Exception as exc:  # a bad instrument must not end the screen
            rows.append({"symbol": symbol, "skipped": f"{type(exc).__name__}: {exc}"})
    table = pd.DataFrame(rows)
    if "edge_over_cost" not in table:
        raise SearchError(f"no instrument could be screened: {rows}")
    table = table.sort_values("edge_over_cost", ascending=False, na_position="last").reset_index(
        drop=True
    )
    out = run.step_dir("instruments")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "screen.csv", index=False)
    if discover:
        try:
            from trading_research.data.discover import candidate_table

            candidate_table().to_csv(out / "candidates.csv", index=False)
        except OSError as exc:
            report(f"  the venue's listing could not be read: {exc}")
    usable = table[np.isfinite(table["edge_over_cost"].astype(float))]
    if usable.empty:
        raise SearchError("no instrument has a finite edge over cost on the search block")
    chosen = str(usable.iloc[0]["symbol"])
    report(
        f"instruments: {chosen} ranks first of {len(table)} by edge over cost on {len(search_days)} search days"
    )
    return run.finish(
        "instruments",
        {"dataset": dataset.as_dict(), "symbol": chosen, "ranked": list(usable["symbol"]),
         "search_days": [str(d) for d in search_days], "held_out_days": [str(d) for d in held_out]},
        inputs={"dataset": dataset.as_dict(), "symbols": names},
        params={"horizon_s": horizon_s, "fee": costs.describe()},
        outputs={"chosen": chosen, "screen": "screen.csv", "rows": len(table)},
    )  # fmt: skip


# ---------------------------------------------------------------------------
# 2. frequency
# ---------------------------------------------------------------------------


def _round_trip(frame: pd.DataFrame, costs: ex.Costs) -> float:
    spread = float(np.nanmedian(frame["spread_bp_now"]))
    return 2 * costs.taker.fee_bp_per_side + spread + 2 * costs.taker.slippage_bp


def frequency(run: SearchRun, *, report: Report = _quiet) -> Path:
    """The hold and the label threshold, chosen by their neighbourhood on validation."""
    from trading_research.pipeline.stages import load_prepared, load_selected, prepare, select

    before = run.previous("frequency")
    registration, _ = run.registration()
    dataset, symbol = Dataset.from_dict(before["dataset"]), before["symbol"]
    days = _days(before["search_days"])
    costs = ex.Costs.named(registration["fee"]["tier"], registration["fee"]["slippage_bp"])
    seen = view(dataset, [symbol], days, run.root / "views" / "search")
    step = _step_seconds(seen, symbol, days[0])
    subsample = max(1, round(GRID_SECONDS / step))
    holds = [int(h) for h in registration["frequency_grid"]["holds_s"]]
    base = holds[len(holds) // 2]
    prepared = prepare(seen.raw, run.root / "prepared" / "search", symbol,
                       horizon=max(1, round(base / step)), subsample=subsample)  # fmt: skip
    spec = sg.spec_for(len(days))
    features = select(prepared, run.root / "features", train_start=days[0],
                      train_end=days[spec.train_days - 1], threshold_bp=0.0, max_features=20)  # fmt: skip
    frame = load_prepared(prepared)
    columns = load_selected(features)
    round_trip = _round_trip(frame, costs)
    rows = []
    for hold_s in holds:
        for share in registration["frequency_grid"]["label_threshold_share_of_round_trip"]:
            candidate = sg.Candidate(
                max(1, hold_s // GRID_SECONDS), round(share * round_trip, 2), "logistic"
            )
            row: dict[str, Any] = {
                "hold_s": hold_s,
                "threshold_bp": candidate.threshold_bp,
                "candidate": candidate.label,
            }
            try:
                row.update(sg.evaluate(frame, columns, days, candidate, costs.taker))
            except Exception as exc:  # a cell that cannot be scored is recorded, not dropped
                row["skipped"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            report(
                f"  {candidate.label}: {row.get('trades', 0):.0f} trades, {row.get('net_per_trade_bp', math.nan):+.2f} bp"
            )
    table = pd.DataFrame(rows)
    dense = table[
        table.get("trades", pd.Series(0.0, index=table.index)) >= registration["minimum_trades"]
    ]
    if dense.empty:
        raise SearchError(
            f"no cell reached {registration['minimum_trades']} trades on the validation days"
        )
    scored = sg.neighbourhood(dense, "threshold_bp", "hold_s")
    winner, peak = sg.choose(scored)
    declined = (peak["hold_s"], peak["threshold_bp"]) != (winner["hold_s"], winner["threshold_bp"])
    out = run.step_dir("frequency")
    out.mkdir(parents=True, exist_ok=True)
    table.merge(scored[["hold_s", "threshold_bp", "neighbourhood_bp"]], how="left").to_csv(
        out / "grid.csv", index=False
    )
    report(f"frequency: hold {winner['hold_s']:.0f}s, label threshold {winner['threshold_bp']:.2f} bp "
           f"({winner['neighbourhood_bp']:+.2f} bp with its neighbours); peak "
           f"{peak['hold_s']:.0f}s/{peak['threshold_bp']:.2f} bp at {peak['net_per_trade_bp']:+.2f} bp "
           + ("declined" if declined else "is the choice"))  # fmt: skip
    keep = ("hold_s", "threshold_bp", "trades", "net_per_trade_bp", "neighbourhood_bp")
    return run.finish(
        "frequency",
        {**before, "step_s": step, "subsample": subsample, "prepared": str(prepared),
         "features": str(features), "columns": columns, "round_trip_bp": round_trip,
         "hold_s": int(winner["hold_s"]), "hold_rows": max(1, int(winner["hold_s"]) // GRID_SECONDS),
         "threshold_bp": float(winner["threshold_bp"]),
         "frequency": {"chosen": {k: winner.get(k) for k in keep}, "peak": {k: peak.get(k) for k in keep},
                       "peak_declined": bool(declined)}},
        inputs={"previous": str(run.step_dir("instruments"))},
        params={"holds_s": holds, "model": "logistic", "spec": asdict(spec)},
        outputs={"grid": "grid.csv", "hold_s": int(winner["hold_s"]), "peak_declined": bool(declined)},
    )  # fmt: skip


# ---------------------------------------------------------------------------
# 3. method and 4. model
# ---------------------------------------------------------------------------


def _score(before: dict[str, Any], run: SearchRun, candidates: Sequence[sg.Candidate],
           report: Report) -> pd.DataFrame:  # fmt: skip
    from trading_research.pipeline.stages import load_prepared

    registration, _ = run.registration()
    costs = ex.Costs.named(registration["fee"]["tier"], registration["fee"]["slippage_bp"])
    frame = load_prepared(before["prepared"])
    days = _days(before["search_days"])
    rows = []
    for candidate in candidates:
        row: dict[str, Any] = {"candidate": candidate.label, "model": candidate.model,
                               "family": candidate.family, "params": json.dumps(dict(candidate.params))}  # fmt: skip
        try:
            row.update(sg.evaluate(frame, before["columns"], days, candidate, costs.taker))
        except Exception as exc:  # recorded with its reason, as the halving records one
            row["skipped"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        report(
            f"  {candidate.label}: {row.get('trades', 0):.0f} trades, {row.get('net_per_trade_bp', math.nan):+.2f} bp"
        )
    return pd.DataFrame(rows)


def _eligible(table: pd.DataFrame, minimum: int) -> pd.DataFrame:
    if "trades" not in table:
        return table.iloc[0:0]
    return table[
        (table["trades"] >= minimum) & np.isfinite(table["net_per_trade_bp"].astype(float))
    ]


def method(run: SearchRun, *, report: Report = _quiet) -> Path:
    """Every rule against every learned model at the chosen frequency."""
    before = run.previous("method")
    registration, _ = run.registration()
    hold, threshold = int(before["hold_rows"]), float(before["threshold_bp"])
    names = [*registration["methods"]["rules"], *registration["methods"]["models"]]
    table = _score(before, run, [sg.Candidate(hold, threshold, name) for name in names], report)
    eligible = _eligible(table, registration["minimum_trades"])
    if eligible.empty:
        raise SearchError(
            f"no rule or model reached {registration['minimum_trades']} trades on validation"
        )
    best = {family: eligible[eligible["family"] == family].sort_values("net_per_trade_bp").tail(1)
            for family in ("rule", "model")}  # fmt: skip
    found = {f: (r.iloc[0].to_dict() if len(r) else None) for f, r in best.items()}
    chosen = max((r for r in found.values() if r), key=lambda r: r["net_per_trade_bp"])
    out = run.step_dir("method")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "method.csv", index=False)
    summary = {f: (None if r is None else {"model": r["model"], "net_per_trade_bp": r["net_per_trade_bp"],
                                           "trades": r["trades"]}) for f, r in found.items()}  # fmt: skip
    report(
        f"method: {chosen['family']} ({chosen['model']}, {chosen['net_per_trade_bp']:+.2f} bp a trade on validation)"
    )
    return run.finish(
        "method",
        {**before, "method": chosen["family"], "method_best": summary},
        inputs={"previous": str(run.step_dir("frequency"))},
        params={"candidates": names, "hold_rows": hold, "threshold_bp": threshold},
        outputs={"method": chosen["family"], "table": "method.csv", "best": summary},
    )


def model(run: SearchRun, *, report: Report = _quiet) -> Path:
    """The chosen family's candidates: the rules as they are, or a model's parameters."""
    before = run.previous("model")
    registration, _ = run.registration()
    hold, threshold = int(before["hold_rows"]), float(before["threshold_bp"])
    if before["method"] == "rule":
        candidates = [
            sg.Candidate(hold, threshold, name) for name in registration["methods"]["rules"]
        ]
    else:
        name = before["method_best"]["model"]["model"]
        candidates = [sg.Candidate(hold, threshold, name, tuple(sorted(p.items())))
                      for p in (dict(q) for q in registration["parameters"].get(name, [{}]))]  # fmt: skip
    table = _score(before, run, candidates, report)
    eligible = _eligible(table, registration["minimum_trades"])
    if eligible.empty:
        raise SearchError(
            f"no {before['method']} candidate reached {registration['minimum_trades']} trades"
        )
    winner = eligible.sort_values("net_per_trade_bp").iloc[-1].to_dict()
    out = run.step_dir("model")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "model.csv", index=False)
    params = json.loads(winner["params"])
    report(
        f"model: {winner['candidate']} ({winner['net_per_trade_bp']:+.2f} bp a trade on validation)"
    )
    return run.finish(
        "model",
        {**before, "model": winner["model"], "params": params,
         "model_validation": {k: winner.get(k) for k in ("trades", "net_per_trade_bp", "hit_rate", "min_confidence")}},
        inputs={"previous": str(run.step_dir("method"))},
        params={"candidates": [c.label for c in candidates]},
        outputs={"model": winner["candidate"], "table": "model.csv"},
    )  # fmt: skip


# ---------------------------------------------------------------------------
# 5. execution
# ---------------------------------------------------------------------------


def _candidate(chosen: dict[str, Any]) -> sg.Candidate:
    return sg.Candidate(int(chosen["hold_rows"]), float(chosen["threshold_bp"]), chosen["model"],
                        tuple(sorted(dict(chosen["params"]).items())))  # fmt: skip


def _prepared_at_hold(run: SearchRun, chosen: dict[str, Any], seen: Dataset, name: str) -> Path:
    from trading_research.pipeline.stages import prepare

    horizon = max(1, round(int(chosen["hold_s"]) / float(chosen["step_s"])))
    return prepare(seen.raw, run.root / "prepared" / name, chosen["symbol"], horizon=horizon,
                   subsample=int(chosen["subsample"]))  # fmt: skip


def _modes(
    family: Sequence[str],
    seen: Dataset,
    chosen: dict[str, Any],
    days: Sequence[date],
    costs: ex.Costs,
    frame: pd.DataFrame,
    decision: np.ndarray,
    plan: Any,
) -> tuple[dict[str, Any], dict[str, str], Callable[[str, np.ndarray], Any]]:
    """Each mode's result over ``days``, the modes that could not run with the reason, and the runner."""
    symbol, hold = chosen["symbol"], int(chosen["hold_rows"])
    strength = float(chosen["round_trip_bp"])
    market = ex.grid_market(seen, symbol, days)

    def run_one(mode: str, decisions: np.ndarray) -> Any:
        if mode == "maker" and plan is None:
            raise ValueError(
                "no market-maker plan: the dataset has no book, prints and funding for it"
            )
        signal = ex.stream(symbol, frame, decisions, hold, strength)
        return ex.run_mode(mode, seen, symbol, days, costs, signal=signal, hold_rows=hold,
                           market=market, plan=plan)  # fmt: skip

    results, skipped = {}, {}
    for mode in family:
        try:
            results[mode] = run_one(mode, decision)
        except Exception as exc:  # a mode that cannot run is recorded with its reason
            skipped[mode] = f"{type(exc).__name__}: {exc}"
    return results, skipped, run_one


def _summary(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for mode, result in results.items():
        a = result.attempts
        row = {"mode": mode, "attempts": len(a)}
        if len(a):
            row.update(fill_rate=float(a["filled"].astype(bool).mean()), gross_per_attempt_bp=float(a["gross_bp"].mean()),
                       cost_per_attempt_bp=float(a["cost_bp"].mean()), net_per_attempt_bp=float(a["net_bp"].mean()),
                       days=int(a["day"].nunique()))  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


def execution(run: SearchRun, *, report: Report = _quiet) -> Path:
    """The chosen signal executed each way on the search block; the mode chosen and everything frozen."""

    before = run.previous("execution")
    registration, _ = run.registration()
    dataset, symbol = Dataset.from_dict(before["dataset"]), before["symbol"]
    days = _days(before["search_days"])
    costs = ex.Costs.named(registration["fee"]["tier"], registration["fee"]["slippage_bp"])
    seen = view(dataset, [symbol], days, run.root / "views" / "search")
    prepared = _prepared_at_hold(run, before, seen, "search_at_hold")
    spec = sg.spec_for(len(days))
    train = (days[0], days[spec.train_days - 1])
    validation_day = days[spec.train_days]
    scored_days = days[spec.train_days + 1 :]
    candidate = _candidate(before)
    out = run.step_dir("execution")
    model_dir = sg.fit_signal(
        prepared, Path(before["features"]), out / "model", train=train, candidate=candidate
    )
    validation = sg.predictions(
        prepared, model_dir, out / "predictions_validation", [validation_day]
    )
    confidence = sg.confidence_on(validation, costs.taker)
    frame = sg.predictions(prepared, model_dir, out / "predictions", scored_days)
    decision = sg.decisions(frame, confidence)
    plan, plan_values = None, None
    if "maker" in registration["family"] and seen.has_event_time(symbol, days):
        plan = ex.maker_plan(seen, symbol, days[: spec.train_days + 1], costs, lean=True)
        plan_values = ex.plan_values(plan)
    results, skipped, _ = _modes(registration["family"], seen, before, scored_days, costs, frame,
                                 decision, plan)  # fmt: skip
    table = _summary(results)
    table.to_csv(out / "execution.csv", index=False)
    if table.empty or "net_per_attempt_bp" not in table:
        raise SearchError(f"no execution mode produced an attempt on the search block: {skipped}")
    ranked = table.dropna(subset=["net_per_attempt_bp"]).sort_values("net_per_attempt_bp")
    chosen_mode = str(ranked.iloc[-1]["mode"])
    frozen = {
        "symbol": symbol, "hold_s": before["hold_s"], "hold_rows": before["hold_rows"],
        "threshold_bp": before["threshold_bp"], "model": before["model"], "params": before["params"],
        "features": before["columns"], "prepare": {"step_s": before["step_s"], "subsample": before["subsample"]},
        "entry_threshold_rule": "chosen on the last search day by choose_confidence, the fit on the train_days before it",
        "train_days": spec.train_days, "execution": chosen_mode, "family": registration["family"],
        "costs": costs.describe(), "search_entry_threshold": confidence, "maker_plan": plan_values,
    }  # fmt: skip
    write_json(out / "frozen.json", frozen)
    report(f"execution: {chosen_mode} chosen on {len(scored_days)} search day(s); "
           + ", ".join(f"{r.mode} {r.net_per_attempt_bp:+.2f} bp" for r in ranked.itertuples()))  # fmt: skip
    return run.finish(
        "execution",
        {**before, "execution": chosen_mode, "frozen": str(out / "frozen.json"),
         "frozen_sha256": sha256_file(out / "frozen.json"), "prepared_at_hold": str(prepared),
         "skipped_modes": skipped},
        inputs={"previous": str(run.step_dir("model"))},
        params={"train": [str(d) for d in train], "validation_day": str(validation_day),
                "scored_days": [str(d) for d in scored_days], "entry_threshold": confidence},
        outputs={"table": "execution.csv", "chosen": chosen_mode, "skipped": skipped},
    )  # fmt: skip


# ---------------------------------------------------------------------------
# 6. read
# ---------------------------------------------------------------------------


def read(run: SearchRun, *, second_read: bool = False, report: Report = _quiet) -> Path:
    """The held-out block, once, through the ledger: everything frozen, the verdict as registered."""
    from trading_research import fees as fee_tables
    from trading_research.pipeline.stages import backtest as taker_backtest

    before = run.previous("read")
    registration, _ = run.registration()
    frozen_path = Path(before["frozen"])
    if sha256_file(frozen_path) != before["frozen_sha256"]:
        raise ReadRefused(f"{frozen_path} changed after the execution step froze it")
    entry = run.open_held_out(frozen_path, second_read=second_read)
    report(f"read: the ledger records the {entry['read']} before the block is opened")
    frozen = json.loads(frozen_path.read_text())
    dataset, symbol = Dataset.from_dict(before["dataset"]), before["symbol"]
    search_days, held_days = _days(before["search_days"]), _days(before["held_out_days"])
    costs = ex.Costs.named(registration["fee"]["tier"], registration["fee"]["slippage_bp"])
    out = run.step_dir("read")
    # The last search day warms the held-out days' rolling windows; it is not scored.
    seen = view(dataset, [symbol], [search_days[-1], *held_days], run.root / "views" / "held_out")
    held_prepared = _prepared_at_hold(run, before, seen, "held_out_at_hold")
    search_prepared = Path(before["prepared_at_hold"])
    train_days = int(frozen["train_days"])
    train = (search_days[-1 - train_days], search_days[-2])
    candidate = _candidate(before)
    model_dir = sg.fit_signal(
        search_prepared, Path(before["features"]), out / "model", train=train, candidate=candidate
    )
    last = sg.predictions(
        search_prepared, model_dir, out / "predictions_last_search_day", [search_days[-1]]
    )
    confidence = sg.confidence_on(last, costs.taker)
    frame = sg.predictions(held_prepared, model_dir, out / "predictions", held_days)
    decision = sg.decisions(frame, confidence)
    family = list(registration["family"])
    plan = ex.plan_from(frozen["maker_plan"], costs) if frozen.get("maker_plan") else None
    results, skipped, run_one = _modes(
        family, seen, before, held_days, costs, frame, decision, plan
    )
    for mode, result in results.items():
        result.attempts.to_csv(out / f"attempts_{mode}.csv", index=False)
    shift_list = vd.shifts(
        len(frame), int(registration["placebos"]["draws"]), int(registration["placebos"]["seed"])
    )
    group = "g1" if symbol in BYBIT_G1_EXAMPLES else "altcoin"
    ladder_rows = fee_tables.ladder(group)
    assessed = {}
    for mode, result in results.items():
        draws = (
            shift_list
            if mode != "maker"
            else shift_list[: int(registration["placebos"]["maker_draws"])]
        )

        def attempts_of(decisions: np.ndarray, mode: str = mode) -> pd.DataFrame:
            return run_one(mode, decisions).attempts

        placebos = vd.placebo_values(attempts_of, decision, draws)
        assessed[mode] = vd.assess_mode(
            mode, result.attempts, placebos, min_days=int(registration["min_days_with_attempt"])
        )
        assessed[mode]["fee_ladder"] = [
            {"tier": r.tier, "maker_bp": r.maker_bp, "taker_bp": r.taker_bp,
             "net_per_attempt_bp": ex.at_fee(result, mode, costs, fee_tier(r.tier))}
            for r in ladder_rows
        ]  # fmt: skip
    decided = vd.decide(assessed, alpha=float(registration["holm"]["alpha"]))
    # The taker's file stage on the same predictions, as a reader would run it.
    cross_check = taker_backtest(out / "predictions", out / "backtest",
                                 min_confidence=confidence, hold_periods=int(before["hold_rows"]),
                                 fee_bp_per_side=costs.tier.taker_bp, slippage_bp=costs.slippage_bp)  # fmt: skip
    backtest_outputs = json.loads((cross_check / "manifest.json").read_text())["outputs"]["thinned"]
    results_payload = {
        "symbol": symbol, "held_out_days": [str(d) for d in held_days], "ledger_entry": entry,
        "entry_threshold": confidence, "execution_chosen_on_search": frozen["execution"],
        "costs": costs.describe(), "fee_group": group, "family": family, "skipped_modes": skipped,
        "modes": decided, "taker_backtest_stage": backtest_outputs,
        "kill_conditions": registration["kill_conditions"], "placebo_shifts_rows": shift_list,
    }  # fmt: skip
    results_path = write_json(out / "results.json", results_payload)
    run.close_held_out(results_path)
    for mode, row in decided.items():
        report(f"  {mode}: {row['net_per_attempt_bp']:+.2f} bp per attempt over {row['attempts']} attempt(s), "
               f"day t {row['day_t']:+.2f}: {row['status']} {row['decided_by']}")  # fmt: skip
    return run.finish(
        "read",
        {**before, "results": str(results_path), "statuses": {m: r["status"] for m, r in decided.items()}},
        inputs={"previous": str(run.step_dir("execution")), "frozen_sha256": before["frozen_sha256"]},
        params={"train": [str(d) for d in train], "entry_threshold": confidence},
        outputs={"results": "results.json", "statuses": {m: r["status"] for m, r in decided.items()}},
    )  # fmt: skip


STEP_FUNCTIONS: dict[str, Callable[..., Path]] = {
    "frequency": frequency,
    "method": method,
    "model": model,
    "execution": execution,
    "read": read,
}


def run_all(run: SearchRun, dataset: Dataset, *, report: Report = _quiet, **options: Any) -> Path:
    """Every step in order; the read last, once."""
    instruments(run, dataset, report=report, **options)
    for step in ("frequency", "method", "model", "execution", "read"):
        STEP_FUNCTIONS[step](run, report=report)
    return run.step_dir("read")
