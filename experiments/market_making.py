"""The market-making study on its development block (D), and the freeze.

Runs everything the pre-registration (``docs/preregistration/market_making.md``)
assigns to block D, 2024-02-01 to 2024-02-25, and nothing on any later day:

1. funding history for the study's instruments and blocks, fetched into
   ``data/funding`` (the simulator reads only D's here);
2. the admission screen, which reads D and checks H's files for presence only;
3. the reversion tape on D for the H2-admitted instruments: theta, beta, the
   triggers, the taker twin, the index autocorrelation and H2.3's gate level;
4. the regime flags on D for the MM-admitted instruments, and their rate;
5. the two successive-halving searches (S1 over 144 cells, S2 over 432) on a
   seeded, nested subset of D's days, each choice made on the neighbourhood of
   the cells measured at the full budget;
6. S3's six cells on whichever of S1 and S2 scored higher, S4's six cells on the
   H2-admitted instruments, X1 and its taker twin, and the twins and baselines
   whose daily spread sets the minimum detectable effects;
7. with ``--freeze``, the chosen values written into the placeholders of
   ``configs/mm_prereg.yaml``, and nothing else.

Every reader goes through :class:`trading_research.market_making.prereg.Access`;
this script holds only :func:`~trading_research.market_making.prereg.development_access`,
so a day outside D raises before any file of it is opened.

Usage::

    uv run python -m experiments.market_making --workers 8
    uv run python -m experiments.market_making --workers 8 --freeze

Results land in ``experiments/results/mm_*_D.csv``; per-day fills and equity
in the gitignored ``artifacts/mm/cache``, from which a re-run is answered.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments._common import RESULTS
from trading_research.market_making import flags as flag_module
from trading_research.market_making import prereg, screen, signals
from trading_research.market_making.quoters import (
    InsideQuoter,
    Quoter,
    RegimeGuard,
    ReversionLean,
    SignalExecutor,
    SkewQuoter,
)
from trading_research.market_making.simulator import Cell, SimConfig, run_cells
from trading_research.pipeline.discovery import UNIVERSE
from trading_research.strategies.reversion import (
    ROWS_PER_DAY,
    ReversionConfig,
    index_level,
    trailing_autocorrelation,
)
from trading_research.validation.search import neighbourhood_scores, successive_halving

STUDY_SYMBOLS = ("BICOUSDT", "CRVUSDT", "XRPUSDT", "BTCUSDT")
BOOK_ROOTS = (Path("data/book"), Path("data/book_fresh"))
TRADES_ROOT = Path("data/trades")
FUNDING_ROOT = Path("data/funding")
UNIVERSE_ROOT = Path("data/universe")
CACHE = Path("artifacts/mm/cache")

#: The amendment's date.
AMENDMENT_DATE = date(2026, 10, 3)

#: Minimum fills a day for a cell to be ranked (the pre-registration's "minimum
#: 20 fills a day"), applied to the mean over the days measured.
MIN_FILLS_PER_DAY = 20.0

#: Days of the held-out block, for the power calculation.
H_DAYS = 13


def report(message: str) -> None:
    print(message, flush=True)


def sig(value: float, digits: int = 6) -> float:
    """``value`` to ``digits`` significant figures, as it is frozen."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    return float(f"{value:.{digits}g}")


def children_peak_mb() -> float:
    """The largest peak resident set of any worker process that has exited."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


@dataclass
class Clock:
    """Wall time per stage, and the largest worker resident set seen."""

    stages: dict[str, float]
    worker_peak_mb: float = 0.0

    def time(self, name: str, start: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - start

    def saw(self, frame: pd.DataFrame) -> None:
        if "worker_peak_rss_mb" in frame.columns and len(frame):
            self.worker_peak_mb = max(self.worker_peak_mb, float(frame["worker_peak_rss_mb"].max()))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def fetch_funding() -> None:
    """Funding for every instrument and block the study uses: D and H for all
    four instruments, F for the two with a book after 9 March. The files are
    written; only D's are read here."""
    from trading_research.data.ensure import ensure_funding

    spans = [(s, date(2024, 2, 1), date(2024, 3, 9)) for s in STUDY_SYMBOLS]
    spans += [(s, date(2024, 3, 12), date(2024, 4, 7)) for s in ("BICOUSDT", "BTCUSDT")]
    for symbol, start, end in spans:
        ensure_funding(symbol, start, end, root=FUNDING_ROOT, on_progress=None)
    d_days = prereg.block_days("D")
    for symbol in STUDY_SYMBOLS:
        missing = [
            d for d in d_days if not (FUNDING_ROOT / symbol / f"{d.isoformat()}.parquet").is_file()
        ]
        if missing:
            raise SystemExit(f"{symbol}: no funding file for D days {missing}")


# ---------------------------------------------------------------------------
# Cells and scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instrument:
    symbol: str
    clip_notional: float
    sigma_ref: float


def config_for(instrument: Instrument, soft_limit_clips: float = 6.0) -> SimConfig:
    return SimConfig(clip_notional=instrument.clip_notional, soft_limit_clips=soft_limit_clips)


def s1_quoter(params: dict[str, Any], instrument: Instrument) -> SkewQuoter:
    return SkewQuoter(
        skew_bp=float(params["skew_bp"]),
        k=float(params["k"]),
        min_edge_bp=float(params["min_edge_bp"]),
        sigma_ref=instrument.sigma_ref,
    )


def s2_quoter(
    params: dict[str, Any], instrument: Instrument, *, inside: bool = True
) -> InsideQuoter:
    return InsideQuoter(
        skew_bp=float(params["skew_bp"]),
        k=float(params["k"]),
        min_edge_bp=float(params["min_edge_bp"]),
        m_ticks=int(params["m_ticks"]),
        sigma_ref=instrument.sigma_ref,
        inside=inside,
        name="S2" if inside else "S2-twin",
    )


def label(params: dict[str, Any]) -> str:
    return "/".join(f"{k}={v}" for k, v in params.items())


def constant(quoter: Quoter) -> Callable[[], Quoter]:
    """A picklable factory for a stateless (frozen) quoter."""
    return partial(_same, quoter)


def _same(quoter: Quoter) -> Quoter:
    return quoter


def daily_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """The objective and its companions over a cell's (instrument, day) rows."""
    usable = frame[(frame["status"] == "ok") & ~frame["excluded"].astype(bool)]
    if usable.empty:
        return {"net_usd_day": float("nan"), "trades": 0.0, "days": 0.0}
    net = usable["net"].to_numpy(dtype=np.float64)
    out = {
        "net_usd_day": float(net.mean()),
        "sd_usd_day": float(net.std(ddof=1)) if len(net) > 1 else float("nan"),
        # Passive fills a day, the quantity the search's minimum applies to.
        "trades": float((usable["fills"] - usable["flattens"]).mean()),
        "days": float(len(usable)),
        "excluded_days": float(len(frame) - len(usable)),
    }
    for name in ("making", "spread", "adverse", "inventory", "fees", "funding"):
        out[f"{name}_usd_day"] = float(usable[name].mean())
    out["flattens_day"] = float(usable["flattens"].mean())
    out["max_abs_position"] = float(usable["max_abs_position"].max())
    return out


@dataclass
class Runner:
    """Runs cells on the admitted instruments, D days only."""

    workers: int
    access: prereg.Access
    clock: Clock

    def run(
        self,
        instrument: Instrument,
        days: Sequence[date],
        cells: Sequence[Cell],
        tapes: dict[str, signals.SignalTape] | None = None,
        *,
        cells_per_job: int = 8,
    ) -> pd.DataFrame:
        frame = run_cells(
            instrument.symbol,
            days,
            cells,
            book_roots=BOOK_ROOTS,
            trades_root=TRADES_ROOT,
            funding_root=FUNDING_ROOT,
            tapes=tapes,
            workers=self.workers,
            cache=CACHE,
            cells_per_job=cells_per_job,
            day_guard=lambda ds: self.access.require(ds, what="simulate"),
        )
        self.clock.saw(frame)
        return frame


def search_grids(registration: prereg.PreRegistration) -> dict[str, dict[str, list[Any]]]:
    """The registered grids, as the YAML holds them."""
    return {
        name: dict(registration.value(("strategies", name, "search")))
        for name in ("S1", "S2", "S3", "S4")
    }


def nested_days(d_days: Sequence[date], budgets: Sequence[int], seed: int) -> dict[int, list[date]]:
    """The seeded, nested subsets: budget b takes the first b of one permutation."""
    order = np.random.default_rng(seed).permutation(len(d_days))
    return {b: sorted(d_days[int(i)] for i in order[:b]) for b in budgets}


def search(
    name: str,
    grid: dict[str, list[Any]],
    make: Callable[[dict[str, Any], Instrument], Cell],
    instruments: Sequence[Instrument],
    subsets: dict[int, list[date]],
    runner: Runner,
    *,
    keep_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """One successive-halving run and its neighbourhood choice.

    Returns every evaluation, the chosen cell with its neighbourhood, and the
    outright peak at the full budget.
    """
    from itertools import product

    axes = list(grid)
    candidates = [dict(zip(axes, values, strict=True)) for values in product(*grid.values())]

    def score_batch(alive: Sequence[dict[str, Any]], budget: int) -> list[dict[str, float]]:
        days = subsets[budget]
        frames = []
        for instrument in instruments:
            cells = [make(params, instrument) for params in alive]
            frames.append(runner.run(instrument, days, cells))
        frame = pd.concat(frames, ignore_index=True)
        report(f"  {name}: {len(alive)} cells x {len(days)} days scored")
        return [daily_metrics(frame[frame["cell"] == label(params)]) for params in alive]

    outcome = successive_halving(
        candidates,
        score_batch=score_batch,
        objective="net_usd_day",
        budgets=sorted(subsets),
        keep_fraction=keep_fraction,
        minimum_trades=MIN_FILLS_PER_DAY,
        label=label,
    )
    table = outcome.table.copy()
    parsed = pd.DataFrame(
        [dict(item.split("=", 1) for item in str(c).split("/")) for c in table["candidate"]]
    )
    for axis in axes:
        table[axis] = pd.to_numeric(parsed[axis])
    full = table[(table["budget"] == max(subsets)) & (table["trades"] >= MIN_FILLS_PER_DAY)].copy()
    full = full.dropna(subset=["net_usd_day"]).reset_index(drop=True)
    if full.empty:
        raise SystemExit(
            f"{name}: no cell reached {MIN_FILLS_PER_DAY} fills a day at the full budget"
        )
    full["neighbourhood_usd_day"] = neighbourhood_scores(full, axes, "net_usd_day")
    ranked = full.sort_values(["neighbourhood_usd_day", "net_usd_day"], kind="stable")
    chosen_row = ranked.iloc[-1]
    peak_row = full.loc[full["net_usd_day"].idxmax()]
    chosen = {axis: _plain(chosen_row[axis]) for axis in axes}
    positions = {
        axis: full[axis].map({v: i for i, v in enumerate(sorted(full[axis].unique()))})
        for axis in axes
    }
    near_mask = np.ones(len(full), dtype=bool)
    for axis in axes:
        near_mask &= (positions[axis] - positions[axis][chosen_row.name]).abs().to_numpy() <= 1
    near = full[near_mask]
    neighbourhood = {
        "axes": axes,
        "chosen": [chosen[a] for a in axes],
        "median": sig(float(chosen_row["neighbourhood_usd_day"])),
        "own": sig(float(chosen_row["net_usd_day"])),
        "cells": [
            [*(_plain(r[a]) for a in axes), sig(float(r["net_usd_day"]))]
            for _, r in near.iterrows()
        ],
        "peak": [*(_plain(peak_row[a]) for a in axes), sig(float(peak_row["net_usd_day"]))],
        "full_budget_cells": len(full),
    }
    table = table.merge(full[[*axes, "neighbourhood_usd_day"]], on=axes, how="left")
    return table, chosen, neighbourhood


def _plain(value: Any) -> Any:
    """A grid value as the YAML holds it: int when whole, else float."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    number = float(value)
    return int(number) if number.is_integer() else number


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def reversion_stage(
    h2_symbols: Sequence[str], d_days: Sequence[date], access: prereg.Access
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, float]]:
    """Tapes, theta, beta and triggers on D; the index's state on D."""
    config = ReversionConfig()
    panel = signals.load_universe_panel(UNIVERSE, d_days, access=access, root=UNIVERSE_ROOT)
    report(f"  universe panel: {panel.log_mid.shape[1]} instruments, {len(panel.log_mid):,} rows")
    out: dict[str, dict[str, Any]] = {}
    rows = []
    for symbol in h2_symbols:
        tape = signals.reversion_tape(panel, symbol, config)
        # Rounded as they are frozen, so D's runs use exactly what H will.
        theta = sig(signals.fit_theta(tape, config))
        beta = sig(signals.fit_beta(panel, symbol, tape, theta))
        triggers = signals.reversion_triggers(tape, theta, config)
        x1 = triggers.eligible()
        twin = signals.taker_twin(x1, panel, symbol)
        out[symbol] = {
            "tape": tape,
            "theta": theta,
            "beta": beta,
            "lean": signals.lean_trigger_tape(tape, theta),
            "triggers": x1,
            "taker": twin,
        }
        strong = np.abs(tape.values) >= theta
        rows.append(
            {
                "symbol": symbol,
                "theta_bp": theta,
                "beta": beta,
                "rows": len(tape),
                "rows_at_or_above_theta_per_day": float(strong.sum()) / len(d_days),
                "triggers_per_day": len(triggers) / len(d_days),
                "x1_eligible_triggers_per_day": len(x1) / len(d_days),
                "taker_twin_trades": len(twin),
                "taker_twin_net_bp_per_trade": float(twin["net_bp"].mean())
                if len(twin)
                else np.nan,
                "taker_twin_gross_bp_per_trade": float(twin["gross_bp"].mean())
                if len(twin)
                else np.nan,
            }
        )
    values = panel.log_mid.to_numpy(dtype=np.float64)
    level = index_level(values)
    lag = config.hold
    past = np.full(len(level), np.nan)
    past[lag:] = (level[lag:] - level[:-lag]) * 1e4
    forward = np.full(len(level), np.nan)
    forward[:-lag] = (level[lag:] - level[:-lag]) * 1e4
    ok = np.isfinite(past) & np.isfinite(forward)
    state = {"index_autocorrelation_D": float(np.corrcoef(past[ok], forward[ok])[0, 1])}

    # H2.3's gate: the trailing one-day autocorrelation known at each day's
    # open, one value per day; its median over D is the frozen threshold.
    labels = pd.DatetimeIndex(panel.log_mid.index)
    first = labels[0]
    if first != pd.Timestamp(d_days[0], tz="UTC") + np.timedelta64(5, "s"):
        raise SystemExit(f"the panel starts at {first}, not at the first bin end of D")
    if len(labels) != ROWS_PER_DAY * len(d_days):
        raise SystemExit(f"the panel has {len(labels)} rows, not {ROWS_PER_DAY} a day")
    gate = trailing_autocorrelation(level, lag, ROWS_PER_DAY, known_at_open=True)
    per_day = gate[::ROWS_PER_DAY]
    state["h2_3_gate_days"] = float(np.isfinite(per_day).sum())
    state["h2_3_gate_threshold"] = float(np.nanmedian(per_day))
    table = pd.DataFrame(rows)
    for key, value in state.items():
        table[key] = value
    return out, table, state


def flag_stage(
    mm_symbols: Sequence[str], d_days: Sequence[date], access: prereg.Access
) -> tuple[dict[str, flag_module.FlagTimeline], pd.DataFrame, pd.DataFrame]:
    timelines: dict[str, flag_module.FlagTimeline] = {}
    frames, rows = [], []
    for symbol in mm_symbols:
        timeline = flag_module.build_flags(symbol, d_days, book_roots=BOOK_ROOTS, access=access)
        timelines[symbol] = timeline
        frames.append(timeline.frame())
        counts = flag_module.flag_counts(timeline)
        row: dict[str, Any] = {
            "symbol": symbol,
            "flags": len(timeline),
            "flags_per_day": timeline.rate_per_day(len(d_days)),
            "testable_K3": 0.2 <= timeline.rate_per_day(len(d_days)) <= 24.0,
        }
        for window in (15, 60, 240):
            row[f"guard_share_{window}min"] = flag_module.guard_share(timeline, window)
        row.update({f"flags_{k}": v for k, v in sorted(counts.items())})
        rows.append(row)
    flags_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not flags_frame.empty:
        flags_frame = flags_frame.drop(columns=["effective_ns"])
    return timelines, pd.DataFrame(rows), flags_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", action="store_true", help="write the frozen values")
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("between 1 and 8 workers")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    started = time.perf_counter()
    clock = Clock(stages={})

    registration = prereg.PreRegistration.load().require_consistent()
    if registration.frozen:
        raise SystemExit("the registration is already frozen; the development block is done")
    access = prereg.development_access()
    d_days = prereg.block_days("D")
    h_days = prereg.block_days("H")
    d_search = registration.value(("d_search",))
    budgets = [int(b) for b in d_search["budgets_days"]]
    seed = int(d_search["day_subset_seed"])
    keep = float(d_search["keep_fraction"])
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. funding ------------------------------------------------------------
    t = time.perf_counter()
    if not args.skip_fetch:
        report("funding: fetching the study's instruments and blocks")
        fetch_funding()
    clock.time("funding", t)

    # 2. admission ----------------------------------------------------------
    t = time.perf_counter()
    report("screen: D statistics, H coverage from the file system")
    table = screen.screen(
        STUDY_SYMBOLS,
        d_days=d_days,
        h_days=h_days,
        book_roots=BOOK_ROOTS,
        trades_root=TRADES_ROOT,
        access=access,
        workers=args.workers,
        on_progress=report,
    )
    table = screen.admit(table, universe=UNIVERSE, exclude=ReversionConfig().exclude)
    table.to_csv(RESULTS / "mm_admission_D.csv", index=False)
    clock.time("screen", t)
    mm_symbols = [s for s, ok in zip(table["symbol"], table["mm_admitted"], strict=True) if ok]
    h2_symbols = [s for s, ok in zip(table["symbol"], table["h2_admitted"], strict=True) if ok]
    report(f"  MM-admitted: {mm_symbols}; H2-admitted: {h2_symbols}")
    if not mm_symbols:
        raise SystemExit("no instrument is MM-admitted on D")
    instruments = {
        row["symbol"]: Instrument(
            row["symbol"], sig(row["clip_notional_usdt"]), sig(row["sigma_ref_bp_1m"])
        )
        for _, row in table.iterrows()
    }

    # 3. reversion ----------------------------------------------------------
    t = time.perf_counter()
    report("reversion: tapes, theta, beta and triggers on D")
    reversion, signal_table, state = reversion_stage(h2_symbols, d_days, access)
    clock.time("reversion tape", t)

    # 4. flags ----------------------------------------------------------------
    t = time.perf_counter()
    report("flags: both detectors on D")
    timelines, flag_table, flags_frame = flag_stage(mm_symbols, d_days, access)
    flag_table.to_csv(RESULTS / "mm_flag_rate_D.csv", index=False)
    flags_frame.to_csv(RESULTS / "mm_flags_D.csv", index=False)
    clock.time("flags", t)

    runner = Runner(args.workers, access, clock)
    subsets = nested_days(d_days, budgets, seed)
    pd.DataFrame(
        [
            {"budget": b, "days": " ".join(d.isoformat() for d in days)}
            for b, days in subsets.items()
        ]
    ).to_csv(RESULTS / "mm_search_days_D.csv", index=False)
    admitted = [instruments[s] for s in mm_symbols]

    # 5. the two searches ---------------------------------------------------
    grids = search_grids(registration)

    def make_s1(params: dict[str, Any], instrument: Instrument) -> Cell:
        return Cell(
            label(params),
            constant(s1_quoter(params, instrument)),
            config_for(instrument, float(params["soft_limit_clips"])),
        )

    def make_s2(params: dict[str, Any], instrument: Instrument) -> Cell:
        return Cell(
            label(params),
            constant(s2_quoter(params, instrument)),
            config_for(instrument, float(params["soft_limit_clips"])),
        )

    t = time.perf_counter()
    report("search S1")
    s1_table, s1_chosen, s1_near = search(
        "S1", grids["S1"], make_s1, admitted, subsets, runner, keep_fraction=keep
    )
    s1_table.to_csv(RESULTS / "mm_search_S1_D.csv", index=False)
    clock.time("search S1", t)
    report(f"  S1 chosen {s1_chosen}")

    t = time.perf_counter()
    report("search S2")
    s2_table, s2_chosen, s2_near = search(
        "S2", grids["S2"], make_s2, admitted, subsets, runner, keep_fraction=keep
    )
    s2_table.to_csv(RESULTS / "mm_search_S2_D.csv", index=False)
    clock.time("search S2", t)
    report(f"  S2 chosen {s2_chosen}")

    # 6. the chosen strategies on every D day --------------------------------
    t = time.perf_counter()
    daily: list[pd.DataFrame] = []

    def tagged(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
        out = frame.copy()
        out["strategy"] = strategy
        daily.append(out)
        return out

    s1_cells = {i.symbol: make_s1(s1_chosen, i) for i in admitted}
    s2_cells = {i.symbol: make_s2(s2_chosen, i) for i in admitted}
    twin_cells = {
        i.symbol: Cell(
            "S2-twin",
            constant(s2_quoter(s2_chosen, i, inside=False)),
            config_for(i, float(s2_chosen["soft_limit_clips"])),
        )
        for i in admitted
    }
    s1_rows = pd.concat(
        [tagged(runner.run(i, d_days, [s1_cells[i.symbol]]), "S1") for i in admitted]
    )
    s2_rows = pd.concat(
        [tagged(runner.run(i, d_days, [s2_cells[i.symbol]]), "S2") for i in admitted]
    )
    twin_rows = pd.concat(
        [tagged(runner.run(i, d_days, [twin_cells[i.symbol]]), "S2-twin") for i in admitted]
    )
    s1_score = daily_metrics(s1_rows)["net_usd_day"]
    s2_score = daily_metrics(s2_rows)["net_usd_day"]
    guards = "S2" if s2_score > s1_score else "S1"
    base_rows = s2_rows if guards == "S2" else s1_rows
    report(f"  S1 {s1_score:+.4f}, S2 {s2_score:+.4f} USDT/day on D: S3 guards {guards}")
    clock.time("chosen strategies", t)

    # 7. S3 -------------------------------------------------------------------
    t = time.perf_counter()
    report("S3: six guard cells on D")
    s3_rows_all = []
    for instrument in admitted:
        base_quoter = (
            s2_quoter(s2_chosen, instrument) if guards == "S2" else s1_quoter(s1_chosen, instrument)
        )
        soft = float((s2_chosen if guards == "S2" else s1_chosen)["soft_limit_clips"])
        cells = [
            Cell(
                f"action={action}/guard_window_min={window}",
                constant(RegimeGuard(base_quoter, action, float(window))),
                config_for(instrument, soft),
            )
            for action in grids["S3"]["action"]
            for window in grids["S3"]["guard_window_min"]
        ]
        tapes = {flag_module.FLAG_SIGNAL: timelines[instrument.symbol].tape()}
        s3_rows_all.append(runner.run(instrument, d_days, cells, tapes, cells_per_job=6))
    s3_rows = pd.concat(s3_rows_all, ignore_index=True)
    s3_scores = []
    for cell, group in s3_rows.groupby("cell", sort=False):
        metrics = daily_metrics(group)
        paired = group.merge(
            base_rows[["symbol", "day", "net"]], on=["symbol", "day"], suffixes=("", "_base")
        )
        diff = (paired["net"] - paired["net_base"]).to_numpy(dtype=np.float64)
        action, window = (part.split("=")[1] for part in str(cell).split("/"))
        s3_scores.append(
            {
                "action": action,
                "guard_window_min": int(window),
                **metrics,
                "minus_guarded_usd_day": float(diff.mean()),
                "minus_guarded_sd": float(diff.std(ddof=1)),
            }
        )
    s3_table = pd.DataFrame(s3_scores)
    s3_table.insert(0, "guards", guards)
    s3_table.to_csv(RESULTS / "mm_guard_S3_D.csv", index=False)
    best = s3_table.loc[s3_table["net_usd_day"].idxmax()]
    s3_chosen = {
        "guards": guards,
        "action": str(best["action"]),
        "guard_window_min": int(best["guard_window_min"]),
    }
    chosen_label = f"action={s3_chosen['action']}/guard_window_min={s3_chosen['guard_window_min']}"
    tagged(s3_rows[s3_rows["cell"] == chosen_label], "S3")
    clock.time("S3", t)
    report(f"  S3 chosen {s3_chosen}")

    # 8. S4 -------------------------------------------------------------------
    t = time.perf_counter()
    report("S4: six lean cells on D, H2-admitted instruments")
    s4_rows_all, s1_h2_rows = [], []
    for symbol in h2_symbols:
        instrument = instruments[symbol]
        soft = float(s1_chosen["soft_limit_clips"])
        base = s1_quoter(s1_chosen, instrument)
        info = reversion[symbol]
        cells = [
            Cell(
                f"lambda={lam}/one_sided={one_sided}",
                constant(ReversionLean(base, float(info["beta"]), float(lam), bool(one_sided))),
                config_for(instrument, soft),
            )
            for lam in grids["S4"]["lambda"]
            for one_sided in grids["S4"]["one_sided"]
        ]
        tapes = {"index_return_bp": info["tape"], "lean_trigger_s": info["lean"]}
        # Three cells a job: a busy instrument-day (XRPUSDT holds about a
        # million events) then stays well inside a worker's memory budget.
        s4_rows_all.append(runner.run(instrument, d_days, cells, tapes, cells_per_job=3))
        baseline = Cell("S1", constant(base), config_for(instrument, soft))
        s1_h2_rows.append(tagged(runner.run(instrument, d_days, [baseline]), "S1-h2"))
    s4_rows = pd.concat(s4_rows_all, ignore_index=True)
    s1_h2 = pd.concat(s1_h2_rows, ignore_index=True)
    s4_scores = []
    for cell, group in s4_rows.groupby("cell", sort=False):
        paired = group.merge(
            s1_h2[["symbol", "day", "net"]], on=["symbol", "day"], suffixes=("", "_s1")
        )
        paired["diff"] = paired["net"] - paired["net_s1"]
        pooled = paired.groupby("day")["diff"].sum()
        lam, one_sided = (part.split("=")[1] for part in str(cell).split("/"))
        row: dict[str, Any] = {
            "lambda": float(lam),
            "one_sided": one_sided == "True",
            **daily_metrics(group),
            "minus_s1_pooled_usd_day": float(pooled.mean()),
            "minus_s1_pooled_sd": float(pooled.std(ddof=1)),
        }
        for symbol, part in paired.groupby("symbol"):
            row[f"minus_s1_{symbol}"] = float(part["diff"].mean())
        s4_scores.append(row)
    s4_table = pd.DataFrame(s4_scores)
    s4_table.to_csv(RESULTS / "mm_lean_S4_D.csv", index=False)
    best4 = s4_table.loc[s4_table["minus_s1_pooled_usd_day"].idxmax()]
    s4_chosen = {"lambda": _plain(best4["lambda"]), "one_sided": bool(best4["one_sided"])}
    s4_label = f"lambda={s4_chosen['lambda']}/one_sided={s4_chosen['one_sided']}"
    pick = s4_rows[s4_rows["cell"] == s4_label]
    if pick.empty:
        raise SystemExit(f"no rows for the chosen S4 cell {s4_label}")
    tagged(pick, "S4")
    clock.time("S4", t)
    report(f"  S4 chosen {s4_chosen}")

    # 9. X1 and its taker twin -------------------------------------------------
    t = time.perf_counter()
    report("X1 and the taker twin on D")
    x1_rows = []
    for symbol in h2_symbols:
        instrument = instruments[symbol]
        tapes = reversion[symbol]["triggers"].tapes()
        cell = Cell("X1", SignalExecutor, config_for(instrument))
        x1_rows.append(tagged(runner.run(instrument, d_days, [cell], tapes), "X1"))
    x1 = pd.concat(x1_rows, ignore_index=True)
    clock.time("X1", t)

    # 10. power -----------------------------------------------------------------
    t = time.perf_counter()
    power_rows = []

    def power(name: str, values: pd.Series | np.ndarray, unit: str, note: str) -> None:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        sd = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
        power_rows.append(
            {
                "quantity": name,
                "unit": unit,
                "days": len(array),
                "mean_D": float(array.mean()) if len(array) else float("nan"),
                "sigma_D": sd,
                "mde_H": 2.0 * sd / math.sqrt(H_DAYS),
                "note": note,
            }
        )

    def per_day(rows: pd.DataFrame) -> pd.Series:
        ok = rows[(rows["status"] == "ok") & ~rows["excluded"].astype(bool)]
        return ok.groupby("day")["net"].sum()

    chosen_daily = per_day(base_rows)
    power(
        f"{guards} (the chosen strategy, guarded by S3) daily net",
        chosen_daily,
        "USDT/day",
        "the registered MDE",
    )
    power(
        "S1 daily net",
        per_day(s1_rows),
        "USDT/day",
        "H2.1's comparator on the MM-admitted instruments",
    )
    power("S2 daily net", per_day(s2_rows), "USDT/day", "H1 K1")
    power("S2 - twin, paired daily", per_day(s2_rows) - per_day(twin_rows), "USDT/day", "H1 K2")
    s3_daily = per_day(s3_rows[s3_rows["cell"] == chosen_label])
    power("S3 - guarded, paired daily", s3_daily - chosen_daily, "USDT/day", "H3 K1")
    s4_daily = per_day(pick)
    power(
        "S4 - S1, paired daily, pooled over H2 instruments",
        s4_daily - per_day(s1_h2),
        "USDT/day",
        "H2.1 K1",
    )
    x1_ok = x1[(x1["status"] == "ok") & ~x1["excluded"].astype(bool)].copy()
    takers = pd.concat([reversion[s]["taker"] for s in h2_symbols], ignore_index=True)
    takers["day"] = takers["day"].astype(str)
    triggers_per_day = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "symbol": s,
                        "day": pd.to_datetime(reversion[s]["triggers"].ts, utc=True).date.astype(
                            str
                        ),
                    }
                )
                for s in h2_symbols
            ]
        )
        .groupby(["symbol", "day"])
        .size()
        .rename("eligible_triggers")
    )
    x1_ok = x1_ok.merge(triggers_per_day.reset_index(), on=["symbol", "day"], how="left")
    x1_ok["bp_per_attempt"] = x1_ok["quoter_attempt_net_bp_sum"] / x1_ok["eligible_triggers"]
    x1_pooled = x1_ok.groupby("day").apply(
        lambda g: float(g["quoter_attempt_net_bp_sum"].sum() / g["eligible_triggers"].sum()),
        include_groups=False,
    )
    taker_pooled = takers.groupby("day")["net_bp"].mean()
    power("X1 net per attempt (misses zero), pooled", x1_pooled, "bp/attempt", "H2.2 K2")
    power(
        "X1 - taker twin, paired daily, pooled", x1_pooled - taker_pooled, "bp/attempt", "H2.2 K1"
    )
    power_table = pd.DataFrame(power_rows)
    power_table.to_csv(RESULTS / "mm_power_D.csv", index=False)
    clock.time("power", t)

    daily_table = pd.concat(daily, ignore_index=True)
    keep_columns = [
        "strategy",
        "symbol",
        "day",
        "cell",
        "status",
        "excluded",
        "flags",
        "net",
        "making",
        "spread",
        "adverse",
        "inventory",
        "fees",
        "funding",
        "fills",
        "fills_queue",
        "fills_through",
        "flattens",
        "max_abs_position",
        "maker_turnover",
        "taker_turnover",
        "clip_over_touch",
    ]
    extra = [c for c in daily_table.columns if c.startswith("quoter_")]
    daily_table = daily_table[[c for c in keep_columns if c in daily_table.columns] + extra]
    daily_table.to_csv(RESULTS / "mm_daily_D.csv", index=False)

    signal_table["x1_attempts"] = signal_table["symbol"].map(
        x1_ok.groupby("symbol")["quoter_attempts"].sum()
    )
    signal_table["x1_fills"] = signal_table["symbol"].map(
        x1_ok.groupby("symbol")["quoter_attempts_filled"].sum()
    )
    signal_table.to_csv(RESULTS / "mm_signals_D.csv", index=False)

    # 11. the frozen values ------------------------------------------------------
    flag_rate = float(sum(len(timelines[s]) for s in mm_symbols) / (len(mm_symbols) * len(d_days)))
    admission_rows = []
    for _, row in table.iterrows():
        admission_rows.append(
            {
                "symbol": row["symbol"],
                "tick_bp": sig(row["tick_bp"], 4),
                "spread_bp": sig(row["spread_bp_time_weighted"], 4),
                "share_two_ticks": sig(row["share_at_two_ticks_or_more"], 4),
                "touch_over_print": sig(row["median_touch_over_median_print"], 4),
                "prints_per_day": round(float(row["prints_per_day"])),
                "markout_1s_bp": sig(row["market_wide_passive_markout_bp_1s"], 4),
                "markout_5s_bp": sig(row["market_wide_passive_markout_bp_5s"], 4),
                "markout_30s_bp": sig(row["market_wide_passive_markout_bp_30s"], 4),
                "coverage_D": sig(row["coverage_D"], 4),
                "coverage_H": sig(row["coverage_H"], 4),
                "mm_admitted": bool(row["mm_admitted"]),
                "h2_admitted": bool(row["h2_admitted"]),
            }
        )
    chosen_sigma = power_rows[0]["sigma_D"]
    values: dict[tuple[str, ...], Any] = {
        ("admission", "table"): admission_rows,
        ("simulator", "clip", "notional_usdt"): {
            s: instruments[s].clip_notional for s in STUDY_SYMBOLS
        },
        ("simulator", "volatility", "sigma_ref"): {
            s: instruments[s].sigma_ref for s in STUDY_SYMBOLS
        },
        ("strategies", "S3", "chosen", "guards"): s3_chosen["guards"],
        ("strategies", "S3", "chosen", "action"): s3_chosen["action"],
        ("strategies", "S3", "chosen", "guard_window_min"): s3_chosen["guard_window_min"],
        ("strategies", "S4", "theta_bp"): {s: sig(reversion[s]["theta"]) for s in h2_symbols},
        ("strategies", "S4", "beta"): {s: sig(reversion[s]["beta"]) for s in h2_symbols},
        ("strategies", "S4", "chosen", "lambda"): s4_chosen["lambda"],
        ("strategies", "S4", "chosen", "one_sided"): s4_chosen["one_sided"],
        ("d_search", "neighbourhood", "S1"): s1_near,
        ("d_search", "neighbourhood", "S2"): s2_near,
        ("metrics", "minimum_detectable_effect", "sigma_d_usd_day"): sig(chosen_sigma),
        ("metrics", "minimum_detectable_effect", "value_usd_day"): sig(
            2.0 * chosen_sigma / math.sqrt(H_DAYS)
        ),
        ("hypotheses", "H2.3", "gate_threshold"): sig(state["h2_3_gate_threshold"]),
        ("hypotheses", "H3", "flag_rate_on_d_per_day"): sig(flag_rate),
        ("amendment", "date"): AMENDMENT_DATE,
    }
    for axis, value in s1_chosen.items():
        values[("strategies", "S1", "chosen", axis)] = value
    for axis, value in s2_chosen.items():
        values[("strategies", "S2", "chosen", axis)] = value

    clock.stages["total"] = time.perf_counter() - started
    compute = pd.DataFrame(
        [{"stage": k, "wall_s": round(v, 1)} for k, v in clock.stages.items()]
        + [
            {
                "stage": "peak RSS, any simulation worker (MB)",
                "wall_s": round(clock.worker_peak_mb, 0),
            },
            {"stage": "peak RSS, any worker process (MB)", "wall_s": round(children_peak_mb(), 0)},
            {"stage": "peak RSS, driver (MB)", "wall_s": round(peak_rss_mb(), 0)},
            {"stage": "workers", "wall_s": args.workers},
        ]
    )
    compute.to_csv(RESULTS / "mm_compute_D.csv", index=False)
    Path("artifacts/mm").mkdir(parents=True, exist_ok=True)
    Path("artifacts/mm/freeze_D.json").write_text(
        json.dumps({".".join(k): v for k, v in values.items()}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    report(json.dumps({".".join(k): v for k, v in values.items()}, indent=1, default=str))
    report(compute.to_string(index=False))

    if args.freeze:
        path = prereg.CONFIG_PATH
        frozen = prereg.freeze_text(path.read_text(encoding="utf-8"), values)
        path.write_text(frozen, encoding="utf-8")
        report(f"frozen: {path} sha256 {prereg.sha256_bytes(frozen.encode())}")


if __name__ == "__main__":
    main()
