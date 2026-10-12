"""Stage 0, before any 2025 fetch: the hard budget and the walk-table parity gate.

On the two 2024 days already on disk (BTCUSDT 2024-02-11, a quiet day, and
2024-03-05, the busiest), read through the development access:

1. **Unit costs.** Every piece of code D runs is timed on those days or on
   synthetic arrays of the registered size: the walk table and each hold's
   table, the features, labels and gate inputs, a configuration's cells scored
   from a tape, Null B's replay of a cell-day on another day's tables, the
   executor, the simulator's signal-entry quoter, a Null A draw of the whole
   choice at the catalogue's size, and the boosted, logistic and ridge fits.
   The network is timed in a process of its own (it does not share a process
   with XGBoost): a training batch and inference per row, each over as many
   rows as the busier timing day holds, as D trains on whole days streamed
   from the store and scores whole test days; a sample's figures are kept
   beside them. The fold-level leakage test's network is timed on the CPU, a
   training batch and inference per row, and a day of a float16 store read and
   written, the unit of the test's mutated scratch root.
2. **The projection** of D's wall time (:func:`projection`), Null B at full size,
   the refit and the fold-level leakage test's fits (twice per fold, the network
   one epoch on the CPU) included, at the registered worker counts (8, the
   simulator 6, the replay's as measured) whatever ``--workers`` says, against
   the six-day ceiling; over it, the
   registered reduction order cuts the 120 s hold, then stratum C's
   micro+tensor inputs, then the ``quiet_out`` gate, until it fits. Cuts are
   not made here: they are written into the record with the cut catalogue's
   sha256, which a dated amendment must register before stage 1 runs.
3. **The walk-table parity test** (``search_engine.null_b.stage0_parity``) on
   both days at every hold; a failure blocks stage B.
4. **The replay's peak memory and seconds**, the two registered timing days'
   archives under ``data/timing_archives/`` (no other), each replayed alone in
   a fresh process, and the replay worker count the largest peak allows; the
   slower archive's seconds stand for a day in the projection, in place of the
   registered 2-3 minutes. Stage 0 refuses to run without both archives.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.taker import TakerBook, TakerSettings, execute_day
from trading_research.data import study_access as sa
from trading_research.data import study_loader as sl
from trading_research.pipeline.short_horizon.context import Study
from trading_research.search_engine.catalogue import catalogue_sha256, parse_key
from trading_research.search_engine.walk_tables import WalkTable, hold_table

NS_PER_S = 1_000_000_000
#: The registered ceiling for D, in hours, and the reduction order.
CEILING_HOURS = 6 * 24
REDUCTION_ORDER: tuple[str, ...] = ("hold_s 120", "C's micro+tensor inputs", "quiet_out gate")


def timed(fn: Callable[[], Any], repeat: int = 1) -> tuple[float, Any]:
    """Seconds per call (the best of ``repeat``) and the last result."""
    best, result = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def load_timing_day(study: Study, day: date) -> tuple[pd.DataFrame, TakerBook, pd.DataFrame]:
    """A 2024 day's book frame, the executor's book with funding, and its prints."""
    access = sa.development_access()
    book = sl.load_book_day(study.symbol, day, study.roots.book, max_sequence_gaps=None,
                            access=access)  # fmt: skip
    if book.frame is None:
        raise FileNotFoundError(f"stage 0 times the code on {day}, which has no book on disk")
    funding = sl.load_funding_day(study.symbol, day, study.roots.funding, access=access)
    prints = sl.load_prints_day(study.symbol, day, study.roots.trades, access=access)
    taker = TakerBook.from_frame(book.frame, day, funding=funding)
    return book.frame, taker, prints


def decisions(rows: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` decision rows in time order with random sides, for the parity
    test and the executor's timing."""
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(rows, size=min(count, rows), replace=False))
    return picked.astype(np.int64), rng.choice(np.array([-1, 1]), size=len(picked))


def parity(books: Sequence[TakerBook], holds_s: Sequence[float]) -> dict[str, Any]:
    """The registered parity test at every hold on every timing day."""
    from trading_research.search_engine.null_b import stage0_parity

    out: dict[str, Any] = {}
    for book in books:
        rows, side = decisions(len(book.ts), 2_000, seed=book.day.toordinal())
        for hold in holds_s:
            found = stage0_parity(book, rows, side, round(hold * NS_PER_S))
            out[f"{book.day}/{hold:g}"] = {
                "trades": found.trades,
                "mismatches": list(found.mismatches),
            }
    return {"ok": all(not v["mismatches"] for v in out.values()), "runs": out}


def walk_costs(book: TakerBook, holds_s: Sequence[float]) -> dict[str, float]:
    settings = TakerSettings()
    walk_s, walk = timed(lambda: WalkTable.from_book(book, settings.clip))
    holds = []
    for hold in holds_s:
        holds.append(timed(partial(hold_table, walk, round(hold * NS_PER_S), settings))[0])
    rows, side = decisions(len(book.ts), 2_000, seed=1)
    executor_s, _ = timed(lambda: execute_day(book, book.ts[rows], side, 50 * NS_PER_S, settings))
    return {
        "walk_table_s": walk_s,
        "hold_table_s": float(np.mean(holds)),
        "executor_2000_s": executor_s,
    }


def catalogue_after(keys: Sequence[str], cuts: Sequence[str]) -> list[str]:
    """The catalogue with the registered cuts applied, in order."""
    out = list(keys)
    for cut in cuts:
        if cut == REDUCTION_ORDER[0]:
            out = [k for k in out if parse_key(k).hold_s != 120]
        elif cut == REDUCTION_ORDER[1]:
            out = [
                k
                for k in out
                if not (parse_key(k).stratum == "C" and parse_key(k).inputs == "micro+tensor")
            ]
        elif cut == REDUCTION_ORDER[2]:
            out = [k for k in out if parse_key(k).gate != "quiet_out"]
        else:
            raise ValueError(f"{cut!r} is not in the reduction order {REDUCTION_ORDER}")
    return out


def cut_sha256(keys: Sequence[str], cuts: Sequence[str]) -> str:
    return catalogue_sha256(catalogue_after(keys, cuts))


def signal_costs(frame: pd.DataFrame, prints: pd.DataFrame) -> dict[str, float]:
    """Features (the 58 timed columns and the tensor), labels and gate inputs on one day."""
    from trading_research.features.lob_tensor import fit_tensor_calibration, lob_tensor
    from trading_research.features.study_set import microstructure_registry
    from trading_research.features.timed import build_timed
    from trading_research.labels.taker_vwap import WalkedTakerLabel, forward_move_bp
    from trading_research.search_engine.market_gate import gate_inputs

    registry = microstructure_registry()
    micro_s, _ = timed(
        lambda: build_timed(frame, registry.names(), prints=prints, registry=registry)
    )
    calibration = fit_tensor_calibration([frame])
    tensor_s, _ = timed(lambda: lob_tensor(frame, calibration))
    label_s, _ = timed(lambda: WalkedTakerLabel(50.0).outcomes(frame))
    forward_s, _ = timed(lambda: forward_move_bp(frame, 50.0))
    gate_s, _ = timed(lambda: gate_inputs(frame))
    return {
        "features_s": micro_s + tensor_s,
        "labels_s": label_s + forward_s,
        "gate_inputs_s": gate_s,
    }


def scoring_costs(
    keys: Sequence[str], frames: Sequence[pd.DataFrame], books: Sequence[TakerBook]
) -> tuple[dict[str, float], dict[str, Any]]:
    """S25's configuration's cells scored from a tape on the first day, and their
    entries replayed on the second day's table (Null B's unit)."""
    from trading_research.search_engine.cells import score_configuration
    from trading_research.search_engine.market_gate import fit_gate_levels, gate_inputs, gate_states
    from trading_research.search_engine.null_b import entries_of, replay
    from trading_research.search_engine.tapes import hold_first_tape
    from trading_research.strategies.hold_first import fit_hold_first

    s25 = parse_key(
        "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none"
    )
    cells = [parse_key(k) for k in keys if parse_key(k).configuration == s25.configuration]
    settings = TakerSettings()
    tables = [
        hold_table(WalkTable.from_book(b, settings.clip), 50 * NS_PER_S, settings) for b in books
    ]
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=len(books[0].ts))
    rates = sorted({c.trades_per_day for c in cells})
    rules = {rate: fit_hold_first(proba, days=1.0, trades_per_day=rate) for rate in rates}
    inputs = gate_inputs(frames[0])
    gates = gate_states(inputs, fit_gate_levels(inputs))
    tape = hold_first_tape(books[0].day, s25.configuration, books[0].ts, proba, rules[max(rates)],
                           gates, np.zeros(0, dtype=np.int64))  # fmt: skip
    score_s, scored = timed(lambda: score_configuration(tape, tables[0], cells, rules))
    entries = [entries_of(tables[0], trades) for trades in scored.values()]
    replay_s, _ = timed(lambda: [replay(tables[1], e) for e in entries])
    trades = scored[s25.key]
    costs = {"cell_day_s": score_s / len(cells), "null_b_cell_day_s": replay_s / len(cells)}
    return costs, {"rows": trades.rows, "side": trades.side, "day": books[0].day}


def simulator_cost(study: Study, entries: dict[str, Any], max_gaps: int) -> float:
    """One signal-entry quoter cell-day on the first timing day."""
    from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
    from trading_research.market_making.simulator import simulate_day
    from trading_research.pipeline.short_horizon_execution import study_config

    roots = study.roots
    events = sl.load_events_day(
        study.symbol, entries["day"], book_roots=[roots.book], trades_root=roots.trades,
        funding_root=roots.funding, access=sa.development_access(),
    )  # fmt: skip
    stamps = events.book_ts[np.minimum(entries["rows"], len(events.book_ts) - 1)]
    tapes = entry_tapes(stamps, entries["side"])
    config = study_config(max_gaps)
    seconds, _ = timed(lambda: simulate_day(events, SignalEntryQuoter(50.0), config, tapes))
    return seconds


def null_a_draw_cost(keys: Sequence[str], test_days: int) -> float:
    """One Null A draw (the whole choice and A on hour sums) at the catalogue's size."""
    from trading_research.search_engine.catalogue import GROUP_COLUMNS, ORDERED_AXES, axes_table
    from trading_research.validation.choice import neighbourhood_index
    from trading_research.validation.placebo_search import HourCells, null_a

    table = axes_table(keys)
    index = neighbourhood_index(table, ORDERED_AXES, by=GROUP_COLUMNS)
    strata = {s: np.flatnonzero(table["stratum"].to_numpy() == s) for s in ("C", "R")}
    rng = np.random.default_rng(0)
    shape = (len(keys), test_days * 24)
    trades = rng.poisson(0.3, size=shape)
    notional = trades * 1_000.0
    hours = HourCells(
        np.repeat(np.arange(test_days), 24), test_days, trades, notional,
        rng.normal(size=shape) * trades, rng.normal(size=shape) * trades,
        rng.normal(size=shape) * trades,
    )  # fmt: skip
    seconds, _ = timed(lambda: null_a(hours, strata, index, draws=2))
    return seconds / 2


def fit_costs(scale: Any, rows: int) -> dict[str, float]:
    """Fits and predictions of the three models that share a process, per row."""
    from trading_research.pipeline.short_horizon.models import bound_ridge, classifier

    rng = np.random.default_rng(0)
    out: dict[str, float] = {"rows": float(rows)}
    for name, model, width in (
        ("gbm", classifier("gbm", scale), 40),
        ("logistic", classifier("logistic", scale), 100),
        ("bound_ridge", bound_ridge(scale), 40),
    ):
        x = pd.DataFrame(rng.normal(size=(rows, width)).astype(np.float32))
        x.columns = [f"f{i}" for i in range(width)]
        y = pd.Series(
            rng.integers(-1, 2, size=rows) if name != "bound_ridge" else rng.normal(size=rows)
        )
        fit_s, fitted = timed(partial(model.fit, x, y))
        predict = fitted.predict_proba if name != "bound_ridge" else fitted.predict_mean_bp
        predict_s, _ = timed(partial(predict, x))
        out[f"{name}_fit_row_s"] = fit_s / rows
        out[f"{name}_predict_row_s"] = predict_s / rows
    return out


def network_costs(params: dict[str, Any], rows: int, width: int, day_rows: int) -> dict[str, float]:
    """The network's training batch and its inference per row, each over a whole
    day of ``day_rows`` rows (one epoch of a fit on them, then a prediction of
    them), as D trains on days streamed from the store and scores whole test
    days; the same two on a ``rows`` sample are kept beside them, since a
    sample's time per batch or per row also carries the device's start-up and
    the first kernels, which a day's rows spread thin. Run in a fresh process.
    The cost does not depend on the values, and random inputs leave every window
    complete, so every row is trained on and scored: an upper bound for a day
    with warm-up or missing inputs."""
    from trading_research.models.tcn import GaussianTCN

    rng = np.random.default_rng(0)
    columns = [f"f{i}" for i in range(width)]
    out: dict[str, float] = {}
    for name, n in (("sample", rows), ("day", day_rows)):
        x = pd.DataFrame(rng.normal(size=(n, width)).astype(np.float32), columns=columns)
        y = pd.Series(rng.normal(size=n))
        model = GaussianTCN(**{**params, "epochs": 1})
        fit_s, _ = timed(partial(model.fit, x, y))
        predict_s, _ = timed(partial(model.predict_gaussian, x))
        out[f"batch_{name}_s"] = fit_s / max(1, -(-n // int(model.batch_size)))
        out[f"inference_{name}_row_s"] = predict_s / n
        out[f"inference_{name}_s"] = predict_s
    return {
        "batch_s": out["batch_day_s"],
        "inference_row_s": out["inference_day_row_s"],
        "inference_day_s": out["inference_day_s"],
        "batch_sample_s": out["batch_sample_s"],
        "inference_sample_row_s": out["inference_sample_row_s"],
    }


def cpu_network_costs(params: dict[str, Any], rows: int, width: int) -> dict[str, float]:
    """The fold-level leakage test's network on the CPU, where it is fitted for one
    epoch: a training batch and inference per row over ``rows`` random rows. The
    CPU has no device start-up for a whole day's rows to spread, so a sample's
    times stand for a fit's. Run in a fresh process."""
    from trading_research.models.tcn import GaussianTCN

    rng = np.random.default_rng(1)
    columns = [f"f{i}" for i in range(width)]
    x = pd.DataFrame(rng.normal(size=(rows, width)).astype(np.float32), columns=columns)
    y = pd.Series(rng.normal(size=rows))
    model = GaussianTCN(**{**params, "epochs": 1, "device": "cpu"})
    fit_s, _ = timed(partial(model.fit, x, y))
    predict_s, _ = timed(partial(model.predict_gaussian, x))
    return {
        "cpu_batch_s": fit_s / max(1, -(-rows // int(model.batch_size))),
        "cpu_inference_row_s": predict_s / rows,
    }


def store_day_cost(folder: Path, rows: int, width: int) -> float:
    """Reading and writing one day of a float16 store of ``rows`` x ``width``: the
    fold-level test's unit for each day it mutates on its scratch root."""
    import shutil

    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(rows, width)).astype(np.float16))
    frame.columns = [f"f{i}" for i in range(width)]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "day.parquet"

    def copy() -> None:
        frame.to_parquet(path, compression="zstd", index=False)
        pd.read_parquet(path)

    seconds, _ = timed(copy)
    shutil.rmtree(folder, ignore_errors=True)
    return seconds


def timing_archives(study: Study) -> list[Path]:
    """The two registered timing days' ``ob500`` archives under ``data/timing_archives/``;
    stage 0 refuses to run without both, and reads no other archive there."""
    folder = study.roots.archives.parent / "timing_archives"
    wanted = [folder / f"{study.symbol}-{day}-ob500.zip" for day in study.scale.timing_days]
    missing = [path.name for path in wanted if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"stage 0 measures the replay on the registered timing days' archives; "
            f"missing under {folder.name}/: {missing}"
        )
    return wanted


def replay_memory(study: Study) -> dict[str, Any]:
    """The replay's peak memory and seconds on each registered 2024 timing archive,
    and the workers the largest peak allows. The seconds are ``reconstruct`` alone
    in a fresh process (the fetch's zip test, message scan and parquet write are
    not in them), the peak that process's resident memory after the day."""
    from trading_research.data.fetch_stage import replay_peak_rss, replay_workers

    found = timing_archives(study)
    peaks = {}
    seconds = {}
    for archive in found:
        seconds[archive.name], peak = timed(partial(replay_peak_rss, archive, symbol=study.symbol))
        peaks[archive.name] = int(peak)
    peak = max(peaks.values())
    return {"measured": True, "peak_rss_bytes": peaks, "replay_s": seconds,
            "workers": replay_workers(peak)}  # fmt: skip


#: The registered study's simulator cell-days on D (stage C) and on H's read, on 150 test days.
SIMULATOR_CELL_DAYS_C = 7_750
SIMULATOR_CELL_DAYS_H = 31_300
FOLDS = 5
#: The registered worker counts the projection uses, whatever ``--workers`` says:
#: 8 for everything but the replay and the simulator, at most 6 for the simulator;
#: the replay's are measured (:func:`replay_memory`).
GENERAL_WORKERS = 8
SIMULATOR_WORKERS = 6


@dataclass(frozen=True)
class Sizes:
    """What the projection multiplies unit costs by, for the run's scale: 150 test
    days, 212 days of W and D, 864,000 rows a day and 4 million training rows on
    the registered study."""

    test_days: int
    development_days: int
    rows_per_day: int
    training_rows: int
    epochs: int
    inner_days: int
    shifts: int
    #: Folds the fold-level leakage test fits in, twice each.
    leakage_folds: int = FOLDS

    @classmethod
    def of(cls, study: Study, rows_per_day: int) -> Sizes:
        from trading_research.pipeline.short_horizon.plan import walk_forward

        d_days = study.scale.days("D")
        test = [d for d in d_days if d.month >= 2]
        development = len(study.scale.days("W")) + len(d_days)
        inner = max((len(f.inner) for f in walk_forward(d_days)), default=0)
        epochs = int(study.scale.params("tcn").get("max_epochs", 20))
        training = min(study.scale.max_training_rows, rows_per_day // 10 * development)
        return cls(len(test), development, rows_per_day, training, epochs, inner,
                   len(study.scale.null_b_shifts(len(test))),
                   min(FOLDS, study.scale.leakage_folds or FOLDS))  # fmt: skip

    @property
    def share(self) -> float:
        """The run's test days as a share of the registered 150."""
        return self.test_days / 150.0


def counts(keys: Sequence[str]) -> dict[str, int]:
    """What the projection multiplies: cells, configurations by model, holds."""
    parsed = [parse_key(k) for k in keys]
    configurations = {p.configuration: p.model for p in parsed}
    out = {"cells": len(parsed), "configurations": len(configurations)}
    out |= {
        m: sum(v == m for v in configurations.values())
        for m in ("gbm", "logistic", "tcn", "bound_ridge")
    }
    out["holds"] = len({p.hold_s for p in parsed})
    out["C_cells"] = sum(p.stratum == "C" for p in parsed)
    return out


def projection(
    costs: dict[str, float], n: dict[str, int], size: Sizes, workers: int, null_a_draws: int
) -> dict[str, float]:
    """D's wall time in hours per stage, from unit costs, the catalogue's counts and sizes;
    ``workers`` the general count, the simulator's at most :data:`SIMULATOR_WORKERS`."""
    w, sim_w = max(1, workers), max(1, min(SIMULATOR_WORKERS, workers))
    fit = {
        m: costs[f"{m}_fit_row_s"] * size.training_rows for m in ("gbm", "logistic", "bound_ridge")
    }
    batches = size.training_rows / costs["network_batch_size"]
    inner_rows = size.inner_days * size.rows_per_day
    fit["tcn"] = size.epochs * (
        batches * costs["network_batch_s"] + inner_rows * costs["network_inference_row_s"]
    )
    predict = {m: costs[f"{m}_predict_row_s"] for m in ("gbm", "logistic", "bound_ridge")}
    predict["tcn"] = costs["network_inference_row_s"]
    scored_rows = (size.test_days + FOLDS * size.inner_days) * size.rows_per_day
    tables_s = costs["walk_table_s"] + n["holds"] * costs["hold_table_s"]
    cell_days = n["cells"] * size.test_days
    # the fold-level leakage test: per fold, its fits twice (the network one epoch on
    # the CPU), and the scratch root's mutated days (a test month and the cutoff's day)
    cpu_network = (n["tcn"] > 0) * (
        size.training_rows / costs["network_batch_size"] * costs["network_cpu_batch_s"]
        + inner_rows * costs["network_cpu_inference_row_s"]
    )
    leak_fits = (
        cpu_network
        + fit["gbm"]
        + fit["logistic"]
        + fit["bound_ridge"]
        + inner_rows * (predict["gbm"] + predict["logistic"])
    )
    month_days = size.test_days / FOLDS
    leak_root = 1.1 * month_days * costs["store_day_s"] + n["holds"] * costs["labels_s"]
    stage = {
        "1": size.development_days * (costs["replay_day_s"] / costs["replay_workers"] + costs["features_s"] / w),
        "1_fold_leakage": size.leakage_folds * (2 * leak_fits + leak_root),
        "A": sum(FOLDS * n[m] * fit[m] + n[m] * scored_rows * predict[m] for m in fit)
        + size.development_days * n["holds"] * costs["labels_s"],
        "B": (cell_days * costs["cell_day_s"] + size.test_days * tables_s) / w,
        "D_null_A": null_a_draws * costs["null_a_draw_s"],
        "D_null_B": (size.shifts * cell_days * costs["null_b_cell_day_s"] + size.test_days * tables_s) / w,
        "D_refit": FOLDS * (n["gbm"] * fit["gbm"] + n["logistic"] * fit["logistic"])
        + (n["gbm"] * predict["gbm"] + n["logistic"] * predict["logistic"]) * scored_rows
        + n["C_cells"] * size.test_days * costs["cell_day_s"] / w,
        "C": SIMULATOR_CELL_DAYS_C * size.share * costs["simulator_cell_day_s"] / sim_w,
        "E": 12 * max(fit["gbm"], fit["logistic"]) + 3 * fit["bound_ridge"] + 8 * fit["tcn"]
        + SIMULATOR_CELL_DAYS_H * size.share * costs["simulator_cell_day_s"] / sim_w,
    }  # fmt: skip
    hours = {name: seconds / 3600.0 for name, seconds in stage.items()}
    hours["total"] = sum(hours.values())
    return hours


def choose_cuts(
    keys: Sequence[str], costs: dict[str, float], size: Sizes, workers: int, null_a_draws: int
) -> tuple[list[str], dict[str, float]]:
    """The reduction order's prefix that brings D under the ceiling (none if it fits)."""
    for k in range(len(REDUCTION_ORDER) + 1):
        cuts = list(REDUCTION_ORDER[:k])
        hours = projection(costs, counts(catalogue_after(keys, cuts)), size, workers, null_a_draws)
        if hours["total"] <= CEILING_HOURS:
            return cuts, hours
    return list(REDUCTION_ORDER), hours


def projected(
    keys: Sequence[str], costs: dict[str, float], size: Sizes, null_a_draws: int
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """The cuts, the hours after them and the uncut hours, at the registered worker
    counts: the command line's ``--workers`` runs a stage and never enters here."""
    cuts, hours = choose_cuts(keys, costs, size, GENERAL_WORKERS, null_a_draws)
    return cuts, hours, projection(costs, counts(keys), size, GENERAL_WORKERS, null_a_draws)


def run(study: Study) -> dict[str, Any]:
    """Stage 0's record: unit costs, the projection and its cuts, parity, replay memory."""
    from trading_research.pipeline.short_horizon.models import in_fresh_process

    registration = study.registration
    catalogue = study.catalogue()
    keys = study.keys
    timing_archives(study)  # refused before anything is timed if either archive is missing
    holds = sorted({parse_key(k).hold_s for k in keys})
    loaded = [load_timing_day(study, day) for day in study.scale.timing_days]
    frames = [frame for frame, _, _ in loaded]
    books = [book for _, book, _ in loaded]
    gate = parity(books, holds)
    costs: dict[str, float] = {}
    for frame, book, prints in loaded:
        day_costs = walk_costs(book, holds) | signal_costs(frame, prints)
        for name, value in day_costs.items():
            costs[name] = max(costs.get(name, 0.0), value)  # the busier day's
    scored, entries = scoring_costs(keys, frames, books)
    costs |= scored
    costs["simulator_cell_day_s"] = simulator_cost(study, entries, max_gaps=10**6)
    test_days = len([d for d in study.scale.days("D") if d.month >= 2])
    costs["null_a_draw_s"] = null_a_draw_cost(keys, test_days)
    fit_rows = min(1_000_000, study.scale.max_training_rows)
    costs |= fit_costs(study.scale, fit_rows)
    params = study.scale.params("tcn")
    network_rows = min(20_000, study.scale.max_training_rows)
    day_rows = max(len(frame) for frame in frames)  # the busier timing day, whole
    network = in_fresh_process(network_costs, params, network_rows, 70, day_rows)
    costs["network_batch_s"] = network["batch_s"]
    costs["network_inference_row_s"] = network["inference_row_s"]
    costs["network_day_rows"] = float(day_rows)
    costs["network_inference_day_s"] = network["inference_day_s"]
    costs["network_sample_rows"] = float(network_rows)
    costs["network_batch_sample_s"] = network["batch_sample_s"]
    costs["network_inference_sample_row_s"] = network["inference_sample_row_s"]
    costs["network_batch_size"] = float(params.get("batch_size", 512))
    cpu_rows = min(50_000, study.scale.max_training_rows)
    cpu = in_fresh_process(cpu_network_costs, params, cpu_rows, 70)
    costs["network_cpu_rows"] = float(cpu_rows)
    costs["network_cpu_batch_s"] = cpu["cpu_batch_s"]
    costs["network_cpu_inference_row_s"] = cpu["cpu_inference_row_s"]
    costs["store_day_s"] = store_day_cost(study.dir("0", "store_timing"), day_rows, 70)
    memory = replay_memory(study)
    costs["replay_workers"] = float(memory["workers"])
    costs["replay_day_s"] = max(memory["replay_s"].values())
    size = Sizes.of(study, max(len(frame) for frame in frames))
    draws = study.scale.null_a_draws
    cuts, hours, uncut = projected(catalogue.keys, costs, size, draws)
    return {
        "scale": study.scale.name,
        "commit": study.commit(),
        "config_sha256": registration.sha256,
        "catalogue_sha256": catalogue.sha256,
        "cells_scored": len(keys),
        "timing_days": [d.isoformat() for d in study.scale.timing_days],
        "workers": study.workers,
        "projection_workers": {
            "general": GENERAL_WORKERS,
            "simulator": SIMULATOR_WORKERS,
            "replay": memory["workers"],
        },
        "unit_costs": costs,
        "sizes": asdict(size),
        "projected_hours": uncut,
        "ceiling_hours": CEILING_HOURS,
        "cuts": cuts,
        "projected_hours_after_cuts": hours,
        "cut_catalogue_sha256": cut_sha256(catalogue.keys, cuts) if cuts else None,
        "parity": gate,
        "replay_memory": memory,
    }


def require_parity(record: dict[str, Any]) -> None:
    """Stage B's gate: the parity test passed at stage 0."""
    if not record["parity"]["ok"]:
        failed = {k: v for k, v in record["parity"]["runs"].items() if v["mismatches"]}
        raise RuntimeError(
            f"the walk-table parity test failed at stage 0 ({sorted(failed)}); stage B is "
            "blocked until a dated amendment fixes it"
        )


def require_no_pending_cuts(record: dict[str, Any], accepted_catalogue_sha256: str) -> None:
    """Stage 1's gate: any cut stage 0 needed is registered by an amendment."""
    if record["cuts"] and record["cut_catalogue_sha256"] != accepted_catalogue_sha256:
        raise RuntimeError(
            f"stage 0 projected D over the ceiling and cut {record['cuts']}; a dated amendment "
            f"registering the cut catalogue {record['cut_catalogue_sha256']} must be committed "
            "before stage 1"
        )
