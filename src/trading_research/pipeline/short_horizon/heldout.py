"""What a block read computes for each slot, once per usable day of the block.

Everything is read under the read session's access, with only frozen values:
the configuration's frozen placeholders (the gap threshold, the selected
features, the detector's hour profile), the fitted-values file, the finalist
files and the frozen weights. For each usable day:

* the day's features from its own rows (rolling state from the previous day of
  the same block only; a block's first day, and a day after a hole, start cold
  and are not scored for N minutes), the frozen models' scores on every row,
  and each configuration's candidate tape under the frozen thresholds, gate
  levels and the frozen detector's flags;
* for the taker slots (:func:`taker_days`): the true slot through the event-time
  executor, and from walk tables its placebos (sign flip, random directions,
  one-day shift, all-long and all-short), its neighbour cells and its re-run at
  110 ms;
* for the passive slots (:mod:`.heldout_passive`): the simulator under the same
  variants and the queue brackets.

The results are :class:`~trading_research.pipeline.short_horizon.statuses.SlotDays`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property, partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import study_access as sa
from trading_research.data import study_loader as sl
from trading_research.pipeline.short_horizon import features as ft
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.fits import Configuration, configurations
from trading_research.pipeline.short_horizon.passive import SimCell as SimCellType
from trading_research.search_engine.catalogue import parse_key


@dataclass
class Block:
    """One block's read: its days under the session's access, and every frozen input."""

    study: Study
    name: str
    access: sa.StudyAccess
    days_planned: tuple[date, ...]
    notes: list[str] = field(default_factory=list)
    #: The frozen values, when the configuration is not frozen yet (the registered
    #: dry run of stage E reads them before the amendment writes them).
    values: dict[tuple[str, ...], Any] | None = None
    #: The last day the detector's stream runs to (the block's own by default).
    end: date | None = None
    #: The folder under ``reads/`` its days are kept in (the block's name by default).
    label: str | None = None

    @cached_property
    def registration(self) -> Any:
        from trading_research.pipeline import short_horizon_registration as shr

        return shr.Registration.load(repo=self.study.repo).require_frozen()

    def value(self, path: tuple[str, ...]) -> Any:
        """A frozen value: the given one, else the frozen configuration's."""
        if self.values is not None:
            return self.values[path]
        return self.registration.value(path)

    @property
    def threshold(self) -> int:
        return int(self.value(("data", "max_sequence_gaps")))

    @cached_property
    def fitted(self) -> dict[str, Any]:
        from trading_research.pipeline.short_horizon_ledger import FITTED_VALUES_PATH

        loaded: dict[str, Any] = json.loads(
            (self.study.repo / FITTED_VALUES_PATH).read_text(encoding="utf-8")
        )
        return loaded

    def finalist(self, slot: str) -> dict[str, Any] | None:
        from trading_research.pipeline.short_horizon_ledger import FINALISTS_DIR

        path = self.study.repo / FINALISTS_DIR / f"{slot}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @property
    def work(self) -> Path:
        return self.study.dir("reads", self.label or self.name)

    @cached_property
    def validated(self) -> dict[str, Any]:
        """Each planned day's exclusions under the frozen threshold, and the usable days."""
        from trading_research.market_making.events import DayUnavailable

        found: dict[str, list[str]] = {}
        for day in self.days_planned:
            book = sl.load_book_day(self.study.symbol, day, self.study.roots.book,
                                    max_sequence_gaps=self.threshold, access=self.access)  # fmt: skip
            reasons = list(book.exclusions)
            if not reasons:
                try:
                    self.events(day)
                except (DayUnavailable, FileNotFoundError) as exc:
                    reasons.append(f"simulator day loader: {exc}")
            found[day.isoformat()] = reasons
        usable = [d for d in self.days_planned if not found[d.isoformat()]]
        excluded = len(self.days_planned) - len(usable)
        return {"reasons": found, "usable": usable,
                "unreadable": sl.block_unreadable(excluded, max(1, len(self.days_planned)))}  # fmt: skip

    @property
    def usable(self) -> list[date]:
        return list(self.validated["usable"])

    def book(self, day: date) -> pd.DataFrame:
        found = sl.load_book_day(self.study.symbol, day, self.study.roots.book,
                                 max_sequence_gaps=self.threshold, access=self.access)  # fmt: skip
        assert found.frame is not None
        return found.frame

    def prints(self, day: date) -> pd.DataFrame:
        return sl.load_prints_day(
            self.study.symbol, day, self.study.roots.trades, access=self.access
        )

    def funding(self, day: date) -> pd.DataFrame:
        return sl.load_funding_day(
            self.study.symbol, day, self.study.roots.funding, access=self.access
        )

    def events(self, day: date) -> Any:
        roots = self.study.roots
        return sl.load_events_day(self.study.symbol, day, book_roots=[roots.book], trades_root=roots.trades,
                                  funding_root=roots.funding, access=self.access)  # fmt: skip

    @cached_property
    def detector(self) -> dict[str, Any]:
        """The frozen detector over the stream to the block's last day, with D's profile."""
        from trading_research.pipeline.short_horizon import flags

        end = self.end or sa.BLOCKS[self.name][1]
        close = flags.hourly_close(self.study.symbol, self.study.roots.bars, end, self.access)
        return flags.frozen(close, end, profile=list(self.value(("detector", "hour_profile"))))

    @property
    def flags_ns(self) -> np.ndarray:
        return np.asarray(self.detector["flags_ns"], dtype=np.int64)

    def cold(self, day: date) -> bool:
        return ft.previous_day(day, self.usable) is None

    def features(self, day: date) -> pd.DataFrame:
        """The day's every-row inputs as the stores hold them (raw sizes scaled, cast
        to float16 or float32), so the frozen models read what they were fitted on."""
        from trading_research.data.feature_store import StoreLayout, to_stored
        from trading_research.features.lob_tensor import TensorCalibration

        def build() -> pd.DataFrame:
            previous = ft.previous_day(day, self.usable)
            history = None if previous is None else (self.book(previous), self.prints(previous))
            calibration = TensorCalibration.from_dict(self.fitted["tensor"])
            frame = ft.day_frame(self.book(day), self.prints(day), calibration, history)
            return to_stored(frame, StoreLayout.from_dict(self.fitted["store_layout"]))

        return ft.read_or_build(self.work / "features" / f"{day}.parquet", build)

    def scored_from(self, day: date) -> int:
        return ft.scored_from_ns(day, self.cold(day))


def slot_keys(block: Block, slot: str) -> list[str]:
    """A slot's cell and its neighbourhood, as its finalist file froze them."""
    record = block.finalist(slot)
    if record is None:
        return []
    return list(dict.fromkeys([record["key"], *record["neighbourhood"], *record.get("cells", [])]))


def frozen_rules(block: Block) -> dict[str, dict[float, Any]]:
    """Every frozen hold-first threshold, by configuration and rate, from the finalist files."""
    from trading_research.strategies.hold_first import HoldFirstRule

    out: dict[str, dict[float, Any]] = {}
    for slot in ("S25", "S-C", "S-R", "S-P", "S-M", "S25-Z"):
        record = block.finalist(slot) or {}
        for key, theta in record.get("thresholds", {}).items():
            cell = parse_key(key)
            out.setdefault(cell.configuration, {})[cell.trades_per_day] = HoldFirstRule(**theta)
    return out


def frozen_gate_levels(block: Block) -> dict[float, Any]:
    from trading_research.search_engine.market_gate import GateLevels

    out: dict[float, Any] = {}
    for slot in ("S25", "S-C", "S-R", "S-P", "S-M", "S25-Z"):
        for hold, levels in (block.finalist(slot) or {}).get("gate_levels", {}).items():
            out[float(hold)] = GateLevels.from_dict(levels)
    return out


def load_models(block: Block, configs: list[Configuration]) -> dict[str, Any]:
    """The frozen boosted, logistic and ridge models (the network is scored elsewhere)."""
    from trading_research.pipeline.short_horizon.freeze import fit_key
    from trading_research.pipeline.short_horizon.models import load
    from trading_research.pipeline.short_horizon_ledger import WEIGHTS_DIR

    out: dict[str, Any] = {}
    for cfg in configs:
        path = block.study.repo / WEIGHTS_DIR / fit_key(cfg)
        if cfg.model != "tcn" and path.exists():
            out[cfg.name] = load(path, kind=cfg.model)
    return out


def network_scores(block: Block, configs: list[Configuration], days: list[date]) -> None:
    """The frozen networks' mean and log-variance on every row of every day; run in a
    fresh process (the network never shares one with XGBoost)."""
    from trading_research.models.windows import Day
    from trading_research.pipeline.short_horizon.freeze import fit_key
    from trading_research.pipeline.short_horizon.models import load
    from trading_research.pipeline.short_horizon_ledger import WEIGHTS_DIR

    columns = [*block.value(("features", "selected")), *ft.TENSOR]
    for cfg in configs:
        path = block.study.repo / WEIGHTS_DIR / fit_key(cfg)
        out = block.work / "network" / cfg.folder
        out.mkdir(parents=True, exist_ok=True)
        for k, day in enumerate(days):
            target = out / f"{day}.npz"
            if target.exists():
                continue
            rows = len(block.features(day))
            if not path.exists():
                np.savez(target, mu=np.full(rows, np.nan), log_variance=np.full(rows, np.nan))
                continue
            model = load(path, kind="tcn")
            follows = k > 0 and (day - days[k - 1]).days == 1 and not block.cold(day)

            def values(d: date = day) -> np.ndarray:
                return block.features(d)[columns].to_numpy(dtype=np.float32)

            sequence = [Day(str(day), values, np.arange(rows), None, follows)]
            if follows:
                previous = days[k - 1]
                n = len(block.features(previous))
                sequence.insert(
                    0, Day(str(previous), partial(values, previous), np.arange(n), None, False)
                )
            mu, log_variance = model.predict_days(sequence)[-1]
            np.savez(target, mu=mu, log_variance=log_variance)


@dataclass
class DayContext:
    """One usable day of a read: its book, tables and every configuration's tape."""

    day: date
    book: Any
    stamps: np.ndarray
    tables: dict[int, Any]
    late: dict[int, Any]
    tapes: dict[str, Any]
    scores: dict[str, np.ndarray]
    forward: dict[float, np.ndarray]


def day_context(block: Block, day: date, configs: list[Configuration], models: dict[str, Any],
                rules: dict[str, dict[float, Any]], levels: dict[float, Any]) -> DayContext:  # fmt: skip
    """Tapes from the frozen fits, and the walk tables at 10 ms and at 110 ms."""
    from trading_research.backtest.taker import TakerBook, TakerSettings
    from trading_research.backtest.walk import walk_cost_bp
    from trading_research.labels.taker_vwap import forward_move_bp
    from trading_research.pipeline.short_horizon.fits import on_complete_rows
    from trading_research.search_engine.market_gate import gate_inputs, gate_states
    from trading_research.search_engine.tapes import ev_gate_tape, hold_first_tape
    from trading_research.search_engine.walk_tables import WalkTable, hold_table
    from trading_research.strategies.hold_first import HoldFirstRule

    rows = block.book(day)
    stamps = ft.stamps_ns(rows)
    frame = block.features(day)
    unscored = stamps < block.scored_from(day)
    given = gate_inputs(rows)
    cost = walk_cost_bp(rows, sl.clip_for(block.study.symbol))
    holds = sorted({cfg.hold_s for cfg in configs})
    tapes, scores = {}, {}
    for cfg in configs:
        gates = gate_states(given, levels[cfg.hold_s])
        x = frame[
            [
                *block.value(("features", "selected")),
                *(ft.TENSOR if "tensor" in cfg.inputs else ()),
            ]
        ]
        x = x.astype(np.float64)
        if cfg.stratum == "C":
            model = models.get(cfg.name)
            proba = (
                on_complete_rows(model.predict_proba, x, 3)
                if model is not None
                else np.full((len(x), 3), np.nan)
            )
            proba[unscored] = np.nan
            known = rules.get(cfg.name, {})
            loosest = HoldFirstRule(min((r.theta_long for r in known.values()), default=np.inf),
                                    min((r.theta_short for r in known.values()), default=np.inf))  # fmt: skip
            tapes[cfg.name] = hold_first_tape(
                day, cfg.name, stamps, proba, loosest, gates, block.flags_ns
            )
            scores[cfg.name] = proba[:, 2] - proba[:, 0]
            continue
        if cfg.model == "tcn":
            with np.load(block.work / "network" / cfg.folder / f"{day}.npz") as saved:
                mu, sigma = saved["mu"].astype(np.float64), np.exp(0.5 * saved["log_variance"])
        elif cfg.name in models:
            mu = on_complete_rows(models[cfg.name].predict_mean_bp, x, 1)
            sigma = on_complete_rows(models[cfg.name].predict_sigma_bp, x, 1)
        else:
            mu = sigma = np.full(len(x), np.nan)
        mu, sigma = np.where(unscored, np.nan, mu), np.where(unscored, np.nan, sigma)
        tapes[cfg.name] = ev_gate_tape(
            day, cfg.name, stamps, mu, sigma, cost, gates, block.flags_ns
        )
        scores[cfg.name] = mu
    book = TakerBook.from_frame(rows, day, funding=block.funding(day))
    walk = WalkTable.from_book(book, sl.clip_for(block.study.symbol))
    late = TakerSettings(latency_ns=110_000_000)
    tables = {round(h * 1e9): hold_table(walk, round(h * 1e9), TakerSettings()) for h in holds}
    later = {round(h * 1e9): hold_table(walk, round(h * 1e9), late) for h in holds}
    forward = {h: forward_move_bp(rows, h).to_numpy() for h in holds}
    return DayContext(day, book, stamps, tables, later, tapes, scores, forward)


def _bp(trades: Any) -> float:
    """A cell-day's ``net_bp_trade`` from walk tables; NaN on a day without a trade."""
    return float(trades.day_values()["net_bp"]) if trades.trades else float("nan")


def _lookup(table: Any, rows: np.ndarray, side: np.ndarray) -> float:
    """The day's value of trades at ``rows`` on ``side`` from the hold table."""
    from trading_research.search_engine.walk_tables import side_index

    if not len(rows):
        return float("nan")
    own = side_index(side)
    net, notional = table.net[own, rows], table.notional[own, rows]
    return float(net.sum() / notional.sum() * 1e4)


@dataclass
class TakerSlot:
    """A taker slot's per-day values, filled one day at a time."""

    slot: str
    key: str
    neighbours: list[str]
    days: list[date]
    draws: int

    def __post_init__(self) -> None:
        n = len(self.days)
        nan = lambda: np.full(n, np.nan)  # noqa: E731
        self.value, self.gross, self.turnover, self.ic = nan(), nan(), nan(), nan()
        self.trades = np.zeros(n, dtype=np.int64)
        self.flipped, self.shifted, self.long_only, self.short_only, self.latency = (
            nan() for _ in range(5)
        )
        self.random = np.full((self.draws, n), np.nan)
        self.cells = np.full((len(self.neighbours), n), np.nan)
        self.rngs = [np.random.default_rng(k) for k in range(self.draws)]
        self.void: set[str] = set()
        self.stale: set[int] = set()
        self.entries: dict[int, Any] = {}
        self.first: Any = None

    def add(self, i: int, ctx: DayContext, rules: dict[str, dict[float, Any]]) -> None:
        from trading_research.backtest.taker import TakerSettings, execute_day
        from trading_research.market_making.accounting import IdentityError
        from trading_research.pipeline.short_horizon.fits import rank_ic
        from trading_research.search_engine.cells import score_cell
        from trading_research.search_engine.null_b import entries_of, replay
        from trading_research.search_engine.walk_tables import execute_rows
        from trading_research.strategies.ev_gate import UnitsError

        def scored(key: str) -> Any:
            cell = parse_key(key)
            rule = (
                rules.get(cell.configuration, {}).get(cell.trades_per_day)
                if cell.stratum == "C"
                else None
            )
            if cell.stratum == "C" and rule is None:
                return None
            return score_cell(
                ctx.tapes[cell.configuration], ctx.tables[round(cell.hold_s * 1e9)], cell, rule
            )

        cell = parse_key(self.key)
        hold = round(cell.hold_s * 1e9)
        table = ctx.tables[hold]
        try:
            true = scored(self.key)
        except UnitsError:
            self.void.add("K-units")
            return
        if true is None:
            return
        rows, side = true.rows, true.side
        try:
            day = execute_day(ctx.book, ctx.stamps[rows], side, hold, TakerSettings())
        except IdentityError:
            self.void.add("K-acct")
            return
        trades = day.trades
        if day.excluded:
            self.stale.add(i)
        if len(trades) > 1 and bool(
            np.any(
                trades["entry_snapshot_ns"].to_numpy()[1:]
                < trades["exit_snapshot_ns"].to_numpy()[:-1]
            )
        ):
            self.void.add("K-cap")
        self.value[i], self.trades[i] = day.day_bp("net"), len(trades)
        self.gross[i], self.turnover[i] = float(trades["gross"].sum()), float(day.turnover)
        self.flipped[i] = _lookup(table, rows, -side)
        self.long_only[i] = _lookup(table, rows, np.ones_like(side))
        self.short_only[i] = _lookup(table, rows, -np.ones_like(side))
        for k, rng in enumerate(self.rngs):
            self.random[k, i] = _lookup(table, rows, rng.choice(np.array([-1, 1]), size=len(rows)))
        self.latency[i] = _bp(execute_rows(ctx.late[hold], rows, side))
        for j, key in enumerate(self.neighbours):
            found = scored(key)
            self.cells[j, i] = np.nan if found is None else _bp(found)
        self.ic[i] = rank_ic(ctx.scores[cell.configuration], ctx.forward[cell.hold_s])
        self.entries[i] = entries_of(table, true)
        if i == 0:
            self.first = table
        else:
            self.shifted[i] = _bp(replay(table, self.entries[i - 1]))

    def finish(self) -> None:
        """The first day's one-day shift: the last usable day's entries, circularly."""
        from trading_research.search_engine.null_b import replay

        n = len(self.days)
        if n > 1 and self.first is not None and n - 1 in self.entries:
            self.shifted[0] = _bp(replay(self.first, self.entries[n - 1]))
        self.first = None


def passive_decisions(ctx: DayContext, key: str, rules: dict[str, dict[float, Any]],
                      lean_scale: dict[str, float]) -> Any:  # fmt: skip
    """A passive cell's decisions on the day under the frozen thresholds."""
    from trading_research.pipeline.short_horizon.passive import Decisions
    from trading_research.search_engine.cells import cell_sides

    cell = parse_key(key)
    tape = ctx.tapes[cell.configuration]
    rule = (
        rules.get(cell.configuration, {}).get(cell.trades_per_day) if cell.stratum == "C" else None
    )
    if cell.stratum == "C" and rule is None:
        return Decisions(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    side, events = cell_sides(tape, cell, rule)
    taken = np.flatnonzero(side != 0)
    if events is not None and len(taken):
        taken = taken[np.r_[True, events[taken][1:] != events[taken][:-1]]]
    if cell.stratum == "R":
        strength = tape.scores["mu_bp"][taken]
    else:
        strength = side[taken] * float(lean_scale.get(f"{cell.hold_s:g}", 0.0))
    return Decisions(
        tape.stamps[taken], side[taken].astype(np.int64), np.asarray(strength, dtype=np.float64)
    )


@dataclass
class PassiveSlot:
    """A passive slot's central cell, its neighbours and their decisions by day."""

    slot: str
    cell: SimCellType
    neighbours: list[SimCellType]
    decisions: dict[str, list[Any]] = field(default_factory=dict)
    ic: list[float] = field(default_factory=list)


def passive_slots(block: Block) -> list[PassiveSlot]:
    """S-P, S-M and S25-Z as their finalist files froze them."""
    from trading_research.pipeline.short_horizon.passive import SimCell

    out = []
    sp, sm, sz = block.finalist("S-P"), block.finalist("S-M"), block.finalist("S25-Z")
    if sp is not None:
        key = sp["key"]
        own = SimCell("S-P", key, "entry", parse_key(key).hold_s, policy=parse_key(key).regime)
        others = [SimCell("S-P", k, "entry", parse_key(k).hold_s, policy=parse_key(k).regime)
                  for k in sp.get("cells", []) if k != key]  # fmt: skip
        out.append(PassiveSlot("S-P", own, others))
    if sm is not None:
        key, guard = sm["key"], sm.get("guard", "none")
        cells = [
            SimCell("S-M", key, "maker", parse_key(key).hold_s, lean=lean, policy=guard)
            for lean in (1.0, 0.5, 2.0)
        ]
        out.append(PassiveSlot("S-M", cells[0], cells[1:]))
    if sz is not None:
        cells = [
            SimCell("S25-Z", k, "entry", parse_key(k).hold_s, fees="zero")
            for k in sz.get("cells", [])
        ]
        if cells:
            central = next((c for c in cells if c.key == sz["key"]), cells[0])
            out.append(PassiveSlot("S25-Z", central, [c for c in cells if c is not central]))
    return out


def passive_series(p: PassiveSlot, rows: pd.DataFrame, days: list[date], draws: int,
                   window_h: int, on_s25: bool) -> Any:  # fmt: skip
    """A passive slot's per-day values from its simulated runs."""
    from trading_research.pipeline.short_horizon.statuses import SlotDays

    mine = rows[rows["run_slot"] == p.slot]
    index = {d.isoformat(): i for i, d in enumerate(days)}
    primary = "gross_bp_trade" if p.slot == "S25-Z" else "net_per_100"

    def series(variant: str, column: str = primary) -> np.ndarray:
        out = np.full(len(days), np.nan)
        part = mine[mine["variant"] == variant]
        for day, excluded, value in zip(part["day"], part["excluded"], part[column], strict=True):
            if not excluded:
                out[index[day]] = float(value)
        return out

    neighbours = [f"neighbour/{c.label}" for c in p.neighbours]
    return SlotDays(
        p.slot, "taker" if p.slot == "S25-Z" else "entry" if p.slot == "S-P" else "maker", days,
        value=series("true"), trades=np.nan_to_num(series("true", "trades")).astype(np.int64),
        gross=series("true", "net_usdt"), turnover=series("true", "passive_notional"), ic=np.asarray(p.ic),
        random=np.array([series(f"random/{k}") for k in range(draws)]).reshape(draws, len(days)),
        flipped=series("flipped"), shifted=series("shifted"), long_only=series("long_only"),
        short_only=series("short_only"),
        neighbours=np.array([series(v) for v in neighbours]).reshape(len(neighbours), len(days)),
        latency=series("latency"), stale=series("stale"),
        brackets={name: series(f"bracket/{name}") for name in ("pessimistic", "none", "all_ahead")},
        making=series("true", "making_usdt") if p.slot == "S-M" else None,
        passive_notional=series("true", "passive_notional"),
        second=series("true", "net_bp_trade") if p.slot == "S25-Z" else None,
        on_s25_signal=on_s25, window_h=window_h,
    )  # fmt: skip


def run_passive(block: Block, slots: list[PassiveSlot], days: list[date]) -> pd.DataFrame:
    """Every passive run of every day, a day per worker process; days on disk are kept."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from trading_research.pipeline.short_horizon.heldout_passive import (
        VariantJob,
        run_variants,
        variants,
    )
    from trading_research.pipeline.short_horizon.plan import midnight_ns

    draws = block.study.scale.random_direction_draws
    rngs = {p.slot: [np.random.default_rng(k) for k in range(draws)] for p in slots}
    out = block.work / "passive"
    jobs = []
    for i, day in enumerate(days):
        runs = []
        for p in slots:
            j = (i - 1) % len(days)
            own, previous = p.decisions[p.cell.label][i], p.decisions[p.cell.label][j]
            near = [(c, p.decisions[c.label][i]) for c in p.neighbours]
            runs += variants(p.slot, p.cell, own, previous, midnight_ns(day), midnight_ns(days[j]),
                             near, draws, rngs[p.slot])  # fmt: skip
        jobs.append(VariantJob(block.study.repo, block.study.symbol, day, block.access, tuple(runs),
                               block.flags_ns, block.threshold, float(block.fitted["sigma_ref"]), out))  # fmt: skip
    workers = max(1, min(6, block.study.workers))
    if workers == 1:
        paths = [run_variants(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            paths = list(pool.map(run_variants, jobs))
    return (
        pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if paths
        else pd.DataFrame()
    )


def compute_block(block: Block) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every slot's per-day values on the block's usable days, a day flagged stale
    in any run left out of every series; and what the read records beside them."""
    from trading_research.pipeline.short_horizon.fits import rank_ic
    from trading_research.pipeline.short_horizon.models import in_fresh_process
    from trading_research.pipeline.short_horizon.statuses import SlotDays, subset
    from trading_research.reporting.search_keys import S25_KEY

    days, draws = block.usable, block.study.scale.random_direction_draws
    takers = [TakerSlot(slot, keys[0], keys[1:], days, draws)
              for slot in ("S25", "S-C", "S-R") if (keys := slot_keys(block, slot))]  # fmt: skip
    passive = passive_slots(block)
    wanted = {k for t in takers for k in (t.key, *t.neighbours)}
    wanted |= {c.key for p in passive for c in (p.cell, *p.neighbours)}
    configs = configurations([k for k in block.study.catalogue().keys if k in wanted])
    models = load_models(block, configs)
    networks = [c for c in configs if c.model == "tcn"]
    if networks and days:
        in_fresh_process(network_scores, block, networks, days)
    rules, levels = frozen_rules(block), frozen_gate_levels(block)
    lean = dict(block.fitted.get("lean_scale_bp", {}))
    for p in passive:
        p.decisions = {c.label: [] for c in (p.cell, *p.neighbours)}
    for i, day in enumerate(days):
        ctx = day_context(block, day, configs, models, rules, levels)
        for t in takers:
            t.add(i, ctx, rules)
        for p in passive:
            for c in (p.cell, *p.neighbours):
                p.decisions[c.label].append(passive_decisions(ctx, c.key, rules, lean))
            cell = parse_key(p.cell.key)
            p.ic.append(rank_ic(ctx.scores[cell.configuration], ctx.forward[cell.hold_s]))
    for t in takers:
        t.finish()
    rows = run_passive(block, passive, days) if passive and days else pd.DataFrame()
    series: dict[str, Any] = {}
    for t in takers:
        series[t.slot] = SlotDays(t.slot, "taker", days, t.value, t.trades, t.gross, t.turnover, t.ic,
                                  t.random, t.flipped, t.shifted, t.long_only, t.short_only, t.cells,
                                  t.latency, void=tuple(sorted(t.void)), on_s25_signal=t.slot == "S25",
                                  window_h=parse_key(t.key).regime_h)  # fmt: skip
    signal_is_s25 = any(p.slot == "S-P" and p.cell.key == S25_KEY for p in passive)
    for p in passive:
        on_s25 = p.slot == "S25-Z" or signal_is_s25
        series[p.slot] = passive_series(
            p, rows, days, draws, parse_key(p.cell.key).regime_h, on_s25
        )
    stale = {days[i] for t in takers for i in t.stale}
    if len(rows):
        stale |= {date.fromisoformat(d) for d in rows.loc[rows["excluded"], "day"]}
    keep = np.array([d not in stale for d in days], dtype=bool)
    series = {slot: subset(s, keep) for slot, s in series.items()}
    in_block = [a for a in block.detector["alarms"]
                if pd.Timestamp(a["timestamp"]).date() >= sa.BLOCKS[block.name][0]]  # fmt: skip
    info = {"usable_days": [d.isoformat() for d in days], "stale_days": sorted(d.isoformat() for d in stale),
            "excluded": block.validated["reasons"], "unreadable": block.validated["unreadable"],
            "alarms_in_block": in_block, "passive_runs": len(rows)}  # fmt: skip
    return series, info
