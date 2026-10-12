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
* for the taker slots (:class:`TakerSlot`): the true slot through the event-time
  executor, and from walk tables its placebos (sign flip, random directions and
  all-long and all-short at its entries, an entry beyond the visible book on the
  placebo's side skipped and counted; the previous usable day's signal tape at
  the same times of day), its neighbour cells and the slot re-run with every
  order live at 110 ms; a day any of these runs flags stale is stale;
* for the passive slots (:mod:`.heldout_passive`): the simulator under the same
  variants and the queue brackets.

Beside them, the per-day values of the registered measurements: for a slot with
a regime window above 0, its window-0 twin and its runs under the random flag
sets (as many flags as the block has inside its usable days, drawn before any
day is left out as stale; a taker slot's three terms from its walk tables, a
simulator slot's from the simulator); S-P's cell at the other regime windows
and S-M's guards; the passive exit of S-P and S25-Z; on H, FILL's two other
fill rules on their streams; and, for S-R on the network, the control
network's IC against its own label and against the realised move. The days
excluded for sequence gaps alone are read again with the threshold lifted
(:meth:`Block.put_back`), the true runs only, so each verdict can be reported
with them put back.

The results are :class:`~trading_research.pipeline.short_horizon.statuses.SlotDays`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
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

#: The sequence-gap threshold of the put-back view: no day is excluded for its gaps.
GAPS_UNBOUNDED = 2**31 - 1
#: The taker placebos that count the entries they skip (reported beside K-placebo and K-drift).
PLACEBO_SKIPS: tuple[str, ...] = ("flipped", "random", "long_only", "short_only")
#: The regime windows S-P's cell is measured at beside its own, and S-M's guards.
REGIME_POLICIES: tuple[str, ...] = ("none", "pull_4h", "pull_24h", "pull_120h")
MAKER_GUARDS: tuple[str, ...] = tuple(
    f"{action}_{w}h" for w in (4, 24, 120) for action in ("pull", "widen")
)


def twin_key(key: str) -> str:
    """The cell's window-0 twin: the same key at ``regime=none``."""
    return key.rsplit("|regime=", 1)[0] + "|regime=none"


def with_regime(key: str, regime: str) -> str:
    return key.rsplit("|regime=", 1)[0] + f"|regime={regime}"


def reguarded(tape: Any, flags_ns: np.ndarray, window_h: int) -> Any:
    """The tape with its guard state at ``window_h`` recomputed from other flags."""
    from dataclasses import replace

    from trading_research.pipeline.execution import guarded_rows
    from trading_research.pipeline.regime import NS_PER_HOUR

    guard = guarded_rows(tape.stamps, np.asarray(flags_ns, dtype=np.int64), window_h * NS_PER_HOUR)
    return replace(tape, guards={**tape.guards, window_h: guard})


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
    #: The put-back view: the days excluded for sequence gaps alone read as usable.
    gaps_put_back: bool = False

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
        if self.gaps_put_back:
            return GAPS_UNBOUNDED
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
            if reasons in ([], [sl.SEQUENCE_GAPS]):
                try:
                    self.events(day)
                except (DayUnavailable, FileNotFoundError) as exc:
                    reasons.append(f"simulator day loader: {exc}")
            found[day.isoformat()] = reasons
        usable = [d for d in self.days_planned if not found[d.isoformat()]]
        gaps = [d for d in self.days_planned if found[d.isoformat()] == [sl.SEQUENCE_GAPS]]
        excluded = len(self.days_planned) - len(usable)
        return {"reasons": found, "usable": usable, "gap_days": gaps,
                "unreadable": sl.block_unreadable(excluded, max(1, len(self.days_planned)))}  # fmt: skip

    @property
    def usable(self) -> list[date]:
        """The usable days (with the gap days among them in the put-back view)."""
        if self.gaps_put_back:
            return sorted([*self.validated["usable"], *self.validated["gap_days"]])
        return list(self.validated["usable"])

    @property
    def gap_days(self) -> list[date]:
        """The days excluded for sequence gaps and nothing else."""
        return list(self.validated["gap_days"])

    def put_back(self) -> Block:
        """The same block with the gap days read as usable: the threshold lifted, the
        validation and the frozen values already found kept, the same work folder."""
        from dataclasses import replace

        view = replace(self, gaps_put_back=True)
        for name in ("registration", "fitted", "validated", "detector", "history"):
            if name in self.__dict__:
                view.__dict__[name] = self.__dict__[name]
        return view

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

    @cached_property
    def history(self) -> list[date]:
        """The earlier block's last day when it is usable and adjacent with the same
        archive (F's for P, F before P), which the rolling state of the block's
        first day may read; none for H and F, whose first days start cold."""
        from trading_research.market_making.events import DayUnavailable

        last = self.days_planned[0] - timedelta(days=1) if self.days_planned else None
        if self.name != "P" or last is None or sa.block_of(last) != "F":
            return []
        try:
            book = sl.load_book_day(self.study.symbol, last, self.study.roots.book,
                                    max_sequence_gaps=int(self.value(("data", "max_sequence_gaps"))),
                                    access=self.access)  # fmt: skip
            usable = not list(book.exclusions)
            if usable:
                self.events(last)
        except (DayUnavailable, FileNotFoundError):
            return []
        return [last] if usable else []

    def previous(self, day: date) -> date | None:
        """The day whose rolling state ``day`` reads (the earlier block's for P's first day)."""
        return ft.previous_day(day, [*self.history, *self.usable])

    def cold(self, day: date) -> bool:
        return self.previous(day) is None

    def features(self, day: date) -> pd.DataFrame:
        """The day's every-row inputs as the stores hold them (raw sizes scaled, cast
        to float16 or float32), so the frozen models read what they were fitted on."""
        from trading_research.data.feature_store import StoreLayout, to_stored
        from trading_research.features.lob_tensor import TensorCalibration

        def build() -> pd.DataFrame:
            previous = self.previous(day)
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
    """The frozen networks' mean and log-variance on every row of every day, and the
    control network's mean (``control_mu``) where its weights were frozen; run in a
    fresh process (the network never shares one with XGBoost)."""
    from trading_research.models.windows import Day
    from trading_research.pipeline.short_horizon.freeze import fit_key
    from trading_research.pipeline.short_horizon.models import load
    from trading_research.pipeline.short_horizon_ledger import WEIGHTS_DIR

    columns = [*block.value(("features", "selected")), *ft.TENSOR]
    for cfg in configs:
        path = block.study.repo / WEIGHTS_DIR / fit_key(cfg)
        control_path = block.study.repo / WEIGHTS_DIR / fit_key(cfg, control=True)
        out = block.work / "network" / cfg.folder
        out.mkdir(parents=True, exist_ok=True)
        model = load(path, kind="tcn") if path.exists() else None
        control = load(control_path, kind="tcn") if model and control_path.exists() else None
        for day in days:
            target = out / f"{day}.npz"
            if target.exists():
                continue
            rows = len(block.features(day))
            if model is None:
                np.savez(target, mu=np.full(rows, np.nan), log_variance=np.full(rows, np.nan))
                continue
            previous = block.previous(day)

            def values(d: date = day) -> np.ndarray:
                return block.features(d)[columns].to_numpy(dtype=np.float32)

            sequence = [Day(str(day), values, np.arange(rows), None, previous is not None)]
            if previous is not None:
                n = len(block.features(previous))
                sequence.insert(
                    0, Day(str(previous), partial(values, previous), np.arange(n), None, False)
                )
            mu, log_variance = model.predict_days(sequence)[-1]
            arrays = {"mu": mu, "log_variance": log_variance}
            if control is not None:
                arrays["control_mu"] = control.predict_days(sequence)[-1][0]
            np.savez(target, **arrays)


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
    #: Per network configuration with a control fit: the control network's IC against
    #: its own (smoothed) label and against the realised move over the hold.
    control: dict[str, tuple[float, float]] = field(default_factory=dict)


def day_context(block: Block, day: date, configs: list[Configuration], models: dict[str, Any],
                rules: dict[str, dict[float, Any]], levels: dict[float, Any]) -> DayContext:  # fmt: skip
    """Tapes from the frozen fits, and the walk tables at 10 ms and at 110 ms."""
    from trading_research.backtest.taker import TakerBook, TakerSettings
    from trading_research.backtest.walk import walk_cost_bp
    from trading_research.labels.taker_vwap import control_move_bp, forward_move_bp
    from trading_research.pipeline.short_horizon.fits import on_complete_rows, rank_ic
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
    forward = {h: forward_move_bp(rows, h).to_numpy() for h in holds}
    tapes, scores, control = {}, {}, {}
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
                if "control_mu" in saved:
                    guess = np.where(unscored, np.nan, saved["control_mu"].astype(np.float64))
                    own = control_move_bp(rows, cfg.hold_s).to_numpy(dtype=np.float64)
                    control[cfg.name] = (rank_ic(guess, own), rank_ic(guess, forward[cfg.hold_s]))
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
    return DayContext(day, book, stamps, tables, later, tapes, scores, forward, control)


def _bp(trades: Any) -> float:
    """A cell-day's ``net_bp_trade`` from walk tables; NaN on a day without a trade."""
    return float(trades.day_values()["net_bp"]) if trades.trades else float("nan")


def _lookup(table: Any, rows: np.ndarray, side: np.ndarray) -> tuple[float, int]:
    """The day's value of a placebo's trades at the true slot's entry ``rows`` on
    ``side`` from the hold table, and how many it skips: an entry whose clip goes
    beyond the visible book on the placebo's own side is skipped as the executor
    skips it (counted, never priced). A skipped entry only frees the placebo
    earlier, so the true slot's later entries stay free for it."""
    from trading_research.search_engine.walk_tables import side_index

    own = side_index(side)
    skipped = table.skip[own, rows] if len(rows) else np.zeros(0, dtype=bool)
    kept, own = rows[~skipped], own[~skipped]
    if not len(kept):
        return float("nan"), int(skipped.sum())
    net, notional = table.net[own, kept], table.notional[own, kept]
    return float(net.sum() / notional.sum() * 1e4), int(skipped.sum())


@dataclass(frozen=True)
class SignalTape:
    """A cell's signal tape on one day as the one-day shift carries it to another:
    each decided row's time of day and side after the cell's rule, gate and guard,
    and in R its event (one entry per event)."""

    tod_ns: np.ndarray
    side: np.ndarray
    events: np.ndarray | None


def signal_tape(tape: Any, table: Any, cell: Any, rule: Any) -> SignalTape:
    """The day's signal tape of ``cell`` (its decisions, not only the entries taken)."""
    from trading_research.search_engine.cells import cell_sides

    side, events = cell_sides(tape, cell, rule)
    decided = side != 0
    tod = table.walk.ts[tape.rows[decided]] - table.walk.start_ns
    return SignalTape(tod.astype(np.int64), side[decided].astype(np.int64),
                      None if events is None else events[decided])  # fmt: skip


def shift_on(table: Any, signal: SignalTape) -> Any:
    """Another day's signal tape on ``table``'s day at the same times of day, each
    decision at the first row stamped at or after it, thinned on this day's table
    with its own one-position state; a decision after the day's last row counts as
    ``entries_no_snapshot``."""
    from trading_research.search_engine.cells import score_sides

    rows = np.searchsorted(table.walk.ts, table.walk.start_ns + signal.tod_ns, side="left")
    inside = rows < table.rows
    events = None if signal.events is None else signal.events[inside]
    found = score_sides(table, rows[inside], signal.side[inside], events)
    found.counters["decisions"] += int((~inside).sum())
    found.counters["entries_no_snapshot"] += int((~inside).sum())
    return found


@dataclass
class TakerSlot:
    """A taker slot's per-day values, filled one day at a time."""

    slot: str
    key: str
    neighbours: list[str]
    days: list[date]
    draws: int
    #: The random flag sets of the regime attribution (a slot with a window above 0).
    flag_sets: list[np.ndarray] = field(default_factory=list)

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
        #: Each day's signal tape, and the day whose tape each day's one-day shift read.
        self.signals: dict[int, SignalTape] = {}
        self.shifted_from: dict[int, int] = {}
        self.first: Any = None
        self.window_h = parse_key(self.key).regime_h
        self.guarded, self.twin = nan(), nan()
        self.flagsets = np.full((len(self.flag_sets), n), np.nan)
        self.control_label, self.control_move = nan(), nan()
        self.has_control = False
        #: Entries each placebo skips (beyond the visible book on its own side), by day;
        #: the random draws' summed over the draws.
        self.skips = {name: np.zeros(n, dtype=np.int64) for name in PLACEBO_SKIPS}
        self.sums: dict[str, np.ndarray] = {}  # per-day sums beside the verdict (``day/...``)

    def extra(self) -> dict[str, np.ndarray]:
        """The per-day measurements beside the status (``SlotDays.extra``)."""
        out: dict[str, np.ndarray] = {
            f"skipped/{k}": v.astype(np.float64) for k, v in self.skips.items()
        }
        if self.window_h > 0:
            out |= {"guarded": self.guarded, "twin": self.twin}
            if self.flag_sets:  # the gap days' pass draws none; merge fills their rows with NaN
                out["flagsets"] = self.flagsets
        if self.has_control:
            out |= {"control_ic_label": self.control_label, "control_ic_move": self.control_move}
        return out | self.sums

    def add(self, i: int, ctx: DayContext, rules: dict[str, dict[float, Any]]) -> None:
        from trading_research.backtest.taker import TakerSettings, execute_day
        from trading_research.market_making.accounting import IdentityError
        from trading_research.pipeline.short_horizon.beside_verdict import taker_day_sums
        from trading_research.pipeline.short_horizon.fits import rank_ic
        from trading_research.search_engine.cells import score_cell
        from trading_research.strategies.ev_gate import UnitsError

        def rule_of(cell: Any) -> Any:
            return (rules.get(cell.configuration, {}).get(cell.trades_per_day)
                    if cell.stratum == "C" else None)  # fmt: skip

        def scored(key: str, tables: dict[int, Any] = ctx.tables) -> Any:
            """The cell's re-run from walk tables (``ctx.late``: every order live at 110 ms)."""
            cell = parse_key(key)
            if cell.stratum == "C" and rule_of(cell) is None:
                return None
            found = score_cell(ctx.tapes[cell.configuration], tables[round(cell.hold_s * 1e9)],
                               cell, rule_of(cell))  # fmt: skip
            self.flag(i, found)
            return found

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
        for name, found in taker_day_sums(day).items():
            self.sums.setdefault(name, np.full(len(self.days), np.nan))[i] = found
        self.gross[i], self.turnover[i] = float(trades["gross"].sum()), float(day.turnover)
        self.flipped[i], self.skips["flipped"][i] = _lookup(table, rows, -side)
        self.long_only[i], self.skips["long_only"][i] = _lookup(table, rows, np.ones_like(side))
        self.short_only[i], self.skips["short_only"][i] = _lookup(table, rows, -np.ones_like(side))
        self.skips["random"][i] = 0
        for k, rng in enumerate(self.rngs):
            drawn = rng.choice(np.array([-1, 1]), size=len(rows))
            self.random[k, i], skipped = _lookup(table, rows, drawn)
            self.skips["random"][i] += skipped
        # K-latency: the slot re-run with every order live 110 ms after its decision
        self.latency[i] = _bp(scored(self.key, ctx.late))
        for j, key in enumerate(self.neighbours):
            found = scored(key)
            self.cells[j, i] = np.nan if found is None else _bp(found)
        self.ic[i] = rank_ic(ctx.scores[cell.configuration], ctx.forward[cell.hold_s])
        self.signals[i] = signal_tape(ctx.tapes[cell.configuration], table, cell, rule_of(cell))
        if i == 0:
            self.first = table
        else:
            self.shift(i, table, i - 1)
        if self.window_h > 0:  # the attribution's three terms, all from the walk tables
            self.guarded[i], self.twin[i] = _bp(true), _bp(scored(twin_key(self.key)))
            for k, flags in enumerate(self.flag_sets):
                tape = reguarded(ctx.tapes[cell.configuration], flags, self.window_h)
                found = score_cell(tape, table, cell, rule_of(cell))
                self.flag(i, found)
                self.flagsets[k, i] = _bp(found)
        if cell.configuration in ctx.control:
            self.has_control = True
            self.control_label[i], self.control_move[i] = ctx.control[cell.configuration]

    def flag(self, i: int, found: Any) -> None:
        """A run of the slot on day ``i`` that walked a stale snapshot makes the day stale."""
        if found is not None and bool(found.stale):
            self.stale.add(i)

    def shift(self, i: int, table: Any, source: int) -> None:
        """The one-day shift on day ``i``: day ``source``'s signal tape on ``i``'s table."""
        self.shifted_from[i] = source
        if source not in self.signals:  # nothing decided on the source day
            self.shifted[i] = np.nan
            return
        found = shift_on(table, self.signals[source])
        self.shifted[i] = _bp(found)
        self.flag(i, found)

    def finish(self) -> None:
        """The first day's one-day shift: the last usable day's signal tape, circularly."""
        n = len(self.days)
        if n > 1 and self.first is not None:
            self.shift(0, self.first, n - 1)
        self.first = None


def passive_decisions(ctx: DayContext, key: str, rules: dict[str, dict[float, Any]],
                      lean_scale: dict[str, float], flags_ns: np.ndarray | None = None) -> Any:  # fmt: skip
    """A passive cell's decisions on the day under the frozen thresholds (its guard
    reading ``flags_ns`` instead of the block's when given: a random flag set)."""
    from trading_research.pipeline.short_horizon.passive import Decisions, tape_decisions

    cell = parse_key(key)
    tape = ctx.tapes[cell.configuration]
    if flags_ns is not None and cell.regime_h > 0:
        tape = reguarded(tape, flags_ns, cell.regime_h)
    rule = (
        rules.get(cell.configuration, {}).get(cell.trades_per_day) if cell.stratum == "C" else None
    )
    if cell.stratum == "C" and rule is None:
        return Decisions(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    return tape_decisions(tape, cell, rule, float(lean_scale.get(f"{cell.hold_s:g}", 0.0)), ctx.day)


@dataclass
class PassiveSlot:
    """A passive slot's central cell, its neighbours, its measured cells (the regime
    measurements, the window-0 twin as ``regime/none`` among them) and their
    decisions by key and day; under each random flag set, the central cell's."""

    slot: str
    cell: SimCellType
    neighbours: list[SimCellType]
    measured: list[tuple[str, SimCellType]] = field(default_factory=list)
    decisions: dict[str, list[Any]] = field(default_factory=dict)
    flagged: list[list[Any]] = field(default_factory=list)
    ic: list[float] = field(default_factory=list)

    @property
    def cells(self) -> list[SimCellType]:
        return [self.cell, *self.neighbours, *(c for _, c in self.measured)]

    @property
    def window_h(self) -> int:
        return int(parse_key(self.cell.key).regime_h)


def passive_slots(block: Block) -> list[PassiveSlot]:
    """S-P, S-M and S25-Z as their finalist files froze them, with the measured cells:
    S-P's at the other regime windows, S-M's guards at lean 1 and, when it has a
    guard, its twin on the unguarded signal."""
    from trading_research.pipeline.short_horizon.passive import SimCell

    out = []
    sp, sm, sz = block.finalist("S-P"), block.finalist("S-M"), block.finalist("S25-Z")
    if sp is not None:
        key, hold = sp["key"], parse_key(sp["key"]).hold_s
        own = SimCell("S-P", key, "entry", hold, policy=parse_key(key).regime)
        others = [SimCell("S-P", k, "entry", parse_key(k).hold_s, policy=parse_key(k).regime)
                  for k in sp.get("cells", []) if k != key]  # fmt: skip
        measured = [(f"regime/{r}", SimCell("S-P", with_regime(key, r), "entry", hold, policy=r))
                    for r in REGIME_POLICIES if r != own.policy]  # fmt: skip
        out.append(PassiveSlot("S-P", own, others, measured))
    if sm is not None:
        key, guard, hold = sm["key"], sm.get("guard", "none"), parse_key(sm["key"]).hold_s
        cells = [
            SimCell("S-M", key, "maker", hold, lean=lean, policy=guard) for lean in (1.0, 0.5, 2.0)
        ]
        measured = [(f"regime/{p}", SimCell("S-M", key, "maker", hold, lean=1.0, policy=p))
                    for p in MAKER_GUARDS if p != guard]  # fmt: skip
        if guard != "none":
            twin = SimCell("S-M", twin_key(key), "maker", hold, lean=1.0, policy="none")
            measured.insert(0, ("regime/none", twin))
        out.append(PassiveSlot("S-M", cells[0], cells[1:], measured))
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
        if column not in part:
            return out
        for day, excluded, value in zip(part["day"], part["excluded"], part[column], strict=True):
            if not excluded:
                out[index[day]] = float(value)
        return out

    from trading_research.pipeline.short_horizon.passive import BESIDE_COLUMNS

    neighbours = [f"neighbour/{c.label}" for c in p.neighbours]
    variants = set(mine["variant"]) if len(mine) else set()
    extra: dict[str, np.ndarray] = {
        f"day/{c}": series("true", c) for c in BESIDE_COLUMNS if c in mine
    }
    for name in sorted(v for v in variants if v.startswith(("regime/", "exit/"))):
        extra[name], extra[f"{name}/trades"] = series(name), series(name, "trades")
    if window_h > 0 and "regime/none" in variants:
        extra["twin"] = series("regime/none")
        sets = sorted(int(v.split("/")[1]) for v in variants if v.startswith("flagset/"))
        extra["flagsets"] = np.array([series(f"flagset/{k}") for k in sets]).reshape(
            len(sets), len(days)
        )
    void = []  # K-acct: an accounting fault in any run of the slot; K-cap: its true run's position
    if (
        "accounting_failed" in mine
        and mine["accounting_failed"].fillna("").astype(str).ne("").any()
    ):
        void.append("K-acct")
    true = mine[mine["variant"] == "true"]
    if "cap_exceeded" in true and true["cap_exceeded"].fillna(False).astype(bool).any():
        void.append("K-cap")

    def claim(column: str) -> SlotDays:
        """The slot's series and every variant's under one claim's column."""
        return SlotDays(
            p.slot, "zero" if p.slot == "S25-Z" else "entry" if p.slot == "S-P" else "maker", days,
            value=series("true", column), trades=np.nan_to_num(series("true", "trades")).astype(np.int64),
            gross=series("true", "net_usdt"), turnover=series("true", "passive_notional"), ic=np.asarray(p.ic),
            random=np.array([series(f"random/{k}", column) for k in range(draws)]).reshape(draws, len(days)),
            flipped=series("flipped", column), shifted=series("shifted", column),
            long_only=series("long_only", column), short_only=series("short_only", column),
            neighbours=np.array([series(v, column) for v in neighbours]).reshape(len(neighbours), len(days)),
            latency=series("latency", column), stale=series("stale", column),
            brackets={name: series(f"bracket/{name}", column) for name in ("pessimistic", "none", "all_ahead")},
            making=series("true", "making_usdt") if p.slot == "S-M" else None,
            passive_notional=series("true", "passive_notional"),
            on_s25_signal=on_s25, window_h=window_h, void=tuple(void),
        )  # fmt: skip

    found = claim(primary)
    found.extra = extra
    if p.slot == "S25-Z":  # its second claim, net at 0/0, with every variant's net
        found.net_claim = claim("net_bp_trade")
        found.second = found.net_claim.value
    return found


def simulate_jobs(block: Block, jobs: list[Any]) -> list[Path]:
    """Each day's simulated runs, a day per worker process: at most 6 workers, and
    fewer while 6 x one day's peak resident memory would pass 24 GB. The first day
    runs alone in a fresh process to measure that peak; each later batch of days
    runs in fresh processes as many as the largest peak so far allows."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from trading_research.pipeline.short_horizon import heldout_passive as hp

    requested = max(1, min(hp.SIMULATOR_WORKERS_MAX, block.study.workers))
    if requested == 1:
        return [hp.run_variants(job) for job in jobs]
    paths: list[Path] = []
    peak, workers = 0, 1
    while len(paths) < len(jobs):
        batch = jobs[len(paths) : len(paths) + workers]
        with ProcessPoolExecutor(max_workers=len(batch), mp_context=get_context("spawn")) as pool:
            done = list(pool.map(hp.run_measured, batch))
        paths += [path for path, _ in done]
        peak = max(peak, *(rss for _, rss in done))
        workers = hp.simulator_workers(requested, peak)
    block.notes.append(f"simulator: peak {peak} bytes for one day, {workers} workers")
    return paths


def run_passive(block: Block, slots: list[PassiveSlot], days: list[date],
                flag_sets: list[np.ndarray], *, kills: bool = True) -> pd.DataFrame:  # fmt: skip
    """Every passive run of every day, a day per worker process; days on disk are kept.
    Without ``kills`` only the true run of each slot (the gap days put back)."""
    from trading_research.pipeline.short_horizon.heldout_passive import (
        Run,
        VariantJob,
        measured_runs,
        variants,
    )
    from trading_research.pipeline.short_horizon.plan import midnight_ns

    draws = block.study.scale.random_direction_draws
    rngs = {p.slot: [np.random.default_rng(k) for k in range(draws)] for p in slots}
    out = block.work / ("passive" if kills else "passive_gap_days")
    jobs = []
    for i, day in enumerate(days):
        runs = []
        for p in slots:
            own = p.decisions[p.cell.key][i]
            if not kills:
                runs.append(Run(p.slot, "true", p.cell, own))
                continue
            j = (i - 1) % len(days)
            near = [(c, p.decisions[c.key][i]) for c in p.neighbours]
            runs += variants(p.slot, p.cell, own, p.decisions[p.cell.key][j], midnight_ns(day),
                             midnight_ns(days[j]), near, draws, rngs[p.slot])  # fmt: skip
            measured = [(name, c, p.decisions[c.key][i]) for name, c in p.measured]
            sets = (
                [(flags, p.flagged[k][i]) for k, flags in enumerate(flag_sets)]
                if p.window_h
                else []
            )
            runs += measured_runs(p.slot, p.cell, own, measured, sets, fill=block.name == "H")
        jobs.append(VariantJob(block.study.repo, block.study.symbol, day, block.access, tuple(runs),
                               block.flags_ns, block.threshold, float(block.fitted["sigma_ref"]), out))  # fmt: skip
    paths = simulate_jobs(block, jobs)
    return (
        pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if paths
        else pd.DataFrame()
    )


def hold_table_of(block: Block, day: date, hold_ns: int) -> Any:
    """The day's hold table at 10 ms, built again as :func:`day_context` builds it."""
    from trading_research.backtest.taker import TakerBook, TakerSettings
    from trading_research.search_engine.walk_tables import WalkTable, hold_table

    book = TakerBook.from_frame(block.book(day), day, funding=block.funding(day))
    walk = WalkTable.from_book(book, sl.clip_for(block.study.symbol))
    return hold_table(walk, hold_ns, TakerSettings())


def step_over_stale(block: Block, takers: list[TakerSlot], passive: list[PassiveSlot],
                    days: list[date], stale: set[date], rows: pd.DataFrame) -> tuple[set[date], pd.DataFrame]:  # fmt: skip
    """The one-day shift from the previous kept day, circularly over the days no run
    flagged stale, so it steps over excluded days. A day whose previous kept day is
    not the day its shift read is shifted again (a taker slot on its rebuilt hold
    table, a passive slot simulated again); a day a new shift run flags stale is
    left out in turn (any run's flag counts), until no day changes."""
    from trading_research.pipeline.short_horizon.heldout_passive import VariantJob, shifted_run
    from trading_research.pipeline.short_horizon.plan import midnight_ns

    n = len(days)
    passive_from = {i: (i - 1) % n for i in range(n)}
    while True:
        kept = [i for i, d in enumerate(days) if d not in stale]
        wanted = {i: kept[k - 1] for k, i in enumerate(kept)} if len(kept) > 1 else {}
        tables: dict[tuple[int, int], Any] = {}
        changed = False
        for t in takers:
            hold = round(parse_key(t.key).hold_s * 1e9)
            for i, j in wanted.items():
                if t.shifted_from.get(i, j) != j:
                    if (i, hold) not in tables:
                        tables[i, hold] = hold_table_of(block, days[i], hold)
                    t.shift(i, tables[i, hold], j)
                    changed = True
        stale |= {days[i] for t in takers for i in t.stale}
        redo = {i: j for i, j in wanted.items() if passive and passive_from[i] != j}
        if redo:
            jobs = [VariantJob(block.study.repo, block.study.symbol, days[i], block.access,
                               tuple(shifted_run(p.slot, p.cell, p.decisions[p.cell.key][j],
                                                 midnight_ns(days[i]), midnight_ns(days[j])) for p in passive),
                               block.flags_ns, block.threshold, float(block.fitted["sigma_ref"]),
                               block.work / "passive_shifted" / days[j].isoformat())
                    for i, j in redo.items()]  # fmt: skip
            new = pd.concat([pd.read_parquet(path) for path in simulate_jobs(block, jobs)])
            again = rows["variant"].eq("shifted") & rows["day"].isin(
                [days[i].isoformat() for i in redo]
            )
            rows = pd.concat([rows[~again], new], ignore_index=True)
            stale |= {date.fromisoformat(d) for d in new.loc[new["excluded"].astype(bool), "day"]}
            passive_from.update(redo)
        if not changed and not redo:
            return stale, rows


def failures(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Each simulated run that raised an accounting fault (its slot is void, K-acct)."""
    if not len(rows) or "accounting_failed" not in rows:
        return []
    failed = rows[rows["accounting_failed"].fillna("").astype(str).ne("")]
    columns = ["day", "run_slot", "variant", "accounting_failed"]
    return [{str(k): v for k, v in row.items()} for row in failed[columns].to_dict("records")]


def unreadable(block: Block, stale: set[date]) -> bool:
    """More than a quarter of the block's planned days excluded: the loader's
    exclusions and the days any run of the read flagged stale, each day once."""
    excluded = (set(block.days_planned) - set(block.validated["usable"])) | stale
    return bool(sl.block_unreadable(len(excluded), max(1, len(block.days_planned))))


def flag_sets(block: Block, days: list[date]) -> tuple[int, list[np.ndarray]]:
    """The regime attribution's random flag sets over ``days``: how many flags the
    block has inside them, and the registered number of sets for a taker slot
    (a simulator slot takes the first of them)."""
    from trading_research.pipeline.regime import flags_inside, random_flag_sets

    if not days:
        return 0, []
    count = flags_inside(block.flags_ns, days)
    return count, random_flag_sets(days, count, sets=block.study.scale.taker_flag_sets)


def compute_block(block: Block, *, days: list[date] | None = None,
                  kills: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:  # fmt: skip
    """Every slot's per-day values on the block's usable days (or on ``days``), a
    day flagged stale in any run left out of every series (the one-day shift
    stepping over it: :func:`step_over_stale`); and what the read
    records beside them. Without ``kills``, the true runs only (the gap days put
    back: the simulator's placebos, measurements and brackets are not run)."""
    from trading_research.pipeline.short_horizon.fits import rank_ic
    from trading_research.pipeline.short_horizon.heldout_measures import fill_rules
    from trading_research.pipeline.short_horizon.models import in_fresh_process
    from trading_research.pipeline.short_horizon.statuses import SlotDays, subset
    from trading_research.reporting.search_keys import S25_KEY

    usable, draws = block.usable, block.study.scale.random_direction_draws
    days = usable if days is None else days
    n_flags, sets = flag_sets(block, usable) if kills else (0, [])
    sim_sets = sets[: block.study.scale.simulator_flag_sets]
    takers = [TakerSlot(slot, keys[0], keys[1:], days, draws, sets)
              for slot in ("S25", "S-C", "S-R") if (keys := slot_keys(block, slot))]  # fmt: skip
    passive = passive_slots(block)
    wanted = {k for t in takers for k in (t.key, *t.neighbours)}
    wanted |= {c.key for p in passive for c in p.cells}
    configs = configurations([k for k in block.study.catalogue().keys if k in wanted])
    models = load_models(block, configs)
    networks = [c for c in configs if c.model == "tcn"]
    if networks and days:
        in_fresh_process(network_scores, block, networks, usable)
    rules, levels = frozen_rules(block), frozen_gate_levels(block)
    lean = dict(block.fitted.get("lean_scale_bp", {}))
    for p in passive:
        p.decisions = {c.key: [] for c in p.cells}
        p.flagged = [[] for _ in sim_sets] if p.window_h else []
    for i, day in enumerate(days):
        ctx = day_context(block, day, configs, models, rules, levels)
        for t in takers:
            t.add(i, ctx, rules)
        for p in passive:
            for key in p.decisions:
                p.decisions[key].append(passive_decisions(ctx, key, rules, lean))
            for k, flags in enumerate(sim_sets if p.window_h else []):
                p.flagged[k].append(passive_decisions(ctx, p.cell.key, rules, lean, flags))
            cell = parse_key(p.cell.key)
            p.ic.append(rank_ic(ctx.scores[cell.configuration], ctx.forward[cell.hold_s]))
    for t in takers:
        t.finish()
    rows = (
        run_passive(block, passive, days, sim_sets, kills=kills)
        if passive and days
        else pd.DataFrame()
    )
    series: dict[str, Any] = {}
    for t in takers:
        series[t.slot] = SlotDays(t.slot, "taker", days, t.value, t.trades, t.gross, t.turnover, t.ic,
                                  t.random, t.flipped, t.shifted, t.long_only, t.short_only, t.cells,
                                  t.latency, void=tuple(sorted(t.void)), on_s25_signal=t.slot == "S25",
                                  window_h=t.window_h, extra=t.extra())  # fmt: skip
    signal_is_s25 = any(p.slot == "S-P" and p.cell.key == S25_KEY for p in passive)
    for p in passive:
        on_s25 = p.slot == "S25-Z" or signal_is_s25
        series[p.slot] = passive_series(
            p, rows, days, draws, parse_key(p.cell.key).regime_h, on_s25
        )
    stale = {days[i] for t in takers for i in t.stale}
    if len(rows):
        stale |= {date.fromisoformat(d) for d in rows.loc[rows["excluded"].astype(bool), "day"]}
    if kills and days:
        stale, rows = step_over_stale(block, takers, passive, days, stale, rows)
    keep = np.array([d not in stale for d in days], dtype=bool)
    series = {slot: subset(s, keep) for slot, s in series.items()}
    in_block = [a for a in block.detector["alarms"]
                if pd.Timestamp(a["timestamp"]).date() >= sa.BLOCKS[block.name][0]]  # fmt: skip
    kept = [d for d in days if d not in stale]
    shift = (
        {d.isoformat(): kept[k - 1].isoformat() for k, d in enumerate(kept)}
        if len(kept) > 1
        else {}
    )
    info = {"usable_days": [d.isoformat() for d in days], "stale_days": sorted(d.isoformat() for d in stale),
            "one_day_shift_from": shift if kills else {},
            "excluded": block.validated["reasons"], "unreadable": unreadable(block, stale),
            "gap_days": [d.isoformat() for d in block.validated["gap_days"]],
            "alarms_in_block": in_block, "passive_runs": len(rows), "accounting_failures": failures(rows),
            "flag_days": [d.isoformat() for d in usable], "flags_in_block": n_flags,
            "flag_sets": {"taker": len(sets), "simulator": len(sim_sets)}, "notes": list(block.notes)}  # fmt: skip
    if kills and block.name == "H" and len(rows):
        info["fill_rules"] = {p.slot: fill_rules(rows, p.slot, kept) for p in passive
                              if p.cell.kind == "entry"}  # fmt: skip
    return series, info
