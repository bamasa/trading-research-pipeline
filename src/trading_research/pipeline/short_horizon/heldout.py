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
from trading_research.pipeline.short_horizon.fits import Configuration
from trading_research.search_engine.catalogue import parse_key


@dataclass
class Block:
    """One block's read: its days under the session's access, and every frozen input."""

    study: Study
    name: str
    access: sa.StudyAccess
    days_planned: tuple[date, ...]
    notes: list[str] = field(default_factory=list)

    @cached_property
    def registration(self) -> Any:
        from trading_research.pipeline import short_horizon_registration as shr

        return shr.Registration.load(repo=self.study.repo).require_frozen()

    @property
    def threshold(self) -> int:
        return int(self.registration.max_sequence_gaps)

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
        return self.study.dir("reads", self.name)

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

        end = sa.BLOCKS[self.name][1]
        close = flags.hourly_close(self.study.symbol, self.study.roots.bars, end, self.access)
        return flags.frozen(
            close, end, profile=list(self.registration.value(("detector", "hour_profile")))
        )

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
        path = block.study.repo / WEIGHTS_DIR / f"{fit_key(cfg)}.pkl"
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

    columns = [*block.registration.value(("features", "selected")), *ft.TENSOR]
    for cfg in configs:
        path = block.study.repo / WEIGHTS_DIR / f"{fit_key(cfg)}.pt"
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
                *block.registration.value(("features", "selected")),
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
