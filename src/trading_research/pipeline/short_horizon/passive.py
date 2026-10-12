"""Stage C: the simulator on D's test days for S-P, S-M and S25-Z; A0 and FILL on O.

* **S-P**: its signal is the admitted C* or R* (both admitted: the larger
  carry-forward G, C* on a tie; neither: S25's). Every cell of the signal's
  neighbourhood at its categorical and regime values enters through the
  signal-entry quoter (one clip at the touch, a taker exit a hold after the
  fill), and the cell at each other regime window is measured.
* **S-M**: S1 leaning on S-P's signal at lean 0.5, 1 and 2, the guard
  ``guard_pull`` at the signal's regime window; ``guard_pull`` and
  ``guard_widen`` at 4, 24 and 120 h at lean 1 are measured.
* **S25-Z**: S25's hold x rate x label fee through the quoter at 0/0.
* **A0 and FILL** on O (:mod:`.a0`).

A cell's decisions are its rows of its configuration's tape after its own
rule, gate and guard, handed to the quoter as they would reach it live. Each
day is simulated for every cell at once (the day's events loaded once), in
worker processes, and kept on disk so a stopped stage resumes. S-P and S-M
are then admitted on (i) and (ii) of ``net_per_100_clip_day``, and S-M also on
a positive making part.

Every simulated cell, the measured ones included, and A0 and FILL's four runs
are trials of the registered bound (N_max = 2,400 + 12 + 30 + 9 + 4): their
cell-days go into the trial ledger as simulator trials
(``search_engine.ledger``), so that N_trials counts them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import BYBIT_BASE, FeeTier

#: The zero-fee venue S25-Z's verdict and A0 are measured at.
ZERO_FEES = FeeTier("zero", 0.0, 0.0, requires="none (a zero-fee venue, not Bybit's)")
LEANS: tuple[float, ...] = (0.5, 1.0, 2.0)
WINDOWS_H: tuple[int, ...] = (4, 24, 120)


@dataclass(frozen=True)
class SimCell:
    """One simulated configuration: the slot, the signal cell and the quoter's settings."""

    slot: str
    key: str
    kind: str  # "entry" (the signal-entry quoter) or "maker" (S1 with the lean)
    hold_s: float
    fees: str = "base"
    lean: float = 0.0
    policy: str = "none"
    measured: bool = False
    #: The passive-exit variant (held-out blocks only): the exit posted at the touch
    #: a hold after the fill, crossed if still open a further hold later.
    passive_exit: bool = False

    @property
    def label(self) -> str:
        parts = [self.slot, self.key, self.kind, f"fees={self.fees}"]
        if self.kind == "maker":
            parts += [f"lean={self.lean:g}", f"policy={self.policy}"]
        if self.passive_exit:
            parts.append("exit=passive")
        return "|".join(parts)


@dataclass(frozen=True)
class Decisions:
    """A cell's decisions on one day: stamps, sides and the lean's signed strength in bp,
    and in stratum R each row's EV-gate event (one entry per event)."""

    stamps_ns: np.ndarray
    side: np.ndarray
    strength_bp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    event: np.ndarray | None = None


def entry_hours(day: date) -> tuple[int, int]:
    """The day's entry hours in epoch ns, [00:01, 23:58] as the taker executor's."""
    from trading_research.backtest.taker import TakerSettings

    settings = TakerSettings()
    midnight = pd.Timestamp(day.isoformat(), tz="UTC").value

    def at(moment: Any) -> int:
        return midnight + (moment.hour * 3600 + moment.minute * 60 + moment.second) * 10**9

    return at(settings.no_entry_before), at(settings.no_entry_after)


def tape_decisions(tape: Any, cell: Any, rule: Any, scale_bp: float, day: date) -> Decisions:
    """A cell's decisions on ``day`` from its tape after its own rule, gate and guard:
    the rows inside the entry hours, and in stratum R every row of each EV-gate event
    with its event id (the quoter enters an event once, at its first free row); the
    lean's strength is mu in R and the side times ``scale_bp`` in C."""
    from trading_research.search_engine.cells import cell_sides

    side, events = cell_sides(tape, cell, rule)
    stamps = np.asarray(tape.stamps, dtype=np.int64)
    first, last = entry_hours(day)
    taken = np.flatnonzero((side != 0) & (stamps >= first) & (stamps <= last))
    strength = tape.scores["mu_bp"][taken] if cell.stratum == "R" else side[taken] * scale_bp
    event = None if events is None else np.asarray(events[taken], dtype=np.int64)
    return Decisions(stamps[taken], side[taken].astype(np.int64),
                     np.asarray(strength, dtype=np.float64), event)  # fmt: skip


def first_of_events(decided: Decisions) -> Decisions:
    """Each event's first row only (the market maker's lean reads one row an event)."""
    if decided.event is None or not len(decided.event):
        return decided
    first = np.r_[True, decided.event[1:] != decided.event[:-1]]
    strength = decided.strength_bp[first] if len(decided.strength_bp) else decided.strength_bp
    return Decisions(decided.stamps_ns[first], decided.side[first], strength, decided.event[first])


@dataclass(frozen=True)
class DayJob:
    """Every cell of one day, run in a worker process."""

    repo: Path
    symbol: str
    day: date
    cells: tuple[SimCell, ...]
    decisions: tuple[Decisions, ...]
    flags_ns: np.ndarray
    max_sequence_gaps: int
    sigma_ref: float
    out: Path


def _config(cell: SimCell, max_gaps: int, flagged: bool) -> Any:
    from trading_research.pipeline.short_horizon_execution import study_config

    fees = ZERO_FEES if cell.fees == "zero" else BYBIT_BASE
    return study_config(max_gaps, fees=fees, cancel_entries_on_flag=flagged)


def quoter_and_tapes(
    cell: SimCell, decided: Decisions, flags_ns: np.ndarray, max_gaps: int, sigma_ref: float
) -> tuple[Any, Any, dict[str, Any]]:
    """The cell's quoter, its settings and its tapes for the day."""
    from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
    from trading_research.pipeline.market_making import make_quoter, signal_tapes
    from trading_research.pipeline.regime import RegimePolicy
    from trading_research.pipeline.short_horizon_execution import flag_tape, lean_plan

    flagged = cell.policy != "none" and len(flags_ns) > 0
    tapes: dict[str, Any] = flag_tape(flags_ns) if cell.policy != "none" else {}
    if cell.kind == "entry":
        config = _config(cell, max_gaps, flagged)
        tapes |= entry_tapes(decided.stamps_ns, decided.side, decided.event)
        return SignalEntryQuoter(cell.hold_s, passive_exit=cell.passive_exit), config, tapes
    config = _config(cell, max_gaps, False)
    decided = first_of_events(decided)
    plan = lean_plan(config, sigma_ref, lean=cell.lean, hold_s=cell.hold_s,
                     policy=RegimePolicy.parse(cell.policy))  # fmt: skip
    tapes |= signal_tapes(decided.stamps_ns, decided.strength_bp)
    return make_quoter(plan), plan.config, tapes


def _flat_returns(fills: pd.DataFrame) -> int:
    """Returns to a flat position: the market maker's round trips (K-trades)."""
    if fills.empty:
        return 0
    position = fills["position_after"].to_numpy(dtype=np.float64)
    before = np.r_[0.0, position[:-1]]
    return int(((np.abs(position) < 1e-12) & (np.abs(before) >= 1e-12)).sum())


def per_100_clip(filled: pd.DataFrame, funding: float) -> float:
    """``net_per_100_clip_day`` of the signal-entry quoter: each filled attempt's
    net in USDT x 100 over its own entry's clip notional, summed, and 0 without
    a fill. The day's funding, which the simulator settles on the day and not on
    an attempt, is taken over the mean notional of the day's filled entries."""
    if not len(filled):
        return 0.0
    notional = filled["notional"].to_numpy(dtype=np.float64)
    net = filled["net"].to_numpy(dtype=np.float64)
    return float((net / notional * 100.0).sum() + funding * 100.0 / notional.mean())


#: What a passive cell-day records beside its series for the measurements beside
#: every verdict, summed over days (``beside_sums``); NaN where it does not apply.
BESIDE_COLUMNS: tuple[str, ...] = (
    "net_usdt", "gross_usdt", "fees_usdt", "funding_usdt", "passive_notional", "taker_notional",
    "entry_notional", "per100_maker_weight", "per100_taker_weight", "making_usdt", "making_bp_sum", "making_fills",
    "net_long_usdt", "net_short_usdt", "notional_long", "notional_short", "trades_long",
    "trades_short", "entries_skipped", "exit_late_snapshot", "flatten_beyond_visible_book",
    "max_abs_position", "time_above_one_clip_s",
)  # fmt: skip


def _making(fills: pd.DataFrame) -> dict[str, float]:
    """``spread + adverse(5 s)`` over the passive fills: in USDT, and as the sum and the
    count of each fill's own value in bp (its 5 s markout), for a mean per fill."""
    from trading_research.market_making.analysis import horizon_column

    maker = fills[fills["maker"].to_numpy(dtype=bool)] if len(fills) else fills
    if not len(maker):
        return {"making_usdt": 0.0, "making_bp_sum": 0.0, "making_fills": 0.0}
    markout = maker[horizon_column(5.0)].to_numpy(dtype=np.float64)
    ok = np.isfinite(markout)
    paid = (maker["price"] * maker["size"]).to_numpy(dtype=np.float64)
    return {"making_usdt": float((markout[ok] * paid[ok]).sum() * 1e-4),
            "making_bp_sum": float(markout[ok].sum()), "making_fills": float(ok.sum())}  # fmt: skip


def _episodes(fills: pd.DataFrame, clip: float) -> pd.DataFrame:
    """The market maker's entries: each run of fills from a flat position back to flat,
    its side, net (fees included), the clip's notional at its first fill and its
    fills' notionals by kind."""
    columns = ["side", "net", "notional", "maker_notional", "taker_notional"]
    if fills.empty:
        return pd.DataFrame(columns=columns)
    position = fills["position_after"].to_numpy(dtype=np.float64)
    flat = np.abs(position) < 1e-12
    episode = np.r_[0, np.cumsum(flat)[:-1]]
    side = fills["side"].to_numpy(dtype=np.float64)
    price = fills["price"].to_numpy(dtype=np.float64)
    paid = price * fills["size"].to_numpy(dtype=np.float64)
    maker = fills["maker"].to_numpy(dtype=bool)
    frame = pd.DataFrame({"episode": episode, "side": side, "price": price,
                          "flow": -side * paid - fills["fee"].to_numpy(dtype=np.float64),
                          "maker_notional": np.where(maker, paid, 0.0),
                          "taker_notional": np.where(maker, 0.0, paid)})  # fmt: skip
    grouped = frame.groupby("episode", sort=True)
    first = grouped.first()
    sums = grouped[["flow", "maker_notional", "taker_notional"]].sum()
    return pd.DataFrame({"side": first["side"], "net": sums["flow"], "notional": first["price"] * clip,
                         "maker_notional": sums["maker_notional"],
                         "taker_notional": sums["taker_notional"]}).reset_index(drop=True)  # fmt: skip


def _by_side(table: pd.DataFrame, notional: str) -> dict[str, float]:
    long, short = table[table["side"] > 0], table[table["side"] < 0]
    return {"net_long_usdt": float(long["net"].sum()), "net_short_usdt": float(short["net"].sum()),
            "notional_long": float(long[notional].sum()), "notional_short": float(short[notional].sum()),
            "trades_long": float(len(long)), "trades_short": float(len(short))}  # fmt: skip


def day_values(cell: SimCell, quoter: Any, result: Any, clip: float) -> dict[str, Any]:
    """One cell-day: ``net_per_100_clip_day`` and what the kills and the measurements
    beside the verdict read: the making part over the passive fills, each side's
    turnover, the weights that move ``net_per_100`` to another fee pair exactly (the
    quoter never reads a fee), the result by side and the execution counters."""
    d = result.decomposition
    fills = result.fills
    counters = result.counters
    maker = fills[fills["maker"]] if len(fills) else fills
    passive_notional = float((maker["price"] * maker["size"]).sum()) if len(maker) else 0.0
    made = _making(fills)
    row: dict[str, Any] = {
        "label": cell.label, "slot": cell.slot, "key": cell.key, "excluded": bool(result.excluded),
        "flags": ",".join(result.flags), "net_usdt": float(d.get("net", 0.0)),
        "gross_usdt": float(d.get("gross", 0.0)), "fees_usdt": float(d.get("fees", 0.0)),
        "funding_usdt": float(d.get("funding", 0.0)), **made,
        "passive_fills": len(maker), "passive_notional": passive_notional,
        "taker_notional": float(counters.get("taker_turnover", 0.0)),
        "making_bp": made["making_bp_sum"] / made["making_fills"] if made["making_fills"] else float("nan"),
    }  # fmt: skip
    for name in ("exit_late_snapshot", "flatten_beyond_visible_book", "time_above_one_clip_s",
                 "entries_cancelled_by_flag", "feed_pauses", "max_abs_position"):  # fmt: skip
        row[name] = float(counters.get(name, 0.0))
    orders = result.orders  # what FILL reads: the posted orders and their fills
    posted = len(orders)
    row |= {"orders_posted": float(posted),
            "orders_filled": float((orders["filled"] > 0).sum()) if posted else 0.0,
            "volume_posted": float(orders["size"].sum()) if posted else 0.0,
            "volume_filled": float(orders["filled"].sum()) if posted else 0.0,
            "notional_filled": float((orders["filled"] * orders["price"]).sum()) if posted else 0.0}  # fmt: skip
    if cell.kind == "entry":
        row |= _entry_values(quoter, fills, row)
    else:
        row |= _maker_values(fills, clip, row["funding_usdt"])
    return row


def _entry_values(quoter: Any, fills: pd.DataFrame, row: dict[str, Any]) -> dict[str, Any]:
    """The signal-entry quoter's day: each filled attempt over its own clip's notional,
    and ``gross_bp_trade`` and ``net_bp_trade`` over the entries' filled notional."""
    table = quoter.attempts(fills)
    filled = table[table["filled"]] if len(table) else table
    entered = float(filled["entry_notional"].sum()) if len(filled) else 0.0
    weight = filled["notional"].to_numpy(dtype=np.float64) / 100.0 if len(filled) else np.ones(0)
    counts = quoter.counts
    out = {"trades": len(filled), "attempts": len(table), "entry_notional": entered,
           "net_per_100": per_100_clip(filled, row["funding_usdt"]),
           "per100_maker_weight": float((filled["maker_notional"] / weight).sum()) if len(filled) else 0.0,
           "per100_taker_weight": float((filled["taker_notional"] / weight).sum()) if len(filled) else 0.0,
           "gross_bp_trade": row["gross_usdt"] / entered * 1e4 if entered > 0 else float("nan"),
           "net_bp_trade": row["net_usdt"] / entered * 1e4 if entered > 0 else float("nan"),
           "entries_skipped": float(counts.get("skipped_busy", 0) + counts.get("expired_unseen", 0)),
           **_by_side(filled, "entry_notional")}  # fmt: skip
    flat = fills.empty or abs(float(fills["position_after"].iloc[-1])) < 1e-12
    cash = float(filled["net"].sum()) if len(filled) else 0.0
    expected = row["gross_usdt"] - row["fees_usdt"]
    if flat and abs(cash - expected) > 1e-9 * max(1.0, entered):
        raise RuntimeError(
            f"{row['label']}: the filled attempts' cash {cash!r} is not the day's {expected!r}"
        )
    return out


def _maker_values(fills: pd.DataFrame, clip: float, funding: float) -> dict[str, Any]:
    """The market maker's day: each entry (flat to flat) over the clip's notional at its
    first fill, the day's funding over their mean, as for the quoter's attempts."""
    table = _episodes(fills, clip)
    weight = table["notional"].to_numpy(dtype=np.float64) / 100.0 if len(table) else np.ones(0)
    return {"trades": _flat_returns(fills), "attempts": len(fills), "entry_notional": float("nan"),
            "net_per_100": per_100_clip(table, funding),
            "per100_maker_weight": float((table["maker_notional"] / weight).sum()) if len(table) else 0.0,
            "per100_taker_weight": float((table["taker_notional"] / weight).sum()) if len(table) else 0.0,
            "gross_bp_trade": float("nan"), "net_bp_trade": float("nan"),
            "entries_skipped": float("nan"), **_by_side(table, "notional")}  # fmt: skip


def simulate_day(job: DayJob) -> Path:
    """Every cell of ``job``'s day; the rows go to one file (run in a worker)."""
    from trading_research.data import study_access as sa
    from trading_research.data import study_loader as sl
    from trading_research.data.fetch_stage import StudyRoots
    from trading_research.market_making.simulator import simulate_day as simulate

    roots = StudyRoots.under(job.repo / "data")
    events = sl.load_events_day(job.symbol, job.day, book_roots=[roots.book], trades_root=roots.trades,
                                funding_root=roots.funding, access=sa.development_access())  # fmt: skip
    clip = sl.clip_for(job.symbol)
    rows = []
    for cell, decided in zip(job.cells, job.decisions, strict=True):
        quoter, config, tapes = quoter_and_tapes(
            cell, decided, job.flags_ns, job.max_sequence_gaps, job.sigma_ref
        )
        result = simulate(events, quoter, config, tapes)
        rows.append(day_values(cell, quoter, result, clip) | {"day": job.day.isoformat()})
    target = job.out / f"{job.day}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    pd.DataFrame(rows).to_parquet(partial, index=False)
    partial.replace(target)
    return target


def sp_signal(record_d: dict[str, Any]) -> tuple[str, str]:
    """S-P's signal: (``C*``, ``R*`` or ``S25``, its cell key)."""
    from trading_research.reporting.search_keys import S25_KEY

    admitted = [s for s in ("C", "R") if record_d["signal_admitted"][s]]
    if not admitted:
        return "S25", S25_KEY
    if len(admitted) == 2:
        g = {s: record_d["carry_forward"][s]["mean_g"] for s in admitted}
        admitted = ["C"] if g["C"] >= g["R"] else ["R"]
    stratum = admitted[0]
    return f"{stratum}*", str(record_d["choice"][f"{stratum}*"]["cell"])


def _with_regime(key: str, regime: str) -> str:
    return key.rsplit("|regime=", 1)[0] + f"|regime={regime}"


def slot_cells(keys: list[str], record_d: dict[str, Any]) -> list[SimCell]:
    """Every simulated cell of stage C: S-P's, S-M's and S25-Z's, and the measured ones."""
    from trading_research.pipeline.short_horizon.search import Grid
    from trading_research.reporting.search_keys import S25_KEY
    from trading_research.search_engine.catalogue import parse_key

    grid = Grid(keys)
    _, signal = sp_signal(record_d)
    at = grid.position(signal)
    if at is None:
        raise RuntimeError(f"S-P's signal {signal} is not among the cells scored")
    cell = parse_key(signal)
    out = [SimCell("S-P", k, "entry", parse_key(k).hold_s, policy=parse_key(k).regime)
           for k in grid.neighbours(at) if parse_key(k).regime == cell.regime]  # fmt: skip
    for regime in ("none", "pull_4h", "pull_24h", "pull_120h"):
        other = _with_regime(signal, regime)
        if regime != cell.regime and other in keys:
            out.append(SimCell("S-P", other, "entry", cell.hold_s, policy=regime, measured=True))
    own = cell.regime  # S-M's guard: pull at the signal's window, none without one
    out += [SimCell("S-M", signal, "maker", cell.hold_s, lean=lean, policy=own) for lean in LEANS]
    for window in WINDOWS_H:
        for action in ("pull", "widen"):
            policy = f"{action}_{window}h"
            if policy != own:
                out.append(
                    SimCell(
                        "S-M", signal, "maker", cell.hold_s, lean=1.0, policy=policy, measured=True
                    )
                )
    s25 = grid.position(S25_KEY)
    if s25 is not None:
        out += [SimCell("S25-Z", k, "entry", parse_key(k).hold_s, fees="zero")
                for k in grid.neighbours(s25) if parse_key(k).regime == "none"]  # fmt: skip
    return out


def cell_decisions(
    inputs: Any, key: str, day: date, record: dict[str, Any], tapes: Path
) -> Decisions:
    """A cell's decisions on one test day (:func:`tape_decisions`), the lean's scale in
    C the median absolute forward move over the fold's training days."""
    from trading_research.pipeline.short_horizon import features as ft
    from trading_research.pipeline.short_horizon.fits import rules_from
    from trading_research.search_engine.catalogue import parse_key
    from trading_research.search_engine.tapes import read_tape, tape_path

    cell = parse_key(key)
    tape = read_tape(
        tape_path(tapes, cell.configuration, day), configuration=cell.configuration, day=day
    )
    rule = None
    if cell.stratum == "C":
        rule = rules_from(record["fits"][cell.configuration]["thresholds"])[cell.trades_per_day]
    scale = 0.0
    if cell.stratum != "R":
        moves = [
            inputs.targets(d)[ft.forward_column(cell.hold_s)].to_numpy()
            for d in record["days"]["train"]
        ]
        scale = float(np.nanmedian(np.abs(np.concatenate(moves)))) if moves else 0.0
    return tape_decisions(tape, cell, rule, scale, day)


def simulate_day_measured(job: DayJob) -> tuple[Path, int]:
    """:func:`simulate_day` in a fresh worker, with that process's peak memory after it."""
    from trading_research.data.study_fetch import peak_rss_bytes

    return simulate_day(job), peak_rss_bytes()


def simulate_all(study: Any, inputs: Any, cells: list[SimCell], days: list[date]) -> pd.DataFrame:
    """Every cell on every day, a day per worker process (at most 6, and fewer while
    6 x one day's peak resident memory would pass 24 GB, sized as the reads size
    theirs: ``heldout_passive.simulate_batched``); days on disk are kept."""
    from trading_research.pipeline.short_horizon import flags
    from trading_research.pipeline.short_horizon.scoring import test_folds

    out = study.dir("C", "days")
    folds = test_folds(inputs)
    sigma_ref = float(inputs.record["fitted"]["sigma_ref"])
    jobs = []
    for day in days:
        if (out / f"{day}.parquet").exists():
            continue
        fold = folds[day]
        record = json.loads(
            (study.work / "A" / "folds" / f"{fold.name}.json").read_text(encoding="utf-8")
        )
        record["days"]["train"] = [date.fromisoformat(d) for d in record["days"]["train"]]
        flag_record = flags.fold_flags(
            pd.Series(dtype=float), fold.name, study.work / "A" / "flags"
        )
        decided = tuple(
            cell_decisions(inputs, c.key, day, record, study.work / "A" / "tapes") for c in cells
        )
        jobs.append(DayJob(study.repo, study.symbol, day, tuple(cells), decided, flags.guard_flags(flag_record),
                           inputs.threshold, sigma_ref, out))  # fmt: skip
    workers = max(1, min(6, study.workers))
    if workers == 1:
        for job in jobs:
            simulate_day(job)
    else:
        from trading_research.pipeline.short_horizon.heldout_passive import simulate_batched

        simulate_batched(jobs, simulate_day_measured, workers)
    return pd.concat([pd.read_parquet(out / f"{d}.parquet") for d in days], ignore_index=True)


def flagged_days(rows: pd.DataFrame) -> set[str]:
    """Every day a simulated run of any cell flagged (stale flatten, sequence gaps or
    no funding): stage C leaves it out of every passive cell's series."""
    return {str(d) for d in rows.loc[rows["excluded"].astype(bool), "day"]} if len(rows) else set()


def kept_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows on days no run flagged."""
    return rows[~rows["day"].astype(str).isin(flagged_days(rows))] if len(rows) else rows


def admit(rows: pd.DataFrame, cells: list[SimCell], days: list[date], slot: str,
          record_d: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """S-P's or S-M's admission on D: (i) and (ii) on ``net_per_100_clip_day`` over
    the days no run flagged, and for S-M a positive making part, the mean of
    ``spread + adverse(5 s)`` over its passive fills in bp per fill (named with (i),
    the payoff condition it adds to)."""
    from trading_research.validation.choice import CellDays, not_admitted, slot_conditions

    signal, key = sp_signal(record_d)
    mine = [c for c in cells if c.slot == slot and not c.measured]
    central = next(
        i for i, c in enumerate(mine) if (c.key == key if slot == "S-P" else c.lean == 1.0)
    )
    where = {c.label: i for i, c in enumerate(mine)}
    flagged = flagged_days(rows)
    days = [d for d in days if d.isoformat() not in flagged]
    shape = (len(mine), len(days))
    net, trades = np.full(shape, np.nan), np.zeros(shape, dtype=np.int64)
    columns = {d.isoformat(): j for j, d in enumerate(days)}
    for label, day, value, count in zip(
        rows["label"], rows["day"], rows["net_per_100"], rows["trades"], strict=True
    ):
        if label in where and day in columns:
            net[where[label], columns[day]] = float(value)
            trades[where[label], columns[day]] = int(count)
    index = np.tile(np.arange(len(mine)), (len(mine), 1))
    found = slot_conditions(CellDays(trades, net, np.full(shape, np.nan)), days, index, central)
    failed = [name for name, ok in (("i", found.i), ("ii", found.ii)) if not ok]
    making = float("nan")
    if slot == "S-M":
        own = kept_rows(rows)
        own = own[own["label"] == mine[central].label]
        fills = float(own["making_fills"].sum())
        making = float(own["making_bp_sum"].sum()) / fills if fills > 0 else float("nan")
        if not making > 0 and "i" not in failed:
            failed.insert(0, "i")
    if record_d["hand_runs"] and signal != "S25":
        status = "not admitted on D (hand run)"
    else:
        status = not_admitted(failed) if failed else "admitted"
    return {"signal": signal, "signal_key": key, "cell": mine[central].label, "S": found.s,
            "monthly": list(found.monthly), "i": found.i, "ii": found.ii,
            "making_bp_per_fill": making, "status_on_D": status,
            "flagged_days": sorted(flagged)}  # fmt: skip


def a0_and_fill(study: Any, inputs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A0 on O (S25's signal under the short-window protocol) crossing at 0/0, with
    its net at the base and every published tier, and FILL: the same decisions
    posted under the three fill rules. Descriptive only; beside the record, its
    four runs' rows for the trial ledger."""
    from functools import partial

    from trading_research.backtest.taker import TakerSettings, execute_day
    from trading_research.data import study_access as sa
    from trading_research.data import study_loader as sl
    from trading_research.labels.taker_vwap import forward_move_bp
    from trading_research.market_making.fill_rules import compare_fill_rules
    from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
    from trading_research.pipeline.short_horizon import features as ft
    from trading_research.pipeline.short_horizon.beside_verdict import crossing_tiers
    from trading_research.pipeline.short_horizon.fits import (
        LOOSEST_RATE,
        FoldData,
        configurations,
        fit_main_process,
        on_complete_rows,
        rank_ic,
        rules_from,
    )
    from trading_research.pipeline.short_horizon.plan import a0_fold
    from trading_research.pipeline.short_horizon_execution import study_config
    from trading_research.reporting.search_keys import S25_KEY

    fold = a0_fold(inputs.usable("D"))
    if not (fold.train and fold.inner and fold.test):
        return {
            "run": False,
            "note": "O, its training or its threshold days have no usable day",
        }, []
    cfg = configurations([S25_KEY])[0]
    model, entry = fit_main_process(
        inputs, fold, cfg, FoldData.load(inputs, fold.train, inputs.columns("micro"))
    )
    rule = rules_from(entry["thresholds"])[LOOSEST_RATE]
    settings = TakerSettings(fees=ZERO_FEES)
    stamps_all: list[np.ndarray] = []
    sides_all: list[np.ndarray] = []
    crossing: list[dict[str, Any]] = []
    ic: dict[str, float] = {}
    events: list[Any] = []
    for day in fold.test:
        book = inputs.book(day)
        frame = inputs.features(day, ["timestamp", *inputs.columns("micro")], every_row=True)
        proba = on_complete_rows(
            model.predict_proba, frame[inputs.columns("micro")].astype(np.float64), 3
        )
        stamps = ft.stamps_ns(book)
        proba[stamps < inputs.scored_from(day)] = np.nan
        side = rule.decide_proba(proba).astype(np.int64)
        rows = np.flatnonzero(side != 0)
        result = execute_day(
            inputs.taker_book(day, book), stamps[rows], side[rows], 50 * 10**9, settings
        )
        trades = result.trades
        crossing.append({"day": day.isoformat(), "trades": len(trades), "gross": float(trades["gross"].sum()),
                         "net": float(trades["net"].sum()), "turnover": float(result.turnover),
                         "notional": float(trades["notional"].sum()), "net_bp": result.day_bp("net")})  # fmt: skip
        ic[day.isoformat()] = rank_ic(
            proba[:, 2] - proba[:, 0], forward_move_bp(book, 50.0).to_numpy()
        )
        stamps_all.append(stamps[rows])
        sides_all.append(side[rows])
        events.append(sl.load_events_day(study.symbol, day, book_roots=[study.roots.book],
                                         trades_root=study.roots.trades, funding_root=study.roots.funding,
                                         access=sa.development_access()))  # fmt: skip
    gross = sum(c["gross"] for c in crossing)
    turnover = sum(c["turnover"] for c in crossing)
    tapes = entry_tapes(np.concatenate(stamps_all), np.concatenate(sides_all))
    config = study_config(inputs.threshold, fees=ZERO_FEES)
    fill = compare_fill_rules(events, partial(SignalEntryQuoter, 50.0), config, tapes)
    mids = {e.day.isoformat(): float(np.mean(e.bid_px[:, 0] + e.ask_px[:, 0]) / 2.0 * e.spec.tick)
            for e in events}  # fmt: skip
    trials = [{"key": "A0|crossing|fees=zero", "day": c["day"], "trades": c["trades"],
               "net_bp": c["net"] / (c["turnover"] / 2.0) * 1e4 if c["turnover"] > 0 else 0.0}
              for c in crossing]  # fmt: skip
    for r in fill.days[~fill.days["excluded"].astype(bool)].to_dict("records"):
        notional = float(r["volume_filled"]) * mids[str(r["day"])]
        trials.append({"key": f"FILL|{r['rule']}|fees=zero", "day": str(r["day"]),
                       "trades": int(r["orders_filled"]),
                       "net_bp": float(r["net"]) / notional * 1e4 if notional > 0 else 0.0})  # fmt: skip
    return {"run": True, "days": [d.isoformat() for d in fold.test], "thresholds": entry["thresholds"],
            "crossing": crossing, "gross_bp_turnover": gross / turnover * 1e4 if turnover else float("nan"),
            "net_at_tiers": crossing_tiers(crossing),
            "ic": ic, "fill_summary": fill.summary.reset_index().to_dict("records"),
            "fill_prediction": fill.prediction()}, trials  # fmt: skip


def trial_rows(rows: pd.DataFrame, run: int, measured: list[dict[str, Any]]) -> pd.DataFrame:
    """Stage C's simulator trials in the trial ledger's form: every simulated cell on
    every day it was not excluded, its net in bp of its filled notional (0 without a
    fill), and A0 and FILL's runs on O (``search_engine.ledger``)."""
    from trading_research.search_engine.ledger import LEDGER_COLUMNS

    kept = rows[~rows["excluded"].astype(bool)]
    entry = kept["label"].str.contains("|entry|", regex=False).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        maker = (
            kept["net_usdt"].to_numpy(dtype=np.float64)
            / kept["passive_notional"].to_numpy(dtype=np.float64)
            * 1e4
        )
    net = np.where(entry, kept["net_bp_trade"].to_numpy(dtype=np.float64), maker)
    frame = pd.DataFrame({"key": kept["label"].to_numpy(), "day": kept["day"].to_numpy(),
                          "net_bp": net, "trades": kept["trades"].to_numpy()})  # fmt: skip
    frame = pd.concat([frame, pd.DataFrame(measured, columns=["key", "day", "net_bp", "trades"])])
    frame["net_bp"] = np.nan_to_num(frame["net_bp"].to_numpy(dtype=np.float64), nan=0.0,
                                    posinf=0.0, neginf=0.0)  # fmt: skip
    frame["day"] = [date.fromisoformat(str(d)) for d in frame["day"]]
    frame["trades"] = frame["trades"].astype(np.int64)
    frame = frame.assign(run=run, G_bp=0.0, pnl_long_bp=0.0, pnl_short_bp=0.0, skips=0)
    frame["skips"] = frame["skips"].astype(np.int64)
    return frame[list(LEDGER_COLUMNS)].reset_index(drop=True)


def record_trials(study: Any, frame: pd.DataFrame) -> dict[str, Any]:
    """Append the run's simulator trials to the trial ledger once (a resumed stage C
    finds them there); the manifest's counts after them."""
    from trading_research.search_engine.ledger import TrialLedger

    ledger = TrialLedger.in_repository(study.repo, study.catalogue())
    manifest = ledger.record()
    if "simulator" not in manifest["runs"].get(str(study.run), {}):
        manifest = ledger.append_simulator(study.run, study.commit(), frame)
    return {"n_trials": int(manifest["n_trials"]), "n_trials_simulator": int(manifest["n_trials_simulator"]),
            "ledger_rows": int(manifest["rows"]), "ledger_sha256": manifest["sha256"]}  # fmt: skip


def passive_dsr(rows: pd.DataFrame, cells: list[SimCell], days: list[date], n_trials: int,
                admission: dict[str, Any]) -> dict[str, float]:  # fmt: skip
    """The deflated Sharpe ratio of S-P's and S-M's daily ``net_per_100_clip_day`` on
    D (their cells, the days no run flagged, 0 without a fill), against N_trials and
    the variance of the Sharpe ratios of every S-P and S-M cell simulated."""
    from trading_research.evaluation.overfitting import deflated_sharpe_ratio, sharpe_variance

    kept = kept_rows(rows)
    wanted = [d.isoformat() for d in days if d.isoformat() not in flagged_days(rows)]
    labels = [c.label for c in cells if c.slot in ("S-P", "S-M") and not c.measured]
    mine = kept[kept["label"].isin(labels)]
    panel = mine.pivot_table(index="day", columns="label", values="net_per_100", aggfunc="sum")
    panel = panel.reindex(index=wanted, columns=labels).fillna(0.0)
    variance = sharpe_variance(panel)
    out: dict[str, float] = {}
    for slot in ("S-P", "S-M"):
        label = admission[slot]["cell"]
        if label in panel and len(panel) >= 3:
            out[f"{slot}/net"] = deflated_sharpe_ratio(
                panel[label].to_numpy(), n_trials, variance
            ).dsr
    return out


def run(study: Any) -> dict[str, Any]:
    """Stage C's record: every simulated cell-day, S-P's and S-M's admission, A0 and
    FILL, and the second detector over D (reported beside R, run in development)."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon.inputs import Inputs
    from trading_research.pipeline.short_horizon.second_detector import run_span

    record_d = study.record("D")
    inputs = Inputs(study)
    days = [date.fromisoformat(d) for d in record_d["usable_test_days"]]
    cells = slot_cells(list(study.keys), record_d)
    rows = simulate_all(study, inputs, cells, days)
    rows.to_parquet(study.dir("C") / "cells.parquet", index=False)
    admission = {slot: admit(rows, cells, days, slot, record_d) for slot in ("S-P", "S-M")}
    summary = kept_rows(rows).groupby("label").agg(
        days=("day", "count"), mean_net_per_100=("net_per_100", "mean"), trades=("trades", "sum"),
        gross_bp_trade=("gross_bp_trade", "mean"))  # fmt: skip
    measured, trials = a0_and_fill(study, inputs)
    recorded = record_trials(study, trial_rows(rows, study.run, trials))
    return {
        "commit": study.commit(),
        "cells": [asdict(c) | {"label": c.label} for c in cells],
        "simulated_cell_days": len(rows),
        "excluded_cell_days": int(rows["excluded"].sum()),
        "admission": admission,
        "summary": summary.reset_index().to_dict("records"),
        "a0_and_fill": measured,
        "trials": recorded,
        "overfitting": {"dsr": passive_dsr(rows, cells, days, recorded["n_trials"], admission)},
        "second_detector_D": run_span(study, "D", sa.development_access()),
    }
