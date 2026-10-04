"""The market-making study on its held-out block (H), read once.

Runs everything the pre-registration (``docs/preregistration/market_making.md``)
assigns to block H, 2024-02-26 to 2024-03-09, with every value frozen by its
Amendment 2 (``configs/mm_prereg.yaml``) and nothing chosen here:

1. the reversion tape over D and H for the H2-admitted instruments, the frozen
   theta and beta, the triggers and their taker twin on H, the index
   autocorrelation on H and H2.3's gate on each H day;
2. the regime flags on the MM-admitted instrument, run continuously from
   2024-02-01 through H;
3. every strategy on its instruments (S0 on all four; S1, S2, its twin and S3
   on BICOUSDT; S1, S4 and X1 on the H2-admitted three), the same under
   pessimistic cancellation attribution (K-pess), and the placebos: H1's stale
   trigger, H2's flipped and 50 shuffled-state tapes, H3's 50 shifted flags;
4. the measurements without a hypothesis: the advantage ladder on all four
   instruments, the fee grid for the gated strategies, and the robustness grid
   on BICOUSDT.

This script only reads and simulates. Every output lands in
``artifacts/mm/heldout/<block>/`` (gitignored); the verdicts and the committed
tables come from ``experiments/market_making_verdicts.py``, which reads those
outputs and no market data, so a fault in the analysis never needs a second
read of the block.

Block H is opened through :meth:`~trading_research.market_making.prereg.HeldOutLedger.open`,
which checks the frozen configuration and the clean committed tree and writes
the ledger entry before the first file of H is read. ``--dry-run`` runs the
same code on the last days of D under the development access, with fewer
placebo seeds and fee points, to exercise every path before H is touched.

Choices the registration does not spell out, fixed here before H is read:

* the shuffled-state placebo rolls the H part of the index tape on its full
  five-second grid; S4 reads the rolled tape and its trigger times, X1 the
  triggers re-derived from it (thinned from the start of the block, as the
  real triggers are), and the taker twin is scored on those same triggers;
* H2.2's flipped placebo posts the side that follows the move at the real
  trigger times; it is compared on X1's net per attempt;
* a gated strategy's fee grid moves the fee its gate reads and the fee it
  pays together, the taker fee staying at the base 5.5 bp; S0 and X1 decide
  without reading the fee, so their break-even is exact analytically;
* the ladder's noise is seeded per instrument-day from the registered seed,
  so rungs 2 to 4 see the same draws.

Usage::

    .venv/bin/python -m experiments.market_making_heldout --dry-run --workers 8
    .venv/bin/python -m experiments.market_making_heldout --block H --workers 8
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
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import (
    BYBIT_BASE,
    BYBIT_STUDY_GROUPS,
    FeeTier,
    best_maker_tier,
)
from trading_research.market_making import flags as flag_module
from trading_research.market_making import heldout, prereg, signals, verdicts
from trading_research.market_making.events import NS_PER_S, day_start_ns, to_ns
from trading_research.market_making.queue import (
    ArrivalGrowth,
    CancelAttribution,
    CrossedPolicy,
    QueuePriority,
)
from trading_research.market_making.quoters import (
    FLAG_SIGNAL,
    ForecastTouchQuoter,
    InsideQuoter,
    Quoter,
    RegimeGuard,
    ReversionLean,
    SignalExecutor,
    SkewQuoter,
    StaleInsideQuoter,
    TouchQuoter,
)
from trading_research.market_making.simulator import SimConfig
from trading_research.pipeline.discovery import UNIVERSE
from trading_research.strategies.reversion import (
    ROWS_PER_DAY,
    ReversionConfig,
    index_level,
    trailing_autocorrelation,
)

STUDY_SYMBOLS = ("BICOUSDT", "CRVUSDT", "XRPUSDT", "BTCUSDT")
BOOK_ROOTS = (Path("data/book"), Path("data/book_fresh"))
TRADES_ROOT = Path("data/trades")
FUNDING_ROOT = Path("data/funding")
UNIVERSE_ROOT = Path("data/universe")
OUT_ROOT = Path("artifacts/mm/heldout")

#: Markouts kept on every main fill: 1/5/30 s, and 60/300/600 s for H2.
MARKOUT_HORIZONS_S = (1.0, 5.0, 30.0, 60.0, 300.0, 600.0)

#: Specs per job: one loaded day each, fewer on the busiest instruments so a
#: worker's peak stays near the development run's.
SPECS_PER_JOB = {"BICOUSDT": 10, "CRVUSDT": 4, "XRPUSDT": 2, "BTCUSDT": 1}

#: Busiest instrument-days first.
JOB_ORDER = ("BTCUSDT", "XRPUSDT", "CRVUSDT", "BICOUSDT")


def report(message: str) -> None:
    print(message, flush=True)


def peak_rss_mb(who: int = resource.RUSAGE_SELF) -> float:
    peak = resource.getrusage(who).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def same(quoter: Quoter) -> Quoter:
    """A picklable factory for a stateless (frozen) quoter."""
    return quoter


def constant(quoter: Quoter) -> Callable[[], Quoter]:
    return partial(same, quoter)


# ---------------------------------------------------------------------------
# The frozen values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frozen:
    """Every value the held-out run reads, from the frozen registration."""

    clip: dict[str, float]
    sigma_ref: dict[str, float]
    mm_symbols: tuple[str, ...]
    h2_symbols: tuple[str, ...]
    s1: dict[str, float]
    s2: dict[str, float]
    s3: dict[str, Any]
    theta: dict[str, float]
    beta: dict[str, float]
    s4_lambda: float
    s4_one_sided: bool
    stale_lag_s: float
    shuffle_seeds: tuple[int, ...]
    shuffle_days: tuple[int, int]
    shuffle_offset_h: tuple[float, float]
    shift_seeds: tuple[int, ...]
    ladder: tuple[dict[str, Any], ...]
    ladder_seed: int
    fee_grid: tuple[float, ...]
    robustness: dict[str, list[Any]]
    sha256: str = field(default="")

    @classmethod
    def load(cls) -> Frozen:
        registration = prereg.PreRegistration.load().require_frozen()
        value = registration.value
        table = value(("admission", "table"))
        h2_placebo = value(("hypotheses", "H2.1", "placebos", "shuffled_state"))
        h3_placebo = value(("hypotheses", "H3", "placebos", "shifted_flag"))
        return cls(
            clip=dict(value(("simulator", "clip", "notional_usdt"))),
            sigma_ref=dict(value(("simulator", "volatility", "sigma_ref"))),
            mm_symbols=tuple(r["symbol"] for r in table if r["mm_admitted"]),
            h2_symbols=tuple(r["symbol"] for r in table if r["h2_admitted"]),
            s1=dict(value(("strategies", "S1", "chosen"))),
            s2=dict(value(("strategies", "S2", "chosen"))),
            s3=dict(value(("strategies", "S3", "chosen"))),
            theta=dict(value(("strategies", "S4", "theta_bp"))),
            beta=dict(value(("strategies", "S4", "beta"))),
            s4_lambda=float(value(("strategies", "S4", "chosen", "lambda"))),
            s4_one_sided=bool(value(("strategies", "S4", "chosen", "one_sided"))),
            stale_lag_s=float(value(("hypotheses", "H1", "placebos", "stale_trigger", "lag_s"))),
            shuffle_seeds=tuple(range(h2_placebo["seed_first"], h2_placebo["seed_last"] + 1)),
            shuffle_days=tuple(h2_placebo["shift_whole_days"]),
            shuffle_offset_h=tuple(h2_placebo["shift_offset_h"]),
            shift_seeds=tuple(range(h3_placebo["seed_first"], h3_placebo["seed_last"] + 1)),
            ladder=tuple(value(("ladder", "rungs"))),
            ladder_seed=int(value(("ladder", "forecast", "noise_seed"))),
            fee_grid=tuple(float(f) for f in value(("fee_breakeven", "maker_fee_grid_bp"))),
            robustness=dict(value(("robustness", "axes"))),
            sha256=registration.sha256,
        )

    def config(self, symbol: str, **changes: Any) -> SimConfig:
        """The registered simulator settings for an instrument, S1's soft limit."""
        base = SimConfig(
            clip_notional=self.clip[symbol],
            soft_limit_clips=float(self.s1["soft_limit_clips"]),
            markout_horizons_s=MARKOUT_HORIZONS_S,
        )
        return base.with_(**changes) if changes else base

    def s1_quoter(self, symbol: str, maker_bp: float = 2.0) -> SkewQuoter:
        p = self.s1
        return SkewQuoter(
            float(p["skew_bp"]),
            float(p["k"]),
            float(p["min_edge_bp"]),
            self.sigma_ref[symbol],
            maker_bp,
        )

    def s2_quoter(self, symbol: str, *, inside: bool = True, maker_bp: float = 2.0) -> InsideQuoter:
        p = self.s2
        return InsideQuoter(
            float(p["skew_bp"]),
            float(p["k"]),
            float(p["min_edge_bp"]),
            int(p["m_ticks"]),
            self.sigma_ref[symbol],
            maker_bp,
            inside=inside,
            name="S2" if inside else "S2-twin",
        )

    def guarded(self, symbol: str, maker_bp: float = 2.0) -> SkewQuoter | InsideQuoter:
        if self.s3["guards"] == "S2":
            return self.s2_quoter(symbol, maker_bp=maker_bp)
        return self.s1_quoter(symbol, maker_bp)

    def s3_quoter(self, symbol: str, maker_bp: float = 2.0) -> RegimeGuard:
        return RegimeGuard(
            self.guarded(symbol, maker_bp),
            str(self.s3["action"]),
            float(self.s3["guard_window_min"]),
        )

    def s4_quoter(self, symbol: str, *, sign: float = 1.0, maker_bp: float = 2.0) -> ReversionLean:
        return ReversionLean(
            self.s1_quoter(symbol, maker_bp),
            self.beta[symbol],
            sign * self.s4_lambda,
            self.s4_one_sided,
            name="S4" if sign > 0 else "S4-flipped",
        )


# ---------------------------------------------------------------------------
# The reversion signal on the block
# ---------------------------------------------------------------------------


@dataclass
class Reversion:
    """One H2-admitted instrument's tapes on the block, real and placebo."""

    symbol: str
    lean: dict[str, signals.SignalTape]
    triggers: signals.Triggers
    flipped: signals.Triggers
    shuffled_lean: dict[int, dict[str, signals.SignalTape]]
    shuffled_triggers: dict[int, signals.Triggers]


def block_grid(start: date, days: int) -> np.ndarray:
    """Bin-end labels of the five-second grid over ``days`` days from ``start``."""
    step = signals.GRID_SECONDS * NS_PER_S
    return day_start_ns(start) + step + np.arange(days * ROWS_PER_DAY, dtype=np.int64) * step


def on_grid(tape: signals.SignalTape, grid: np.ndarray) -> np.ndarray:
    """The tape's values on ``grid``, NaN where it has no label."""
    out = np.full(len(grid), np.nan)
    where = np.searchsorted(tape.ts, grid)
    hit = (where < len(tape.ts)) & (tape.ts[np.minimum(where, len(tape.ts) - 1)] == grid)
    out[hit] = tape.values[where[hit]]  # noqa: PD011 - a SignalTape
    return out


def tape_from_grid(name: str, grid: np.ndarray, values: np.ndarray) -> signals.SignalTape:
    keep = np.isfinite(values)
    return signals.SignalTape(name, grid[keep], values[keep])


def reversion_stage(
    frozen: Frozen,
    block: Sequence[date],
    lookback: Sequence[date],
    access: prereg.Access,
    seeds: Sequence[int],
) -> tuple[dict[str, Reversion], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tapes, triggers, taker twins and the index's state on the block.

    The panel covers ``lookback`` and the block, continuously, so the block's
    first ten minutes have their lookback and H2.3's gate its previous day.
    Returns the tapes, the taker-twin trades (real, flipped and every placebo,
    by ``variant``), the trigger tables, and the state statistics.
    """
    config = ReversionConfig()
    days = sorted(set(lookback) | set(block))
    panel = signals.load_universe_panel(UNIVERSE, days, access=access, root=UNIVERSE_ROOT)
    report(f"  universe panel: {panel.log_mid.shape[1]} instruments, {len(panel.log_mid):,} rows")
    grid = block_grid(block[0], len(block))
    # A rolled value lands only on labels the panel has, so its triggers can be
    # scored by the taker twin, which reads the panel at each trigger.
    on_panel = np.isin(grid, to_ns(pd.DatetimeIndex(panel.log_mid.index)))
    report(f"  block grid: {len(grid):,} labels, {int(on_panel.sum()):,} on the panel")
    out: dict[str, Reversion] = {}
    twins, trigger_rows = [], []
    shifts = {
        seed: verdicts.state_shift_rows(
            seed,
            whole_days=(frozen.shuffle_days[0], min(frozen.shuffle_days[1], len(block) - 1)),
            offset_h=frozen.shuffle_offset_h,
        )
        for seed in seeds
    }
    for symbol in frozen.h2_symbols:
        theta, beta = frozen.theta[symbol], frozen.beta[symbol]
        tape = signals.reversion_tape(panel, symbol, config)
        values = on_grid(tape, grid)
        own = tape_from_grid("index_return_bp", grid, values)
        real = signals.reversion_triggers(own, theta, config).eligible()
        flipped = real.flipped()
        lean = {"index_return_bp": tape, "lean_trigger_s": signals.lean_trigger_tape(tape, theta)}
        shuffled_lean, shuffled_triggers = {}, {}
        variants = [("real", real), ("flipped", flipped)]
        for seed, shift in shifts.items():
            moved = np.where(on_panel, verdicts.roll_values(values, shift), np.nan)
            rolled = tape_from_grid("index_return_bp", grid, moved)
            shuffled_lean[seed] = {
                "index_return_bp": rolled,
                "lean_trigger_s": signals.lean_trigger_tape(rolled, theta),
            }
            shuffled_triggers[seed] = signals.reversion_triggers(rolled, theta, config).eligible()
            variants.append((f"shuffle_{seed:02d}", shuffled_triggers[seed]))
        for variant, triggers in variants:
            twin = signals.taker_twin(triggers, panel, symbol)
            twin["variant"] = variant
            twins.append(twin)
            trigger_rows.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "variant": variant,
                        "ts": triggers.ts,
                        "day": pd.to_datetime(triggers.ts, utc=True).date.astype(str),
                        "direction": triggers.direction,
                        "signal_bp": triggers.signal_bp,
                    }
                )
            )
        out[symbol] = Reversion(symbol, lean, real, flipped, shuffled_lean, shuffled_triggers)
        report(f"  {symbol}: theta {theta}, beta {beta}, {len(real)} triggers on the block")
    state = reversion_state(panel, block, days, config)
    state["shift_rows"] = {str(k): int(v) for k, v in shifts.items()}
    twin_frame = pd.concat(twins, ignore_index=True)
    twin_frame["day"] = twin_frame["day"].astype(str)
    return out, twin_frame, pd.concat(trigger_rows, ignore_index=True), state


def reversion_state(
    panel: signals.UniversePanel,
    block: Sequence[date],
    days: Sequence[date],
    config: ReversionConfig,
) -> dict[str, Any]:
    """The index's ten-minute autocorrelation over the block, and H2.3's gate
    (the trailing one-day autocorrelation known at each block day's open)."""
    labels = pd.DatetimeIndex(panel.log_mid.index)
    complete = len(labels) == ROWS_PER_DAY * len(days)
    if not complete:
        # The gate's daily blocks assume a complete grid; without one it is
        # reported as missing rather than computed on misaligned days.
        report(f"  the panel has {len(labels)} rows, not {ROWS_PER_DAY} a day: no H2.3 gate")
    level = index_level(panel.log_mid.to_numpy(dtype=np.float64))
    lag = config.hold
    inside = level[np.asarray(labels > pd.Timestamp(block[0], tz="UTC"))]
    past = np.full(len(inside), np.nan)
    past[lag:] = (inside[lag:] - inside[:-lag]) * 1e4
    forward = np.full(len(inside), np.nan)
    forward[:-lag] = (inside[lag:] - inside[:-lag]) * 1e4
    ok = np.isfinite(past) & np.isfinite(forward)
    gate = trailing_autocorrelation(level, lag, ROWS_PER_DAY, known_at_open=True)
    per_day = gate[::ROWS_PER_DAY]
    return {
        "index_autocorrelation_block": float(np.corrcoef(past[ok], forward[ok])[0, 1]),
        "h2_3_gate": {
            d.isoformat(): float(per_day[days.index(d)]) if complete else math.nan for d in block
        },
    }


# ---------------------------------------------------------------------------
# The regime flags
# ---------------------------------------------------------------------------


def flag_stage(
    frozen: Frozen,
    block: Sequence[date],
    lookback: Sequence[date],
    access: prereg.Access,
    seeds: Sequence[int],
) -> tuple[
    dict[str, signals.SignalTape],
    dict[int, dict[str, signals.SignalTape]],
    pd.DataFrame,
    dict[str, Any],
]:
    """Both detectors from the first lookback day through the block, the real
    flag tape per MM-admitted instrument, and each seed's shifted-flag tape."""
    real: dict[str, signals.SignalTape] = {}
    shifted: dict[int, dict[str, signals.SignalTape]] = {seed: {} for seed in seeds}
    frames, meta = [], {}
    start = day_start_ns(block[0])
    end = day_start_ns(block[-1]) + 86_400 * NS_PER_S
    for symbol in frozen.mm_symbols:
        days = sorted(set(lookback) | set(block))
        timeline = flag_module.build_flags(symbol, days, book_roots=BOOK_ROOTS, access=access)
        real[symbol] = timeline.tape()
        mask = (timeline.effective_ns >= start) & (timeline.effective_ns < end)
        on_block = flag_module.FlagTimeline(
            symbol,
            timeline.effective_ns[mask],
            timeline.detector[mask],
            timeline.statistic[mask],
            (start, end),
        )
        offsets = {seed: on_block.placebo_offset(seed) for seed in seeds}
        for seed, offset in offsets.items():
            shifted[seed][symbol] = on_block.shifted(offset).tape()
        frame = on_block.frame().drop(columns=["effective_ns"])
        frames.append(frame)
        meta[symbol] = {
            "flags_block": len(on_block),
            "flags_per_day_block": on_block.rate_per_day(len(block)),
            "flags_lookback": int((timeline.effective_ns < start).sum()),
            "counts": dict(flag_module.flag_counts(on_block)),
            "guard_share": flag_module.guard_share(on_block, float(frozen.s3["guard_window_min"])),
            "offset_h": {str(k): v / 3600 / NS_PER_S for k, v in offsets.items()},
        }
        report(f"  {symbol}: {len(on_block)} flags on the block")
    return real, shifted, pd.concat(frames, ignore_index=True), meta


# ---------------------------------------------------------------------------
# What runs on each instrument
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """The specs of one instrument, and the tapes they read."""

    symbol: str
    frozen: Frozen
    reversion: Reversion | None
    flags: signals.SignalTape | None
    shifted_flags: dict[int, signals.SignalTape]
    seeds: Sequence[int]
    fee_grid: Sequence[float]
    specs: list[heldout.Spec] = field(default_factory=list)

    def add(self, label: str, quoter: Quoter | type[SignalExecutor], group: str, **kw: Any) -> None:
        config = kw.pop("config", None) or self.frozen.config(self.symbol)
        make = SignalExecutor if quoter is SignalExecutor else constant(quoter)  # type: ignore[arg-type]
        self.specs.append(heldout.Spec(label, make, config, group=group, **kw))

    @property
    def mm(self) -> bool:
        return self.symbol in self.frozen.mm_symbols

    @property
    def h2(self) -> bool:
        return self.reversion is not None

    def lean_tapes(self) -> dict[str, signals.SignalTape]:
        assert self.reversion is not None
        return self.reversion.lean

    def x1_tapes(self, triggers: signals.Triggers) -> dict[str, signals.SignalTape]:
        return triggers.tapes()

    def guard_tapes(self, tape: signals.SignalTape | None = None) -> dict[str, signals.SignalTape]:
        chosen = tape if tape is not None else self.flags
        assert chosen is not None
        return {FLAG_SIGNAL: chosen}


def main_specs(plan: Plan, group: str = "main", tag: str = "", **config: Any) -> None:
    """The strategies on their instruments; ``config`` changes the simulator
    settings (the pessimistic bracket, a robustness cell) and ``tag`` names it."""
    f, s = plan.frozen, plan.symbol
    tag = f"|{tag}" if tag else ""
    settings = f.config(s, **config)
    keep = group == "main"
    if group in ("main", "robust"):
        plan.add(
            f"S0{tag}", TouchQuoter(), group, config=settings.with_(soft_limit_clips=6.0), keep=keep
        )
    if plan.mm:
        plan.add(f"S1{tag}", f.s1_quoter(s), group, config=settings, keep=keep)
        plan.add(f"S2{tag}", f.s2_quoter(s), group, config=settings, keep=keep)
        if group != "robust":
            plan.add(
                f"S2-twin{tag}", f.s2_quoter(s, inside=False), group, config=settings, keep=keep
            )
        plan.add(
            f"S3{tag}", f.s3_quoter(s), group, config=settings, tapes=plan.guard_tapes(), keep=keep
        )
    elif plan.h2:
        plan.add(f"S1{tag}", f.s1_quoter(s), group, config=settings, keep=keep)
    if plan.h2:
        assert plan.reversion is not None
        plan.add(
            f"S4{tag}", f.s4_quoter(s), group, config=settings, tapes=plan.lean_tapes(), keep=keep
        )
        plan.add(
            f"X1{tag}",
            SignalExecutor,
            group,
            config=settings,
            tapes=plan.x1_tapes(plan.reversion.triggers),
            keep=keep,
        )


def placebo_specs(plan: Plan) -> None:
    f, s = plan.frozen, plan.symbol
    if plan.mm:
        stale = StaleInsideQuoter(f.s2_quoter(s), lag_s=f.stale_lag_s)
        plan.add("S2-stale", stale, "placebo_h1", built=(("stale_spread", f.stale_lag_s),))
        for seed in plan.seeds:
            tapes = plan.guard_tapes(plan.shifted_flags[seed])
            plan.add(f"S3|shift={seed:02d}", f.s3_quoter(s), "placebo_h3", tapes=tapes)
    if plan.reversion is not None:
        rev = plan.reversion
        plan.add("S4-flipped", f.s4_quoter(s, sign=-1.0), "placebo_h2", tapes=rev.lean)
        plan.add("X1-flipped", SignalExecutor, "placebo_h2", tapes=rev.flipped.tapes())
        for seed in plan.seeds:
            lean = rev.shuffled_lean[seed]
            plan.add(f"S4|shuffle={seed:02d}", f.s4_quoter(s), "placebo_h2", tapes=lean)
            triggers = rev.shuffled_triggers[seed].tapes()
            plan.add(f"X1|shuffle={seed:02d}", SignalExecutor, "placebo_h2", tapes=triggers)


def fee_specs(plan: Plan) -> None:
    """The gated strategies over the fee grid: the gate reads the fee it pays."""
    f, s = plan.frozen, plan.symbol
    for fee in plan.fee_grid:
        if fee == BYBIT_BASE.maker_bp:
            continue  # the main run
        tier = FeeTier(f"grid_{fee:+.2f}", fee, BYBIT_BASE.taker_bp)
        config = f.config(s, fees=tier)
        tag = f"fee={fee:+.2f}"
        if plan.mm or plan.h2:
            plan.add(f"S1|{tag}", f.s1_quoter(s, fee), "fee", config=config)
        if plan.mm:
            plan.add(f"S2|{tag}", f.s2_quoter(s, maker_bp=fee), "fee", config=config)
            plan.add(
                f"S3|{tag}", f.s3_quoter(s, fee), "fee", config=config, tapes=plan.guard_tapes()
            )
        if plan.h2:
            plan.add(
                f"S4|{tag}",
                f.s4_quoter(s, maker_bp=fee),
                "fee",
                config=config,
                tapes=plan.lean_tapes(),
            )


def robust_specs(plan: Plan) -> None:
    """One axis at a time on the MM-admitted instrument, the rest at defaults."""
    if not plan.mm:
        return
    axes = plan.frozen.robustness
    cells: list[tuple[str, dict[str, Any]]] = []
    for value in axes["cancel_attribution"]:
        if value != CancelAttribution.PROPORTIONAL.value:
            cells.append((f"cancel={value}", {"cancel_attribution": CancelAttribution(value)}))
    for value in axes["arrival_growth"]:
        if value != ArrivalGrowth.PRO_RATA_TIME.value:
            cells.append((f"growth={value}", {"arrival_growth": ArrivalGrowth(value)}))
    for value in axes["latency_ms"]:
        if value != 10:
            ns = int(value) * 1_000_000
            cells.append((f"latency={value}ms", {"order_latency_ns": ns, "cancel_latency_ns": ns}))
    for value in axes["feed_latency_ms"]:
        if value != 0:
            cells.append((f"feed={value}ms", {"feed_latency_ns": int(value) * 1_000_000}))
    for value in axes["crossed_policy"]:
        if value != CrossedPolicy.TRADE_TAPE.value:
            cells.append((f"crossed={value}", {"crossed_policy": CrossedPolicy(value)}))
    existing = {spec.label for spec in plan.specs}
    for tag, config in cells:
        before = len(plan.specs)
        main_specs(plan, "robust", tag, **config)
        added = plan.specs[before:]
        plan.specs[before:] = [spec for spec in added if spec.label not in existing]


def ladder_specs(plan: Plan) -> None:
    """S0 up the advantage ladder, each rung adding to the one below."""
    f, s = plan.frozen, plan.symbol
    settings: dict[str, Any] = {"soft_limit_clips": 6.0}
    r2: float | None = None
    tier = BYBIT_BASE
    for rung in f.ladder:
        number = int(rung["rung"])
        if "queue_priority" in rung:
            settings["queue_priority"] = QueuePriority(rung["queue_priority"])
        if "latency_ms" in rung:
            ns = int(rung["latency_ms"]) * 1_000_000
            settings.update(order_latency_ns=ns, cancel_latency_ns=ns)
        if "forecast_r2" in rung:
            r2 = float(rung["forecast_r2"])
        if rung.get("fee_tier") == "best_published":
            tier = best_maker_tier(BYBIT_STUDY_GROUPS[s])
        if number == 0:
            continue  # rung 0 is S0 itself, in the main run
        config = f.config(s, fees=tier, **settings)
        label = f"ladder|rung={number}"
        if r2 is None:
            plan.add(label, TouchQuoter(), "ladder", config=config)
        else:
            quoter = ForecastTouchQuoter(maker_bp=tier.maker_bp)
            plan.add(label, quoter, "ladder", config=config, built=(("forecast", r2),), oracle=True)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def open_block(args: argparse.Namespace) -> tuple[prereg.Access, list[date], list[date], str]:
    """The access, the block's days, the lookback days and the output name."""
    d_days = prereg.block_days("D")
    if args.dry_run:
        block = d_days[-args.dry_days :]
        return prereg.development_access(), block, d_days[: -args.dry_days], "dry"
    if args.block != "H":
        raise SystemExit("this script reads block H; F is the next pull request's")
    ledger = prereg.HeldOutLedger()
    access = ledger.open("H", force=args.force)
    report(f"ledger: {json.dumps(ledger.entries()['H'][-1])}")
    return access, prereg.block_days("H"), d_days, "H"


def plan_all(
    frozen: Frozen,
    reversion: dict[str, Reversion],
    flags: dict[str, signals.SignalTape],
    shifted: dict[int, dict[str, signals.SignalTape]],
    seeds: Sequence[int],
    fee_grid: Sequence[float],
) -> list[Plan]:
    plans = []
    for symbol in STUDY_SYMBOLS:
        plan = Plan(
            symbol,
            frozen,
            reversion.get(symbol),
            flags.get(symbol),
            {seed: shifted[seed][symbol] for seed in seeds if symbol in shifted.get(seed, {})},
            seeds,
            fee_grid,
        )
        main_specs(plan)
        main_specs(
            plan, "pess", "cancel=pessimistic", cancel_attribution=CancelAttribution.PESSIMISTIC
        )
        placebo_specs(plan)
        fee_specs(plan)
        robust_specs(plan)
        ladder_specs(plan)
        plans.append(plan)
        report(f"  {symbol}: {len(plan.specs)} specs a day")
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="H")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="the last days of D, not H")
    parser.add_argument("--dry-days", type=int, default=3)
    parser.add_argument("--dry-seeds", type=int, default=3)
    parser.add_argument("--dry-symbols", default=",".join(STUDY_SYMBOLS))
    parser.add_argument("--force", action="store_true", help="a second read, stamped as such")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("between 1 and 8 workers")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    started = time.perf_counter()
    stages: dict[str, float] = {}
    frozen = Frozen.load()
    access, block, lookback, name = open_block(args)
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(frozen.shuffle_seeds)
    fee_grid = list(frozen.fee_grid)
    if args.dry_run:
        seeds = seeds[: args.dry_seeds]
        fee_grid = [fee_grid[0], fee_grid[-3], fee_grid[-1]]
    report(f"block {name}: {block[0]} to {block[-1]}, {len(seeds)} placebo seeds")

    t = time.perf_counter()
    report("reversion: tapes, triggers and taker twins")
    reversion, twins, triggers, state = reversion_stage(frozen, block, lookback, access, seeds)
    twins.to_parquet(out / "taker_twins.parquet", index=False)
    triggers.to_parquet(out / "triggers.parquet", index=False)
    stages["reversion"] = time.perf_counter() - t

    t = time.perf_counter()
    report("flags: both detectors from 2024-02-01 through the block")
    flags, shifted, flag_frame, flag_meta = flag_stage(frozen, block, lookback, access, seeds)
    flag_frame.to_csv(out / "flags.csv", index=False)
    stages["flags"] = time.perf_counter() - t

    plans = plan_all(frozen, reversion, flags, shifted, seeds, fee_grid)
    wanted = set(args.dry_symbols.split(",")) if args.dry_run else set(STUDY_SYMBOLS)
    jobs = []
    for symbol in JOB_ORDER:
        plan = next(p for p in plans if p.symbol == symbol)
        if symbol not in wanted:
            continue
        jobs += heldout.build_jobs(
            symbol,
            block,
            plan.specs,
            book_roots=BOOK_ROOTS,
            trades_root=TRADES_ROOT,
            funding_root=FUNDING_ROOT,
            out=out,
            specs_per_job=SPECS_PER_JOB[symbol],
            market_horizons_s=MARKOUT_HORIZONS_S,
            noise_base=frozen.ladder_seed,
        )
    report(f"simulating: {len(jobs)} jobs on {args.workers} workers")
    t = time.perf_counter()
    rows = heldout.run_jobs(
        jobs,
        workers=args.workers,
        day_guard=lambda days: access.require(days, what="simulate"),
        on_progress=report,
    )
    stages["simulation"] = time.perf_counter() - t
    rows["stamp"] = access.stamp
    rows.to_parquet(out / "rows.parquet", index=False)
    stages["total"] = time.perf_counter() - started
    meta = {
        "block": name,
        "days": [d.isoformat() for d in block],
        "stamp": access.stamp,
        "config_sha256": frozen.sha256,
        "seeds": seeds,
        "fee_grid": fee_grid,
        "state": state,
        "flags": flag_meta,
        "stages_s": stages,
        "workers": args.workers,
        "jobs": len(jobs),
        "peak_rss_mb_worker": float(rows["worker_peak_rss_mb"].max()),
        "peak_rss_mb_any_child": peak_rss_mb(resource.RUSAGE_CHILDREN),
        "peak_rss_mb_driver": peak_rss_mb(),
    }
    if not args.dry_run:
        meta["ledger"] = prereg.HeldOutLedger().entries()["H"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    report(json.dumps({k: meta[k] for k in ("stages_s", "jobs", "peak_rss_mb_worker")}, indent=1))
    report("done")


if __name__ == "__main__":
    main()
