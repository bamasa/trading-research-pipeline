"""The market-making study's verdicts on its held-out block, from the run's outputs.

Reads what ``experiments/market_making_heldout.py`` wrote under
``artifacts/mm/heldout/<block>/`` — the per-day rows of every simulated cell,
the kept fills, orders and equity of the main strategies, the triggers and
taker twins, the flags and the block's state — and no market data. Computes
exactly what the pre-registration registers, with the frozen values:

* H1, H2.1, H2.2 and H3, each kill condition evaluated mechanically, the day
  level one-sided t of each, Holm's procedure across the four, and the status;
* H2.3 on the block's days (its registered test spans H and F, so its status
  waits for F);
* the advantage ladder, the fee break-even against the transcribed Bybit
  tiers, the robustness grid, the markouts by fill path against the
  market-wide passive benchmark, the decomposition, inventory and fills.

Writes the committed tables ``experiments/results/mm_*_<suffix>.csv``.

How the registered statistics are read, fixed before the block was read:

* daily series pool instruments by summing each day's paired differences over
  the instruments that have both days usable (as on D);
* H1's and H2.2's statements make two claims each (a positive result, and
  more than the comparator), so each is tested at the larger of the two
  one-sided p-values;
* K-pess, K-fund and K-dir are applied to every sign claim of the hypothesis;
* H2.2's net per attempt is the block's total over its eligible triggers,
  misses and skipped triggers counting zero; its daily series is the same
  ratio per day, as on D;
* H1's K3 compares the volume-weighted 5 s markout of S2's fills from orders
  placed inside the spread with that of the twin's fills from orders placed at
  the touch; with no inside fill it fires.

Usage::

    .venv/bin/python -m experiments.market_making_verdicts --block H
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments._common import RESULTS
from trading_research.backtest.costs import (
    BYBIT_FEE_CAVEAT,
    BYBIT_MM_PROGRAMME_MAKER_BP,
    BYBIT_STUDY_GROUPS,
    breakeven_maker_bp,
    tiers_for,
)
from trading_research.evaluation.significance import SignificanceError, assess
from trading_research.market_making import heldout
from trading_research.market_making.verdicts import (
    INCONCLUSIVE,
    KillCheck,
    day_t,
    exceeds_placebos,
    holm,
    k_dir,
    k_pess,
    status,
)

OUT_ROOT = Path("artifacts/mm/heldout")
HORIZONS_S = (1.0, 5.0, 30.0, 60.0, 300.0, 600.0)
PNL = ("net", "making", "spread", "adverse", "inventory", "fees", "funding")

#: Seconds a day the quoters may quote: 00:10 to 23:58.
QUOTING_S = 86_280.0 - 600.0


class Run:
    """The outputs of one run of the held-out script."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.meta: dict[str, Any] = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        rows = pd.read_parquet(out / "rows.parquet")
        rows["excluded"] = rows["excluded"].astype(bool)
        self.market = rows[rows["group"] == "market"].copy()
        self.rows = rows[rows["group"] != "market"].copy()
        self.twins = pd.read_parquet(out / "taker_twins.parquet")
        self.triggers = pd.read_parquet(out / "triggers.parquet")
        self.days = list(self.meta["days"])

    def usable(self, label: str, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Rows of one cell that may enter a verdict: simulated and not excluded."""
        frame = self.rows[self.rows["label"] == label]
        if symbols is not None:
            frame = frame[frame["symbol"].isin(list(symbols))]
        return frame[(frame["status"] == "ok") & ~frame["excluded"]]

    def paired(self, label: str, base: str, symbols: Sequence[str]) -> pd.DataFrame:
        """``label`` minus ``base`` per instrument-day, where both are usable."""
        a = self.usable(label, symbols)[["symbol", "day", *PNL]]
        b = self.usable(base, symbols)[["symbol", "day", *PNL]]
        both = a.merge(b, on=["symbol", "day"], suffixes=("", "_base"))
        for name in PNL:
            both[f"d_{name}"] = both[name] - both[f"{name}_base"]
        return both

    def kept(self, label: str, symbol: str, what: str) -> pd.DataFrame:
        return heldout.kept(self.out, label, symbol, what)


def pooled(paired: pd.DataFrame, column: str = "d_net") -> pd.Series:
    """Each day's sum over instruments."""
    return paired.groupby("day")[column].sum()


def fmt(value: float) -> float:
    return float(value) if value is not None and math.isfinite(value) else math.nan


# ---------------------------------------------------------------------------
# What each strategy did
# ---------------------------------------------------------------------------

MAIN = ("S0", "S1", "S2", "S2-twin", "S3", "S4", "X1")


def inventory(run: Run, label: str, symbol: str) -> dict[str, float]:
    """RMS and maximum |position| in the quoting hours (minute samples, in
    base units and in notional), and the maximum drawdown of the block's
    cumulative minute equity."""
    equity = run.kept(label, symbol, "equity")
    if equity.empty:
        return {}
    seconds = (equity["ts"] // 1_000_000_000) % 86_400
    quoting = equity[(seconds >= 600) & (seconds <= 86_280)]
    position = quoting["position"].to_numpy(dtype=np.float64)
    notional = np.abs(position * quoting["mid"].to_numpy(dtype=np.float64))
    curve = []
    carried = 0.0
    for _, day in equity.groupby("day", sort=True):
        values = day["equity"].to_numpy(dtype=np.float64)
        curve.append(carried + values)
        carried += values[-1]
    total = np.concatenate(curve)
    drawdown = float(np.max(np.maximum.accumulate(np.r_[0.0, total]) - np.r_[0.0, total]))
    return {
        "rms_position": float(np.sqrt(np.nanmean(position**2))),
        "max_abs_position_minute": float(np.nanmax(np.abs(position))),
        "rms_notional_usdt": float(np.sqrt(np.nanmean(notional**2))),
        "max_drawdown_usdt": drawdown,
    }


def strategies(run: Run) -> pd.DataFrame:
    """One row per strategy and instrument: the decomposition per day, fills,
    fill ratio, time quoted, inventory, flattens and drawdown."""
    rows = []
    main = run.rows[run.rows["group"] == "main"]
    for (label, symbol), frame in main.groupby(["label", "symbol"], sort=False):
        ok = frame[(frame["status"] == "ok") & ~frame["excluded"]]
        row: dict[str, Any] = {"strategy": label, "symbol": symbol, "days": len(ok)}
        row["excluded_days"] = len(frame) - len(ok)
        for name in PNL:
            row[f"{name}_usd_day"] = float(ok[name].mean())
        turnover = float((ok["maker_turnover"] + ok["taker_turnover"]).sum())
        row["net_bp_turnover"] = float(ok["net"].sum() / turnover * 1e4) if turnover else math.nan
        row["maker_turnover_usd_day"] = float(ok["maker_turnover"].mean())
        row["passive_fills_day"] = float((ok["fills"] - ok["flattens"]).mean())
        row["flattens_day"] = float(ok["flattens"].mean())
        live = float(ok["orders_live"].sum())
        row["fill_ratio_per_order"] = (
            float(ok["orders_with_fill"].sum() / live) if live else math.nan
        )
        row["bid_quoted_share"] = float(ok["bid_quoted_s"].mean() / QUOTING_S)
        row["ask_quoted_share"] = float(ok["ask_quoted_s"].mean() / QUOTING_S)
        row["max_abs_position"] = float(ok["max_abs_position"].max())
        row["clip_over_touch"] = float(ok["clip_over_touch"].mean())
        row["fills_inside_day"] = float(ok["fills_inside"].mean())
        row["fills_touch_day"] = float(ok["fills_touch"].mean())
        row["fills_behind_day"] = float(ok["fills_behind"].mean())
        row.update(inventory(run, str(label), str(symbol)))
        rows.append(row)
    order = {name: i for i, name in enumerate(MAIN)}
    table = pd.DataFrame(rows)
    table["_o"] = table["strategy"].map(order)
    return table.sort_values(["symbol", "_o"]).drop(columns="_o").reset_index(drop=True)


def markouts(run: Run) -> pd.DataFrame:
    """Volume-weighted markouts by strategy, instrument and fill path, beside
    the market-wide passive benchmark at the same horizons."""
    rows = []
    for label in MAIN:
        for symbol in sorted(run.rows.loc[run.rows["label"] == label, "symbol"].unique()):
            fills = run.kept(label, symbol, "fills")
            if fills.empty:
                continue
            passive = fills[fills["maker"]].assign(path="all passive")
            for path, group in pd.concat([fills, passive]).groupby("path"):
                weights = group["size"].to_numpy(dtype=np.float64)
                row: dict[str, Any] = {"strategy": label, "symbol": symbol, "path": path}
                row["fills"] = len(group)
                for h in HORIZONS_S:
                    values = group[f"markout_{h:g}s"].to_numpy(dtype=np.float64)
                    ok = np.isfinite(values)
                    row[f"markout_bp_{h:g}s"] = (
                        float(np.average(values[ok], weights=weights[ok])) if ok.any() else math.nan
                    )
                if path == "all passive":
                    try:
                        stats = assess(group, value="markout_5s", cluster="day", series=None)
                        row["fill_level_t_5s"] = stats.trade_t
                        row["day_level_t_5s"] = stats.cluster_t
                    except SignificanceError:
                        pass
                rows.append(row)
    for symbol, frame in run.market.groupby("symbol"):
        row = {"strategy": "market-wide passive", "symbol": symbol, "path": "benchmark"}
        row["fills"] = float(frame["prints"].sum())
        for h in HORIZONS_S:
            weights = frame[f"benchmark_volume_{h:g}s"].to_numpy(dtype=np.float64)
            values = frame[f"benchmark_bp_{h:g}s"].to_numpy(dtype=np.float64)
            row[f"markout_bp_{h:g}s"] = float(np.average(values, weights=weights))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The primaries
# ---------------------------------------------------------------------------


class Verdict:
    """One hypothesis: its kill checks, its deciding daily series and p-value."""

    def __init__(self, name: str, metric: str, unit: str) -> None:
        self.name, self.metric, self.unit = name, metric, unit
        self.checks: list[KillCheck] = []
        self.value = math.nan
        self.daily: pd.Series = pd.Series(dtype=float)
        self.p = math.nan
        self.beside: dict[str, Any] = {}

    def check(
        self,
        name: str,
        test: str,
        value: float,
        reference: float,
        fired: bool,
        outcome: str = "killed",
        note: str = "",
    ) -> None:
        self.checks.append(
            KillCheck(self.name, name, test, fmt(value), fmt(reference), bool(fired), outcome, note)
        )

    def common(
        self,
        claims: dict[str, tuple[float, float, float, float]],
        direction: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """K-pess, K-fund and K-dir over the sign claims, each given as
        (net, net under pessimistic attribution, net minus funding, making).
        ``direction`` replaces the claims for K-dir with (net, making) pairs,
        for a claim whose comparator has no making part."""
        names = list(claims)
        proportional = [claims[c][0] > 0 for c in names]
        pessimistic = [claims[c][1] > 0 for c in names]
        worst = min(claims[c][1] for c in names)
        self.check(
            "K-pess",
            "positive under proportional, not under pessimistic attribution",
            worst,
            0.0,
            k_pess(proportional, pessimistic),
            note="; ".join(f"{c}: {claims[c][0]:+.4g} -> {claims[c][1]:+.4g}" for c in names),
        )
        ex_funding = min(claims[c][2] for c in names)
        self.check(
            "K-fund",
            "net minus funding <= 0",
            ex_funding,
            0.0,
            ex_funding <= 0,
            note="; ".join(f"{c}: {claims[c][2]:+.4g}" for c in names),
        )
        pairs = direction or {c: (claims[c][0], claims[c][3]) for c in names}
        fired = [c for c, (net, making) in pairs.items() if k_dir(making, net)]
        self.check(
            "K-dir",
            "making <= 0 while net > 0",
            min(making for _, making in pairs.values()),
            0.0,
            bool(fired),
            note="; ".join(f"{c}: making {m:+.4g}, net {n:+.4g}" for c, (n, m) in pairs.items()),
        )


def mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else math.nan


def h1(run: Run, mm: Sequence[str]) -> Verdict:
    verdict = Verdict("H1", "S2 daily net, and S2 minus its twin, paired daily", "USDT/day")
    s2 = run.usable("S2", mm).groupby("day")[list(PNL)].sum()
    pair = run.paired("S2", "S2-twin", mm)
    diff = pooled(pair)
    verdict.check(
        "K1", "sum of S2's daily net over H <= 0", s2["net"].sum(), 0.0, s2["net"].sum() <= 0
    )
    verdict.check("K2", "mean paired daily S2 - twin <= 0", diff.mean(), 0.0, diff.mean() <= 0)
    inside, touch = inside_against_touch(run, mm)
    verdict.check(
        "K3",
        "5 s markout per inside fill not greater than per twin touch fill",
        inside,
        touch,
        not (inside > touch),
        note="no inside fill" if math.isnan(inside) else "",
    )
    stale = pooled(run.paired("S2-stale", "S2-twin", mm))
    verdict.check(
        "K4",
        "the stale-trigger placebo gains at least as much over the twin",
        stale.mean(),
        diff.mean(),
        stale.mean() >= diff.mean(),
    )
    s2_pess = run.usable("S2|cancel=pessimistic", mm)["net"].sum()
    diff_pess = pooled(run.paired("S2|cancel=pessimistic", "S2-twin|cancel=pessimistic", mm))
    verdict.common(
        {
            "S2 net": (
                s2["net"].sum(),
                s2_pess,
                (s2["net"] - s2["funding"]).sum(),
                s2["making"].sum(),
            ),
            "S2 - twin": (
                diff.mean(),
                mean_or_nan(diff_pess),
                mean_or_nan(pooled(pair, "d_net") - pooled(pair, "d_funding")),
                mean_or_nan(pooled(pair, "d_making")),
            ),
        }
    )
    t_net, t_diff = day_t(s2["net"]), day_t(diff)
    verdict.value, verdict.daily = float(diff.mean()), diff
    verdict.p = max(t_net.p_one_sided, t_diff.p_one_sided)
    verdict.beside = {"S2 net t": t_net.t, "S2 - twin t": t_diff.t, "S2 net sum": s2["net"].sum()}
    return verdict


def inside_against_touch(run: Run, mm: Sequence[str]) -> tuple[float, float]:
    """Volume-weighted 5 s markout of S2's inside fills and of the twin's touch fills."""
    out = []
    for label, where in (("S2", heldout.INSIDE), ("S2-twin", heldout.TOUCH)):
        frames = [run.kept(label, s, "fills") for s in mm]
        fills = pd.concat([f for f in frames if not f.empty] or [pd.DataFrame()])
        chosen = fills[fills["placement"] == where] if len(fills) else fills
        if chosen.empty:
            out.append(math.nan)
            continue
        out.append(float(np.average(chosen["markout_5s"], weights=chosen["size"])))
    return out[0], out[1]


def placebo_means(run: Run, prefix: str, base: str, symbols: Sequence[str]) -> dict[int, float]:
    """Each placebo seed's mean pooled daily difference against ``base``."""
    labels = sorted(label for label in run.rows["label"].unique() if str(label).startswith(prefix))
    return {
        int(label.split("=")[1]): float(pooled(run.paired(label, base, symbols)).mean())
        for label in labels
    }


def h2_1(run: Run, h2: Sequence[str]) -> tuple[Verdict, dict[int, float]]:
    verdict = Verdict("H2.1", "S4 - S1, paired daily, pooled over instruments", "USDT/day")
    pair = run.paired("S4", "S1", h2)
    diff = pooled(pair)
    verdict.check("K1", "mean paired difference <= 0", diff.mean(), 0.0, diff.mean() <= 0)
    flipped = pooled(run.paired("S4-flipped", "S1", h2))
    verdict.check(
        "K2",
        "the flipped placebo does at least as well",
        flipped.mean(),
        diff.mean(),
        flipped.mean() >= diff.mean(),
    )
    shuffled = placebo_means(run, "S4|shuffle=", "S1", h2)
    level, above = exceeds_placebos(float(diff.mean()), list(shuffled.values()))
    verdict.check(
        "K3",
        "the true difference does not exceed the 95th percentile of the shuffled-state placebos",
        diff.mean(),
        level,
        not above,
        note=f"{len(shuffled)} placebos",
    )
    pess = pooled(run.paired("S4|cancel=pessimistic", "S1|cancel=pessimistic", h2))
    verdict.common(
        {
            "S4 - S1": (
                diff.mean(),
                mean_or_nan(pess),
                mean_or_nan(pooled(pair, "d_net") - pooled(pair, "d_funding")),
                mean_or_nan(pooled(pair, "d_making")),
            )
        }
    )
    t = day_t(diff)
    verdict.value, verdict.daily, verdict.p = float(diff.mean()), diff, t.p_one_sided
    for symbol, part in pair.groupby("symbol"):
        verdict.beside[f"{symbol} S4 - S1"] = float(part["d_net"].mean())
    if len(set(pair["day"])) > 1:
        stats = assess(pair, value="d_net", cluster="day", series="symbol")
        verdict.beside["instrument-day t"] = stats.trade_t
        verdict.beside["correlation across instruments"] = stats.correlation
    return verdict, shuffled


def h3(
    run: Run, mm: Sequence[str], flags_block: int, rate_d: float
) -> tuple[Verdict, dict[int, float]]:
    guarded = "S1"
    verdict = Verdict("H3", f"S3 - {guarded}, paired daily", "USDT/day")
    pair = run.paired("S3", guarded, mm)
    diff = pooled(pair)
    verdict.check("K1", "mean paired daily difference <= 0", diff.mean(), 0.0, diff.mean() <= 0)
    shifted = placebo_means(run, "S3|shift=", guarded, mm)
    level, above = exceeds_placebos(float(diff.mean()), list(shifted.values()))
    verdict.check(
        "K2",
        "the true difference does not exceed the 95th percentile of the shifted-flag placebos",
        diff.mean(),
        level,
        not above,
        note=f"{len(shifted)} placebos",
    )
    verdict.check(
        "K3",
        "flag rate on D outside 0.2-24 a day (declared before H)",
        rate_d,
        24.0,
        not 0.2 <= rate_d <= 24.0,
        outcome="untestable",
    )
    verdict.check(
        "K4", "fewer than 5 flags in H", flags_block, 5.0, flags_block < 5, outcome=INCONCLUSIVE
    )
    pess = pooled(run.paired("S3|cancel=pessimistic", f"{guarded}|cancel=pessimistic", mm))
    verdict.common(
        {
            "S3 - S1": (
                diff.mean(),
                mean_or_nan(pess),
                mean_or_nan(pooled(pair, "d_net") - pooled(pair, "d_funding")),
                mean_or_nan(pooled(pair, "d_making")),
            )
        }
    )
    t = day_t(diff)
    verdict.value, verdict.daily, verdict.p = float(diff.mean()), diff, t.p_one_sided
    return verdict, shifted


def x1_daily(run: Run, label: str, variant: str, symbols: Sequence[str]) -> pd.DataFrame:
    """Per day, pooled over instruments: X1's attempt net (bp, misses zero)
    over the eligible triggers, and the taker twin's mean net on the same
    triggers."""
    x1 = run.usable(label, symbols).copy()
    for column in ("quoter_attempt_net_bp_sum", "quoter_attempts_filled", "quoter_attempt_net_sum"):
        if column not in x1.columns:
            x1[column] = 0.0
        x1[column] = x1[column].fillna(0.0)
    triggers = run.triggers[
        (run.triggers["variant"] == variant) & run.triggers["symbol"].isin(list(symbols))
    ]
    counts = triggers.groupby(["symbol", "day"]).size().rename("triggers").reset_index()
    x1 = x1.merge(counts, on=["symbol", "day"], how="left").fillna({"triggers": 0.0})
    day = x1.groupby("day").agg(
        bp_sum=("quoter_attempt_net_bp_sum", "sum"),
        triggers=("triggers", "sum"),
        filled=("quoter_attempts_filled", "sum"),
        net=("net", "sum"),
        funding=("funding", "sum"),
        making=("making", "sum"),
    )
    twins = run.twins[(run.twins["variant"] == variant) & run.twins["symbol"].isin(list(symbols))]
    day["taker_bp"] = twins.groupby("day")["net_bp"].mean()
    day["per_attempt_bp"] = day["bp_sum"] / day["triggers"].where(day["triggers"] > 0)
    day["minus_taker_bp"] = day["per_attempt_bp"] - day["taker_bp"]
    return day


def per_attempt(day: pd.DataFrame) -> float:
    total = float(day["triggers"].sum())
    return float(day["bp_sum"].sum() / total) if total else math.nan


def h2_2(run: Run, h2: Sequence[str]) -> tuple[Verdict, dict[int, float]]:
    verdict = Verdict(
        "H2.2", "X1 net per attempt minus the taker twin's, paired daily", "bp/attempt"
    )
    day = x1_daily(run, "X1", "real", h2)
    diff = day["minus_taker_bp"].dropna()
    x1_bp = per_attempt(day)
    verdict.check("K1", "mean paired daily X1 - taker <= 0", diff.mean(), 0.0, diff.mean() <= 0)
    verdict.check("K2", "X1's net per attempt <= 0", x1_bp, 0.0, x1_bp <= 0)
    flipped = per_attempt(x1_daily(run, "X1-flipped", "flipped", h2))
    verdict.check(
        "K3",
        "the flipped placebo (following the move) does at least as well per attempt",
        flipped,
        x1_bp,
        flipped >= x1_bp,
    )
    shuffled = {}
    labels = [str(x) for x in run.rows["label"].unique() if str(x).startswith("X1|shuffle=")]
    for label in sorted(labels):
        seed = int(label.split("=")[1])
        placebo = x1_daily(run, label, f"shuffle_{seed:02d}", h2)["minus_taker_bp"].dropna()
        shuffled[seed] = float(placebo.mean()) if len(placebo) else math.nan
    level, above = exceeds_placebos(float(diff.mean()), list(shuffled.values()))
    verdict.check(
        "K4",
        "the true difference does not exceed the 95th percentile of the shuffled-state placebos",
        diff.mean(),
        level,
        not above,
        note=f"{len(shuffled)} placebos",
    )
    filled = float(day["filled"].sum())
    verdict.check(
        "K5", "fewer than 100 fills in H", filled, 100.0, filled < 100, outcome=INCONCLUSIVE
    )
    pess = x1_daily(run, "X1|cancel=pessimistic", "real", h2)
    verdict.common(
        {
            "X1 - taker": (
                diff.mean(),
                mean_or_nan(pess["minus_taker_bp"].dropna()),
                # The attempt net is the cash of its fills: it holds no funding.
                diff.mean(),
                day["making"].sum(),
            ),
            "X1 per attempt": (x1_bp, per_attempt(pess), x1_bp, day["making"].sum()),
        },
        # The taker twin has no making part: K-dir reads X1's own day totals.
        direction={"X1 (USDT)": (float(day["net"].sum()), float(day["making"].sum()))},
    )
    t_diff, t_x1 = day_t(diff), day_t(day["per_attempt_bp"])
    verdict.value, verdict.daily = float(diff.mean()), diff
    verdict.p = max(t_diff.p_one_sided, t_x1.p_one_sided)
    verdict.beside = {
        "X1 net per attempt (bp)": x1_bp,
        "taker twin net per trade (bp)": float(
            run.twins.loc[run.twins["variant"] == "real", "net_bp"].mean()
        ),
        "attempts (eligible triggers)": float(day["triggers"].sum()),
        "attempts filled": filled,
        "X1 net (USDT)": float(day["net"].sum()),
        "X1 making (USDT)": float(day["making"].sum()),
        "X1 - taker t": t_diff.t,
        "X1 per attempt t": t_x1.t,
    }
    return verdict, shuffled


# ---------------------------------------------------------------------------
# Reported beside the primaries
# ---------------------------------------------------------------------------


def guard_hours(run: Run, symbol: str, label: str, window_min: float) -> dict[str, float]:
    """The unguarded strategy's net per hour inside and outside the guard
    windows of the real flags, from its minute equity."""
    equity = run.kept(label, symbol, "equity")
    flags = pd.read_csv(run.out / "flags.csv")
    flags = flags[flags["symbol"] == symbol]
    if equity.empty:
        return {}
    stamps = pd.to_datetime(flags["effective"], utc=True).astype("int64").to_numpy()
    stamps.sort()
    window = int(window_min * 60 * 1_000_000_000)
    inside_hours = outside_hours = inside_net = outside_net = 0.0
    for _, day in equity.groupby("day", sort=True):
        ts = day["ts"].to_numpy(dtype=np.int64)
        change = np.diff(np.r_[0.0, day["equity"].to_numpy(dtype=np.float64)])
        # A minute is inside if its start lies within a window: (e, e + G].
        start = ts - 60_000_000_000
        last = np.searchsorted(stamps, start, side="left") - 1
        inside = (last >= 0) & (start - stamps[np.maximum(last, 0)] <= window)
        inside_net += float(change[inside].sum())
        outside_net += float(change[~inside].sum())
        inside_hours += inside.sum() / 60.0
        outside_hours += (~inside).sum() / 60.0
    return {
        "hours_inside": inside_hours,
        "hours_outside": outside_hours,
        "net_per_hour_inside": inside_net / inside_hours if inside_hours else math.nan,
        "net_per_hour_outside": outside_net / outside_hours if outside_hours else math.nan,
    }


def h2_3(run: Run, threshold: float, symbol: str = "BICOUSDT") -> pd.DataFrame:
    """H2.3 on the block's days: the gate known at each open, and S4 - S1."""
    gate = pd.Series(run.meta["state"]["h2_3_gate"], name="gate")
    diff = run.paired("S4", "S1", [symbol]).set_index("day")["d_net"]
    table = pd.DataFrame({"gate": gate, "s4_minus_s1": diff})
    table["admitted"] = table["gate"] <= threshold
    return table.rename_axis("day").reset_index()


def ladder(run: Run) -> pd.DataFrame:
    """S0 up the advantage ladder: rung 0 is the main S0."""
    rows = []
    for symbol in sorted(run.rows["symbol"].unique()):
        labels = [("S0", 0)] + [(f"ladder|rung={k}", k) for k in range(1, 7)]
        for label, rung in labels:
            ok = run.usable(label, [symbol])
            if ok.empty:
                continue
            turnover = float((ok["maker_turnover"] + ok["taker_turnover"]).sum())
            rows.append(
                {
                    "symbol": symbol,
                    "rung": rung,
                    "upper_bound": rung >= 2,
                    "days": len(ok),
                    "net_usd_day": float(ok["net"].mean()),
                    "net_bp_turnover": float(ok["net"].sum() / turnover * 1e4)
                    if turnover
                    else math.nan,
                    "passive_fills_day": float((ok["fills"] - ok["flattens"]).mean()),
                    "maker_turnover_usd_day": float(ok["maker_turnover"].mean()),
                    "fees_usd_day": float(ok["fees"].mean()),
                }
            )
    return pd.DataFrame(rows)


def fee_breakeven(run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The maker fee at which each strategy's net is zero, and the grid.

    S0 and X1 decide without reading the fee, so their break-even is exact
    from one run; the gated strategies were re-run at every fee of the grid.
    """
    from trading_research.backtest.costs import BYBIT_BASE
    from trading_research.market_making.verdicts import breakeven_on_grid

    grid_rows, rows = [], []
    for symbol in sorted(run.rows["symbol"].unique()):
        group = BYBIT_STUDY_GROUPS[symbol]
        tiers = tiers_for(group)
        for strategy in ("S0", "S1", "S2", "S3", "S4", "X1"):
            base = run.usable(strategy, [symbol])
            if base.empty:
                continue
            net_day = float(base["net"].mean())
            turnover = float(base["maker_turnover"].sum())
            if strategy in ("S0", "X1"):
                method = "analytic"
                value = (
                    breakeven_maker_bp(float(base["net"].sum()), turnover, BYBIT_BASE.maker_bp)
                    if turnover > 0
                    else math.nan
                )
            else:
                method = "re-run on the grid"
                fees, nets = [BYBIT_BASE.maker_bp], [net_day]
                for label in run.rows.loc[run.rows["group"] == "fee", "label"].unique():
                    if str(label).startswith(f"{strategy}|fee="):
                        part = run.usable(str(label), [symbol])
                        if len(part):
                            fees.append(float(str(label).split("=")[1]))
                            nets.append(float(part["net"].mean()))
                for fee, net in zip(fees, nets, strict=True):
                    grid_rows.append(
                        {
                            "strategy": strategy,
                            "symbol": symbol,
                            "maker_fee_bp": fee,
                            "net_usd_day": net,
                        }
                    )
                value = breakeven_on_grid(fees, nets)
            paying = [t.name for t in tiers if t.maker_bp < value]
            rows.append(
                {
                    "strategy": strategy,
                    "symbol": symbol,
                    "fee_group": group,
                    "method": method,
                    "net_usd_day_at_base": net_day,
                    "maker_turnover_usd_day": float(base["maker_turnover"].mean()),
                    "breakeven_maker_bp": value,
                    "best_published_maker_bp": min(t.maker_bp for t in tiers),
                    "mm_programme_maker_bp": BYBIT_MM_PROGRAMME_MAKER_BP,
                    "published_tiers_that_would_pay": " ".join(paying) or "none",
                    "caveat": BYBIT_FEE_CAVEAT,
                }
            )
    grid = pd.DataFrame(grid_rows)
    if len(grid):
        grid = grid.sort_values(["strategy", "symbol", "maker_fee_bp"]).reset_index(drop=True)
    return pd.DataFrame(rows), grid


def robustness(run: Run, mm: Sequence[str]) -> pd.DataFrame:
    """Every strategy on the MM-admitted instrument, one axis at a time."""
    rows = []
    for label in run.rows.loc[run.rows["group"].isin(["main", "pess", "robust"]), "label"].unique():
        name = str(label)
        strategy, _, cell = name.partition("|")
        if strategy not in ("S0", "S1", "S2", "S3", "S4", "X1"):
            continue
        for symbol in run.rows.loc[run.rows["label"] == name, "symbol"].unique():
            if symbol not in mm:
                continue
            ok = run.usable(name, [symbol])
            default = run.usable(strategy, [symbol])
            rows.append(
                {
                    "strategy": strategy,
                    "symbol": symbol,
                    "cell": cell or "default",
                    "net_usd_day": float(ok["net"].mean()),
                    "minus_default_usd_day": float(ok["net"].mean() - default["net"].mean()),
                    "passive_fills_day": float((ok["fills"] - ok["flattens"]).mean()),
                    "days": len(ok),
                    "enters_a_verdict": cell == "cancel=pessimistic",
                }
            )
    return pd.DataFrame(rows).sort_values(["strategy", "cell"]).reset_index(drop=True)


def inventory_series(run: Run, label: str, symbol: str, every_min: int = 5) -> pd.DataFrame:
    """The headline strategy's position every few minutes, with the soft and
    hard limits in force at its latest fill."""
    equity = run.kept(label, symbol, "equity")
    fills = run.kept(label, symbol, "fills")
    if equity.empty:
        return pd.DataFrame()
    equity = equity.iloc[::every_min].copy()
    limits = fills[["ts", "soft_limit", "hard_limit"]].sort_values("ts")
    out = pd.merge_asof(equity.sort_values("ts"), limits, on="ts", direction="backward")
    out["time"] = pd.to_datetime(out["ts"], utc=True).dt.strftime("%Y-%m-%d %H:%M")
    out = out[["time", "position", "soft_limit", "hard_limit", "mid"]]
    out.insert(0, "symbol", symbol)
    out.insert(0, "strategy", label)
    return out.round(6)


def market_table(run: Run) -> pd.DataFrame:
    rows = []
    for symbol, frame in run.market.groupby("symbol"):
        seconds = float(frame["seconds"].sum())
        row = {
            "symbol": symbol,
            "days": len(frame),
            "tick_bp": float(frame["tick_bp_seconds"].sum() / seconds),
            "spread_bp_time_weighted": float(frame["spread_bp_seconds"].sum() / seconds),
            "share_at_two_ticks_or_more": float(frame["two_tick_seconds"].sum() / seconds),
            "prints_per_day": float(frame["prints"].mean()),
            "sequence_gap_days": int((frame["sequence_gaps"] > 0).sum()),
        }
        for h in HORIZONS_S:
            weights = frame[f"benchmark_volume_{h:g}s"].to_numpy(dtype=np.float64)
            values = frame[f"benchmark_bp_{h:g}s"].to_numpy(dtype=np.float64)
            row[f"market_wide_passive_markout_bp_{h:g}s"] = float(
                np.average(values, weights=weights)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def signals_table(run: Run, h2: Sequence[str], frozen: dict[str, Any]) -> pd.DataFrame:
    rows = []
    real = run.triggers[run.triggers["variant"] == "real"]
    twins = run.twins[run.twins["variant"] == "real"]
    for symbol in h2:
        x1 = run.usable("X1", [symbol])
        mine = twins[twins["symbol"] == symbol]
        rows.append(
            {
                "symbol": symbol,
                "theta_bp": frozen["theta"][symbol],
                "beta": frozen["beta"][symbol],
                "triggers": int((real["symbol"] == symbol).sum()),
                "x1_attempts": float(x1.get("quoter_attempts", pd.Series(dtype=float)).sum()),
                "x1_attempts_filled": float(
                    x1.get("quoter_attempts_filled", pd.Series(dtype=float)).sum()
                ),
                "x1_net_bp_per_attempt": float(
                    x1.get("quoter_attempt_net_bp_sum", pd.Series(dtype=float)).sum()
                    / max(1, int((real["symbol"] == symbol).sum()))
                ),
                "taker_twin_trades": len(mine),
                "taker_twin_net_bp_per_trade": float(mine["net_bp"].mean())
                if len(mine)
                else math.nan,
                "index_autocorrelation_block": run.meta["state"]["index_autocorrelation_block"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# All of it
# ---------------------------------------------------------------------------


def main() -> None:
    from trading_research.market_making.prereg import PreRegistration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="H", help="H, or dry for the dry run")
    parser.add_argument("--results", default=None, help="where the tables go")
    args = parser.parse_args()
    run = Run(OUT_ROOT / args.block)
    results = Path(args.results) if args.results else RESULTS
    results.mkdir(parents=True, exist_ok=True)
    suffix = args.block

    def save(frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(results / f"mm_{name}_{suffix}.csv", index=False, float_format="%.6g")

    registration = PreRegistration.load().require_frozen()
    value = registration.value
    table = value(("admission", "table"))
    mm = [r["symbol"] for r in table if r["mm_admitted"]]
    h2 = [r["symbol"] for r in table if r["h2_admitted"]]
    frozen = {
        "theta": value(("strategies", "S4", "theta_bp")),
        "beta": value(("strategies", "S4", "beta")),
    }
    flags_block = sum(int(run.meta["flags"][s]["flags_block"]) for s in mm)
    rate_d = float(value(("hypotheses", "H3", "flag_rate_on_d_per_day")))

    one, (two_one, shuffled_s4), (two_two, shuffled_x1), (three, shifted) = (
        h1(run, mm),
        h2_1(run, h2),
        h2_2(run, h2),
        h3(run, mm, flags_block, rate_d),
    )
    primaries = [one, two_one, two_two, three]
    passes = holm({v.name: v.p for v in primaries})
    verdict_rows, checks = [], []
    for v in primaries:
        state, deciding = status(v.checks, passes[v.name])
        t = day_t(v.daily)
        verdict_rows.append(
            {
                "hypothesis": v.name,
                "metric": v.metric,
                "unit": v.unit,
                "value": v.value,
                "days": t.days,
                "positive_days": t.positive_days,
                "day_t": t.t,
                "p_one_sided": v.p,
                "passes_holm": passes[v.name],
                "kill_conditions_fired": " ".join(c.name for c in v.checks if c.fired) or "none",
                "status": state,
                "decided_by": " ".join(deciding),
                "beside": json.dumps({k: fmt(x) for k, x in v.beside.items()}),
                "stamp": str(run.rows["stamp"].iloc[0]),
            }
        )
        checks += [asdict(c) for c in v.checks]
    save(pd.DataFrame(verdict_rows), "hypotheses")
    save(pd.DataFrame(checks), "kill_conditions")

    placebo_rows = []
    for name, values, true in (
        ("H2.1 shuffled state", shuffled_s4, two_one.value),
        ("H2.2 shuffled state", shuffled_x1, two_two.value),
        ("H3 shifted flags", shifted, three.value),
    ):
        for seed, x in sorted(values.items()):
            placebo_rows.append({"placebo": name, "seed": seed, "value": x, "true_value": true})
    save(pd.DataFrame(placebo_rows), "placebos")

    daily_groups = run.rows[
        run.rows["group"].isin(["main", "pess", "placebo_h1"])
        | run.rows["label"].isin(["S4-flipped", "X1-flipped"])
    ]
    keep = [
        "label",
        "group",
        "symbol",
        "day",
        "status",
        "excluded",
        "flags",
        *PNL,
        "fills",
        "flattens",
        "fills_inside",
        "fills_touch",
        "fills_behind",
        "maker_turnover",
        "taker_turnover",
        "max_abs_position",
    ]
    keep += [c for c in daily_groups.columns if c.startswith("quoter_")]
    save(daily_groups[[c for c in keep if c in daily_groups.columns]], "daily")
    save(strategies(run), "strategies")
    save(markouts(run), "markouts")
    save(ladder(run), "ladder")
    breakeven, grid = fee_breakeven(run)
    save(breakeven, "fee_breakeven")
    save(grid, "fee_grid")
    save(robustness(run, mm), "robustness")
    save(market_table(run), "market")
    save(signals_table(run, h2, frozen), "signals")
    gate = float(value(("hypotheses", "H2.3", "gate_threshold")))
    save(h2_3(run, gate), "h2_3")
    window = float(value(("strategies", "S3", "chosen", "guard_window_min")))
    guard = [{"symbol": s, "strategy": "S1", **guard_hours(run, s, "S1", window)} for s in mm]
    save(pd.DataFrame(guard), "guard_hours")
    save(pd.concat([inventory_series(run, "S1", s) for s in mm]), "inventory")
    compute = [{"stage": k, "seconds": round(float(x), 1)} for k, x in run.meta["stages_s"].items()]
    for key in (
        "peak_rss_mb_worker",
        "peak_rss_mb_any_child",
        "peak_rss_mb_driver",
        "workers",
        "jobs",
    ):
        compute.append({"stage": key, "seconds": round(float(run.meta[key]), 0)})
    save(pd.DataFrame(compute), "compute")
    print(
        pd.DataFrame(verdict_rows)[
            ["hypothesis", "value", "day_t", "p_one_sided", "kill_conditions_fired", "status"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
