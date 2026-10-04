"""The second market-making round's verdicts, from a held-out read's outputs.

Reads what ``experiments/market_making_round2_heldout.py`` wrote under
``artifacts/mm/round2/heldout/<read>/`` and the frozen registration, and no
market data. Computes what ``docs/preregistration/market_making_round2.md``
registers, with the frozen values:

* B1, B2, B3a, B3b on the admitted basket and B4 on BICOUSDT: every kill
  condition evaluated mechanically (K1 to K5 as registered, K-pess, K-queue,
  K-fund, K-dir, K-nbhd), the day-level one-sided t of the pooled net per 100
  USDT of clip a day, Holm's procedure at 5% across the five primaries, and
  the status;
* on a later read (P for the basket, Q), the registered transitions from the
  statuses committed after the read before it;
* beside them: every instrument's own net with its day t, the pooled net with
  each instrument left out and without GALAUSDT and ALGOUSDT, the pooled USDT
  sum, the fee break-even, the robustness axes, the room and the net by hour,
  every pooled number by admission criterion, and the gate's open share.

How the registered statistics are read, fixed before any held-out block is
read:

* a claim's value is the mean over the block's days of its pooled daily series
  (per 100 USDT of clip); a paired claim pools the per instrument-day
  differences where both days are usable;
* K-pess and K-queue compare those means under the default rules and under the
  bracket, for every sign claim; K-fund reads the pooled net less funding,
  K-dir the pooled making part, each claim by claim;
* K-nbhd: for S1, each instrument's median over its neighbourhood cells of its
  own mean per 100 USDT of clip, pooled by the mean over instruments; for the
  gate, the median over its neighbourhood cells of the pooled claim (net, and
  the paired difference with the base); it fires if any claim's median is at
  or below zero;
* K4 pools by the mean over instruments of each instrument's volume-weighted
  markout of the prints that arrived while the gate was open, and while it was
  closed, over the instruments with both; the maker fee cancels;
* the minimum-count conditions read passive fills (flattens excluded) summed
  over the block and instruments, and the gate's open share as the mean over
  instruments of its share of quoting time;
* Holm's procedure counts all five primaries; a primary not tested has no
  p-value and fails.

Usage::

    .venv/bin/python -m experiments.market_making_round2_verdicts --read first
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments import _round2 as r2
from experiments._common import RESULTS
from experiments.market_making_round2_heldout import ROBUST
from trading_research.backtest.costs import BYBIT_FEE_CAVEAT, breakeven_maker_bp
from trading_research.market_making import heldout, round2
from trading_research.market_making import round2_stats as stats
from trading_research.market_making.verdicts import (
    INCONCLUSIVE,
    KillCheck,
    breakeven_on_grid,
    day_t,
    exceeds_placebos,
    holm,
    k_dir,
    k_pess,
    status,
)

OUT_ROOT = r2.OUT_ROOT / "heldout"
BICO = round2.B4_SYMBOL
PNL = ("net", "making", "spread", "adverse", "inventory", "fees", "funding")
#: Seconds a day the quoters may quote: 00:10 to 23:58.
QUOTING_S = 86_280.0 - 600.0
#: The two index members whose mids round one read (item 3b).
INDEX_MEMBERS = ("GALAUSDT", "ALGOUSDT")
MIN_FILLS = 100
MIN_OPEN_SHARE = 0.01


def fmt(value: Any) -> float:
    number = float(value) if value is not None else math.nan
    return number if math.isfinite(number) else math.nan


class Run:
    """The outputs of one held-out read, and the frozen values it used."""

    def __init__(self, out: Path, registration: round2.Registration) -> None:
        self.out = out
        self.meta: dict[str, Any] = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        rows = pd.read_parquet(out / "rows.parquet")
        rows["excluded"] = rows["excluded"].astype(bool)
        rows["net_ex_funding"] = rows["net"] - rows["funding"]
        self.market = rows[rows["group"] == "market"].copy()
        self.rows = rows[rows["group"] != "market"].copy()
        self.persistence = pd.read_parquet(out / "persistence.parquet")
        self.hours = pd.read_parquet(out / "hours.parquet")
        self.registration = registration
        symbols = sorted(self.meta["days"])
        self.clips = {s: registration.clip(s) for s in symbols}
        self.basket = [s for s in registration.admitted if s in symbols]

    def rows_of(self, label: str, symbols: Sequence[str]) -> pd.DataFrame:
        frame = self.rows[self.rows["label"] == label]
        return frame[frame["symbol"].isin(list(symbols))]

    def daily(self, label: str, symbols: Sequence[str], column: str = "net") -> pd.Series:
        return stats.pooled_daily(self.rows_of(label, symbols), self.clips, column)

    def paired(
        self, label: str, base: str, symbols: Sequence[str], column: str = "net"
    ) -> pd.Series:
        a, b = self.rows_of(label, symbols), self.rows_of(base, symbols)
        return stats.paired_daily(a, b, self.clips, column)

    def mean(self, label: str, symbols: Sequence[str], column: str = "net") -> float:
        daily = self.daily(label, symbols, column)
        return float(daily.mean()) if len(daily) else math.nan

    def paired_mean(
        self, label: str, base: str, symbols: Sequence[str], column: str = "net"
    ) -> float:
        daily = self.paired(label, base, symbols, column)
        return float(daily.mean()) if len(daily) else math.nan

    def passive_fills(self, label: str, symbols: Sequence[str]) -> float:
        return float(r2.passive_fills(self.rows_of(label, symbols)).sum())

    def own_mean(self, label: str, symbol: str) -> float:
        """One instrument's mean per 100 USDT of clip over the block."""
        frame = stats.per_clip(self.rows_of(label, [symbol]), self.clips)
        return float(frame["y"].mean()) if len(frame) else math.nan

    def kept(self, label: str, symbol: str, what: str) -> pd.DataFrame:
        folder = self.meta["folders"].get(symbol)
        return pd.DataFrame() if folder is None else heldout.kept(Path(folder), label, symbol, what)


# ---------------------------------------------------------------------------
# A verdict and the common conditions
# ---------------------------------------------------------------------------


class Verdict:
    """One hypothesis on one block: its kill checks, its deciding p-value."""

    def __init__(self, name: str, metric: str, block: str) -> None:
        self.name, self.metric, self.block = name, metric, block
        self.checks: list[KillCheck] = []
        self.p = math.nan
        self.value = math.nan
        self.daily: pd.Series = pd.Series(dtype=float)
        self.beside: dict[str, Any] = {}
        self.tested = True
        self.claims_positive = False

    def check(self, name: str, test: str, value: float, reference: float, fired: bool,
              outcome: str = "killed", note: str = "") -> None:  # fmt: skip
        self.checks.append(
            KillCheck(self.name, name, test, fmt(value), fmt(reference), bool(fired), outcome, note)
        )

    def common(
        self,
        run: Run,
        symbols: Sequence[str],
        claims: dict[str, tuple[str, str | None]],
        nbhd: dict[str, tuple[float, bool]],
    ) -> None:
        """K-pess, K-queue, K-fund, K-dir and K-nbhd over the sign claims, each
        claim a label and, for a paired claim, the base it is paired with."""

        def value(label: str, base: str | None, column: str = "net") -> float:
            if base is None:
                return run.mean(label, symbols, column)
            return run.paired_mean(label, base, symbols, column)

        names = list(claims)
        default = [value(*claims[c]) for c in names]
        brackets = {}
        for tag in ("pess", "queue"):
            brackets[tag] = [
                value(f"{label}|{tag}", None if base is None else f"{base}|{tag}")
                for label, base in (claims[c] for c in names)
            ]
        self.claims_positive = all(v > 0 for v in default)
        notes = {
            tag: "; ".join(
                f"{c}: {d:+.4g} -> {b:+.4g}" for c, d, b in zip(names, default, values, strict=True)
            )
            for tag, values in brackets.items()
        }
        self.check("K-pess", "positive under proportional attribution, not under pessimistic",
                   min(brackets["pess"]), 0.0,
                   k_pess([d > 0 for d in default], [b > 0 for b in brackets["pess"]]),
                   note=notes["pess"])  # fmt: skip
        self.check("K-queue", "positive by default, not under the joint pessimistic queue",
                   min(brackets["queue"]), 0.0, stats.k_queue(default, brackets["queue"]),
                   note=notes["queue"])  # fmt: skip
        ex = [value(*claims[c], column="net_ex_funding") for c in names]
        self.check("K-fund", "net minus funding <= 0", min(ex), 0.0, min(ex) <= 0,
                   note="; ".join(f"{c}: {v:+.4g}" for c, v in zip(names, ex, strict=True)))  # fmt: skip
        making = [value(*claims[c], column="making") for c in names]
        fired = [c for c, n, m in zip(names, default, making, strict=True) if k_dir(m, n)]
        self.check("K-dir", "making <= 0 while net > 0", min(making), 0.0, bool(fired),
                   note="; ".join(f"{c}: making {m:+.4g}, net {n:+.4g}"
                                  for c, n, m in zip(names, default, making, strict=True)))  # fmt: skip
        medians = [m for m, _ in nbhd.values()]
        self.check("K-nbhd", "median over the D neighbourhood <= 0",
                   min(medians) if medians else math.nan, 0.0, any(f for _, f in nbhd.values()),
                   note="; ".join(f"{c}: {m:+.4g}" for c, (m, _) in nbhd.items()))  # fmt: skip

    def minimum(
        self, run: Run, label: str, symbols: Sequence[str], share: float | None = None
    ) -> None:
        """K5: too few passive fills (or too little time open) to say anything."""
        fills = run.passive_fills(label, symbols)
        note = f"{fills:.0f} passive fills"
        fired = fills < MIN_FILLS
        if share is not None:
            note += f"; open {share:.4f} of quoting time"
            fired = fired or not share >= MIN_OPEN_SHARE
        self.check("K5", f"fewer than {MIN_FILLS} passive fills, or open under 1%", fills,
                   MIN_FILLS, fired, INCONCLUSIVE, note)  # fmt: skip


def not_tested(name: str, block: str) -> Verdict:
    verdict = Verdict(name, "not tested", block)
    verdict.tested = False
    verdict.beside["status"] = round2.NOT_TESTED
    return verdict


# ---------------------------------------------------------------------------
# The primaries
# ---------------------------------------------------------------------------


def labels_like(run: Run, prefix: str, symbol: str | None = None) -> list[str]:
    frame = run.rows if symbol is None else run.rows[run.rows["symbol"] == symbol]
    return sorted(str(x) for x in frame["label"].unique() if str(x).startswith(prefix))


def s1_nbhd(run: Run, label: str, symbols: Sequence[str]) -> tuple[float, bool]:
    """K-nbhd for an S1 claim: each instrument's own neighbourhood."""
    values = {
        s: [run.own_mean(c, s) for c in [label, *labels_like(run, f"{label}|nbhd|", s)]]
        for s in symbols
    }
    return stats.k_nbhd_per_instrument(values)


def beside_basket(run: Run, label: str, symbols: Sequence[str]) -> dict[str, Any]:
    """Each instrument's net with its day t, leave one out, without the two
    index members, and the pooled USDT sum."""
    out: dict[str, Any] = {}
    for s in symbols:
        frame = stats.per_clip(run.rows_of(label, [s]), run.clips)
        t = day_t(frame.groupby("day")["y"].mean())
        out[f"{s} per 100 clip"] = t.mean
        out[f"{s} day t"] = t.t
        if len(symbols) > 1:
            out[f"without {s}"] = run.mean(label, [x for x in symbols if x != s])
    rest = [s for s in symbols if s not in INDEX_MEMBERS]
    out["without GALAUSDT and ALGOUSDT"] = run.mean(label, rest) if rest else math.nan
    usdt = stats.usdt_daily(run.rows_of(label, symbols))
    out["pooled USDT a day"] = float(usdt.mean()) if len(usdt) else math.nan
    return out


def b1(run: Run, block: str) -> Verdict:
    symbols = run.basket
    verdict = Verdict("B1", "S1, net per 100 USDT of clip a day, pooled", block)
    daily = run.daily("S1", symbols)
    mean = float(daily.mean())
    verdict.check("K1", "pooled mean daily net <= 0", mean, 0.0, mean <= 0)
    verdict.common(run, symbols, {"net": ("S1", None)}, {"net": s1_nbhd(run, "S1", symbols)})
    verdict.minimum(run, "S1", symbols)
    t = day_t(daily)
    verdict.p, verdict.value, verdict.daily = t.p_one_sided, mean, daily
    verdict.beside = {"day t": t.t, **beside_basket(run, "S1", symbols)}
    for s in symbols:
        verdict.beside[f"S0 {s} per 100 clip"] = run.own_mean("S0", s)
    return verdict


def k4(run: Run, symbols: Sequence[str]) -> tuple[float, float, dict[str, Any]]:
    """The room's persistence, pooled over instruments: the markout of prints
    that arrived while the gate was open, and while it was closed."""
    opened, closed, per = [], [], {}
    for s in symbols:
        frame = run.persistence[run.persistence["symbol"] == s]
        ov, cv = frame["open_volume"].sum(), frame["closed_volume"].sum()
        if ov > 0 and cv > 0:
            o = float(frame["open_weighted"].sum() / ov)
            c = float(frame["closed_weighted"].sum() / cv)
            opened.append(o)
            closed.append(c)
            per[f"{s} room open"] = o - 2.0
            per[f"{s} room closed"] = c - 2.0
    if not opened:
        return math.nan, math.nan, per
    return float(np.mean(opened)), float(np.mean(closed)), per


def gated(run: Run, name: str, symbols: Sequence[str], block: str) -> Verdict:
    """B2 on the basket, or B4 on BICOUSDT: G(base) positive, and more than
    its base on the same days."""
    cell = run.registration.gate_cell(BICO if name == "B4" else None)
    base = "S1" if cell.base == "S1" else "S1-touch"
    verdict = Verdict(
        name, f"G(base) {cell.label}, and G(base) - {base}, per 100 USDT of clip", block
    )
    daily = run.daily("G", symbols)
    paired = run.paired("G", base, symbols)
    mean, diff = float(daily.mean()), float(paired.mean())
    verdict.check("K1", "pooled mean daily net of G(base) <= 0", mean, 0.0, mean <= 0)
    verdict.check("K2", "mean paired daily G(base) - base <= 0", diff, 0.0, diff <= 0)
    placebos = [run.paired_mean(label, base, symbols) for label in labels_like(run, "G|shift=")]
    level, above = exceeds_placebos(diff, placebos)
    verdict.check("K3", "paired difference not above the placebos' 95th percentile", diff, level,
                  not above, note=f"{len(placebos)} shifted-gate placebos")  # fmt: skip
    opened, closed, per = k4(run, symbols)
    verdict.check("K4", "room while open not greater than while closed", opened, closed,
                  not opened > closed, note="markout before the fee; the fee cancels")  # fmt: skip
    cells = ["G", *labels_like(run, "G|nbhd|")]
    nbhd = {
        "net": stats.k_nbhd([run.mean(c, symbols) for c in cells]),
        "minus_base": stats.k_nbhd([run.paired_mean(c, base, symbols) for c in cells]),
    }
    verdict.common(run, symbols, {"net": ("G", None), "minus_base": ("G", base)}, nbhd)
    shares = [float(run.meta["open_share"][s]["real"]) for s in symbols]
    verdict.minimum(run, "G", symbols, share=float(np.mean(shares)) if shares else math.nan)
    t_net, t_diff = day_t(daily), day_t(paired)
    verdict.p = max(t_net.p_one_sided, t_diff.p_one_sided)
    verdict.value, verdict.daily = diff, paired
    verdict.beside = {"net per 100 clip": mean, "net day t": t_net.t, "minus base day t": t_diff.t,
                      **per, **beside_basket(run, "G", symbols)}  # fmt: skip
    for s, share in zip(symbols, shares, strict=True):
        verdict.beside[f"{s} open share"] = share
    if base == "S1-touch" and name == "B2":
        verdict.beside["G(base) - S1"] = run.paired_mean("G", "S1", symbols)
    return verdict


def b3(run: Run, name: str, tier: str, block: str) -> Verdict:
    """B3a or B3b: the tier's strategy chosen on D, positive at the tier."""
    symbols = run.basket
    pick = run.registration.value(("d_search", "b3_choice", "chosen"))[tier]["strategy"]
    label = f"B3|{tier}"
    verdict = Verdict(name, f"{pick} at {tier}, net per 100 USDT of clip a day, pooled", block)
    daily = run.daily(label, symbols)
    mean = float(daily.mean())
    verdict.check("K1", f"pooled mean daily net at {tier} <= 0", mean, 0.0, mean <= 0)
    if pick == "S1":
        nbhd = s1_nbhd(run, label, symbols)
    else:
        cells = [label, *labels_like(run, f"{label}|nbhd|")]
        nbhd = stats.k_nbhd([run.mean(c, symbols) for c in cells])
    verdict.common(run, symbols, {"net": (label, None)}, {"net": nbhd})
    verdict.minimum(run, label, symbols)
    t = day_t(daily)
    verdict.p, verdict.value, verdict.daily = t.p_one_sided, mean, daily
    verdict.beside = {
        "day t": t.t,
        "the same at the base fee": run.mean("S1" if pick == "S1" else "G", symbols),
        "the same with the taker at 5.5 bp": run.mean(f"{label}|taker=5.5", symbols),
        "caveat": BYBIT_FEE_CAVEAT,
        **beside_basket(run, label, symbols),
    }
    return verdict


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------

PRIMARIES = ("B1", "B2", "B3a", "B3b", "B4")


def first_statuses(verdicts: Sequence[Verdict]) -> dict[str, tuple[str, str]]:
    """The first read's statuses: killed, candidate or inconclusive, Holm at
    5% across all five primaries (a primary not tested fails it)."""
    passed = holm({v.name: v.p if v.tested else math.nan for v in verdicts})
    out = {}
    for v in verdicts:
        if not v.tested:
            out[v.name] = (round2.NOT_TESTED, "")
            continue
        found, deciding = status(v.checks, passed[v.name])
        out[v.name] = (found, ", ".join(deciding))
    return out


def taken_forward(previous: pd.DataFrame, names: Sequence[str]) -> list[str]:
    """The hypotheses a later read takes: candidate, or inconclusive with
    every claim positive, on the read before."""
    taken = []
    for _, row in previous[previous["hypothesis"].isin(list(names))].iterrows():
        if row["status"] == "candidate" or (
            row["status"] == INCONCLUSIVE and bool(row["claims_positive"])
        ):
            taken.append(str(row["hypothesis"]))
    return taken


def later_statuses(
    verdicts: Sequence[Verdict], previous: pd.DataFrame, read: str
) -> dict[str, tuple[str, str]]:
    """The registered transitions on P for the basket, or on Q."""
    before = dict(zip(previous["hypothesis"], previous["status"], strict=True))
    passed = holm({v.name: v.p for v in verdicts})
    out = {}
    for v in verdicts:
        killed = [c.name for c in v.checks if c.fired and c.outcome == "killed"]
        if killed:
            out[v.name] = (f"killed on {'P' if read == 'basket_P' else 'Q'}", ", ".join(killed))
            continue
        was = str(before.get(v.name, ""))
        short = [c.name for c in v.checks if c.fired and c.outcome == INCONCLUSIVE]
        passed[v.name] = passed[v.name] and not short
        if read == "Q":
            out[v.name] = ("confirmed", "") if passed[v.name] else ("candidate", "Holm")
        elif was == "candidate":
            out[v.name] = (
                ("confirmed", "") if passed[v.name] else ("candidate, not confirmed", "Holm")
            )
        else:
            out[v.name] = ("candidate", "first significant on P") if passed[v.name] else (
                INCONCLUSIVE, "Holm")  # fmt: skip
    if read == "basket_P" and out.get("B1", ("",))[0] != "confirmed":
        for name in ("B3a", "B3b"):
            if out.get(name, ("",))[0] == "confirmed":
                out[name] = ("conditional on the fee tier", "B1 not confirmed")
    return out


def verdict_table(
    verdicts: Sequence[Verdict], statuses: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    for v in verdicts:
        found, deciding = statuses[v.name]
        rows.append(
            {
                "hypothesis": v.name,
                "block": v.block,
                "metric": v.metric,
                "status": found,
                "decided_by": deciding,
                "value": fmt(v.value),
                "p_one_sided": fmt(v.p),
                "days": int(v.daily.size),
                "claims_positive": v.claims_positive,
                "fired": ", ".join(c.name for c in v.checks if c.fired),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The measurements without a hypothesis
# ---------------------------------------------------------------------------


def fee_breakeven(run: Run) -> pd.DataFrame:
    """The maker fee at which each instrument's S0 (exactly), S1 and G(base)
    (over the grid, the gate reading the fee) would have broken even."""
    grid = [float(f) for f in run.meta["fee_grid"]]
    rows = []
    for s in sorted(run.clips):
        s0 = stats.usable(run.rows_of("S0", [s]))
        row: dict[str, Any] = {"symbol": s}
        turnover = float(s0["maker_turnover"].sum())
        net = float(s0["net"].sum())
        row["S0"] = breakeven_maker_bp(net, turnover, 2.0) if turnover > 0 else math.nan
        for strategy in ("S1", "G"):
            nets = []
            for fee in grid:
                label = strategy if fee == 2.0 else f"{strategy}|fee={fee:+.2f}"
                nets.append(float(stats.usable(run.rows_of(label, [s]))["net"].mean()))
                row[f"{strategy} net at {fee:+.2f}"] = nets[-1]
            row[strategy] = breakeven_on_grid(grid, nets)
        rows.append(row)
    return pd.DataFrame(rows)


def robustness(run: Run) -> pd.DataFrame:
    """S1 and G(base) on each axis alone, per instrument and pooled."""
    cells = ["default", "pess", "queue", *(tag for tag, _ in ROBUST)]
    rows = []
    for strategy in ("S1", "G"):
        for cell in cells:
            label = strategy if cell == "default" else f"{strategy}|{cell}"
            groups = [[s] for s in sorted(run.clips)] + ([run.basket] if run.basket else [])
            for members in groups:
                if not len(run.rows_of(label, members)):
                    continue
                name = members[0] if len(members) == 1 else "pooled basket"
                rows.append({"strategy": strategy, "cell": cell, "instruments": name,
                             "per_100_clip_day": run.mean(label, members)})  # fmt: skip
    return pd.DataFrame(rows)


def by_hour(run: Run) -> pd.DataFrame:
    """Per instrument and UTC hour: the market's room at the base fee and S1's
    mean net, from its kept minute equity."""
    rows = []
    for s in sorted(run.clips):
        hours = run.hours[run.hours["symbol"] == s].set_index("hour")
        equity = run.kept("S1", s, "equity")
        net = pd.Series(dtype=float)
        if not equity.empty:
            equity = equity.sort_values(["day", "ts"])
            equity["change"] = equity.groupby("day")["equity"].diff().fillna(equity["equity"])
            equity["hour"] = (equity["ts"] // 1_000_000_000 % 86_400) // 3600
            days = equity["day"].nunique()
            net = equity.groupby("hour")["change"].sum() / max(days, 1)
        for hour in range(24):
            rows.append({
                "symbol": s, "hour": hour,
                "room_base_bp": fmt(hours["markout_bp"].get(hour, math.nan)) - 2.0,
                "volume": fmt(hours["volume"].get(hour, math.nan)),
                "s1_net_usd": fmt(net.get(hour, math.nan)),
            })  # fmt: skip
    return pd.DataFrame(rows)


def by_criterion(run: Run) -> pd.DataFrame:
    """Every pooled number split by the admission criterion (A1, A2)."""
    table = {r["symbol"]: r["criterion"] for r in run.registration.value(("admission", "table"))}
    rows = []
    for criterion in ("A1", "A2"):
        members = [s for s in run.basket if criterion in str(table.get(s, ""))]
        for label in ("S0", "S1", "G", *labels_like(run, "B3|")):
            if members and "|" not in label.removeprefix("B3|"):
                rows.append({"criterion": criterion, "instruments": " ".join(members),
                             "strategy": label, "per_100_clip_day": run.mean(label, members)})  # fmt: skip
    return pd.DataFrame(rows)


MAIN = ("S0", "S1", "S1-touch", "G")


def strategies(run: Run) -> pd.DataFrame:
    """Per strategy and instrument: the decomposition, the metric, fills, time
    quoted, inventory, flattens and drawdown."""
    rows = []
    kept = [*MAIN, *(x for x in labels_like(run, "B3|") if x.count("|") == 1)]
    for label in kept:
        for s in sorted(run.clips):
            frame = run.rows_of(label, [s])
            ok = stats.usable(frame)
            if frame.empty:
                continue
            row: dict[str, Any] = {"strategy": label, "symbol": s, "days": len(ok)}
            row["excluded_days"] = len(frame) - len(ok)
            row["per_100_clip_day"] = run.own_mean(label, s)
            for name in PNL:
                row[f"{name}_usd_day"] = float(ok[name].mean())
            turnover = float((ok["maker_turnover"] + ok["taker_turnover"]).sum())
            row["net_bp_turnover"] = (
                float(ok["net"].sum() / turnover * 1e4) if turnover else math.nan
            )
            row["passive_fills_day"] = float((ok["fills"] - ok["flattens"]).mean())
            row["flattens_day"] = float(ok["flattens"].mean())
            row["bid_quoted_share"] = float(ok["bid_quoted_s"].mean() / QUOTING_S)
            row["ask_quoted_share"] = float(ok["ask_quoted_s"].mean() / QUOTING_S)
            row["max_abs_position"] = float(ok["max_abs_position"].max())
            row["clip_over_touch"] = float(ok["clip_over_touch"].mean())
            row.update(inventory(run, label, s))
            rows.append(row)
    return pd.DataFrame(rows)


def inventory(run: Run, label: str, symbol: str) -> dict[str, float]:
    """RMS and maximum |position| at the minute in the quoting hours, and the
    maximum drawdown of the block's cumulative minute equity."""
    equity = run.kept(label, symbol, "equity")
    if equity.empty:
        return {}
    seconds = (equity["ts"] // 1_000_000_000) % 86_400
    quoting = equity[(seconds >= 600) & (seconds <= 86_280)]
    position = quoting["position"].to_numpy(dtype=np.float64)
    curve, carried = [], 0.0
    for _, day in equity.groupby("day", sort=True):
        values = day["equity"].to_numpy(dtype=np.float64)
        curve.append(carried + values)
        carried += values[-1]
    total = np.r_[0.0, np.concatenate(curve)]
    return {
        "rms_position": float(np.sqrt(np.nanmean(position**2))),
        "max_abs_position_minute": float(np.nanmax(np.abs(position))),
        "max_drawdown_usdt": float(np.max(np.maximum.accumulate(total) - total)),
    }


def markouts(run: Run) -> pd.DataFrame:
    """Volume-weighted markouts by strategy, instrument and fill path, beside
    the market-wide passive benchmark."""
    rows = []
    for label in MAIN:
        for s in sorted(run.clips):
            fills = run.kept(label, s, "fills")
            if fills.empty:
                continue
            passive = fills[fills["maker"]].assign(path="all passive")
            for path, group in pd.concat([fills, passive]).groupby("path"):
                weights = group["size"].to_numpy(dtype=np.float64)
                row: dict[str, Any] = {
                    "strategy": label,
                    "symbol": s,
                    "path": path,
                    "fills": len(group),
                }
                for h in r2.MARKOUT_HORIZONS_S:
                    values = group[f"markout_{h:g}s"].to_numpy(dtype=np.float64)
                    ok = np.isfinite(values)
                    row[f"markout_bp_{h:g}s"] = (
                        float(np.average(values[ok], weights=weights[ok])) if ok.any() else math.nan
                    )
                rows.append(row)
    for s, frame in run.market.groupby("symbol"):
        row = {"strategy": "market-wide passive", "symbol": s, "path": "benchmark"}
        row["fills"] = float(frame["prints"].sum())
        for h in r2.MARKOUT_HORIZONS_S:
            weights = frame[f"benchmark_volume_{h:g}s"].to_numpy(dtype=np.float64)
            values = frame[f"benchmark_bp_{h:g}s"].to_numpy(dtype=np.float64)
            row[f"markout_bp_{h:g}s"] = float(np.average(values, weights=weights))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def hypotheses(run: Run, names: Sequence[str], basket_block: str, bico_block: str) -> list[Verdict]:
    out = []
    for name in names:
        if name == "B4":
            out.append(gated(run, "B4", [BICO], bico_block))
        elif not run.basket:
            out.append(not_tested(name, basket_block))
        elif name == "B1":
            out.append(b1(run, basket_block))
        elif name == "B2":
            out.append(gated(run, "B2", run.basket, basket_block))
        else:
            out.append(
                b3(run, name, "pro1_altcoin" if name == "B3a" else "programme", basket_block)
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read", default="first", choices=["dry", *sorted(round2.READS)])
    args = parser.parse_args()
    registration = round2.Registration.load().require_frozen()
    run = Run(OUT_ROOT / args.read, registration)
    if args.read in ("first", "dry"):
        basket_block, bico_block = ("D", "D") if args.read == "dry" else ("H", "P")
        verdicts = hypotheses(run, PRIMARIES, basket_block, bico_block)
        statuses = first_statuses(verdicts)
        suffix = "dry" if args.read == "dry" else "H"
    elif args.read == "basket_P":
        previous = pd.read_csv(RESULTS / "mm_round2_verdicts_H.csv")
        taken = taken_forward(previous, PRIMARIES[:4])
        verdicts = hypotheses(run, taken, "P", "P")
        statuses = later_statuses(verdicts, previous, "basket_P")
        suffix = "P"
    else:
        frames = [pd.read_csv(RESULTS / f"mm_round2_verdicts_{s}.csv") for s in ("H", "P")
                  if (RESULTS / f"mm_round2_verdicts_{s}.csv").exists()]  # fmt: skip
        previous = pd.concat(frames, ignore_index=True).drop_duplicates("hypothesis", keep="last")
        candidates = previous[previous["status"] == "candidate"]["hypothesis"].tolist()
        verdicts = hypotheses(run, [h for h in candidates if h in PRIMARIES], "Q", "Q")
        statuses = later_statuses(verdicts, previous, "Q")
        suffix = "Q"
    target = OUT_ROOT / "dry" / "verdicts" if args.read == "dry" else RESULTS
    target.mkdir(parents=True, exist_ok=True)
    table = verdict_table(verdicts, statuses)
    table["stamp"] = run.meta["stamp"]
    table.to_csv(target / f"mm_round2_verdicts_{suffix}.csv", index=False)
    checks = pd.DataFrame([c.__dict__ for v in verdicts for c in v.checks])
    checks.to_csv(target / f"mm_round2_checks_{suffix}.csv", index=False)
    beside = {v.name: {k: (fmt(x) if isinstance(x, int | float) else x) for k, x in v.beside.items()}
              for v in verdicts}  # fmt: skip
    (target / f"mm_round2_beside_{suffix}.json").write_text(
        json.dumps(beside, indent=2, default=str) + "\n", encoding="utf-8"
    )
    daily = pd.concat(
        [
            v.daily.rename("value").to_frame().assign(hypothesis=v.name)
            for v in verdicts
            if len(v.daily)
        ]
        or [pd.DataFrame()]
    )
    daily.to_csv(target / f"mm_round2_daily_{suffix}.csv")
    for name, make in (
        ("strategies", strategies),
        ("markouts", markouts),
        ("fee_breakeven", fee_breakeven),
        ("robustness", robustness),
        ("by_hour", by_hour),
        ("by_criterion", by_criterion),
    ):
        make(run).to_csv(target / f"mm_round2_{name}_{suffix}.csv", index=False)
    print(
        table[["hypothesis", "block", "status", "decided_by", "value", "p_one_sided"]].to_string()
    )


if __name__ == "__main__":
    main()
