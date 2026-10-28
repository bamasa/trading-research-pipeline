"""``verify pairs``: the pairs screen and backtest on a planted pair and its two nulls.

Each path generates three pairs on the same hourly clock
(:mod:`trading_research.synthetic.pairs`): a planted cointegration, two
independent random walks, and a pair whose cointegration breaks halfway. The
screen (:func:`trading_research.pairs.screen.screen`) is run on the first
formation window of the planted and the independent pair together, four
symbols and six pairs, Holm-corrected; the walk-forward backtest
(:func:`trading_research.pairs.backtest.backtest`, no gate) on each pair.
Trades are pooled over paths and clustered by path-day.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from trading_research.synthetic.verify import T_BAR, CheckResult, _bp, _t

A, B, C, D = "SYNTHAUSDT", "SYNTHBUSDT", "SYNTHCUSDT", "SYNTHDUSDT"

#: Relative tolerance on the recovered half-life, and absolute on the hedge ratio.
HALF_LIFE_TOLERANCE = 0.25
BETA_TOLERANCE = 0.05


def _pooled(parts: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [p for p in parts if not p.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stats(trades: pd.DataFrame) -> dict[str, float]:
    from trading_research.evaluation.significance import SignificanceError, assess

    out = {"trades": float(len(trades))}
    if trades.empty:
        return {
            **out,
            "gross_bp": math.nan,
            "net_bp": math.nan,
            "gross_t": math.nan,
            "net_t": math.nan,
            "days": 0.0,
        }
    out["gross_bp"] = float(trades["gross_bp"].mean())
    out["net_bp"] = float(trades["net_bp"].mean())
    out["days"] = float(trades["day"].nunique())
    for value in ("gross", "net"):
        try:
            out[f"{value}_t"] = assess(
                trades, value=f"{value}_bp", cluster="day", series=None
            ).cluster_t
        except SignificanceError:
            out[f"{value}_t"] = math.nan
    return out


def _line(s: dict[str, float], value: str) -> str:
    return (
        f"{_bp(s[f'{value}_bp'])} a trade, day t {_t(s[f'{value}_t'])}, {int(s['trades'])} trades"
    )


def verify_pairs(
    *,
    paths: int = 3,
    days: int = 120,
    seed: int = 20200106,
    progress: Callable[[str], None] | None = None,
) -> CheckResult:
    """The screen and the walk-forward backtest on a planted pair, independent walks and a break."""
    from trading_research.pairs import data
    from trading_research.pairs.backtest import PairRule, backtest
    from trading_research.pairs.costs import leg_costs
    from trading_research.pairs.screen import screen
    from trading_research.pairs.stats import half_life
    from trading_research.strategies.spread import fit_hedge
    from trading_research.synthetic.pairs import PairSpec, generate

    started = time.perf_counter()
    spec0 = PairSpec(days=days, seed=seed)
    rule = PairRule()
    costs = leg_costs()
    result = CheckResult(
        what="pairs",
        title="the pairs screen and walk-forward backtest on a planted pair and its nulls",
        planted=(
            f"{paths} paths of three pairs on {spec0.interval} bars over {days} days. Planted: "
            f"log a = {spec0.alpha:g} + {spec0.beta:g} log b + s, b a random walk of "
            f"{spec0.vol_b_bp:g} bp a bar, s an AR(1) with half-life {spec0.half_life_bars:g} "
            f"bars and sd {spec0.spread_sd_bp:g} bp. Independent: a is a random walk of the "
            "same volatility, unrelated to b. Break: planted until "
            f"{spec0.break_fraction:.0%} of the bars, then s a random walk with the same "
            "innovations. The screen reads the first formation window of the planted and the "
            "independent pair together (four symbols, six pairs, Holm at 5%). The backtest: "
            f"formation {rule.formation} bars, trading {rule.trading}, z bands entry "
            f"{rule.entry_z:g}, exit {rule.exit_z:g}, stop {rule.stop_z:g}, max hold "
            f"{rule.max_hold} bars, no gate; fills at the next bar's open; "
            f"{costs.per_fill_bp:g} bp a fill on each leg ({costs.source})."
        ),
        proves=(
            "The screen ranks a planted cointegration first and does not pass independent "
            "walks after correction; the hedge ratio and half-life come back; the backtest "
            "earns the planted reversion before costs, finds nothing after costs on "
            "independent walks or after a break, and takes no decision from a later bar."
        ),
        does_not_prove=(
            "That any two real instruments cointegrate, at what half-life, or that a "
            "relation found on one window holds on the next."
        ),
        command=f"trading-research verify pairs --paths {paths} --days {days} --seed {seed}",
        seeds={
            "planted": [seed + k for k in range(paths)],
            "independent": [seed + 1_000_000 + k for k in range(paths)],
            "break": [seed + 2_000_000 + k for k in range(paths)],
        },
        settings={
            "days": days,
            **{f"rule_{k}": v for k, v in rule.__dict__.items()},
            **costs.describe(),
        },
    )
    ranked_first = planted_passes = independent_passes = others_pass = 0
    betas, lives, causal = [], [], []
    trades: dict[str, list[pd.DataFrame]] = {"planted": [], "independent": [], "break": []}
    pre_break: list[pd.DataFrame] = []
    first_screen = pd.DataFrame()
    for k in range(paths):
        if progress:
            progress(f"pairs: path {k}")
        s = seed + k
        planted = generate(PairSpec(days=days, seed=s))
        independent = generate(
            PairSpec(days=days, seed=s + 1_000_000, variant="independent", symbol_a=C, symbol_b=D)
        )
        broken = generate(PairSpec(days=days, seed=s + 2_000_000, variant="break"))
        bars = data.align({**planted.bars, **independent.bars}, planted.spec.interval)
        table = screen(data.log_closes(bars).iloc[: rule.formation], costs)
        if k == 0:
            first_screen = table
        top = table.iloc[0]
        ranked_first += int(top["a"] == A and top["b"] == B)
        is_planted = (table["a"] == A) & (table["b"] == B)
        is_independent = (table["a"] == C) & (table["b"] == D)
        planted_passes += int(table.loc[is_planted, "passes"].iloc[0])
        independent_passes += int(table.loc[is_independent, "passes"].iloc[0])
        others_pass += int(table.loc[~is_planted, "passes"].sum())

        log_a = np.log(planted.bars[A]["close"].to_numpy())
        log_b = np.log(planted.bars[B]["close"].to_numpy())
        hedge = fit_hedge(log_a, log_b)
        betas.append(hedge.beta)
        lives.append(half_life(hedge.residual(log_a, log_b)))

        for label, dataset in (
            ("planted", planted),
            ("independent", independent),
            ("break", broken),
        ):
            pair = list(dataset.bars)
            aligned = data.align(dataset.bars, dataset.spec.interval)
            run = backtest(aligned, pair[0], pair[1], rule, costs)
            got = run.trades.assign(day=[f"{k}:{d}" for d in run.trades["day"]])
            if label == "break":
                cut = dataset.bars[pair[0]]["timestamp"].iloc[int(dataset.spec.break_bar or 0)]
                pre_break.append(got[got["entry_decision"] < cut])
                got = got[got["entry_decision"] >= cut]
            trades[label].append(got)
            if label == "planted":
                causal.append(_unchanged_before(aligned, pair, rule, costs, s, run.decisions))

    result.add(
        "screen: the planted pair ranks first",
        "rank 1",
        f"rank 1 on {ranked_first} of {paths} paths",
        "every path",
        ranked_first == paths,
    )
    result.add(
        "screen: the planted pair passes after Holm",
        "cointegrated",
        f"passes on {planted_passes} of {paths} paths",
        "every path, Holm 5%",
        planted_passes == paths,
    )
    result.add(
        "screen: independent walks do not pass after Holm",
        "not cointegrated",
        f"passes on {independent_passes} of {paths} paths",
        "no path, Holm 5%",
        independent_passes == 0,
    )
    result.add(
        "screen: any other pair passing",
        "none cointegrated",
        f"{others_pass} pass(es) over {paths} paths",
        "info",
        True,
        binding=False,
    )
    beta = float(np.mean(betas))
    result.add(
        "hedge ratio, whole planted series",
        f"{spec0.beta:g}",
        f"{beta:.3f} (mean)",
        f"within {BETA_TOLERANCE:g}",
        abs(beta - spec0.beta) <= BETA_TOLERANCE,
    )
    life = float(np.mean(lives))
    result.add(
        "half-life of the residual, bars",
        f"{spec0.half_life_bars:g}",
        f"{life:.2f} (mean)",
        f"within {HALF_LIFE_TOLERANCE:.0%}",
        abs(life / spec0.half_life_bars - 1.0) <= HALF_LIFE_TOLERANCE,
    )
    result.add(
        "no decision reads a later bar",
        "decisions before a perturbed bar unchanged",
        f"unchanged on {sum(causal)} of {paths} paths",
        "every path",
        all(causal),
    )

    rows = []
    stats = {label: _stats(_pooled(parts)) for label, parts in trades.items()}
    stats["break, before"] = _stats(_pooled(pre_break))
    for label, measured in stats.items():
        rows.append({"dataset": label, **measured})
    p, i, b = stats["planted"], stats["independent"], stats["break"]
    result.add(
        "backtest: planted, positive before costs",
        "a reverting residual",
        _line(p, "gross"),
        f"day-clustered t >= {T_BAR:g}",
        math.isfinite(p["gross_t"]) and p["gross_t"] >= T_BAR,
    )
    result.add("backtest: planted, after costs", "", _line(p, "net"), "info", True, binding=False)
    result.add(
        "backtest: independent walks, not positive after costs",
        "nothing to earn",
        _line(i, "net"),
        f"day-clustered t < {T_BAR:g}",
        not math.isfinite(i["net_t"]) or i["net_t"] < T_BAR,
    )
    result.add(
        "backtest: independent walks, nothing before costs",
        "0 bp",
        _line(i, "gross"),
        f"|day-clustered t| < {T_BAR:g}",
        not math.isfinite(i["gross_t"]) or abs(i["gross_t"]) < T_BAR,
    )
    result.add(
        "backtest: after the break, not positive after costs",
        "nothing to earn",
        _line(b, "net"),
        f"day-clustered t < {T_BAR:g}",
        not math.isfinite(b["net_t"]) or b["net_t"] < T_BAR,
    )
    result.add(
        "backtest: before the break, before costs",
        "a reverting residual",
        _line(stats["break, before"], "gross"),
        "info",
        True,
        binding=False,
    )
    result.tables.append(("Pooled over paths, clustered by path-day", pd.DataFrame(rows)))
    if not first_screen.empty:
        result.tables.append(("The screen on path 0's first formation window", first_screen))
    result.seconds = time.perf_counter() - started
    return result


def _unchanged_before(
    bars: pd.DataFrame,
    pair: list[str],
    rule: Any,
    costs: Any,
    seed: int,
    decisions: pd.DataFrame,
) -> bool:
    """Scramble every price from the middle bar on; the decisions before it must not move."""
    from trading_research.pairs.backtest import backtest

    cut = len(bars) // 2
    changed = bars.copy()
    rng = np.random.default_rng(seed)
    for field in ("open", "close"):
        for symbol in pair:
            values = changed[(field, symbol)].to_numpy().copy()
            values[cut:] *= np.exp(rng.normal(0.0, 0.05, len(values) - cut))
            changed[(field, symbol)] = values
    again = backtest(changed, pair[0], pair[1], rule, costs).decisions
    when = bars.index[cut]
    before, after = decisions[decisions["timestamp"] < when], again[again["timestamp"] < when]
    return bool(
        len(before) > 0 and before.reset_index(drop=True).equals(after.reset_index(drop=True))
    )
