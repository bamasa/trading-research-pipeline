"""The market-making study's charts, each drawn from a committed results table.

Six charts, one per question the README asks of the held-out block: where the
money goes (markouts by fill path, the profit decomposition), what inventory
the quoter carried, whether the hypotheses beat their placebos, what each
advantage of the ladder is worth, and at which maker fee each strategy would
break even. Theme-blind: :func:`trading_research.reporting.plots.both_themes`
renders each twice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.reporting.plots import COLOURS, _finish, _pyplot

STRATEGY_ORDER = ("S0", "S1", "S2", "S2-twin", "S3", "S4", "X1")
PATH_COLOURS = {
    "queue": COLOURS["signal"],
    "through": COLOURS["accent"],
    "flatten": COLOURS["cost"],
}
PARTS = (
    ("spread", COLOURS["edge"]),
    ("adverse", COLOURS["cost"]),
    ("inventory", COLOURS["accent"]),
    ("fees", COLOURS["neutral"]),
    ("funding", COLOURS["signal"]),
)


def markouts_by_path(
    table: pd.DataFrame,
    path: Path | str,
    *,
    symbol: str,
    horizons: Sequence[str] = ("1s", "5s", "30s"),
    title: str = "",
) -> Path:
    """One panel per horizon: each strategy's markout by fill path, against the
    market-wide passive benchmark at that horizon (dashed)."""
    plt = _pyplot()
    mine = table[(table["symbol"] == symbol) & table["path"].isin(["queue", "through"])]
    mine = mine[mine["strategy"].isin(STRATEGY_ORDER)]
    bench = table[(table["symbol"] == symbol) & (table["path"] == "benchmark")]
    labels = [
        (s, p)
        for s in STRATEGY_ORDER
        for p in ("queue", "through")
        if ((mine["strategy"] == s) & (mine["path"] == p)).any()
    ]
    fig, axes = plt.subplots(1, len(horizons), figsize=(10, 0.36 * len(labels) + 1.8), sharey=True)
    for ax, horizon in zip(np.atleast_1d(axes), horizons, strict=True):
        column = f"markout_bp_{horizon}"
        values = [
            float(mine.loc[(mine["strategy"] == s) & (mine["path"] == p), column].iloc[0])
            for s, p in labels
        ]
        y = np.arange(len(labels))
        ax.barh(y, values, color=[PATH_COLOURS[p] for _, p in labels], height=0.62)
        if len(bench):
            ax.axvline(
                float(bench[column].iloc[0]), color=COLOURS["cost"], linestyle="--", linewidth=1.1
            )
        ax.axvline(0.0, color=COLOURS["neutral"], linewidth=0.8)
        ax.set_yticks(y, [f"{s} {p}" for s, p in labels])
        ax.invert_yaxis()
        ax.set_xlabel(f"markout at {horizon}, bp")
        ax.set_title(horizon, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.suptitle(
        title or f"Markouts by fill path, {symbol}; dashed: market-wide passive",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110)
    plt.close(fig)
    return target


def decomposition(table: pd.DataFrame, path: Path | str, *, symbol: str, title: str = "") -> Path:
    """Stacked bars per strategy: spread, adverse (5 s), inventory, fees (as a
    cost) and funding, USDT a day, with the net marked."""
    plt = _pyplot()
    mine = table[table["symbol"] == symbol].set_index("strategy")
    names = [s for s in STRATEGY_ORDER if s in mine.index]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(names))
    up = np.zeros(len(names))
    down = np.zeros(len(names))
    for part, colour in PARTS:
        values = mine.loc[names, f"{part}_usd_day"].to_numpy(dtype=np.float64)
        if part == "fees":
            values = -values
        base = np.where(values >= 0, up, down)
        ax.bar(
            x,
            values,
            bottom=base,
            color=colour,
            width=0.62,
            label=part if part != "fees" else "fees (paid)",
        )
        up += np.where(values >= 0, values, 0.0)
        down += np.where(values < 0, values, 0.0)
    net = mine.loc[names, "net_usd_day"].to_numpy(dtype=np.float64)
    ax.scatter(x, net, color="#805ad5", zorder=3, marker="D", s=28, label="net")
    for xi, value in zip(x, net, strict=True):
        ax.annotate(
            f"{value:+.2f}",
            (xi, value),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
        )
    ax.axhline(0.0, color=COLOURS["neutral"], linewidth=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("USDT a day")
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="lower left")
    return _finish(
        fig, ax, title or f"Where the money goes, {symbol}: the decomposition", Path(path)
    )


def inventory(table: pd.DataFrame, path: Path | str, *, title: str = "") -> Path:
    """The headline strategy's position over the block against its soft and
    hard limits, with the distribution of the position beside it."""
    plt = _pyplot()
    fig, (ax, side) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [4, 1]}, sharey=True
    )
    time = pd.to_datetime(table["time"])
    position = table["position"].to_numpy(dtype=np.float64)
    ax.plot(time, position, color=COLOURS["signal"], linewidth=0.7, label="position")
    for column, style in (("soft_limit", "--"), ("hard_limit", ":")):
        limit = table[column].ffill().to_numpy(dtype=np.float64)
        label = column.replace("_", " ")
        ax.plot(time, limit, color=COLOURS["cost"], linestyle=style, linewidth=0.9, label=label)
        ax.plot(time, -limit, color=COLOURS["cost"], linestyle=style, linewidth=0.9)
    ax.axhline(0.0, color=COLOURS["neutral"], linewidth=0.6)
    ax.set_ylabel(f"position, {table['symbol'].iloc[0].removesuffix('USDT')}")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=3)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    side.hist(
        position[np.isfinite(position)], bins=40, orientation="horizontal", color=COLOURS["signal"]
    )
    side.set_xlabel("five-minute samples")
    for axis in (ax, side):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    strategy = table["strategy"].iloc[0]
    return _finish(
        fig, ax, title or f"{strategy}'s inventory over the held-out fortnight", Path(path)
    )


def placebos(table: pd.DataFrame, path: Path | str, *, title: str = "") -> Path:
    """One histogram per placebo family, the true value marked."""
    plt = _pyplot()
    names = list(dict.fromkeys(table["placebo"]))
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6))
    for ax, name in zip(np.atleast_1d(axes), names, strict=True):
        mine = table[table["placebo"] == name]
        values = mine["value"].to_numpy(dtype=np.float64)
        true = float(mine["true_value"].iloc[0])
        finite = values[np.isfinite(values)]
        ax.hist(
            finite, bins=15, color=COLOURS["neutral"], alpha=0.8, label=f"{len(finite)} placebos"
        )
        if len(finite):
            ax.axvline(
                float(np.percentile(finite, 95)),
                color=COLOURS["accent"],
                linestyle=":",
                label="95th percentile",
            )
        if math.isfinite(true):
            ax.axvline(true, color=COLOURS["cost"], linewidth=2.0, label="true")
        ax.set_title(name, fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        title or "Each true difference against its placebos", x=0.01, ha="left", fontsize=11
    )
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110)
    plt.close(fig)
    return target


def ladder(table: pd.DataFrame, path: Path | str, *, title: str = "") -> Path:
    """Net bp of turnover at each rung of the advantage ladder, per instrument;
    rungs from 2 read the future and are upper bounds (shaded)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    palette = [COLOURS["signal"], COLOURS["accent"], COLOURS["edge"], COLOURS["cost"]]
    for colour, (symbol, mine) in zip(palette, table.groupby("symbol", sort=True), strict=False):
        mine = mine.sort_values("rung")
        ax.plot(mine["rung"], mine["net_bp_turnover"], marker="o", color=colour, label=symbol)
    ax.axhline(0.0, color=COLOURS["neutral"], linewidth=1.0, linestyle="--")
    ax.axvspan(
        1.5, 6.5, color=COLOURS["neutral"], alpha=0.12, label="upper bounds (read the future)"
    )
    names = ["0 tail", "1 front", "2 R² .1", "3 R² .3", "4 1 ms", "5 R² 1", "6 best tier"]
    ax.set_xticks(range(7), names, fontsize=8)
    ax.set_ylabel("net, bp of turnover")
    ax.set_xlim(-0.4, 6.6)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, ax, title or "What each advantage is worth: S0 up the ladder", Path(path))


def fee_breakeven(
    table: pd.DataFrame,
    path: Path | str,
    *,
    grid: tuple[float, float] = (-1.5, 2.0),
    base_bp: float = 2.0,
    title: str = "",
) -> Path:
    """Each strategy's break-even maker fee, against the published tiers.

    A break-even outside the grid is drawn at the grid's edge with an arrow:
    left when the strategy loses at every fee tried, right when it pays at
    every fee tried. Lines mark the base fee, the best published maker fee and
    the market-maker programme's rebate.
    """
    plt = _pyplot()
    rows = table.assign(name=table["strategy"] + " " + table["symbol"].str.removesuffix("USDT"))
    rows = rows.sort_values("breakeven_maker_bp").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(rows) + 1.8))
    low, high = grid[0] - 0.4, grid[1] + 0.4
    for y, row in rows.iterrows():
        value = float(row["breakeven_maker_bp"])
        if math.isnan(value):
            continue
        shown = min(max(value, low), high)
        marker = "<" if value < low else (">" if value > high else "o")
        colour = COLOURS["edge"] if value >= base_bp else COLOURS["cost"]
        ax.scatter([shown], [y], marker=marker, color=colour, s=40, zorder=3)
    ax.axvline(
        base_bp,
        color=COLOURS["neutral"],
        linestyle="--",
        linewidth=1,
        label=f"base maker fee {base_bp:g} bp",
    )
    best = float(rows["best_published_maker_bp"].min()) if len(rows) else 0.0
    ax.axvline(
        best,
        color=COLOURS["accent"],
        linestyle=":",
        linewidth=1.2,
        label=f"best published tier {best:g} bp",
    )
    rebate = float(rows["mm_programme_maker_bp"].iloc[0]) if len(rows) else -1.0
    ax.axvline(
        rebate,
        color=COLOURS["signal"],
        linestyle="-.",
        linewidth=1,
        label=f"market-maker programme, up to {rebate:g} bp",
    )
    ax.set_yticks(range(len(rows)), rows["name"])
    ax.invert_yaxis()
    ax.set_xlim(low - 0.1, high + 0.1)
    ax.set_xlabel("maker fee at which the net is zero, bp (arrows: outside the grid)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    return _finish(fig, ax, title or "The maker fee that would make each strategy pay", Path(path))
