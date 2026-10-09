"""The 2025 search's figures, each drawn from one of its result tables.

Theme-blind, like every chart of the repository:
:func:`trading_research.reporting.plots.both_themes` renders each twice, and
the README picks one with ``<picture>`` (:func:`picture`). Every D number on
them is labelled "on development data, selected"; the cost line, or the bar a
number has to clear, is drawn on the axis it belongs to.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making.fill_rules import QUEUE_AT_MOST, TOUCH_AT_LEAST
from trading_research.reporting.leaderboard import SELECTED
from trading_research.reporting.plots import COLOURS, _pyplot

#: Each stratum's colour; blue and amber stay apart under every colour-vision type.
STRATUM_COLOURS = {"C": COLOURS["signal"], "R": COLOURS["accent"]}
DEVELOPMENT = f"on development data, {SELECTED}"


def _save(fig: Any, path: Path | str, title: str, note: str = "") -> Path:
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    if note:
        fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=8, color=COLOURS["neutral"])
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0.0, 0.05 if note else 0.0, 1.0, 1.0))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110)
    _pyplot().close(fig)
    return target


def picture(name: str, alt: str, images: str = "docs/images") -> str:
    """The README's ``<picture>`` block for a chart rendered by ``both_themes``."""
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{images}/{name}_dark.png">\n'
        f'  <img alt="{alt}" src="{images}/{name}_light.png">\n'
        "</picture>"
    )


def placebo_search_chart(
    nulls: pd.DataFrame, path: Path | str, *, subject: str = "search", title: str = ""
) -> Path:
    """The real A against both null distributions, with the refit point.

    ``nulls`` is the nulls file (``reporting.nulls``). One panel per null: the
    histogram of each replication's maximum over strata (or S25's own A), its
    95th percentile dashed, each stratum's chosen cell's A as a line labelled
    with its p-value, stratum C's refit A dotted, and p_search beside.
    """
    plt = _pyplot()
    mine = nulls.loc[nulls["subject"] == subject]
    observed = mine.loc[mine["row"] == "observed"]
    reps = mine.loc[mine["row"] == "replication"]
    top = "max" if (reps["stratum"] == "max").any() else str(reps["stratum"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharex=True)
    for ax, null in zip(axes, ("A", "B"), strict=True):
        picked = reps.loc[(reps["null"] == null) & (reps["stratum"] == top), "value"]
        values = picked.to_numpy(dtype=np.float64)
        ax.hist(
            values,
            bins=30,
            color=COLOURS["neutral"],
            alpha=0.75,
            label=f"{len(values)} replications",
        )
        q95 = float(np.quantile(values, 0.95)) if len(values) else math.nan
        ax.axvline(
            q95, color=COLOURS["cost"], linestyle="--", linewidth=1.1, label="95th percentile"
        )
        for _, row in observed.iterrows():
            if subject == "search" and not row["key"]:
                continue
            colour = STRATUM_COLOURS.get(str(row["stratum"]), COLOURS["signal"])
            p = mine.loc[
                (mine["row"] == "p") & (mine["null"] == null) & (mine["stratum"] == row["stratum"]),
                "value",
            ]
            name = f"{row['stratum']}*" if subject == "search" else subject
            label = f"{name} A {row['value']:.2f}" + (
                f", p {float(p.iloc[0]):.3g}" if len(p) else ""
            )
            ax.axvline(float(row["value"]), color=colour, linewidth=2.0, label=label)
        for _, row in mine.loc[mine["row"] == "refit"].iterrows():
            ax.axvline(
                float(row["value"]),
                color=STRATUM_COLOURS.get(str(row["stratum"]), COLOURS["signal"]),
                linestyle=":",
                linewidth=1.6,
                label=f"{row['stratum']} refit A {row['value']:.2f}",
            )
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)  # headroom: the legend sits above the bars
        ax.set_title(f"Null {null}", fontsize=10, loc="left")
        ax.set_xlabel("A, the neighbourhood median of the day-level t of G")
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("replications")
    found = mine.loc[mine["row"] == "p_search", "value"]
    note = f"p_search = {float(found.iloc[0]):.3g}, " if len(found) else ""
    footnote = (
        f"{note}{DEVELOPMENT}" if subject == "search" else "on development data, no selection"
    )
    default = (
        "The placebo search: the chosen cells against the search re-run on noise"
        if subject == "search"
        else f"{subject} against its own nulls, no selection"
    )
    return _save(fig, path, title or default, footnote)


def _mark(
    ax: Any, x: float, y: float, text: str, colour: str, marker: str = "o", below: bool = False
) -> None:
    size = 140 if marker == "*" else 60
    ax.scatter([x], [y], s=size, marker=marker, color=colour, edgecolors="none", zorder=4)
    offset = (6, -14) if below else (6, 8)
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=8, zorder=5)


def leaderboard_chart(board: pd.DataFrame, path: Path | str, *, title: str = "") -> Path:
    """Every cell's S by its rank within its stratum, one panel per stratum.

    ``board`` is the leaderboard (``reporting.leaderboard``). Admissible cells
    (at least five trades a day) are filled, the others hollow; the chosen cell
    (the best-ranked admissible one), S25 with its rank r of N, and each
    region's best cell are named. The dashed line is a net of 0 at the base fee.
    """
    plt = _pyplot()
    strata = [s for s in ("C", "R") if s in set(board["key"].str.split("|").str[1])]
    fig, axes = plt.subplots(1, len(strata), figsize=(5.0 * len(strata), 4.0), squeeze=False)
    stratum_of = board["key"].str.split("|").str[1]
    n = len(board)
    for ax, stratum in zip(axes[0], strata, strict=True):
        mine = board.loc[stratum_of == stratum].sort_values("rank_in_stratum")
        colour = STRATUM_COLOURS[stratum]
        ok = mine["admissible"].astype(bool).to_numpy()
        x, y = mine["rank_in_stratum"].to_numpy(), mine["S"].to_numpy(dtype=np.float64)
        ax.scatter(x[ok], y[ok], s=7, color=colour, linewidths=0, label="admissible")
        ax.scatter(
            x[~ok],
            y[~ok],
            s=7,
            facecolors="none",
            edgecolors=COLOURS["neutral"],
            linewidths=0.6,
            label="fewer than 5 trades a day",
        )
        ax.axhline(
            0.0, color=COLOURS["cost"], linestyle="--", linewidth=1.0, label="net 0 at the base fee"
        )
        chosen = mine.loc[mine["admissible"].astype(bool)]
        if len(chosen):
            best = chosen.iloc[0]
            _mark(ax, best["rank_in_stratum"], best["S"], f"{stratum}*", colour)
        for _, row in mine.loc[mine["S25_badge"] != ""].iterrows():
            _mark(
                ax,
                row["rank_in_stratum"],
                row["S"],
                f"S25, rank {row['rank_overall']} of {n}",
                COLOURS["accent"],
                marker="*",
            )
        for region in ("T1", "T2"):
            tagged = mine.loc[mine["region"].str.split("+").apply(lambda tags, r=region: r in tags)]
            if len(tagged):
                row = tagged.iloc[0]
                _mark(
                    ax,
                    row["rank_in_stratum"],
                    row["S"],
                    region,
                    COLOURS["neutral"],
                    "D",
                    below=True,
                )
        ax.set_title(f"stratum {stratum}, {len(mine)} cells", fontsize=10, loc="left")
        ax.set_xlabel("rank in stratum by S")
        ax.legend(
            frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2
        )
    axes[0][0].set_ylabel("S (neighbourhood median net), bp")
    return _save(
        fig,
        path,
        title or "The D leaderboard: every cell by S, with S25 and the regions",
        DEVELOPMENT,
    )


def marginals_chart(marginals: pd.DataFrame, path: Path | str, *, title: str = "") -> Path:
    """The median net per axis value, one panel per stratum, with each axis's
    range (how far the choice moves the median cell) beside its name."""
    plt = _pyplot()
    scopes = [s for s in ("C", "R") if (marginals["scope"] == s).any()]
    sizes = [int((marginals["scope"] == s).sum()) for s in scopes]
    fig, axes = plt.subplots(
        1,
        len(scopes),
        figsize=(5.0 * len(scopes), 0.24 * max(sizes) + 1.6),
        squeeze=False,
        sharex=True,
    )
    for ax, scope in zip(axes[0], scopes, strict=True):
        mine = marginals.loc[marginals["scope"] == scope].reset_index(drop=True)
        spans = mine.groupby("axis", sort=False)["net"].agg(lambda v: v.max() - v.min())
        labels = [f"{a} = {v}" for a, v in zip(mine["axis"], mine["value"], strict=True)]
        y = np.arange(len(mine))
        ax.scatter(mine["net"], y, s=28, color=STRATUM_COLOURS[scope], zorder=3)
        ax.set_yticks(y, labels, fontsize=7.5)
        ax.invert_yaxis()
        for axis, rows in mine.groupby("axis", sort=False):
            first = int(rows.index[0])
            ax.annotate(
                f"range {spans[axis]:.2f}",
                (1.02, first),
                xycoords=("axes fraction", "data"),
                ha="left",
                va="center",
                fontsize=7,
                color=COLOURS["neutral"],
            )
            if first:
                ax.axhline(first - 0.5, color=COLOURS["neutral"], linewidth=0.5, alpha=0.4)
        ax.axvline(0.0, color=COLOURS["cost"], linestyle="--", linewidth=1.0)
        ax.set_title(f"stratum {scope}", fontsize=10, loc="left")
        ax.set_xlabel("median over cells of mean daily net, bp", fontsize=8)
    title = title or "What each design choice moves: the median cell per axis value"
    return _save(fig, path, title, DEVELOPMENT)


def edge_ladder_chart(ladder: pd.DataFrame, path: Path | str, *, title: str = "") -> Path:
    """S25's edge ladder (``reporting.edge_ladder``): each step's values in bp
    of traded notional, and the break-even per side against every tier.

    Rows inside S25's information set are hatched, a row not read yet is
    named and left empty, and the dashed line is a net of 0.
    """
    plt = _pyplot()
    bars = ladder.loc[ladder["unit"] == "bp of traded notional"].reset_index(drop=True)
    tiers = ladder.loc[ladder["unit"] == "bp per side"].reset_index(drop=True)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 0.32 * len(bars) + 1.8), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    y = np.arange(len(bars))
    values = bars["value"].to_numpy(dtype=np.float64)
    inside = bars["label"].str.contains("information set").to_numpy()
    shown = np.where(np.isfinite(values), values, 0.0)
    left.barh(y[~inside], shown[~inside], height=0.6, color=COLOURS["signal"])
    left.barh(
        y[inside],
        shown[inside],
        height=0.6,
        color="none",
        edgecolor=COLOURS["signal"],
        hatch="///",
        linewidth=0.8,
        label="inside S25's information set",
    )
    for i in np.flatnonzero(~np.isfinite(values)):
        left.text(0.0, i, " not read yet", va="center", fontsize=7.5, color=COLOURS["neutral"])
    blocks = bars["block"].str.replace("H before 2025-08-01", "H to 07-31", regex=False)
    labels = [f"{s}. {r} ({b})" for s, r, b in zip(bars["step"], bars["row"], blocks, strict=True)]
    left.set_yticks(y, labels, fontsize=7.5)
    left.invert_yaxis()
    left.axvline(0.0, color=COLOURS["cost"], linestyle="--", linewidth=1.0)
    left.set_xlabel("bp of traded notional")
    if inside.any():
        left.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    fees = tiers["reference"].to_numpy(dtype=np.float64)
    names = [str(r).rsplit(", ", 1)[-1] for r in tiers["row"]]
    right.scatter(
        fees,
        np.arange(len(tiers)),
        s=24,
        color=COLOURS["cost"],
        zorder=3,
        label="taker fee of the tier",
    )
    right.set_yticks(np.arange(len(tiers)), names, fontsize=7.5)
    right.invert_yaxis()
    even = tiers["value"].to_numpy(dtype=np.float64)
    if len(even) and np.isfinite(even[0]):
        right.axvline(
            even[0], color=COLOURS["signal"], linewidth=2.0, label=f"S25's break-even {even[0]:.2f}"
        )
    elif len(even):
        right.text(
            0.98,
            0.02,
            "break-even not read yet",
            transform=right.transAxes,
            ha="right",
            fontsize=8,
            color=COLOURS["neutral"],
        )
    right.set_xlabel("bp per side")
    right.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    return _save(fig, path, title or "S25's edge ladder: one change at each step")


#: How the fill-rule chart names each rule.
RULE_NAMES = {"touch": "on touch", "trade_through": "trade-through", "queue": "queue"}


def fill_rule_chart(streams: dict[str, pd.DataFrame], path: Path | str, *, title: str = "") -> Path:
    """FILL: the share of posted orders with a fill under each rule, one bar
    per order stream (A0 on O; S-P and S25-Z on H), each bar labelled with
    its value, and the registered prediction drawn over its rule.

    Each table is ``market_making.fill_rules.FillComparison.summary``.
    """
    plt = _pyplot()
    rules = list(RULE_NAMES)
    palette = [COLOURS["signal"], COLOURS["accent"], COLOURS["edge"]]
    if len(streams) > len(palette):
        raise ValueError(f"at most {len(palette)} order streams, got {len(streams)}")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.8 / max(len(streams), 1)
    for k, (name, summary) in enumerate(streams.items()):
        ratio = summary.set_index("rule")["fill_ratio"].reindex(rules).to_numpy(dtype=np.float64)
        x = np.arange(len(rules)) - 0.4 + width * (k + 0.5)
        ax.bar(x, np.nan_to_num(ratio), width=width * 0.9, color=palette[k], label=name)
        for xi, value in zip(x, ratio, strict=True):
            text = "n/a" if not math.isfinite(value) else f"{value:.2f}"
            ax.text(
                xi, (value if math.isfinite(value) else 0.0) + 0.02, text, ha="center", fontsize=7.5
            )
    for i, (bound, words) in ((0, (TOUCH_AT_LEAST, "at least")), (2, (QUEUE_AT_MOST, "at most"))):
        ax.hlines(bound, i - 0.45, i + 0.45, color=COLOURS["cost"], linestyles="--", linewidth=1.2)
        ax.text(
            i,
            1.05,
            f"prediction, dashed: {words} {bound:.2f}",
            ha="center",
            fontsize=7.5,
            color=COLOURS["cost"],
        )
    ax.set_xticks(np.arange(len(rules)), [RULE_NAMES[r] for r in rules])
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("fill ratio: share of posted orders with a fill")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.95))
    return _save(fig, path, title or "The fill rule: the same orders under three rules")
