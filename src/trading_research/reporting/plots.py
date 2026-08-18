"""Charts for the report.

Six figures, each carrying one claim. The constraint that shaped them: a reader
scanning a repository gives a chart a few seconds, so anything that needs a
caption to be understood has failed. Every one here puts the cost line on the
axis it belongs to, because the whole result is a comparison against that line
and a chart without it invites the reader to judge the numbers on their own
scale.

Deliberately plain — no grid decoration, no colour beyond what distinguishes
series, no styling that would need explaining. Matplotlib is an optional
dependency: ``uv sync --extra report``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Muted palette, distinguishable in greyscale by lightness as well as hue.
COLOURS: dict[str, str] = {
    "signal": "#2b6cb0",
    "cost": "#c53030",
    "edge": "#2f855a",
    "neutral": "#718096",
    "accent": "#b7791f",
}

FIGSIZE = (7.0, 4.0)
DPI = 110


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display needed, and none available in CI
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "plots need matplotlib, an optional dependency. Install with: uv sync --extra report"
        ) from exc
    return plt


def _finish(fig: Any, ax: Any, title: str, path: Path) -> Path:
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    _pyplot().close(fig)
    return path


def _seconds_label(horizons: Sequence[int], grid_ms: int = 100) -> list[str]:
    out = []
    for h in horizons:
        seconds = h * grid_ms / 1000
        out.append(f"{seconds:g}s" if seconds < 60 else f"{seconds / 60:g}m")
    return out


def signal_decay(
    horizons: Sequence[int],
    coefficients: dict[str, Sequence[float]],
    path: Path | str,
    *,
    grid_ms: int = 100,
) -> Path:
    """Information coefficient against horizon, one line per feature.

    The chart behind the central claim: prediction is real and short-lived.
    Log x-axis because the horizons span four orders of magnitude and a linear
    axis would compress everything interesting into the left edge.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (name, values) in enumerate(coefficients.items()):
        ax.plot(
            horizons,
            values,
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=name,
            color=list(COLOURS.values())[i % len(COLOURS)],
        )
    ax.set_xscale("log")
    ax.set_xticks(list(horizons))
    ax.set_xticklabels(_seconds_label(horizons, grid_ms))
    ax.axhline(0, color=COLOURS["neutral"], linewidth=0.8)
    ax.set_xlabel("horizon")
    ax.set_ylabel("correlation with forward return")
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, "Prediction works, and decays", Path(path))


def edge_against_cost(
    horizons: Sequence[int],
    edge_bp: Sequence[float],
    cost_bp: float,
    path: Path | str,
    *,
    grid_ms: int = 100,
) -> Path:
    """Expected edge per trade against the cost of trading, by horizon.

    The single most important figure in the study. Edge is flat because signal
    decays at the rate volatility grows; the cost line does not move. The gap
    between them never closes, which is the result stated as a picture.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        horizons,
        edge_bp,
        marker="o",
        markersize=4,
        linewidth=1.8,
        color=COLOURS["edge"],
        label="expected edge per trade",
    )
    ax.axhline(
        cost_bp,
        color=COLOURS["cost"],
        linewidth=1.8,
        linestyle="--",
        label=f"round-trip cost ({cost_bp:.1f} bp)",
    )
    ax.fill_between(horizons, edge_bp, cost_bp, color=COLOURS["cost"], alpha=0.07)
    ax.set_xscale("log")
    ax.set_xticks(list(horizons))
    ax.set_xticklabels(_seconds_label(horizons, grid_ms))
    ax.set_ylim(0, max(cost_bp * 1.25, float(np.max(edge_bp)) * 1.25))
    ax.set_xlabel("horizon")
    ax.set_ylabel("basis points per trade")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    return _finish(fig, ax, "The gap that never closes", Path(path))


def breakeven_share(
    horizons: Sequence[int],
    shares: dict[str, Sequence[float]],
    path: Path | str,
    *,
    grid_ms: int = 100,
) -> Path:
    """Share of moments whose move clears the cost, per horizon and instrument.

    An upper bound under perfect foresight, so the reader can see how little
    room exists before any model is involved.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (name, values) in enumerate(shares.items()):
        ax.plot(
            horizons,
            [v * 100 for v in values],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=name,
            color=[COLOURS["signal"], COLOURS["accent"]][i % 2],
        )
    ax.set_xscale("log")
    ax.set_xticks(list(horizons))
    ax.set_xticklabels(_seconds_label(horizons, grid_ms))
    ax.set_xlabel("horizon")
    ax.set_ylabel("% of moments clearing the cost")
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, "Room to trade, assuming perfect foresight", Path(path))


def model_comparison(results: pd.DataFrame, cost_bp: float, path: Path | str) -> Path:
    """Gross edge per trade by model, against the cost line.

    Bars rather than a table because the comparison a reader wants — every bar
    against one line — is spatial. The models are far enough below the line that
    the ordering between them is visibly beside the point.
    """
    plt = _pyplot()
    frame = results.sort_values("gross_per_trade_bp")
    labels = [f"{m}\ncd={int(c)}" for m, c in zip(frame["model"], frame["cooldown"], strict=True)]
    values = frame["gross_per_trade_bp"].to_numpy()

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 4.4))
    colours = [COLOURS["edge"] if v > 0 else COLOURS["neutral"] for v in values]
    ax.bar(labels, values, color=colours, width=0.62)
    ax.axhline(
        cost_bp,
        color=COLOURS["cost"],
        linewidth=1.8,
        linestyle="--",
        label=f"cost to beat ({cost_bp:.1f} bp)",
    )
    ax.axhline(0, color=COLOURS["neutral"], linewidth=0.8)
    ax.set_ylim(min(values.min() * 1.3, -1), cost_bp * 1.2)
    ax.set_ylabel("gross bp per trade")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, "Every model, against the cost of trading", Path(path))


def fold_spread(per_fold: pd.DataFrame, path: Path | str) -> Path:
    """Net result per fold, one line per model.

    Included to make instability visible rather than described. The vertical
    spread within a model is larger than the distance between models, which is
    the caveat that governs how any other chart here should be read.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (name, group) in enumerate(per_fold.groupby("model")):
        ax.plot(
            group["fold"],
            group["net_bp"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=str(name),
            color=list(COLOURS.values())[i % len(COLOURS)],
        )
    ax.axhline(0, color=COLOURS["neutral"], linewidth=1.0)
    ax.set_xlabel("fold (each shifted one day later)")
    ax.set_ylabel("net basis points")
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, "Folds disagree more than models do", Path(path))


def thinning_effect(summary: pd.DataFrame, path: Path | str) -> Path:
    """Trades and gross edge per trade, as thinning tightens.

    Two axes because the point is the trade-off: fewer trades, more edge in
    each. Sharing one axis would hide it, since the quantities differ by three
    orders of magnitude.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = [str(c) for c in summary["cooldown"]]

    ax.bar(labels, summary["trades"], color=COLOURS["neutral"], width=0.5, label="trades")
    ax.set_ylabel("trades", color=COLOURS["neutral"])
    ax.set_xlabel("cooldown (observations)")

    twin = ax.twinx()
    twin.plot(
        labels,
        summary["gross_per_trade_bp"],
        marker="o",
        markersize=6,
        linewidth=2,
        color=COLOURS["edge"],
        label="gross bp per trade",
    )
    twin.axhline(0, color=COLOURS["neutral"], linewidth=0.8, linestyle=":")
    twin.set_ylabel("gross bp per trade", color=COLOURS["edge"])
    twin.spines["top"].set_visible(False)

    handles = [*ax.get_legend_handles_labels()[0], *twin.get_legend_handles_labels()[0]]
    labels_ = [*ax.get_legend_handles_labels()[1], *twin.get_legend_handles_labels()[1]]
    ax.legend(handles, labels_, frameon=False, fontsize=9, loc="upper center")
    return _finish(fig, ax, "Trading less, earning more per trade", Path(path))


# ---------------------------------------------------------------------------
# Backtest charts
# ---------------------------------------------------------------------------
#
# These show what the strategy did rather than what the study concluded: where
# it entered, how long it held, what it earned, and how the account would have
# moved. They are the charts a reader looks at first and the ones most likely
# to mislead, so each carries its cost line or its gross-versus-net pair — an
# equity curve without costs subtracted is the single most common way a
# backtest flatters itself.


def price_with_trades(
    timestamps: Sequence[Any],
    mid: Sequence[float],
    trades: pd.DataFrame,
    path: Path | str,
    *,
    max_trades: int = 400,
) -> Path:
    """Mid price with entry and exit markers, coloured by outcome.

    Up triangles are longs, down triangles are shorts; filled means the trade
    made money gross, hollow means it did not. Exits are joined to their entry
    by a line, so holding period is visible as horizontal extent rather than
    something to infer.

    Capped at ``max_trades`` because a chart with a thousand markers shows
    density, not behaviour, and density is already in the other figures.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.0, 4.6))

    times = pd.to_datetime(pd.Series(list(timestamps)))
    ax.plot(times, mid, linewidth=0.8, color=COLOURS["neutral"], alpha=0.85, zorder=1)

    shown = trades.head(max_trades)
    for _, trade in shown.iterrows():
        entry, exit_ = int(trade["entry_index"]), int(trade["exit_index"])
        if entry >= len(times) or exit_ >= len(times):
            continue
        gross = trade["direction"] * trade["move_bp"]
        colour = COLOURS["edge"] if gross > 0 else COLOURS["cost"]
        marker = "^" if trade["direction"] > 0 else "v"

        ax.plot(
            [times.iloc[entry], times.iloc[exit_]],
            [mid[entry], mid[exit_]],
            linewidth=1.0,
            color=colour,
            alpha=0.55,
            zorder=2,
        )
        ax.scatter(
            times.iloc[entry],
            mid[entry],
            marker=marker,
            s=34,
            facecolors=colour if gross > 0 else "none",
            edgecolors=colour,
            linewidths=1.1,
            zorder=3,
        )

    ax.set_ylabel("mid price")
    note = (
        f"{len(shown)} of {len(trades)} trades"
        if len(shown) < len(trades)
        else f"{len(trades)} trades"
    )
    ax.text(
        0.995,
        0.02,
        f"{note} · up = long, down = short · filled = gross win",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=COLOURS["neutral"],
    )
    fig.autofmt_xdate()
    return _finish(fig, ax, "Where the model traded", Path(path))


def equity_curve(
    trades: pd.DataFrame, cost_bp_per_trade: Sequence[float], path: Path | str
) -> Path:
    """Cumulative gross and net profit, in basis points.

    Both lines, always. A gross curve on its own is the most common way a
    backtest flatters itself: the shape is identical and the sign is not.
    """
    plt = _pyplot()
    gross = (trades["direction"] * trades["move_bp"]).to_numpy()
    net = gross - np.asarray(cost_bp_per_trade)
    x = np.arange(1, len(gross) + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x, np.cumsum(gross), linewidth=1.8, color=COLOURS["edge"], label="gross")
    ax.plot(x, np.cumsum(net), linewidth=1.8, color=COLOURS["cost"], label="net of costs")
    ax.axhline(0, color=COLOURS["neutral"], linewidth=1.0)
    ax.fill_between(x, np.cumsum(gross), np.cumsum(net), color=COLOURS["cost"], alpha=0.08)
    ax.set_xlabel("trade")
    ax.set_ylabel("cumulative basis points")
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, "Gross against net: the cost of trading", Path(path))


def drawdown(trades: pd.DataFrame, cost_bp_per_trade: Sequence[float], path: Path | str) -> Path:
    """Net drawdown from the running peak."""
    plt = _pyplot()
    net = (trades["direction"] * trades["move_bp"]).to_numpy() - np.asarray(cost_bp_per_trade)
    equity = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    under = equity - peak

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(1, len(net) + 1)
    ax.fill_between(x, under, 0, color=COLOURS["cost"], alpha=0.28)
    ax.plot(x, under, linewidth=1.2, color=COLOURS["cost"])
    ax.set_xlabel("trade")
    ax.set_ylabel("basis points below peak")
    ax.text(
        0.995,
        0.06,
        f"worst: {under.min():,.0f} bp",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=COLOURS["cost"],
    )
    return _finish(fig, ax, "Drawdown, net of costs", Path(path))


def trade_outcomes(trades: pd.DataFrame, cost_bp: float, path: Path | str) -> Path:
    """Distribution of gross outcomes per trade, against the cost line.

    The figure that explains the whole result in one look: the mass of the
    distribution sits well inside the cost, so most trades were never going to
    pay for themselves however the direction came out.
    """
    plt = _pyplot()
    gross = (trades["direction"] * trades["move_bp"]).to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    limit = float(np.percentile(np.abs(gross), 99)) if len(gross) else 1.0
    ax.hist(gross, bins=60, range=(-limit, limit), color=COLOURS["signal"], alpha=0.75)
    ax.axvline(
        cost_bp,
        color=COLOURS["cost"],
        linewidth=1.8,
        linestyle="--",
        label=f"cost to clear (+{cost_bp:.1f} bp)",
    )
    ax.axvline(0, color=COLOURS["neutral"], linewidth=1.0)
    share = float((gross > cost_bp).mean()) if len(gross) else 0.0
    ax.set_xlabel("gross outcome per trade, bp")
    ax.set_ylabel("trades")
    ax.text(
        0.995,
        0.9,
        f"{share:.1%} of trades cleared the cost",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=COLOURS["cost"],
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _finish(fig, ax, "What a trade actually earned", Path(path))


def holding_periods(
    trades: pd.DataFrame, path: Path | str, *, grid_ms: int = 100, subsample: int = 1
) -> Path:
    """How long positions stayed open, in wall-clock terms."""
    plt = _pyplot()
    held = (trades["exit_index"] - trades["entry_index"]).to_numpy() * grid_ms * subsample / 1000

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(held, bins=40, color=COLOURS["accent"], alpha=0.8)
    ax.set_xlabel("seconds held")
    ax.set_ylabel("trades")
    ax.text(
        0.995,
        0.9,
        f"median {np.median(held):,.0f}s",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color=COLOURS["neutral"],
    )
    return _finish(fig, ax, "How long a position stayed open", Path(path))


def confidence_against_outcome(
    confidence: Sequence[float],
    gross_bp: Sequence[float],
    cost_bp: float,
    path: Path | str,
    *,
    bins: int = 8,
) -> Path:
    """Mean gross outcome by how confident the model was.

    The check that decides whether a confidence threshold means anything: if
    the model knows what it knows, this rises left to right. Flat says the
    probability carries no information about size, which is what selecting on
    it would be relying on.
    """
    plt = _pyplot()
    frame = pd.DataFrame({"confidence": list(confidence), "gross": list(gross_bp)})
    frame["bucket"] = pd.qcut(frame["confidence"], q=bins, duplicates="drop")
    grouped = frame.groupby("bucket", observed=True).agg(
        mean_gross=("gross", "mean"), n=("gross", "size"), centre=("confidence", "mean")
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(range(len(grouped)), grouped["mean_gross"], color=COLOURS["signal"], width=0.65)
    ax.axhline(
        cost_bp,
        color=COLOURS["cost"],
        linewidth=1.8,
        linestyle="--",
        label=f"cost ({cost_bp:.1f} bp)",
    )
    ax.axhline(0, color=COLOURS["neutral"], linewidth=0.9)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([f"{c:.2f}" for c in grouped["centre"]], fontsize=8)
    ax.set_xlabel("model confidence")
    ax.set_ylabel("mean gross bp per trade")
    ax.legend(frameon=False, fontsize=9)
    return _finish(fig, ax, "Does confidence predict size?", Path(path))


def selection_leak(
    cutoffs: Sequence[str],
    chosen_on_test: Sequence[float],
    chosen_on_validation: Sequence[float],
    path: Path | str,
) -> Path:
    """The same strategy scored two ways: cutoff picked on test, and honestly.

    The most instructive figure here, and the one a reader is most likely to
    need. Trading only the most confident signals looks profitable when the
    cutoff is chosen after seeing how each one scored — the red line. Choose it
    on validation and apply it once to test, which is the only version that
    means anything, and the same strategy loses money.

    Same code, same data, same model. The difference is entirely *when* the
    cutoff was decided.
    """
    plt = _pyplot()
    x = np.arange(len(cutoffs))

    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 4.2))
    ax.plot(
        x,
        chosen_on_test,
        marker="o",
        markersize=6,
        linewidth=2,
        color=COLOURS["cost"],
        label="cutoff chosen after seeing test",
    )
    ax.plot(
        x,
        chosen_on_validation,
        marker="s",
        markersize=6,
        linewidth=2,
        color=COLOURS["edge"],
        label="cutoff chosen on validation",
    )
    ax.axhline(0, color=COLOURS["neutral"], linewidth=1.2)
    ax.fill_between(x, chosen_on_test, chosen_on_validation, color=COLOURS["cost"], alpha=0.08)
    ax.set_xticks(x)
    ax.set_xticklabels(cutoffs)
    ax.set_xlabel("how selective")
    ax.set_ylabel("net bp per trade")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(
        0.99,
        0.04,
        "same code, same data — only the timing of one decision differs",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=COLOURS["neutral"],
    )
    return _finish(fig, ax, "Why 'trade only the best signals' looks profitable", Path(path))


def retrain_schedules(
    results: pd.DataFrame,
    path: Path | str,
    *,
    label: str = "",
) -> Path:
    """Net result per trade by retraining schedule, against break-even.

    The chart the retraining sweep exists for. Grouped by how much history each
    fit sees, one line per how long the fit is kept, so the two questions the
    sweep asks are separated: does more history help, and does refitting more
    often help.

    Plotted net rather than gross because the comparison is between schedules
    that trade very different amounts, and gross alone would flatter whichever
    one traded least.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 4.2))

    palette = [COLOURS["signal"], COLOURS["accent"], COLOURS["edge"]]
    for i, (apply_days, group) in enumerate(results.groupby("apply_days")):
        ordered = group.sort_values("train_days")
        ax.plot(
            ordered["train_days"],
            ordered["net_per_trade_bp"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=palette[i % len(palette)],
            label=f"refit every {apply_days}d",
        )

    ax.axhline(0, color=COLOURS["cost"], linewidth=1.8, linestyle="--", label="break-even")
    ax.set_xticks(sorted(results["train_days"].unique()))
    ax.set_xlabel("days of history each fit sees")
    ax.set_ylabel("net bp per trade")
    ax.legend(frameon=False, fontsize=9)
    title = "Retraining schedule against break-even"
    if label:
        title += f" — {label}"
    return _finish(fig, ax, title, Path(path))


def maker_tradeoff(results: pd.DataFrame, path: Path | str) -> Path:
    """Fill rate against adverse selection, as patience is varied.

    The figure behind the maker question. Waiting longer fills more orders, and
    the extra fills are the ones the market had to move to reach — so the two
    axes pull against each other and the chart is the shape of that pull rather
    than a single number.

    Fill rate on x because it is the quantity a strategy actually chooses: how
    much of the signal it is willing to give up.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 4.2))

    palette = [COLOURS["signal"], COLOURS["accent"]]
    for i, (symbol, group) in enumerate(results.groupby("symbol")):
        ordered = group.sort_values("fill_rate")
        ax.plot(
            ordered["fill_rate"] * 100,
            ordered["adverse_selection_bp"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=palette[i % len(palette)],
            label=str(symbol),
        )
        for _, row in ordered.iterrows():
            ax.annotate(
                f"{row['patience_s']:g}s",
                (row["fill_rate"] * 100, row["adverse_selection_bp"]),
                textcoords="offset points",
                xytext=(0, 7),
                fontsize=7.5,
                ha="center",
                color=COLOURS["neutral"],
            )

    # One line per instrument: what posting saves depends on the spread, and a
    # single line would invite reading XRPUSDT's points against BTCUSDT's
    # threshold. The saving is the whole comparison, so it cannot be averaged.
    for i, (symbol, group) in enumerate(results.groupby("symbol")):
        saved = float(group["cost_saved_bp"].iloc[0])
        ax.axhline(
            saved,
            color=palette[i % len(palette)],
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            label=f"{symbol}: cost saved by posting ({saved:.1f} bp)",
        )

    ax.set_xlabel("% of posted orders that filled")
    ax.set_ylabel("adverse selection, bp")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(
        0.99,
        0.04,
        "below its own dashed line, posting pays for its adverse selection",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color=COLOURS["neutral"],
    )
    return _finish(fig, ax, "What posting costs, and what it saves", Path(path))
