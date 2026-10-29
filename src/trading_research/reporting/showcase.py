"""The README's first figure: where a strategy here made money net of fees.

Four panels, each from a committed table or the reversion run's per-trade
file: the futures reversion rule's cumulative net on its held-out fortnight,
the same block by instrument, the exit and sizing models on its entries, and
the second market-making round's quoter at the professional maker tiers. Every
number drawn is net of the fee its study states; what each one does not
establish is said in the table beside it in the README, not on the figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.reporting.plots import DPI, _pyplot

GAIN = "#2da44e"
LOSS = "#cf222e"
LINE = "#2b6cb0"
SECOND = "#bf8700"
MUTED = "#8c959f"

#: The reversion variants shown, in the order drawn, with their short labels.
VARIANTS = {
    "baseline: fixed 10-minute clock": "rule alone, 10-minute exit",
    "control: take profit at 80 bp (chosen on search)": "rule, take profit 80 bp",
    "xgboost: per-trade take-profit level": "XGBoost sets the take profit",
    "lightgbm: hold through trigger when confident": "LightGBM holds when confident",
    "xgboost: hold through trigger when confident": "XGBoost holds when confident",
}

#: The market-making hypotheses shown and their labels.
QUOTER = {
    "B3a": "quoter at the 0 bp maker tier",
    "B3b": "quoter at the \N{MINUS SIGN}1 bp maker rebate",
    "B1": "quoter at the base fee (2 bp)",
}


def _tidy(ax: object, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, pad=8)  # type: ignore[attr-defined]
    ax.spines["top"].set_visible(False)  # type: ignore[attr-defined]
    ax.spines["right"].set_visible(False)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]


def overview(
    trades: pd.DataFrame,
    variants: pd.DataFrame,
    quoter: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Draw the four panels.

    ``trades`` is the reversion run's held-out block (``symbol``, ``net_bp``,
    in entry order); ``variants`` is ``reversion_boost_ALL.csv``; ``quoter``
    is ``mm_round2_daily_H.csv`` (``day``, ``value``, ``hypothesis``).
    """
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))

    ax = axes[0, 0]
    net = trades["net_bp"].to_numpy(dtype="float64")
    equity = np.concatenate([[0.0], np.cumsum(net)])
    ax.fill_between(np.arange(equity.size), equity, 0.0, color=GAIN, alpha=0.15, linewidth=0)
    ax.plot(equity, color=GAIN, linewidth=1.8)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("trade, in entry order")
    ax.set_ylabel("cumulative net, bp")
    ax.text(
        0.02,
        0.93,
        f"{net.size:,} trades · {net.mean():+.2f} bp a trade after fees",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    _tidy(ax, "Futures, taker: cross-sectional reversion, 26 perpetuals, held-out fortnight")

    ax = axes[0, 1]
    by_symbol = trades.groupby("symbol")["net_bp"].mean().sort_values()
    colours = [GAIN if value > 0 else LOSS for value in by_symbol]
    ax.barh([s.removesuffix("USDT") for s in by_symbol.index], by_symbol, color=colours, height=0.7)
    ax.axvline(0.0, color=MUTED, linewidth=0.8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("net per trade, bp")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    _tidy(ax, f"The same block by instrument: {(by_symbol > 0).sum()} of {by_symbol.size} positive")

    ax = axes[1, 0]
    shown = variants.set_index("variant").loc[list(VARIANTS)]
    labels = list(VARIANTS.values())
    values = shown["net_per_trade_bp"].to_numpy(dtype="float64")
    errors = shown["two_se_bp"].to_numpy(dtype="float64")
    colours = [MUTED if i == 0 else LINE for i in range(len(values))]
    ax.barh(labels, values, xerr=errors, color=colours, height=0.6,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3})  # fmt: skip
    for y, value in enumerate(values):
        ax.text(value + errors[y] + 0.4, y, f"{value:+.1f}", va="center", fontsize=9)
    ax.axvline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("net per trade on the held-out block, bp (bar: ±2 s.e.)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    _tidy(ax, "Trees on the same entries: a model decides how long to hold")

    ax = axes[1, 1]
    for (name, label), colour in zip(QUOTER.items(), (LINE, GAIN, MUTED), strict=True):
        series = quoter[quoter["hypothesis"] == name].sort_values("day")
        days = pd.to_datetime(series["day"])
        cumulative = series["value"].cumsum().to_numpy(dtype="float64")
        ax.plot(days, cumulative, marker="o", markersize=3, linewidth=1.8, color=colour,
                label=label)  # fmt: skip
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("cumulative net, USDT per 100 USDT of clip")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    _tidy(ax, "Market making, maker: six altcoin perpetuals, held-out fortnight")

    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=DPI)
    plt.close(fig)
    return target
