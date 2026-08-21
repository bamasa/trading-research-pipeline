"""Every chart the README shows, rendered for both GitHub themes.

The README's images are the first thing a reader sees, and half of GitHub reads
in dark mode, where a white matplotlib canvas is a glaring rectangle. So each
chart here is rendered twice — ``_light`` and ``_dark`` — and the README selects
with a ``<picture>`` block. One script owns all of them so a styling change is a
rerun rather than an archaeology dig.

Every input is a committed CSV under ``experiments/results``, with one
exception: the reversion equity curves need the per-trade table, which is 32 MB
and gitignored. When it is absent the script says which experiment regenerates
it and renders everything else.

    uv run python scripts/render_readme_charts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trading_research.reporting import plots  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
IMAGES = ROOT / "docs" / "images"


def reversion_variants() -> None:
    table = pd.read_csv(RESULTS / "reversion_boost_ALL.csv")
    shown = table[
        (~table["variant"].str.startswith("  "))
        & table["vs_baseline_bp"].notna()
        & (~table["variant"].str.startswith("oracle"))
    ].copy()
    shown["variant"] = shown["variant"].str.replace(": ", " — ", regex=False)
    baseline = float(
        table.loc[table["variant"].str.startswith("baseline"), "net_per_trade_bp"].iloc[0]
    )
    plots.both_themes(
        lambda path: plots.variant_comparison(
            shown,
            path,
            baseline=baseline,
            title="Exit and sizing variants on the held-out block, 26 instruments",
        ),
        IMAGES / "reversion_variants_ALL.png",
    )


def reversion_horizon() -> None:
    table = pd.read_csv(RESULTS / "market_reversion.csv").dropna(
        subset=["search_net_bp", "final_net_bp"]
    )
    profile = (
        table.groupby("hold_s")
        .agg(search=("search_net_bp", "median"), final=("final_net_bp", "median"))
        .reset_index()
    )
    plots.both_themes(
        lambda path: plots.two_block_profile(
            profile,
            path,
            x_column="hold_s",
            search_column="search",
            final_column="final",
            x_label="holding period (seconds, log scale)",
            title="Market reversion: net per trade by holding period",
        ),
        IMAGES / "reversion_horizon_profile.png",
    )


def reversion_equity() -> bool:
    trades = RESULTS / "reversion_trades.csv"
    if not trades.exists():
        return False
    frame = pd.read_csv(trades)
    final = frame[frame["block"] == "final"]
    curves = {"baseline: fixed 10-minute clock": final["net_bp"].to_numpy()}
    plots.both_themes(
        lambda path: plots.equity_curves(
            curves, path, title="Cumulative result on the held-out block"
        ),
        IMAGES / "reversion_equity_ALL.png",
    )
    return True


def rules_against_models() -> None:
    """The §14 chart, rebuilt from the committed per-strategy tables."""
    rows = []
    for file in sorted(RESULTS.glob("strategies_*_BTCUSDT.csv")):
        name = file.stem.removeprefix("strategies_").removesuffix("_BTCUSDT")
        table = pd.read_csv(file)
        if "gross_per_trade_bp" not in table.columns or table.empty:
            continue
        rows.append(
            {
                "variant": name.replace("_", " "),
                "net_per_trade_bp": float(table["gross_per_trade_bp"].iloc[0]),
            }
        )
    table = pd.DataFrame(rows).dropna()
    plots.both_themes(
        lambda path: plots.variant_comparison(
            table,
            path,
            value_column="net_per_trade_bp",
            baseline=0.0,
            baseline_label="break-even, before costs",
            title="Gross edge per trade, BTCUSDT: rules against models",
        ),
        IMAGES / "strategies_btc.png",
    )


def gross_against_net() -> None:
    """The one-figure summary of the negative result, from the best run's trades."""
    trades = pd.read_csv(RESULTS / "best_trades_BTCUSDT.csv")
    gross = trades["gross_bp"].to_numpy()
    net = trades["net_bp"].to_numpy()
    curves = {"gross": gross, "net of costs": net}
    plots.both_themes(
        lambda path: plots.equity_curves(
            curves, path, title="Gross against net: the cost of trading, BTCUSDT"
        ),
        IMAGES / "bt_equity.png",
    )


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    reversion_variants()
    print("variants done")
    reversion_horizon()
    print("horizon done")
    if reversion_equity():
        print("equity done")
    else:
        print(
            "equity skipped: experiments/results/reversion_trades.csv is absent "
            "(32 MB, gitignored) — regenerate with "
            "`uv run python -m experiments.reversion_boost --symbol ALL`"
        )
    rules_against_models()
    print("strategies done")
    gross_against_net()
    print("gross/net done")


if __name__ == "__main__":
    main()
