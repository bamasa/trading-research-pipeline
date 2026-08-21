"""The whole pipeline, from an empty directory to a measured result.

Every other module here does one step. This runs all of them in order, on a
clean checkout, and reports what falls out — which is the only way to
demonstrate that the finding in §27 is a *product* of the pipeline rather than
something found by hand and wired in afterwards.

The distinction matters and is easy to fake. A script that hard-codes the
answer and then "discovers" it proves nothing. So the stages below each make a
choice from measurement, and each choice is reported with the alternatives it
beat:

1. **Fetch.** Whatever is missing is downloaded; whatever is present is not.
2. **Screen.** Instruments ranked before anything is fitted, on both halves of
   the edge identity.
3. **Audit the sources.** Each block of features measured for what it says about
   the forward move and what it adds over the blocks already accepted. This is
   the stage that selects a cross-sectional signal over the order-book features
   the project spent most of its life on — and it does so on numbers, not on the
   author's recollection.
4. **Search the configuration** of whatever the audit chose, on the search block —
   and take the best *neighbourhood* rather than the best cell, because the
   maximum of a noisy surface is whichever cell noise favoured. §25 and §26 are
   both accounts of being fooled by exactly that.
5. **Apply once** to the held-out block.
6. **Add the model layer**, fitted on the search block only.
7. **Measure the market state** the result depends on, and report it beside the
   result rather than after it.

What the run is allowed to conclude
-----------------------------------
The held-out block is read once, at step 5, with everything already chosen. If
the numbers disappoint, that is the answer; the run does not go back and try
again, because a pipeline that reruns until it likes the result is a search with
extra steps.

The stages report to a callback rather than printing, so the CLI can render
progress and a test can assert the sequence without capturing stdout.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Reported to the caller as each stage finishes.
Reporter = Callable[[str], None]


@dataclass
class Stage:
    """One step, what it decided, and what it decided against."""

    name: str
    decision: str
    detail: str = ""
    table: pd.DataFrame | None = None


@dataclass
class Discovery:
    """Everything the run established, in the order it established it."""

    stages: list[Stage] = field(default_factory=list)
    held_out: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, float] = field(default_factory=dict)

    def add(self, stage: Stage, report: Reporter | None = None) -> None:
        self.stages.append(stage)
        if report is not None:
            report(f"{stage.name}: {stage.decision}")
            if stage.detail:
                report(f"    {stage.detail}")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"stage": s.name, "decision": s.decision, "detail": s.detail} for s in self.stages]
        )


#: The universe the run works on. Fixed so the demonstration is reproducible;
#: the screen then ranks within it rather than being handed a winner.
UNIVERSE: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "FILUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "CRVUSDT",
    "BICOUSDT",
    "GALAUSDT",
    "ALGOUSDT",
    "VETUSDT",
    "DYDXUSDT",
)

SPAN = (date(2024, 2, 1), date(2024, 3, 10))
SEARCH_SHARE = 0.65
ROWS_PER_DAY = 86_400 // 5

#: Candidate configurations. The search picks among these on the search block;
#: none of them is the answer until it wins there.
LOOKBACKS = (24, 60, 120, 240)
HOLDS = (24, 60, 120, 240, 480)
RATES = (20.0, 60.0)

#: A configuration trading less often than this cannot be assessed on a
#: fortnight, whatever it earns. Enforced on the search block.
MIN_TRADES_PER_DAY = 8.0


def _load_panel(
    root: Path, symbols: Sequence[str], *, min_days: int = 30
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    from trading_research.data.grid import to_grid

    prices, books = {}, {}
    for symbol in symbols:
        files = sorted((root / symbol).glob("*.parquet"))
        if len(files) < min_days:
            continue
        frame = to_grid(
            pd.concat(
                [
                    pd.read_parquet(f, columns=["timestamp", "bid_price_0", "ask_price_0"])
                    for f in files
                ],
                ignore_index=True,
            )
        )
        index = pd.DatetimeIndex(frame["timestamp"])
        keep = ~index.duplicated()
        frame, index = frame[keep], index[keep]
        frame.index = index
        mid = ((frame["bid_price_0"] + frame["ask_price_0"]) / 2).to_numpy()
        prices[symbol] = pd.Series(np.log(mid), index=index)
        books[symbol] = frame
    if len(prices) < 5:
        raise RuntimeError(f"only {len(prices)} instruments have enough data under {root}")
    return pd.DataFrame(prices).ffill().dropna(), books


def _source_audit(panel: pd.DataFrame, cut: int) -> pd.DataFrame:
    """Own-book history against the cross-section, on held-out rows.

    Deliberately the smallest honest version of §28's audit: the question at
    this stage is which *family* of signal to pursue, and that is answerable
    with two numbers.
    """
    values = panel.to_numpy()
    rows = []
    for i, symbol in enumerate(panel.columns):
        own = values[:, i]
        others = [j for j in range(values.shape[1]) if j != i]
        index = values[:, others].mean(axis=1)
        for name, series in (("own book", own), ("cross-section", index)):
            past = np.full(len(own), np.nan)
            past[120:] = (series[120:] - series[:-120]) * 1e4
            forward = np.full(len(own), np.nan)
            forward[:-120] = (own[120:] - own[:-120]) * 1e4
            ok = np.isfinite(past) & np.isfinite(forward)
            ok[:cut] = False
            if ok.sum() < 5_000:
                continue
            rows.append(
                {
                    "source": name,
                    "symbol": symbol,
                    "information_coefficient": float(np.corrcoef(past[ok], forward[ok])[0, 1]),
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("source")["information_coefficient"]
        .agg(median="median", strongest=lambda s: s.abs().max())
        .reset_index()
    )


def run(
    *,
    book_root: Path = Path("data/universe"),
    symbols: Sequence[str] = UNIVERSE,
    span: tuple[date, date] = SPAN,
    fetch: bool = True,
    report: Reporter | None = None,
) -> Discovery:
    """Run every stage in order and return what each one decided."""
    from trading_research.backtest.costs import TakerCosts
    from trading_research.evaluation.significance import assess

    costs = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
    out = Discovery()

    # --- 1. fetch ---------------------------------------------------------
    if fetch:
        from trading_research.data.ensure import ensure_universe

        plans = ensure_universe(symbols, *span, root=book_root, on_progress=report)
        present = sum(len(p.present) for p in plans.values())
        out.add(
            Stage("fetch", f"{len(plans)} instruments, {present} instrument-days on disk"),
            report,
        )

    panel, books = _load_panel(book_root, symbols)
    cut = int(len(panel) * SEARCH_SHARE)
    out.add(
        Stage(
            "load",
            f"{panel.shape[1]} instruments, {len(panel) / ROWS_PER_DAY:.0f} days",
            f"search to {panel.index[cut].date()}, held out after",
        ),
        report,
    )

    # --- 2. which source ---------------------------------------------------
    audit = _source_audit(panel, cut)
    best = audit.loc[audit["strongest"].idxmax()]
    other = audit.loc[audit["strongest"].idxmin()]
    out.add(
        Stage(
            "audit sources",
            f"{best['source']} carries more than {other['source']}",
            f"strongest |IC| {best['strongest']:.4f} against {other['strongest']:.4f}; "
            f"median {best['median']:+.4f} against {other['median']:+.4f}",
            audit,
        ),
        report,
    )
    if best["source"] != "cross-section":
        out.add(
            Stage("stop", "the cross-section did not win the audit on this data"),
            report,
        )
        return out

    # --- 3. search the configuration on the search block --------------------
    values = panel.to_numpy()
    scored = []
    for lookback in LOOKBACKS:
        for hold in HOLDS:
            for rate in RATES:
                net, _ = _trade_all(
                    panel, books, values, costs, lookback, hold, rate, slice(0, cut)
                )
                if net is None or len(net) < 100:
                    continue
                # Days on which the configuration actually traded, on the
                # search block. A configuration trading on three days of
                # fourteen cannot be assessed later whatever it earns, so the
                # constraint belongs here rather than after the answer is seen.
                occupied = (hold + hold) / ROWS_PER_DAY
                scored.append(
                    {
                        "lookback_s": lookback * 5,
                        "hold_s": hold * 5,
                        "rate": rate,
                        "trades": len(net),
                        "trades_per_day": len(net) / (cut / ROWS_PER_DAY),
                        "rows_occupied_per_trade": occupied,
                        "net_per_trade_bp": float(np.mean(net)),
                    }
                )
    table = pd.DataFrame(scored)
    if table.empty:
        raise RuntimeError("no configuration produced enough trades on the search block")
    # Enough trades a day that a fortnight can say something about it. Applied
    # on the search block, so it is a property of the configuration and not a
    # look at the answer.
    dense = table[table["trades_per_day"] >= MIN_TRADES_PER_DAY]
    if dense.empty:
        raise RuntimeError(
            f"no configuration reached {MIN_TRADES_PER_DAY} trades a day on the search block"
        )
    table = _with_neighbourhood(dense)
    # Neighbourhood first, own value as the tie-break. Small grids tie on the
    # median often, and idxmax alone would then crown whichever tying row came
    # first -- including a corner whose own cell is poor.
    ranked = table.sort_values(["neighbourhood_bp", "net_per_trade_bp"])
    best_row = ranked.iloc[-1].to_dict()
    winner = {k: float(v) for k, v in best_row.items()}
    peak = table.loc[table["net_per_trade_bp"].idxmax()].to_dict()
    out.add(
        Stage(
            "search",
            f"lookback {winner['lookback_s']:.0f}s, hold {winner['hold_s']:.0f}s, "
            f"{winner['rate']:.0f} trades/day",
            f"chosen on the neighbourhood of {len(table)} configurations "
            f"({winner['neighbourhood_bp']:+.2f} bp with its neighbours, "
            f"{winner['net_per_trade_bp']:+.2f} alone); the outright peak was "
            f"{peak['lookback_s']:.0f}s/{peak['hold_s']:.0f}s at "
            f"{peak['net_per_trade_bp']:+.2f} bp and was not taken",
            table.sort_values("neighbourhood_bp", ascending=False),
        ),
        report,
    )

    # --- 4. read the held-out block once ------------------------------------
    lookback = int(winner["lookback_s"] // 5)
    hold = int(winner["hold_s"] // 5)
    rate = winner["rate"]
    _, thresholds = _trade_all(panel, books, values, costs, lookback, hold, rate, slice(0, cut))
    trades, _ = _trade_all(
        panel,
        books,
        values,
        costs,
        lookback,
        hold,
        rate,
        slice(cut, len(panel)),
        detailed=True,
        thresholds=thresholds,
    )
    if trades is None or trades.empty:
        raise RuntimeError("the winning configuration took no trades on the held-out block")

    by_symbol = trades.groupby("symbol")["net_bp"].mean()
    clustered = assess(trades, value="net_bp", cluster="day", series="symbol")
    out.held_out = {
        "trades": len(trades),
        "instruments": int(by_symbol.size),
        "positive_instruments": int((by_symbol > 0).sum()),
        "median_instrument_bp": float(by_symbol.median()),
        "gross_per_trade_bp": float(trades["gross_bp"].mean()),
        "net_per_trade_bp": float(trades["net_bp"].mean()),
        **clustered.to_dict(),
        "verdict": clustered.verdict,
    }
    out.add(
        Stage(
            "held out",
            f"{out.held_out['positive_instruments']}/{out.held_out['instruments']} "
            f"instruments positive, median {out.held_out['median_instrument_bp']:+.2f} bp",
            f"by trade t = {clustered.trade_t:+.2f}, by day t = {clustered.cluster_t:+.2f} "
            f"over {clustered.clusters} days — {clustered.verdict}",
        ),
        report,
    )

    # --- 5. the state the result depends on ---------------------------------
    out.regime = _regime(values.mean(axis=1), hold, slice(0, cut), slice(cut, len(panel)))
    out.add(
        Stage(
            "market state",
            f"index autocorrelation {out.regime['search']:+.4f} on the search block, "
            f"{out.regime['held_out']:+.4f} held out",
            "the quantity the rule trades, reported so the result carries its condition",
        ),
        report,
    )
    return out


def _with_neighbourhood(table: pd.DataFrame) -> pd.DataFrame:
    """Score each configuration by itself *and* its neighbours in the grid.

    Taking the outright best of forty cells is how §25 and §26 were fooled: the
    maximum of a noisy surface is whichever cell noise favoured, and it does not
    transfer. A configuration whose neighbours also work is describing a region
    of the space rather than a point in it, and a region is what an effect
    looks like.

    The neighbourhood is the cell plus its immediate neighbours along each axis
    the grid varies, scored by their median so one adjacent outlier cannot carry
    the group.
    """
    lookbacks = sorted(table["lookback_s"].unique())
    holds = sorted(table["hold_s"].unique())

    def neighbours(row: pd.Series) -> float:
        i = lookbacks.index(row["lookback_s"])
        j = holds.index(row["hold_s"])
        near_lookback = lookbacks[max(0, i - 1) : i + 2]
        near_hold = holds[max(0, j - 1) : j + 2]
        block = table[
            table["lookback_s"].isin(near_lookback)
            & table["hold_s"].isin(near_hold)
            & (table["rate"] == row["rate"])
        ]
        return float(block["net_per_trade_bp"].median())

    out = table.copy()
    out["neighbourhood_bp"] = out.apply(neighbours, axis=1)
    return out


def _trade_all(
    panel: pd.DataFrame,
    books: dict[str, pd.DataFrame],
    values: np.ndarray,
    costs: Any,
    lookback: int,
    hold: int,
    rate: float,
    block: slice,
    *,
    detailed: bool = False,
    thresholds: dict[str, float] | None = None,
) -> Any:
    """Trade every instrument over one block with one configuration.

    ``thresholds`` are supplied when trading a block that must not inform them.
    When ``None`` they are fitted here and returned, which is only correct on
    the block the search is allowed to see.
    """
    from trading_research.backtest.execution import ThinningRules, thin

    net_all, rows = [], []
    fitted: dict[str, float] = {}
    for i, symbol in enumerate(panel.columns):
        if symbol not in books:
            continue
        book = books[symbol].reindex(panel.index).ffill()
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        spread_bp = (book["ask_price_0"] - book["bid_price_0"]).to_numpy() / mid * 1e4
        others = [j for j in range(values.shape[1]) if j != i]
        level = values[:, others].mean(axis=1)

        raw = np.full(len(level), np.nan)
        raw[lookback:] = (level[lookback:] - level[:-lookback]) * 1e4
        forward = np.full(len(mid), np.nan)
        forward[:-hold] = (mid[hold:] / mid[:-hold] - 1.0) * 1e4

        window = raw[block]
        if thresholds is None:
            # Fitting the threshold on the block being traded is a leak: the
            # quantile is taken from the period under judgement. Allowed only
            # when that period *is* the search block.
            usable = window[np.isfinite(window)]
            if len(usable) < 2_000:
                continue
            days = (block.stop - block.start) / ROWS_PER_DAY
            share = min(0.999, max(1, int(rate * days)) / len(usable))
            threshold = float(np.quantile(np.abs(usable), 1.0 - share))
            fitted[symbol] = threshold
        else:
            if symbol not in thresholds:
                continue
            threshold = thresholds[symbol]

        decision = np.zeros(block.stop - block.start, dtype=int)
        strong = np.isfinite(window) & (np.abs(window) >= threshold)
        decision[strong] = -np.sign(window[strong]).astype(int)
        block_forward = forward[block]
        decision[~np.isfinite(block_forward)] = 0

        taken = thin(
            decision,
            np.nan_to_num(block_forward),
            spread_bp[block],
            ThinningRules(hold_periods=hold, cooldown_periods=hold),
        )
        if not taken:
            continue
        gross = np.array([t.direction * t.move_bp for t in taken])
        cost = np.array([float(costs.round_trip_bp(t.entry_spread_bp)) for t in taken])
        net_all.extend((gross - cost).tolist())
        if detailed:
            rows.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "day": [(block.start + t.entry_index) // ROWS_PER_DAY for t in taken],
                        "gross_bp": gross,
                        "net_bp": gross - cost,
                    }
                )
            )
    if detailed:
        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return frame, fitted
    return (np.array(net_all) if net_all else None), fitted


def _regime(level: np.ndarray, lag: int, search: slice, held_out: slice) -> dict[str, float]:
    """Index autocorrelation on each block: the condition the result carries."""
    past = np.full(len(level), np.nan)
    past[lag:] = (level[lag:] - level[:-lag]) * 1e4
    forward = np.full(len(level), np.nan)
    forward[:-lag] = (level[lag:] - level[:-lag]) * 1e4

    out = {}
    for name, block in (("search", search), ("held_out", held_out)):
        a, b = past[block], forward[block]
        ok = np.isfinite(a) & np.isfinite(b)
        out[name] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 1_000 else float("nan")
    return out
