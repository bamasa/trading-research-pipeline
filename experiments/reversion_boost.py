"""Can a model improve a rule that already works?

The reversion rule of :mod:`trading_research.strategies.reversion` is the one
thing in this project that made money on a block it had not seen. It is also
completely undiscriminating: it enters whenever the index has moved far enough,
holds for exactly ten minutes, and stakes the same amount every time. Three
decisions, none of them informed by anything.

That is the natural place for a model — not to replace the rule, but to answer
the questions the rule does not ask. This is meta-labelling in López de Prado's
sense: the rule picks the side, the model decides whether the trade is worth
taking, and the two are trained separately because they are answering different
questions on different sample sizes.

Where the money might be
------------------------
Before fitting anything, the rule's own trades are decomposed. For each one the
path is recorded, so four distinct failures can be told apart:

* **Bad entries.** Trades that were never in front. A filter fixes these.
* **Late exits.** Trades that were in front and gave it back. The clock is
  wrong, not the signal.
* **Early exits.** Trades still improving when the clock closed them.
* **Flat sizing.** Trades that were right and small next to trades that were
  wrong and equally large.

The diagnosis says which of the four is worth attacking, and the answer decides
which targets are worth training.

Five targets
------------
1. ``profitable`` — will this trade clear its cost? The classic meta-label.
2. ``net_bp`` — how much will it make? Regression, for sizing rather than
   filtering.
3. ``best_hold`` — how many rows until the position peaks? Turns the fixed
   clock into a predicted one.
4. ``max_adverse`` — how far will it go against us first? A stop that is placed
   rather than guessed.
5. ``reverts_fully`` — will the index deviation close? The mechanism the rule
   is betting on, asked directly.

Protocol
--------
Trades are generated on all 26 instruments across the whole span. Models train
on the search block only — every instrument, which is what makes the sample
large enough to fit on — and are applied to DOGEUSDT's held-out block, which
nothing in the fitting has seen. The baseline is the unmodified rule on the same
trades, so every number is a difference against something that already works.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.information_audit import to_grid
from trading_research.backtest.costs import TakerCosts
from trading_research.data.ensure import ensure_universe
from trading_research.strategies.reversion import (
    ReversionConfig,
    index_level,
    signal,
    threshold_for_rate,
)

COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
ROWS_PER_DAY = 86_400 // 5
SEARCH_SHARE = 0.65

#: How far past the nominal hold the path is recorded, so "we exited too early"
#: is answerable rather than assumed.
LOOKAHEAD = 360

#: Every row of the path is stored, not a one-minute grid. The grid version was
#: wrong in a way that flattered the result: a take-profit was booked at
#: whatever the path happened to be worth at the next checkpoint, so a trade
#: that touched the level and ran to 150 bp within the minute was credited 150.
#: A limit order fills at its level. Full paths make that exact.
CHECKPOINTS = tuple(range(LOOKAHEAD + 1))


#: What this experiment needs. Stated here so a clean checkout fetches it
#: rather than failing on a path, and so a reader can see the cost before
#: starting: 26 instruments over 39 days of top-of-book, about eleven seconds
#: each.
UNIVERSE = (
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


def load_panel(root: Path, *, min_days: int = 35):
    prices, books = {}, {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(directory.glob("*.parquet"))
        if len(files) < min_days:
            continue
        frame = to_grid(
            pd.concat(
                [
                    pd.read_parquet(
                        f,
                        columns=[
                            "timestamp",
                            "bid_price_0",
                            "ask_price_0",
                            "bid_size_0",
                            "ask_size_0",
                        ],
                    )
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
        prices[directory.name] = pd.Series(np.log(mid), index=index)
        books[directory.name] = frame
    panel = pd.DataFrame(prices).ffill().dropna()
    return panel, books


def make_trades(
    panel: pd.DataFrame, books: dict, config: ReversionConfig, cut: int
) -> pd.DataFrame:
    """Every trade the rule would take, with its features and its whole path."""
    names = list(panel.columns)
    values = panel.to_numpy()
    rows = []

    for i, symbol in enumerate(names):
        book = books[symbol].reindex(panel.index).ffill()
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        spread_bp = (book["ask_price_0"] - book["bid_price_0"]).to_numpy() / mid * 1e4
        bid_size = book["bid_size_0"].to_numpy()
        ask_size = book["ask_size_0"].to_numpy()

        level = index_level(values, exclude=i)
        raw = signal(level, config)
        # The threshold comes from the search block for both blocks, so the
        # held-out trades are the ones a live run would have taken.
        threshold = threshold_for_rate(raw[:cut], ROWS_PER_DAY, config)

        own = values[:, i]
        own_return = np.full(len(own), np.nan)
        own_return[config.lookback :] = (own[config.lookback :] - own[: -config.lookback]) * 1e4
        dispersion = np.full(len(own), np.nan)
        step = np.diff(values, axis=0, prepend=values[:1])
        dispersion[config.lookback :] = (
            pd.DataFrame(step)
            .rolling(config.lookback)
            .std()
            .mean(axis=1)
            .to_numpy()[config.lookback :]
            * 1e4
        )
        realised = pd.Series(np.diff(own, prepend=own[0]) * 1e4).rolling(120).std().to_numpy()

        free_until = -1
        for t in range(config.lookback, len(own) - LOOKAHEAD):
            if t < free_until or not np.isfinite(raw[t]) or abs(raw[t]) < threshold:
                continue
            direction = int(-np.sign(raw[t]))
            if direction == 0:
                continue
            path = (mid[t : t + LOOKAHEAD + 1] / mid[t] - 1.0) * 1e4 * direction
            cost = float(COSTS.round_trip_bp(spread_bp[t]))
            held = path[config.hold]
            checkpoints = {f"p{step}": float(path[step]) for step in CHECKPOINTS}
            rows.append(
                {
                    "symbol": symbol,
                    "at": t,
                    "block": "search" if t < cut else "final",
                    "direction": direction,
                    # --- features, all known at entry ---
                    "signal_bp": abs(raw[t]),
                    "signal_over_threshold": abs(raw[t]) / threshold,
                    "own_return_bp": own_return[t] * direction,
                    "own_vs_index": (own_return[t] - raw[t]) * direction,
                    "dispersion_bp": dispersion[t],
                    "realised_vol_bp": realised[t],
                    "spread_bp": spread_bp[t],
                    "queue_imbalance": (bid_size[t] - ask_size[t])
                    / (bid_size[t] + ask_size[t])
                    * direction,
                    "hour": panel.index[t].hour,
                    "cost_bp": cost,
                    # --- outcomes ---
                    "gross_bp": held,
                    "net_bp": held - cost,
                    "best_bp": float(np.max(path[: config.hold + 1])),
                    "worst_bp": float(np.min(path[: config.hold + 1])),
                    "best_hold": int(np.argmax(path[: config.hold + 1])),
                    "best_extended_bp": float(np.max(path)),
                    "best_extended_hold": int(np.argmax(path)),
                    "gross_at_2x": float(path[min(2 * config.hold, LOOKAHEAD)]),
                    "gross_at_half": float(path[config.hold // 2]),
                    **checkpoints,
                }
            )
            free_until = t + config.hold + config.cooldown
    return pd.DataFrame(rows)


FEATURES = [
    "signal_over_threshold",
    "own_return_bp",
    "own_vs_index",
    "dispersion_bp",
    "realised_vol_bp",
    "spread_bp",
    "queue_imbalance",
    "hour",
]


def diagnose(trades: pd.DataFrame, config: ReversionConfig) -> pd.DataFrame:
    """Where the rule's money goes, before anything is fitted."""
    rows = []
    for label, part in (
        ("search", trades[trades.block == "search"]),
        ("final", trades[trades.block == "final"]),
    ):
        if part.empty:
            continue
        net = part["net_bp"]
        winners = part[net > 0]
        losers = part[net <= 0]
        rows.append(
            {
                "block": label,
                "trades": len(part),
                "net_bp": float(net.mean()),
                "share_winning": float((net > 0).mean()),
                "mean_win_bp": float(winners["net_bp"].mean()) if len(winners) else np.nan,
                "mean_loss_bp": float(losers["net_bp"].mean()) if len(losers) else np.nan,
                # Was the trade ever in front? A trade that never was is an
                # entry problem; one that was and gave it back is an exit
                # problem, and they need different fixes.
                "never_in_front": float((part["best_bp"] <= part["cost_bp"]).mean()),
                "gave_back_bp": float((part["best_bp"] - part["gross_bp"]).mean()),
                "mean_best_hold": float(part["best_hold"].mean()),
                "still_improving": float((part["best_extended_hold"] > config.hold).mean()),
                # What a perfect exit inside the window would have earned, and
                # what one twice as long would have.
                "oracle_exit_net_bp": float((part["best_bp"] - part["cost_bp"]).mean()),
                "double_hold_net_bp": float((part["gross_at_2x"] - part["cost_bp"]).mean()),
                "half_hold_net_bp": float((part["gross_at_half"] - part["cost_bp"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/universe"))
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--models", nargs="+", default=["xgboost", "lightgbm", "tcn"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    config = ReversionConfig()
    ensure_universe(UNIVERSE, *SPAN, root=args.universe)
    panel, books = load_panel(args.universe)
    cut = int(len(panel) * SEARCH_SHARE)
    print(
        f"{panel.shape[1]} instruments, {len(panel):,} rows, "
        f"search to {panel.index[cut].date()}, held out after"
    )

    trades = make_trades(panel, books, config, cut)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Gitignored: 32 MB of per-trade paths, regenerable by re-running this.
    trades.to_csv(RESULTS / "reversion_trades.csv", index=False)
    print(f"{len(trades):,} trades across all instruments")

    focus = trades[trades.symbol == args.symbol]
    emit(diagnose(focus, config).assign(symbol=args.symbol), f"reversion_diagnosis_{args.symbol}")
    emit(diagnose(trades, config).assign(symbol="all"), "reversion_diagnosis_all")

    # Models train on every instrument's search block -- 1,259 trades, enough to
    # fit on -- and are applied to one instrument's held-out block, which
    # nothing in the fitting has seen.
    scope = (
        trades
        if args.symbol.upper() == "ALL"
        else trades[(trades.block == "search") | (trades.symbol == args.symbol)]
    )
    print(
        f"\ntraining on {int((scope.block == 'search').sum())} trades, "
        f"testing on {int((scope.block == 'final').sum())} {args.symbol} trades"
    )
    table = evaluate(scope, config, tuple(args.models))
    table.insert(0, "symbol", args.symbol)
    emit(table, f"reversion_boost_{args.symbol}")


# ---------------------------------------------------------------------------
# The model layer
# ---------------------------------------------------------------------------

#: Path checkpoints, as column names.
PATH_COLUMNS = [f"p{step}" for step in CHECKPOINTS]


def path_of(row: pd.Series) -> np.ndarray:
    return row[PATH_COLUMNS].to_numpy(dtype="float64")


def paths_of(frame: pd.DataFrame) -> np.ndarray:
    """Every trade's path as one array.

    Pulled out once rather than per row. Reading 361 columns through
    ``iterrows`` for each of several thousand trades, for each of seven levels,
    for each of two exit rules, turned a second of arithmetic into hours of
    pandas indexing.
    """
    return frame[PATH_COLUMNS].to_numpy(dtype="float64")


def take_profit_net(paths: np.ndarray, cost: np.ndarray, level: float, hold: int) -> np.ndarray:
    """A limit order at ``level``: it fills at its price, or the clock closes it.

    Booking the level rather than wherever the path had reached is the whole
    point. The first version of this credited a trade that touched 80 bp and ran
    to 150 within the minute with 150, which no resting order would have earned.
    """
    reached = (paths >= level).any(axis=1)
    return np.where(reached, level, paths[:, hold]) - cost


def trailing_net(paths: np.ndarray, cost: np.ndarray, give_back: float, hold: int) -> np.ndarray:
    """A stop that follows the peak down by ``give_back`` and fills at the stop."""
    window = paths[:, : hold + 1]
    peak = np.maximum.accumulate(window, axis=1)
    triggered = (peak > 0) & (window <= peak - give_back)
    fired = triggered.any(axis=1)
    first = triggered.argmax(axis=1)
    at_stop = peak[np.arange(len(paths)), first] - give_back
    return np.where(fired, at_stop, paths[:, hold]) - cost


def exit_at(row: pd.Series, step: float) -> float:
    """Net result of closing at ``step`` rows."""
    path = path_of(row)
    index = int(np.clip(float(step), 0, len(path) - 1))
    return float(path[index]) - float(row["cost_bp"])


def targets(trades: pd.DataFrame) -> pd.DataFrame:
    """The five things a model could usefully predict about a rule's trade."""
    out = pd.DataFrame(index=trades.index)
    out["profitable"] = (trades["net_bp"] > 0).astype(int)
    out["net_bp"] = trades["net_bp"]
    # Where the peak was, capped at the checkpoint grid. Predicting this turns
    # the fixed clock into a predicted one, which the diagnosis says is where
    # the money is: trades give back 51 to 86 bp from their peak.
    out["best_hold"] = trades["best_extended_hold"].clip(12, CHECKPOINTS[-1])
    out["max_adverse"] = -trades["worst_bp"]
    # Did the position clear its cost at any point? A trade that never did is an
    # entry to filter, not an exit to time.
    out["ever_in_front"] = (trades["best_bp"] > trades["cost_bp"]).astype(int)
    # How far the trade runs in total, for setting a take-profit per trade.
    out["best_extended"] = trades["best_extended_bp"]
    # Once a trade is 40 bp in front, does it go on to make half as much again?
    # This is what the "hold through the trigger" nudge needs answered.
    out["runs_further"] = (trades["best_extended_bp"] > 60.0).astype(int)
    return out


def fit_predict(name: str, train_x, train_y, test_x, *, classify: bool):
    from trading_research.models.forests import LightGBMBaseline
    from trading_research.pipeline.stages import build_model

    if name == "tcn":
        from trading_research.models.tcn import TCNBaseline

        model = TCNBaseline(window=16, channels=16, levels=3, max_epochs=12, batch_size=128)
        labels = np.sign(train_y.to_numpy()).astype(int) if not classify else train_y
        model.fit(train_x, pd.Series(labels, index=train_x.index))
        proba = model.predict_proba(test_x)
        return proba[:, 2] - proba[:, 0]

    if classify:
        model = build_model(name) if name != "lightgbm" else LightGBMBaseline()
        model.fit(train_x, pd.Series(train_y.to_numpy(), index=train_x.index))
        proba = model.predict_proba(test_x)
        return proba[:, 2] - proba[:, 0]

    from sklearn.ensemble import GradientBoostingRegressor
    from xgboost import XGBRegressor

    regressor = (
        XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, n_jobs=-1)
        if name == "xgboost"
        else GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    )
    regressor.fit(train_x, train_y)
    return regressor.predict(test_x)


def evaluate(
    trades: pd.DataFrame, config: ReversionConfig, models: tuple[str, ...]
) -> pd.DataFrame:
    """Baseline against every model-improved variant, on the held-out block."""
    train = trades[trades.block == "search"].dropna(subset=FEATURES)
    test = trades[trades.block == "final"].dropna(subset=FEATURES)
    if len(train) < 200 or len(test) < 30:
        raise SystemExit(f"{len(train)} training and {len(test)} test trades is not enough")

    y_train = targets(train)
    train_x, test_x = train[FEATURES], test[FEATURES]

    def paired(net: np.ndarray) -> dict[str, float]:
        """Compare against the baseline on the same trades.

        The variants trade the same 113 entries and differ only in how they
        exit or size, so their results are paired. An unpaired comparison of two
        means with a common component is far less powerful than it needs to be,
        and would call a real improvement noise.
        """
        if len(net) != len(baseline):
            return {}
        difference = net - baseline
        error = float(np.std(difference, ddof=1) / np.sqrt(len(difference)))
        return {
            "vs_baseline_bp": float(np.mean(difference)),
            "paired_t": float(np.mean(difference) / error) if error > 0 else float("nan"),
        }

    def summarise(label: str, net: np.ndarray, taken: int) -> dict:
        if taken == 0:
            return {"variant": label, "trades": 0, "net_per_trade_bp": np.nan, "total_bp": 0.0}
        equity = np.cumsum(net)
        peak = np.maximum.accumulate(np.maximum(equity, 0.0))
        return {
            "variant": label,
            "trades": int(taken),
            "net_per_trade_bp": float(np.mean(net)),
            "total_bp": float(np.sum(net)),
            "hit_rate": float(np.mean(net > 0)),
            "max_drawdown_bp": float(np.max(peak - equity)),
            "two_se_bp": float(2 * np.std(net) / np.sqrt(len(net))),
            **paired(net),
        }

    train_paths, test_paths = paths_of(train), paths_of(test)
    train_cost = train["cost_bp"].to_numpy()
    test_cost = test["cost_bp"].to_numpy()

    baseline = test["net_bp"].to_numpy()
    rows = [summarise("baseline: fixed 10-minute clock", baseline, len(baseline))]

    # A ceiling, so every improvement can be read as a share of what was there.
    oracle = (test["best_extended_bp"] - test["cost_bp"]).to_numpy()
    rows.append(summarise("oracle: perfect exit (not achievable)", oracle, len(oracle)))

    for name in models:
        # --- 1. predicted exit time ---
        try:
            predicted_hold = fit_predict(
                name, train_x, y_train["best_hold"], test_x, classify=False
            )
            steps = np.clip(predicted_hold, 12, test_paths.shape[1] - 1).astype(int)
            net = test_paths[np.arange(len(test_paths)), steps] - test_cost
            rows.append(summarise(f"{name}: predicted exit time", net, len(net)))
        except Exception as exc:
            rows.append(
                {"variant": f"{name}: predicted exit time", "trades": 0, "note": str(exc)[:60]}
            )

        # --- 2. meta-label filter ---
        try:
            score = fit_predict(name, train_x, y_train["profitable"], test_x, classify=True)
            keep = score > np.quantile(score, 0.4)
            rows.append(summarise(f"{name}: filtered entries", baseline[keep], int(keep.sum())))
        except Exception as exc:
            rows.append(
                {"variant": f"{name}: filtered entries", "trades": 0, "note": str(exc)[:60]}
            )

        # --- 3. size by predicted profit ---
        try:
            edge = fit_predict(name, train_x, y_train["net_bp"], test_x, classify=False)
            scale = np.clip(edge / max(np.std(edge), 1e-9), 0.0, 2.0)
            rows.append(
                summarise(f"{name}: sized by predicted edge", baseline * scale, len(baseline))
            )
        except Exception as exc:
            rows.append(
                {"variant": f"{name}: sized by predicted edge", "trades": 0, "note": str(exc)[:60]}
            )

        # --- 4. filter and predicted exit together ---
        try:
            score = fit_predict(name, train_x, y_train["profitable"], test_x, classify=True)
            keep = score > np.quantile(score, 0.4)
            held = fit_predict(name, train_x, y_train["best_hold"], test_x, classify=False)
            steps = np.clip(held, 12, test_paths.shape[1] - 1).astype(int)
            net = test_paths[np.arange(len(test_paths)), steps] - test_cost
            rows.append(summarise(f"{name}: filtered + predicted exit", net[keep], int(keep.sum())))
        except Exception as exc:
            rows.append(
                {
                    "variant": f"{name}: filtered + predicted exit",
                    "trades": 0,
                    "note": str(exc)[:60],
                }
            )

    # Fixed exit rules, as the controls a model has to beat. Their levels are
    # chosen on the search block and applied unchanged, because picking the best
    # of several levels by looking at the held-out result is the selection
    # mistake this project keeps catching itself making.
    def run_take_profit(frame: pd.DataFrame, level: float) -> np.ndarray:
        paths = train_paths if frame is train else test_paths
        cost = train_cost if frame is train else test_cost
        return take_profit_net(paths, cost, level, config.hold)

    def run_trailing(frame: pd.DataFrame, give_back: float) -> np.ndarray:
        paths = train_paths if frame is train else test_paths
        cost = train_cost if frame is train else test_cost
        return trailing_net(paths, cost, give_back, config.hold)

    LEVELS = (10.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0)

    # The combination worth testing: let the rule take the profit, and let the
    # model decide which trades to open at all. They address different failures
    # -- the diagnosis found 11% of trades never in front (an entry problem) and
    # 86 bp given back from the peak (an exit problem) -- so they should compose
    # rather than compete.
    def take_profit_curve(frame: pd.DataFrame, level: float) -> np.ndarray:
        return run_take_profit(frame, level)

    best_level = max(LEVELS, key=lambda level: float(np.mean(take_profit_curve(train, level))))
    for name in models:
        try:
            score = fit_predict(name, train_x, y_train["profitable"], test_x, classify=True)
            keep = score > np.quantile(score, 0.4)
            combined = take_profit_curve(test, best_level)[keep]
            rows.append(
                summarise(
                    f"{name}: filtered + take profit at {best_level:g} bp",
                    combined,
                    int(keep.sum()),
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "variant": f"{name}: filtered + take profit",
                    "trades": 0,
                    "note": str(exc)[:60],
                }
            )

    for label, run in (("take profit", run_take_profit), ("trailing stop", run_trailing)):
        on_search = {level: float(np.mean(run(train, level))) for level in LEVELS}
        chosen = max(on_search, key=lambda level: on_search[level])
        result = summarise(
            f"control: {label} at {chosen:g} bp (chosen on search)",
            run(test, chosen),
            len(test),
        )
        result["search_net_bp"] = on_search[chosen]
        rows.append(result)
        # The whole curve, so a reader can see whether the choice sat on a peak
        # or on a cliff.
        for level in LEVELS:
            rows.append(
                {
                    "variant": f"  {label} at {level:g} bp",
                    "trades": len(test),
                    "net_per_trade_bp": float(np.mean(run(test, level))),
                    "search_net_bp": on_search[level],
                }
            )

    for name in models:
        for label, net in modulated(
            train, test, y_train, FEATURES, best_level, config.hold, name
        ).items():
            rows.append(summarise(f"{name}: {label}", net, len(net)))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The model as a modulator, not a replacement
# ---------------------------------------------------------------------------


def modulated(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.DataFrame,
    features: list[str],
    level: float,
    hold: int,
    name: str,
) -> dict[str, np.ndarray]:
    """Let the rule run the exit and the model adjust it at the margin.

    Asking a model to name the exit row outright failed on all three
    architectures — the peak moves too much between trades to be predicted from
    eight features. Asking it to *nudge* a rule that already works is a smaller
    question with more sample behind it, and three nudges are worth separating:

    * **Per-trade level.** Predict how far this trade will run and set its
      take-profit from that, so a trade the model expects little from books
      early and one it expects a lot from is given room.
    * **Override the trigger.** When the level is touched, ask whether the move
      has further to go. If the model is confident it has, hold; otherwise book.
    * **Cut the dead ones.** Trades whose level is never touched currently run
      the full clock. Ask, part-way through, whether this one is going anywhere,
      and close it early if not.

    Every prediction is made from features known at entry, and every model is
    fitted on the search block only.
    """
    train_x, test_x = train[features], test[features]
    out: dict[str, np.ndarray] = {}

    paths = paths_of(test)
    cost = test["cost_bp"].to_numpy()

    # --- per-trade take-profit level -------------------------------------
    try:
        reach = fit_predict(name, train_x, y_train["best_extended"], test_x, classify=False)
        # Book a fraction of what the model thinks the trade will reach, floored
        # and capped so a wild prediction cannot set a level nobody would use.
        levels = np.clip(0.7 * reach, 20.0, 200.0)
        reached = (paths >= levels[:, None]).any(axis=1)
        out["per-trade take-profit level"] = np.where(reached, levels, paths[:, hold]) - cost
    except Exception:
        pass

    # --- hold through the trigger when the model expects more -------------
    try:
        more = fit_predict(name, train_x, y_train["runs_further"], test_x, classify=True)
        confident = more > np.quantile(more, 0.75)
        hit_mask = paths >= level
        reached = hit_mask.any(axis=1)
        first = hit_mask.argmax(axis=1)

        # Everything that does not extend books the level, or the clock.
        gross = np.where(reached, level, paths[:, hold])
        # Those that do extend run a trailing stop from the trigger onward, so
        # the extension has a way to end that is not hindsight.
        give_back = 0.25 * level
        for i in np.flatnonzero(confident & reached):
            rest = paths[i, first[i] :]
            peak = np.maximum.accumulate(rest)
            stopped = np.flatnonzero(rest <= peak - give_back)
            gross[i] = peak[stopped[0]] - give_back if len(stopped) else float(rest[-1])
        out["hold through trigger when confident"] = gross - cost
    except Exception:
        pass

    # --- close the dead ones early ----------------------------------------
    try:
        alive = fit_predict(name, train_x, y_train["ever_in_front"], test_x, classify=True)
        dead = alive < np.quantile(alive, 0.3)
        reached = (paths >= level).any(axis=1)
        # Out at a third of the clock rather than riding a dead trade to the end.
        early = paths[:, hold // 3]
        gross = np.where(reached, level, np.where(dead, early, paths[:, hold]))
        out["cut the dead ones early"] = gross - cost
    except Exception:
        pass

    return out


if __name__ == "__main__":
    main()
