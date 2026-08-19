"""Everything searched again, on all the data, with the answer held back.

§25 ended a run of individually reasonable decisions in a result that did not
survive contact with new data. The response asked for here is not a smaller
search but a larger and better-protected one: re-select the features, the model,
the target, the horizon, the exit rule, the refit policy and the entry threshold
together, on the whole eighty-four days rather than the thirty-nine most of the
earlier work used, and let the winner be whatever wins.

The protection is the part that matters, because a wider search is a stronger
generator of false positives, not a weaker one. Two things guard it.

**A block the search never sees.** The eighty-four days are cut once, at the
start, into a search block and a final block. Every choice — every feature, every
hyperparameter, every threshold, the winner itself — is made using the search
block alone. The final block is read exactly once, after the search has
committed, and its number is the one reported. Nothing is re-run against it,
because a held-out block consulted twice is a validation block with a better
name.

**Segments instead of a fixed horizon of validity.** §25's failure was a model
outliving the conditions it was fitted to, and a fixed refit cadence is a guess
about how long those conditions last. The CUSUM detector in
:mod:`trading_research.validation.changepoint` supplies the alternative: refit
when the series says it changed. On this data it reports a break every three or
four days, so *refit at breaks* competes directly against fixed cadences of one,
two and five days, and the search decides which is better rather than the author.

What is searched
----------------
Eleven axes, sampled rather than enumerated: feature plane, model, target,
label horizon, holding period, cooldown, exit rule, entry-threshold objective,
refit policy, training window, and market gate. The product is over a hundred
thousand combinations; a few hundred are drawn and put through successive
halving, so a candidate that is obviously poor after four windows does not cost
twenty.

What is not searched, and why
-----------------------------
Costs. The taker fee, the spread treatment and the slippage assumption are held
at the published Bybit numbers throughout. They are the one part of this that is
not a modelling choice, and letting a search touch them would be letting it
choose its own scoreboard.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.backtest.gating import MarketGate
from trading_research.features.depth import build_depth_features
from trading_research.features.normalise import RollingNormaliser
from trading_research.labels.targets import build_target
from trading_research.models.base import CLASSES
from trading_research.pipeline.stages import REGRESSION_MODELS, build_model
from trading_research.validation.changepoint import segment
from trading_research.validation.search import successive_halving

#: Bybit USD-M perpetual taker, 0.055% a side, plus the slippage assumed
#: throughout this project. Not searched: see the module docstring.
COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)

#: One row is five seconds after subsampling a 100 ms stream by fifty.
SUBSAMPLE = 50
ROWS_PER_DAY = 86_400 / 5

#: Share of the span the search may look at. The rest is read once, at the end.
SEARCH_SHARE = 0.65

# --------------------------------------------------------------------------- #
# Feature planes
# --------------------------------------------------------------------------- #

#: The three columns the earlier work used, kept as the control: any wider plane
#: has to beat the hand-picked one to justify itself.
TOUCH = ("queue_imbalance", "spread_bp", "log_mid_ret20")

#: Short-horizon quantities from the touch alone, all scale-invariant.
MICRO = (
    "queue_imbalance",
    "spread_bp",
    "log_mid_ret20",
    "log_mid_ret50",
    "log_mid_vol50",
    "imbalance_change_10",
    "mid_reversal_10",
    "size_shock_10",
)

#: Columns the rule-based models read by name. Kept in every plane so a rule is
#: never handicapped by a plane that happens to exclude its input.
RULE_COLUMNS = ("queue_imbalance", "spread_bp", "log_mid_ret20", "log_mid_ret50", "log_mid_vol50")

#: Columns already bounded or centred, which gain nothing from rescaling.
BOUNDED = frozenset({"queue_imbalance", "depth_imbalance", "weighted_depth_imbalance"})


def build_features(book: pd.DataFrame) -> pd.DataFrame:
    """Every candidate column, computed once for the whole span.

    Building these per window would be both slower and subtly different — a
    rolling statistic restarted at a window boundary is not the same series as
    one that ran through it — so they are built over the full span and sliced.
    That is safe because every column here is causal: each value reads its own
    past and nothing else.
    """
    bid, ask = book["bid_price_0"].to_numpy(), book["ask_price_0"].to_numpy()
    bid_size, ask_size = book["bid_size_0"].to_numpy(), book["ask_size_0"].to_numpy()
    mid = (bid + ask) / 2.0
    log_mid = pd.Series(np.log(mid))

    out = pd.DataFrame(index=book.index)
    out["mid"] = mid
    out["spread_bp"] = (ask - bid) / mid * 1e4
    out["queue_imbalance"] = (bid_size - ask_size) / (bid_size + ask_size)

    for span in (10, 20, 50, 100):
        out[f"log_mid_ret{span}"] = (log_mid.diff(span) * 1e4).to_numpy()
    for span in (50, 200):
        out[f"log_mid_vol{span}"] = (log_mid.diff().rolling(span).std() * 1e4).to_numpy()

    imbalance = out["queue_imbalance"]
    out["imbalance_change_10"] = imbalance.diff(10)
    out["imbalance_mean_50"] = imbalance.rolling(50).mean()
    out["size_shock_10"] = (
        pd.Series(bid_size + ask_size).pct_change(10).replace([np.inf, -np.inf], np.nan)
    )
    # How much of the last move has already been given back: positive when the
    # price has reverted, negative when it has run on.
    out["mid_reversal_10"] = (log_mid.diff(10) - log_mid.diff(20) / 2) * 1e4
    out["spread_change_20"] = out["spread_bp"].diff(20)
    out["quote_intensity_20"] = (log_mid.diff() != 0).rolling(20).mean().to_numpy()

    depth = build_depth_features(book)
    for column in depth.columns:
        out[column] = depth[column].to_numpy()
    return out


def normalise(frame: pd.DataFrame, window: int = 4000) -> pd.DataFrame:
    """Rolling robust normalisation of everything that is not already bounded.

    The scale floor is the spread: a trailing dispersion smaller than one tick
    means the price was pinned, and dividing by it turns a single tick into a
    large normalised value.
    """
    columns = [c for c in frame.columns if c not in BOUNDED and c != "mid"]
    normaliser = RollingNormaliser(window=window, exclude=frozenset(BOUNDED | {"mid"}))
    return normaliser.transform(frame, columns, floor=frame["spread_bp"])


def planes(frame: pd.DataFrame) -> dict[str, list[str]]:
    """The feature sets a configuration may choose between."""
    depth_columns = [
        c
        for c in frame.columns
        if c.startswith(("depth_", "weighted_", "concentration", "slope", "impact", "book_"))
        or c == "log_visible_depth"
    ]
    wide = [c for c in frame.columns if c != "mid"]
    return {
        "touch": list(TOUCH),
        "micro": list(MICRO),
        "depth": sorted({*RULE_COLUMNS, *depth_columns}),
        "wide": sorted(wide),
    }


# --------------------------------------------------------------------------- #
# The search space
# --------------------------------------------------------------------------- #

CLASSIFIERS = (
    "logistic",
    "xgboost",
    "order_flow",
    "momentum",
    "mean_reversion",
    "breakout",
    "spread_capture",
    "ensemble",
    "agree",
    "rule_gated_by_model",
    "meta_rule_model",
    "meta_rule_tree",
)
REGRESSORS = ("ridge", "xgboost_regressor", "bound_ridge", "bound_mixed")

CLASSIFICATION_TARGETS = ("direction", "triple_barrier", "smoothed_direction")
REGRESSION_TARGETS = ("net_pnl", "forward_smoothed", "normalised_magnitude", "magnitude")

#: Rules read raw columns and fit nothing, so a plane wider than their inputs
#: changes neither their signal nor their cost. Sampling them across four planes
#: would spend three quarters of their share of the budget re-measuring the same
#: candidate.
RULES = frozenset({"order_flow", "momentum", "mean_reversion", "breakout", "spread_capture"})

SPACE: dict[str, Sequence[Any]] = {
    "plane": ("touch", "micro", "depth", "wide"),
    "model": (*CLASSIFIERS, *REGRESSORS),
    # In five-second rows: one minute to twenty.
    "horizon": (12, 24, 60, 120, 240),
    "hold": (12, 24, 60, 120, 240),
    "cooldown_multiple": (0.0, 1.0, 2.0),
    "exit": ("clock", "take_profit_stop", "trailing", "flip", "confidence"),
    "objective": ("net_bp", "net_per_trade_bp"),
    "refit": ("1d", "2d", "5d", "breaks"),
    "train_days": (5, 10, 20),
    "gate": ("open", "quiet_out", "tight_only"),
}


@dataclass(frozen=True)
class Config:
    """One point in the space, hashable so the search can carry it around."""

    plane: str
    model: str
    target: str
    horizon: int
    hold: int
    cooldown: int
    exit: str
    objective: str
    refit: str
    train_days: int
    gate: str

    def label(self) -> str:
        return (
            f"{self.model}/{self.plane}/{self.target}/h{self.horizon}"
            f"/hold{self.hold}/cd{self.cooldown}/{self.exit}"
            f"/{self.objective}/{self.refit}/tr{self.train_days}/{self.gate}"
        )


def draw(n: int, seed: int) -> list[Config]:
    """Sample configurations, pairing each model with a target it can be fitted on.

    Drawing the target independently would spend most of the budget on
    combinations that cannot be evaluated — a classifier handed a basis-point
    label treats every distinct value as its own class — so the target is drawn
    from the ones the sampled model admits.
    """
    rng = np.random.default_rng(seed)
    seen: set[Config] = set()
    out: list[Config] = []
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        choice = {key: values[int(rng.integers(len(values)))] for key, values in SPACE.items()}
        model = choice["model"]
        regression = model in REGRESSORS
        targets = REGRESSION_TARGETS if regression else CLASSIFICATION_TARGETS
        plane = "micro" if model in RULES else choice["plane"]
        config = Config(
            plane=plane,
            model=model,
            target=str(targets[int(rng.integers(len(targets)))]),
            horizon=int(choice["horizon"]),
            hold=int(choice["hold"]),
            cooldown=int(choice["cooldown_multiple"] * choice["hold"]),
            exit=str(choice["exit"]),
            objective=str(choice["objective"]),
            refit=str(choice["refit"]),
            train_days=int(choice["train_days"]),
            gate=str(choice["gate"]),
        )
        if config in seen:
            continue
        seen.add(config)
        out.append(config)
    return out


def rules_for(config: Config) -> ThinningRules:
    """Turn the chosen exit into the execution rules that implement it."""
    common = {"hold_periods": config.hold, "cooldown_periods": config.cooldown}
    if config.exit == "take_profit_stop":
        return ThinningRules(**common, take_profit_bp=8.0, stop_loss_bp=8.0)
    if config.exit == "trailing":
        return ThinningRules(**common, trailing_stop_bp=6.0)
    if config.exit == "flip":
        return ThinningRules(**common, exit_on_flip=True)
    if config.exit == "confidence":
        return ThinningRules(**common, exit_below_confidence=0.34)
    return ThinningRules(**common)


def gate_for(config: Config) -> MarketGate | None:
    if config.gate == "quiet_out":
        return MarketGate(min_volatility_quantile=0.3)
    if config.gate == "tight_only":
        return MarketGate(max_spread_quantile=0.7)
    return None


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class Data:
    """The span, its features, and the targets built lazily and kept."""

    def __init__(self, features: pd.DataFrame, spread_bp: np.ndarray, mid: np.ndarray) -> None:
        self.features = features
        self.spread_bp = spread_bp
        self.mid = mid
        self.plane = planes(features)
        self._targets: dict[tuple[str, int], pd.Series] = {}
        self._forward: dict[int, np.ndarray] = {}

    def target(self, name: str, horizon: int) -> pd.Series:
        key = (name, horizon)
        if key not in self._targets:
            frame = pd.DataFrame(
                {
                    "mid": self.mid,
                    "spread_bp_now": self.spread_bp,
                    "forward_bp": self.forward(horizon),
                }
            )
            values, _ = build_target(
                name,
                frame,
                cost_bp=float(COSTS.round_trip_bp(float(np.median(self.spread_bp)))),
                horizon=horizon,
            )
            self._targets[key] = values
        return self._targets[key]

    def forward(self, horizon: int) -> np.ndarray:
        """The realised move a trade opened now would see, in basis points."""
        if horizon not in self._forward:
            values = np.full(len(self.mid), np.nan)
            values[:-horizon] = (self.mid[horizon:] / self.mid[:-horizon] - 1.0) * 1e4
            self._forward[horizon] = values
        return self._forward[horizon]


def refit_points(config: Config, start: int, stop: int, breaks: Sequence[int]) -> list[int]:
    """Where the model is refitted, between ``start`` and ``stop``."""
    if config.refit == "breaks":
        inside = [b for b in breaks if start < b < stop]
        return [start, *inside]
    days = {"1d": 1, "2d": 2, "5d": 5}[config.refit]
    stride = int(days * ROWS_PER_DAY)
    return list(range(start, stop, stride))


def run_block(
    config: Config,
    data: Data,
    bounds: list[tuple[int, int]],
    breaks: Sequence[int],
    *,
    max_train_rows: int = 120_000,
) -> pd.DataFrame:
    """Walk one or more windows and return every trade taken.

    The threshold is chosen on the tail of the training window and applied
    forward; the model never sees the rows it trades, and neither does the
    threshold.
    """
    columns = data.plane[config.plane]
    target = data.target(config.target, config.horizon)
    forward = data.forward(config.hold)
    regression = config.model in REGRESSION_MODELS or config.model.startswith("bound_")
    gate = gate_for(config)
    rules = rules_for(config)

    usable = data.features[columns].notna().all(axis=1).to_numpy()
    labelled = usable & target.notna().to_numpy()
    tradeable = usable & np.isfinite(forward)

    rows: list[dict[str, Any]] = []
    for window_start, window_end in bounds:
        for point in refit_points(config, window_start, window_end, breaks):
            train_from = max(window_start, point - int(config.train_days * ROWS_PER_DAY))
            # Purge the label horizon: the last rows of the training block read
            # forward into the rows about to be traded.
            train_to = point - config.horizon
            if train_to - train_from < 20_000:
                continue
            apply_to = min(window_end, point + int(5 * ROWS_PER_DAY))

            train_mask = np.zeros(len(data.features), dtype=bool)
            train_mask[train_from:train_to] = True
            train_mask &= labelled
            apply_mask = np.zeros(len(data.features), dtype=bool)
            apply_mask[point:apply_to] = True
            apply_mask &= tradeable
            if train_mask.sum() < 10_000 or apply_mask.sum() < 2_000:
                continue

            if gate is not None:
                gate_frame = pd.DataFrame(
                    {
                        "spread_bp_now": data.spread_bp,
                        "volatility": data.features["log_mid_vol50"].to_numpy(),
                    }
                )
                levels = gate.thresholds(gate_frame[train_mask])
                keep = gate.mask(gate_frame, levels).to_numpy()
                train_mask &= keep
                apply_mask &= keep
                if train_mask.sum() < 10_000 or apply_mask.sum() < 1_000:
                    continue

            # Front fits, tail sets the threshold — chronologically, because
            # neighbouring five-second rows are near-duplicates and a random
            # split would put the same moment on both sides.
            index = np.flatnonzero(train_mask)
            cut = int(len(index) * 0.75)
            fit_index, inner_index = index[:cut][-max_train_rows:], index[cut:]
            if len(fit_index) < 5_000 or len(inner_index) < 3_000:
                continue

            y = target.to_numpy()
            if not regression and len(np.unique(y[fit_index])) < 2:
                continue

            estimator = build_model(config.model)
            estimator.fit(
                data.features.iloc[fit_index][columns], pd.Series(y[fit_index], index=fit_index)
            )

            inner_proba = estimator.predict_proba(data.features.iloc[inner_index][columns])
            confidence, _ = choose_confidence(
                inner_proba,
                pd.Series(forward[inner_index]),
                pd.Series(data.spread_bp[inner_index]),
                COSTS,
                objective=config.objective,
            )

            apply_index = np.flatnonzero(apply_mask)
            proba = estimator.predict_proba(data.features.iloc[apply_index][columns])
            decision = decide(proba, min_confidence=confidence)
            trades = thin(
                decision,
                forward[apply_index],
                data.spread_bp[apply_index],
                rules,
                mid=data.mid[apply_index] if rules.needs_price_path else None,
                p_buy=proba[:, CLASSES.index(1)] if rules.needs_probabilities else None,
                p_sell=proba[:, CLASSES.index(-1)] if rules.needs_probabilities else None,
            )
            days = len(apply_index) / ROWS_PER_DAY
            for trade in trades:
                cost = float(COSTS.round_trip_bp(trade.entry_spread_bp))
                rows.append(
                    {
                        "at": int(apply_index[trade.entry_index]),
                        "exit_reason": trade.exit_reason,
                        "net_bp": trade.direction * trade.move_bp - cost,
                        "days": days,
                        "refit_at": point,
                    }
                )
    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame, days: float) -> dict[str, float]:
    """The figures a configuration is judged on."""
    if trades.empty:
        return {"trades": 0.0, "net_per_trade_bp": float("nan"), "net_bp": 0.0}
    net = trades["net_bp"]
    equity = net.cumsum()
    return {
        "trades": float(len(net)),
        "trades_per_day": float(len(net) / days) if days else float("nan"),
        "net_per_trade_bp": float(net.mean()),
        "net_bp": float(net.sum()),
        "hit_rate": float((net > 0).mean()),
        "max_drawdown_bp": float((equity.cummax() - equity).max()),
        "dispersion_bp": float(net.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--candidates", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20240819)
    parser.add_argument("--book", type=Path, default=Path("data/book"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    files = sorted((args.book / args.symbol).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no book data for {args.symbol} under {args.book}")
    book = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    book = book.iloc[::SUBSAMPLE].reset_index(drop=True)
    print(f"{args.symbol}: {len(book):,} rows, {book.timestamp.min()} .. {book.timestamp.max()}")

    features = normalise(build_features(book))
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
    spread_bp = ((book["ask_price_0"] - book["bid_price_0"]) / mid * 1e4).to_numpy()
    features["mid"] = mid
    data = Data(features, spread_bp, mid)

    cut = int(len(book) * SEARCH_SHARE)
    final_bounds = [(cut, len(book))]
    search_days = cut / ROWS_PER_DAY
    final_days = (len(book) - cut) / ROWS_PER_DAY
    print(
        f"search block: {search_days:.0f} days to {book.timestamp.iloc[cut].date()}; "
        f"final block: {final_days:.0f} days, read once"
    )

    # Breaks are detected on the search block alone and, for the final block,
    # re-detected forward from its start — the detector is causal, so this is
    # what a run on the day would have had.
    cuts = segment(mid[:cut], spread_bp[:cut], minimum_rows=40_000)
    breaks = [b.index for b in cuts.breaks]
    print(f"{len(breaks)} breaks in the search block, {len(cuts)} segments")

    candidates = draw(args.candidates, args.seed)
    print(f"{len(candidates)} candidates drawn from {len(SPACE)} axes\n")

    started = time.time()
    seen = {"n": 0}

    def score(config: Config, budget: int) -> dict[str, float]:
        span = int(budget / 24 * cut) if budget < 24 else cut
        trades = run_block(config, data, [(0, span)], breaks)
        seen["n"] += 1
        result = summarise(trades, span / ROWS_PER_DAY)
        if seen["n"] % 10 == 0:
            print(
                f"  [{seen['n']:4d}] {time.time() - started:6.0f}s "
                f"{config.label()[:60]:60s} "
                f"{result['trades']:5.0f} trades {result['net_per_trade_bp']:+7.2f} bp"
            )
        return result

    outcome = successive_halving(
        candidates,
        score,
        objective="net_per_trade_bp",
        budgets=(6, 12, 24),
        keep_fraction=0.3,
        minimum_trades=30.0,
        label=lambda c: c.label(),
    )
    print(f"\n{outcome.summary()}")
    print(f"winner: {outcome.best.label()}")
    print(f"on the search block: {outcome.best_metrics}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    outcome.table.to_csv(RESULTS / "grand_search_rungs.csv", index=False)

    # The one reading of the final block.
    final_cuts = segment(mid[cut:], spread_bp[cut:], minimum_rows=40_000)
    final_breaks = [cut + b.index for b in final_cuts.breaks]
    final_trades = run_block(outcome.best, data, final_bounds, final_breaks)
    final = summarise(final_trades, final_days)

    rows = [
        {"block": "search (chose the winner)", "days": round(search_days), **outcome.best_metrics},
        {"block": "final (never searched)", "days": round(final_days), **final},
    ]
    table = pd.DataFrame(rows)
    if final["trades"]:
        table["two_se_bp"] = 2 * table["dispersion_bp"] / np.sqrt(table["trades"])
    emit(table, "grand_search")

    (RESULTS / "grand_search_winner.json").write_text(
        json.dumps({"config": outcome.best.__dict__, "final": final}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
