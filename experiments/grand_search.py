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

Four axes the first pass left at defaults
-----------------------------------------
The first run of this searched which model, not how that model was configured;
which of four fixed feature planes, not which columns within them; and which
exit rule, not at what level. Those defaults were inherited from earlier
sections and were never themselves searched, which makes "no configuration
works" a weaker claim than it sounds. So four more axes were added:

* **Model hyperparameters** — tree depth, learning rate, regularisation, the
  number of estimators, the ridge penalty, the logistic penalty.
* **Feature selection inside each training window** — the variance,
  correlation and information-coefficient stages of
  :mod:`trading_research.features.selection`, keeping the top 8, 16 or 40
  columns, refitted on every training block so the choice never sees the rows
  it will trade.
* **Exit levels** — where the take-profit, the stop and the trailing stop sit,
  rather than one hard-coded number for each.
* **The normalisation window** — a rolling normaliser is a claim about how fast
  the scale of a feature moves, and 4,000 rows was an assumption. Three
  windows are pre-computed and a configuration chooses among them.

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
from trading_research.features.selection import FeatureSelector
from trading_research.labels.targets import build_target
from trading_research.models.base import CLASSES
from trading_research.pipeline.stages import REGRESSION_MODELS, build_model
from trading_research.validation.changepoint import segment
from trading_research.validation.search import successive_halving

#: Bybit USD-M perpetual taker, 0.055% a side, plus the slippage assumed
#: throughout this project. Not searched: see the module docstring.
COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)

#: Seconds per row. Every instrument is resampled onto this grid rather than
#: subsampled by a counter, so a horizon expressed in rows means the same
#: elapsed time everywhere.
#:
#: Counting rows was wrong across instruments and only looked right on BTCUSDT.
#: Taking every fiftieth book update gives five-second rows on an instrument
#: that updates ten times a second, and fourteen-second rows on one that updates
#: three times a minute — so "twenty days of training" became thirty-three days,
#: and "a two-minute horizon" became five. Comparing instruments that way
#: compares their update rates.
GRID_SECONDS = 5
ROWS_PER_DAY = 86_400 / GRID_SECONDS

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


def to_grid(book: pd.DataFrame, seconds: int = GRID_SECONDS) -> pd.DataFrame:
    """Resample a stream of book updates onto a fixed time grid.

    The last update in each interval, carried forward across intervals with no
    update at all. Carrying forward is the honest reading — between updates the
    book is what it last was — and it is causal, since a row only ever repeats
    something already observed.

    The cost is that a quiet instrument produces repeated rows whose returns are
    zero, which dilutes any per-row statistic. That is a true statement about
    the instrument rather than an artefact: an instrument whose book stands still
    for a minute at a time is one where a two-minute horizon contains very little.
    """
    frame = book.set_index("timestamp").sort_index()
    # Per day, not across the span. Resampling the whole range at once builds a
    # continuous grid over days that were never downloaded and forward-fills a
    # stale book into them — on BTCUSDT that invented sixteen days out of a
    # hundred. A gap in the data has to stay a gap.
    parts = [
        day.resample(f"{seconds}s").last().ffill()
        for _, day in frame.groupby(frame.index.date, sort=True)
    ]
    grid = pd.concat(parts)
    return grid.dropna(subset=["bid_price_0", "ask_price_0"]).reset_index()


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


def is_basis_points(name: str) -> bool:
    """Whether a column is denominated in basis points of price."""
    return name.endswith("_bp") or any(
        token in name
        for token in ("ret", "vol", "reversal", "span", "impact", "distance", "spread")
    )


def normalise(frame: pd.DataFrame, window: int = 4000) -> pd.DataFrame:
    """Rolling robust normalisation of everything that is not already bounded.

    The scale floor is the spread: a trailing dispersion smaller than one tick
    means the price was pinned, and dividing by it turns a single tick into a
    large normalised value.

    The floor applies **only to columns measured in basis points**. Applying it
    to everything was a bug worth recording: a quote-intensity share lives in
    [0, 1] and an imbalance change in [-2, 2], so their natural dispersion is far
    below one basis point, and flooring their scale at the spread divided them
    by a number five to ten times too large. Trees are invariant to that;
    logistic and ridge at fixed regularisation are not, which quietly
    handicapped every wide plane against the hand-picked three columns and
    biased the plane comparison in favour of the narrow one.
    """
    exclude = frozenset(BOUNDED | {"mid"})
    scaled = [c for c in frame.columns if c not in exclude]
    in_bp = [c for c in scaled if is_basis_points(c)]
    unitless = [c for c in scaled if not is_basis_points(c)]

    normaliser = RollingNormaliser(window=window, exclude=exclude)
    out = normaliser.transform(frame, in_bp, floor=frame["spread_bp"])
    return normaliser.transform(out, unitless)


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

#: The two-sided FI-2010 smoothing is deliberately absent. §21 measured its
#: label correlating +0.586 with a quantity known before the decision, because
#: the average it compares against includes the past. A model fitted to it
#: spends its capacity predicting something already known, which does not
#: corrupt the P&L — trades are scored on realised forward moves — but does make
#: the target axis a comparison against a target the project has documented as
#: defective. ``forward_smoothed`` is the leak-free version and is offered
#: instead.
CLASSIFICATION_TARGETS = ("direction", "triple_barrier")
REGRESSION_TARGETS = ("net_pnl", "forward_smoothed", "magnitude")

#: Rules read raw columns and fit nothing, so a plane wider than their inputs
#: changes neither their signal nor their cost. Sampling them across four planes
#: would spend three quarters of their share of the budget re-measuring the same
#: candidate.
RULES = frozenset({"order_flow", "momentum", "mean_reversion", "breakout", "spread_capture"})

#: Hyperparameters offered per model family. A configuration draws one entry
#: from the family its model belongs to, so a ridge is never handed a tree
#: depth and the budget is not spent on combinations that do nothing.
HYPERPARAMETERS: dict[str, tuple[dict[str, Any], ...]] = {
    "tree": (
        {},
        {"max_depth": 2, "n_estimators": 150, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.05},
        {"max_depth": 6, "n_estimators": 300, "learning_rate": 0.03},
        {"max_depth": 4, "n_estimators": 600, "learning_rate": 0.02, "min_child_weight": 100.0},
        {"max_depth": 8, "n_estimators": 200, "learning_rate": 0.05, "reg_lambda": 20.0},
    ),
    "logistic": ({}, {"C": 0.01}, {"C": 0.1}, {"C": 10.0}, {"class_weight": None}),
    "ridge": ({}, {"alpha": 0.01}, {"alpha": 10.0}, {"alpha": 1000.0}),
    # GradientBoostedRegressor accepts a narrower constructor than the
    # classifier, so it gets the subset. Handing it the full tree family made
    # a third of its draws crash on keywords it does not take — and the search
    # swallowed those as "skipped", so a slice of the space was silently never
    # searched.
    "tree_small": (
        {},
        {"max_depth": 2, "n_estimators": 150, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.05},
        {"max_depth": 6, "n_estimators": 300, "learning_rate": 0.03},
    ),
    "none": ({},),
}

#: Which family each model draws its hyperparameters from. Composites and
#: ensembles pass their parameters to every member, so they take the family of
#: whatever they are mostly made of.
#: Composites pass their parameters to *every* member, and several members are
#: rules whose constructors take none — so any composite with a rule inside is
#: "none". ``bound_mixed`` pairs a ridge with a tree regressor, and a keyword
#: for one crashes the other, so it is "none" too.
FAMILY: dict[str, str] = {
    "logistic": "logistic",
    "xgboost": "tree",
    "ridge": "ridge",
    "xgboost_regressor": "tree_small",
    "bound_ridge": "ridge",
    "bound_mixed": "none",
    "ensemble": "none",
    "agree": "none",
    "rule_gated_by_model": "none",
    "meta_rule_model": "none",
    "meta_rule_tree": "none",
}

#: Models that read named columns — the rules, and every composite with a rule
#: inside. Feature selection must not run for them: the selector keeps whatever
#: scores well, and a plane that scores well can still be missing the one
#: column the rule reads by name, which crashes the fit.
RULE_BACKED = frozenset(
    {
        *(),
        "agree",
        "rule_gated_by_model",
        "meta_rule_model",
        "meta_rule_tree",
    }
)

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
    # Added in the second pass: see the module docstring.
    "select": ("all", "top8", "top16", "top40"),
    "exit_level": (3.0, 6.0, 12.0, 25.0),
    "norm_window": (1000, 4000, 16000),
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
    #: Hyperparameters as a sorted tuple of pairs, so a Config stays hashable
    #: and the search can put it in a set.
    params: tuple[tuple[str, Any], ...] = ()
    select: str = "all"
    exit_level: float = 8.0
    norm_window: int = 4000

    def kwargs(self) -> dict[str, Any]:
        return dict(self.params)

    def label(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in self.params) or "default"
        return (
            f"{self.model}/{self.plane}/{self.target}/h{self.horizon}"
            f"/hold{self.hold}/cd{self.cooldown}/{self.exit}{self.exit_level:g}"
            f"/{self.objective}/{self.refit}/tr{self.train_days}/{self.gate}"
            f"/{self.select}/nw{self.norm_window}/[{params}]"
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
        # Rules fit nothing and read named columns, so hyperparameters and
        # feature selection are not theirs to vary.
        family = HYPERPARAMETERS[FAMILY.get(model, "none")] if model not in RULES else ({},)
        params = family[int(rng.integers(len(family)))]
        select = "all" if model in RULES or model in RULE_BACKED else str(choice["select"])
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
            params=tuple(sorted(params.items())),
            select=select,
            exit_level=float(choice["exit_level"]),
            norm_window=int(choice["norm_window"]),
        )
        if config in seen:
            continue
        seen.add(config)
        out.append(config)
    return out


def rules_for(config: Config) -> ThinningRules:
    """Turn the chosen exit into the execution rules that implement it."""
    common = {"hold_periods": config.hold, "cooldown_periods": config.cooldown}
    level = config.exit_level
    if config.exit == "take_profit_stop":
        return ThinningRules(**common, take_profit_bp=level, stop_loss_bp=level)
    if config.exit == "trailing":
        return ThinningRules(**common, trailing_stop_bp=level)
    if config.exit == "flip":
        return ThinningRules(**common, exit_on_flip=True)
    if config.exit == "confidence":
        # The level axis is in basis points elsewhere; here it maps onto the
        # probability floor a position must keep to stay open.
        floor = {3.0: 0.34, 6.0: 0.36, 12.0: 0.40, 25.0: 0.45}[level]
        return ThinningRules(**common, exit_below_confidence=floor)
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

    def __init__(
        self,
        frames: dict[int, pd.DataFrame],
        spread_bp: np.ndarray,
        mid: np.ndarray,
        *,
        search_end: int | None = None,
    ) -> None:
        self.frames = frames
        self.spread_bp = spread_bp
        self.mid = mid
        # The cost baked into cost-aware labels comes from the search block
        # alone. The median over the whole span includes the held-out days,
        # which is a small look-ahead in exactly the place that claims there
        # is none.
        self.search_end = search_end if search_end is not None else len(mid)
        self.plane = planes(next(iter(frames.values())))
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
                cost_bp=float(
                    COSTS.round_trip_bp(float(np.median(self.spread_bp[: self.search_end])))
                ),
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


@dataclass(frozen=True)
class Refit:
    """One refit: when it happens, how far back it may train, how long it trades.

    ``floor`` and ``until`` are what two bugs in the first version of this cost.

    ``until`` was a fixed five days regardless of how often the policy refitted,
    so a daily-refit candidate traded every row five times over with five
    different models. Its trade count was inflated fivefold, its trades were
    heavily overlapping rather than independent, and the "one position at a
    time" premise of :func:`thin` was violated across windows. Apply windows now
    run to the next refit, so they tile the span exactly once whatever the
    cadence.

    ``floor`` was absent, so a model refitted *at* a detected regime break still
    trained on up to twenty days of pre-break data — the very data the break had
    just declared stale. That is not a test of regime-aware refitting; it is a
    test of an irregular cadence with a contaminated training window. Under the
    ``breaks`` policy the floor is now the previous break.
    """

    at: int
    floor: int
    until: int


def refit_points(config: Config, start: int, stop: int, breaks: Sequence[int]) -> list[Refit]:
    """Where the model is refitted between ``start`` and ``stop``, and on what."""
    if config.refit == "breaks":
        points = [start, *[b for b in breaks if start < b < stop]]
        # Under this policy the previous break is the floor: everything before
        # it belongs to a regime the detector says has ended.
        floors = [0, *points[:-1]]
    else:
        days = {"1d": 1, "2d": 2, "5d": 5}[config.refit]
        points = list(range(start, stop, int(days * ROWS_PER_DAY)))
        # A fixed cadence makes no claim about staleness, so training may reach
        # as far back as the window asks for.
        floors = [0] * len(points)

    edges = [*points[1:], stop]
    return [Refit(at=a, floor=f, until=u) for a, f, u in zip(points, floors, edges, strict=True)]


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
    features = data.frames[config.norm_window]
    columns = data.plane[config.plane]
    target = data.target(config.target, config.horizon)
    forward = data.forward(config.hold)
    regression = config.model in REGRESSION_MODELS or config.model.startswith("bound_")
    gate = gate_for(config)
    rules = rules_for(config)

    usable = features[columns].notna().all(axis=1).to_numpy()
    labelled = usable & target.notna().to_numpy()
    tradeable = usable & np.isfinite(forward)

    rows: list[dict[str, Any]] = []
    for window_start, window_end in bounds:
        for refit in refit_points(config, window_start, window_end, breaks):
            point = refit.at
            # Training reaches back past the start of the block being traded.
            # Clamping it there was a bug: on the final block it forbade fitting
            # on data that is simply in the past, skipped the first refit
            # entirely, and left the first days untraded while still counting
            # them in the denominator — so the one number the whole document
            # rests on was measured under a handicap no live run would have.
            train_from = max(refit.floor, point - int(config.train_days * ROWS_PER_DAY))
            # Purge what the labels read forward. The label horizon is the
            # obvious part; the holding period matters too, because the
            # threshold is chosen on outcomes measured over ``hold`` rows and
            # those extend past the end of the training block.
            train_to = point - max(config.horizon, config.hold)
            if train_to - train_from < 20_000:
                continue
            apply_to = refit.until

            train_mask = np.zeros(len(features), dtype=bool)
            train_mask[train_from:train_to] = True
            train_mask &= labelled
            apply_mask = np.zeros(len(features), dtype=bool)
            apply_mask[point:apply_to] = True
            apply_mask &= tradeable
            if train_mask.sum() < 10_000 or apply_mask.sum() < 2_000:
                continue

            if gate is not None:
                # Columns named exactly as the gate reads them. A frame that
                # spelt the volatility column differently made thresholds()
                # return nothing and the gate silently open — every quiet_out
                # candidate in three instrument-wide searches ran ungated, and
                # the gate axis compared open against open.
                gate_frame = pd.DataFrame(
                    {
                        gate.spread_column: data.spread_bp,
                        gate.volatility_column: features["log_mid_vol50"].to_numpy(),
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

            # Feature selection, refitted on this training block alone. Choosing
            # columns once over the whole span would let the test block vote on
            # which features exist, which is the most effective way there is to
            # manufacture an edge that is not there.
            chosen = columns
            if config.select != "all":
                selector = FeatureSelector(max_features=int(config.select[3:]))
                try:
                    selector.fit(
                        features.iloc[fit_index][columns],
                        pd.Series(y[fit_index], index=fit_index),
                        forward=pd.Series(forward[fit_index], index=fit_index),
                    )
                except ValueError:
                    continue
                chosen = selector.selected_ or columns

            estimator = build_model(config.model, **config.kwargs())
            estimator.fit(
                features.iloc[fit_index][chosen], pd.Series(y[fit_index], index=fit_index)
            )

            inner_proba = estimator.predict_proba(features.iloc[inner_index][chosen])
            confidence, _ = choose_confidence(
                inner_proba,
                pd.Series(forward[inner_index]),
                pd.Series(data.spread_bp[inner_index]),
                COSTS,
                objective=config.objective,
            )

            # Trade over the contiguous block, standing aside where a row is
            # unusable or the gate is shut — rather than deleting those rows.
            #
            # Deleting them was a bug. :func:`thin` counts the holding period
            # and the cooldown in rows and reads the price path row by row, so
            # on a compacted series a "five-minute hold" spanned however much
            # wall-clock time the surviving rows happened to cover, and a
            # trailing stop watched the price teleport across the removed
            # stretches. A gate that removes thirty per cent of rows changes
            # what every duration in the configuration means.
            block = slice(point, apply_to)
            usable_here = np.flatnonzero(apply_mask[block])
            if len(usable_here) < 1_000:
                continue
            proba_usable = estimator.predict_proba(features.iloc[point + usable_here][chosen])

            length = apply_to - point
            proba = np.zeros((length, len(CLASSES)))
            proba[:, CLASSES.index(0)] = 1.0
            proba[usable_here] = proba_usable
            # Between usable rows the model's opinion is its last opinion.
            # Leaving the filler at "all hold" handed p_buy = p_sell = 0 to the
            # exit logic, so the moment a gate closed, every confidence exit
            # fired at once — positions were closed by the gate blinking, not
            # by the model changing its mind.
            last = np.maximum.accumulate(
                np.where(np.isin(np.arange(length), usable_here), np.arange(length), -1)
            )
            filled = last >= 0
            proba[filled] = proba[last[filled]]
            decision = np.zeros(length, dtype=int)
            decision[usable_here] = decide(proba_usable, min_confidence=confidence)

            trades = thin(
                decision,
                forward[block],
                data.spread_bp[block],
                rules,
                mid=data.mid[block] if rules.needs_price_path else None,
                p_buy=proba[:, CLASSES.index(1)] if rules.needs_probabilities else None,
                p_sell=proba[:, CLASSES.index(-1)] if rules.needs_probabilities else None,
            )
            days = length / ROWS_PER_DAY
            for trade in trades:
                cost = float(COSTS.round_trip_bp(trade.entry_spread_bp))
                rows.append(
                    {
                        "at": point + int(trade.entry_index),
                        "gross_bp": trade.direction * trade.move_bp,
                        "cost_bp": cost,
                        "exit_reason": trade.exit_reason,
                        "net_bp": trade.direction * trade.move_bp - cost,
                        "days": days,
                        "refit_at": point,
                        "features_used": len(chosen),
                    }
                )
    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame, days: float) -> dict[str, float]:
    """The figures a configuration is judged on."""
    if trades.empty:
        return {
            "trades": 0.0,
            "gross_per_trade_bp": float("nan"),
            "net_per_trade_bp": float("nan"),
            "net_bp": 0.0,
        }
    # Chronological, not append order: a drawdown computed over trades sorted
    # by refit block rather than by time is not a drawdown.
    net = trades.sort_values("at")["net_bp"]
    equity = net.cumsum()
    # Peak anchored at zero, so a losing opening run counts as drawdown.
    peak = equity.cummax().clip(lower=0.0)
    return {
        "trades": float(len(net)),
        "trades_per_day": float(len(net) / days) if days else float("nan"),
        # Reported alongside net because the two carry different information:
        # net says whether the strategy pays, gross says whether there was
        # anything to pay with. An instrument with three times the cost can
        # show a worse net on a better signal.
        "gross_per_trade_bp": float(trades["gross_bp"].mean()),
        "cost_per_trade_bp": float(trades["cost_bp"].mean()),
        "net_per_trade_bp": float(net.mean()),
        "net_bp": float(net.sum()),
        "hit_rate": float((net > 0).mean()),
        "max_drawdown_bp": float((peak - equity).max()),
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
    book = to_grid(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
    print(f"{args.symbol}: {len(book):,} rows, {book.timestamp.min()} .. {book.timestamp.max()}")

    raw = build_features(book)
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
    spread_bp = ((book["ask_price_0"] - book["bid_price_0"]) / mid * 1e4).to_numpy()

    # One normalised copy per window on the axis. Held as float32: three copies
    # of 1.4 million rows in float64 is most of a gigabyte, and nothing here
    # needs more precision than a normalised feature carries.
    frames: dict[int, pd.DataFrame] = {}
    for window in SPACE["norm_window"]:
        frame = normalise(raw, window=int(window)).astype("float32")
        frame["mid"] = mid
        frames[int(window)] = frame
        print(f"  normalised at {window}: {frame.notna().all(axis=1).mean():.1%} complete rows")
    del raw
    cut = int(len(book) * SEARCH_SHARE)
    data = Data(frames, spread_bp, mid, search_end=cut)
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
    stem = f"grand_search_{args.symbol}"
    outcome.table.to_csv(RESULTS / f"{stem}_rungs.csv", index=False)

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
    table.insert(0, "symbol", args.symbol)
    emit(table, stem)

    (RESULTS / f"{stem}_winner.json").write_text(
        json.dumps({"config": outcome.best.__dict__, "final": final}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
