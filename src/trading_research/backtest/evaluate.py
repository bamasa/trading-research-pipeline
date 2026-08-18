"""Turning probabilities into decisions, and decisions into money.

Two steps, kept apart on purpose.

**The decision rule** converts class probabilities into trade or no-trade. It
has one parameter — how confident the model must be — and that parameter is
chosen on validation and then applied unchanged to test. Choosing it on test is
the most common way a short-horizon result is overstated, and it is invisible
in the resulting numbers.

**The accounting** applies the cost model to the trades the rule produced. It
is deliberately simple, and its simplifications are stated rather than hidden:
each signal is treated as an independent round trip held for the label horizon,
entered and exited by taking liquidity. There is no position netting, no queue
position, no market impact, and no limit on concurrent exposure.

That overstates achievable size and understates the cost of trading in size. It
is honest about direction and cost per trade, which is what decides whether an
edge exists at all — and if it does not survive here, no amount of execution
sophistication will rescue it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_research.backtest.costs import TakerCosts
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.models.base import CLASSES


def decide(proba: np.ndarray, *, min_confidence: float) -> np.ndarray:
    """Convert probabilities into -1 / 0 / +1 decisions.

    Trades only when the stronger directional probability clears
    ``min_confidence`` **and** beats the opposite direction. Requiring both
    matters: a model can be confident that the market will move while being
    nearly indifferent about which way, and acting on that is a coin flip that
    still pays full costs.
    """
    if proba.shape[1] != len(CLASSES):
        raise ValueError(f"expected {len(CLASSES)} probability columns, got {proba.shape[1]}")

    p_sell = proba[:, CLASSES.index(SELL)]
    p_buy = proba[:, CLASSES.index(BUY)]

    decision = np.full(len(proba), HOLD, dtype=int)
    decision[(p_buy >= min_confidence) & (p_buy > p_sell)] = BUY
    decision[(p_sell >= min_confidence) & (p_sell > p_buy)] = SELL
    return decision


@dataclass
class Result:
    """What one model did on one block."""

    model: str
    fold: int
    block: str
    n_rows: int
    n_trades: int
    trade_rate: float
    gross_bp: float
    cost_bp: float
    net_bp: float
    net_bp_per_trade: float
    hit_rate: float
    min_confidence: float
    extra: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        out: dict[str, object] = {
            "model": self.model,
            "fold": self.fold,
            "block": self.block,
            "rows": self.n_rows,
            "trades": self.n_trades,
            "trade_rate": self.trade_rate,
            "gross_bp": self.gross_bp,
            "cost_bp": self.cost_bp,
            "net_bp": self.net_bp,
            "net_per_trade_bp": self.net_bp_per_trade,
            "hit_rate": self.hit_rate,
            "min_confidence": self.min_confidence,
        }
        out.update(self.extra)
        return out


def evaluate(
    decision: np.ndarray,
    forward_move_bp: pd.Series,
    spread_bp: pd.Series,
    costs: TakerCosts,
    *,
    model: str = "",
    fold: int = -1,
    block: str = "",
    min_confidence: float = 0.0,
) -> Result:
    """Score a set of decisions against what the market actually did.

    ``forward_move_bp`` is the realised move over the label horizon, so a
    decision is scored against exactly the outcome it was trained to predict.
    Costs are charged per trade at the spread prevailing when the trade was
    entered, rather than at an average, because spreads widen precisely when
    the model is most tempted to act.
    """
    decision = np.asarray(decision)
    move = forward_move_bp.to_numpy()
    spread = spread_bp.to_numpy()

    usable = ~np.isnan(move) & ~np.isnan(spread)
    decision = np.where(usable, decision, HOLD)

    traded = decision != HOLD
    n_trades = int(traded.sum())
    n_rows = int(usable.sum())

    if n_trades == 0:
        return Result(
            model=model,
            fold=fold,
            block=block,
            n_rows=n_rows,
            n_trades=0,
            trade_rate=0.0,
            gross_bp=0.0,
            cost_bp=0.0,
            net_bp=0.0,
            net_bp_per_trade=0.0,
            hit_rate=float("nan"),
            min_confidence=min_confidence,
        )

    gross = decision[traded] * move[traded]
    cost = np.asarray(costs.round_trip_bp(pd.Series(spread[traded])))
    net = gross - cost

    return Result(
        model=model,
        fold=fold,
        block=block,
        n_rows=n_rows,
        n_trades=n_trades,
        trade_rate=n_trades / max(n_rows, 1),
        gross_bp=float(gross.sum()),
        cost_bp=float(cost.sum()),
        net_bp=float(net.sum()),
        net_bp_per_trade=float(net.mean()),
        hit_rate=float((gross > 0).mean()),
        min_confidence=min_confidence,
    )


def choose_confidence(
    proba: np.ndarray,
    forward_move_bp: pd.Series,
    spread_bp: pd.Series,
    costs: TakerCosts,
    *,
    grid: np.ndarray | None = None,
    min_trades: int = 50,
    objective: str = "net_bp",
) -> tuple[float, pd.DataFrame]:
    """Pick the confidence threshold that maximises ``objective`` on validation.

    ``objective`` decides what "best" means, and the two options answer
    different questions. ``net_bp`` maximises the total, which favours trading
    often at a small edge. ``net_per_trade_bp`` maximises the edge on each
    trade, which pushes the threshold up until only the strongest signals
    survive — the selective strategy, with the cut made here on validation and
    carried to test unchanged.

    Returns the chosen threshold and the whole curve. The curve is the useful
    part: a net figure that peaks sharply at one threshold and is negative
    either side of it is a fitted artefact, whereas a broad plateau is
    something that might survive out of sample. Reporting only the peak hides
    exactly that distinction.

    Thresholds producing fewer than ``min_trades`` trades are excluded. With a
    handful of trades the average is dominated by one or two lucky moves, and
    picking that threshold is selecting noise.
    """
    if grid is None:
        grid = np.round(np.arange(0.34, 0.95, 0.02), 3)

    rows = []
    for threshold in grid:
        decision = decide(proba, min_confidence=float(threshold))
        result = evaluate(
            decision,
            forward_move_bp,
            spread_bp,
            costs,
            block="validation",
            min_confidence=float(threshold),
        )
        rows.append(result.as_row())

    curve = pd.DataFrame(rows)
    eligible = curve[curve["trades"] >= min_trades]
    if eligible.empty:
        # Nothing traded enough to judge. Returning the most permissive
        # threshold would quietly pick the noisiest option, so the caller is
        # told to stand aside instead.
        return float(grid.max()) + 1.0, curve

    if objective not in eligible.columns:
        raise ValueError(f"unknown objective {objective!r}; have {list(eligible.columns)}")
    best_row = int(eligible[objective].idxmax())
    return float(eligible["min_confidence"].loc[best_row]), curve
