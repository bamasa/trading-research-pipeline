"""What a strategy is judged on, beyond the average trade.

Profit per trade is the right number for asking whether a signal is worth
acting on, and the wrong one for asking whether a strategy is worth running. A
strategy earning half a basis point per trade twice a day is not a business; one
earning the same figure four hundred times a day is a different proposition
entirely, and neither is separable from how far underwater it goes on the way.

Three additions, all of them things a desk asks before anything else.

**Trades per day.** Sets capacity and it sets how quickly a result becomes
knowable. At three trades a month, a year of live trading cannot distinguish a
real edge from luck; at four hundred a day, a fortnight can.

**Drawdown.** The largest peak-to-trough fall of the cumulative result. A
strategy is not run by someone who sees only the final number, and the path
decides whether it survives to produce that number.

**Profit and loss in total**, not only per trade, because per-trade figures are
scale-free and a business is not.

Everything is in basis points of notional, which keeps it independent of
position size. Converting to currency needs a sizing rule, and this project
does not have one — every trade is one unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_bp(net_per_trade: np.ndarray) -> tuple[float, int]:
    """Largest peak-to-trough fall of the cumulative result, and its length.

    Computed on the trade sequence rather than on a clock, because the trades
    are what the strategy did; a calendar version would show flat stretches
    where nothing happened and read as recovery.

    Returns the depth in basis points as a positive number, and how many trades
    the drawdown lasted from peak to trough.
    """
    if len(net_per_trade) == 0:
        return 0.0, 0
    equity = np.cumsum(net_per_trade)
    peak = np.maximum.accumulate(equity)
    underwater = equity - peak
    trough = int(np.argmin(underwater))
    if underwater[trough] >= 0:
        return 0.0, 0
    # Where the peak that this trough fell from was set.
    start = int(np.argmax(equity[: trough + 1]))
    return float(-underwater[trough]), trough - start


def summarise(
    net_per_trade: np.ndarray,
    *,
    days: float,
) -> dict[str, float]:
    """The full set: profit, per trade, per day, and the worst of the path.

    ``days`` is the length of the period the trades were taken over. Passed in
    rather than inferred, because a strategy that stood aside for a week still
    spent that week not earning, and inferring the span from the trades would
    quietly delete it.
    """
    if len(net_per_trade) == 0:
        return {
            "trades": 0.0,
            "trades_per_day": 0.0,
            "net_bp": 0.0,
            "net_per_trade_bp": float("nan"),
            "net_bp_per_day": 0.0,
            "max_drawdown_bp": 0.0,
            "drawdown_trades": 0.0,
            "hit_rate": float("nan"),
            "profit_factor": float("nan"),
            "days": float(days),
        }

    depth, length = drawdown_bp(net_per_trade)
    won = net_per_trade[net_per_trade > 0].sum()
    lost = -net_per_trade[net_per_trade < 0].sum()

    return {
        "trades": float(len(net_per_trade)),
        "trades_per_day": float(len(net_per_trade) / days) if days else float("nan"),
        "net_bp": float(net_per_trade.sum()),
        "net_per_trade_bp": float(net_per_trade.mean()),
        "net_bp_per_day": float(net_per_trade.sum() / days) if days else float("nan"),
        "max_drawdown_bp": depth,
        "drawdown_trades": float(length),
        # Ratio of what the winners made to what the losers cost. Above one is
        # profitable; it says the same thing as the mean but survives being
        # compared across strategies that trade at different rates.
        "profit_factor": float(won / lost) if lost > 0 else float("inf"),
        "hit_rate": float((net_per_trade > 0).mean()),
        "days": float(days),
        # How far the profit would have to fall before the drawdown swallows it.
        "return_over_drawdown": float(net_per_trade.sum() / depth) if depth > 0 else float("nan"),
    }


def equity_curve(net_per_trade: np.ndarray) -> pd.DataFrame:
    """Cumulative result, its running peak and how far underwater it is."""
    equity = np.cumsum(net_per_trade)
    peak = np.maximum.accumulate(equity) if len(equity) else equity
    return pd.DataFrame(
        {
            "trade": np.arange(len(equity)),
            "equity_bp": equity,
            "peak_bp": peak,
            "underwater_bp": equity - peak,
        }
    )
