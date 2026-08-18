"""When the market is worth trading at all.

Every result so far treats each observation as a candidate: the model scores it,
the confidence threshold decides, and the state of the market itself is never
consulted. That is a gap, and the arithmetic in the results says where it hurts.
The edge per trade is roughly the information coefficient times the volatility
of the move; the cost is fixed. In a quiet stretch the second term collapses and
the trade cannot pay for itself however good the prediction is.

A gate makes that explicit. It reads the market state — how much the price has
been moving, how wide the spread is — and closes trading when the state cannot
support it, before the model is consulted.

Applied on both sides
---------------------
The same gate is used when fitting and when trading, which is the point. Fitting
on every observation and trading only some means the model spent most of its
capacity learning a regime it will never act in: 99% of rows are HOLD already,
and the quiet ones are the emptiest of them. Restricting the training sample to
the rows the strategy would actually act on gives the fit a target that matches
the job.

The threshold is not knowable in advance and is not guessed here. It is a
quantile of the training window's own distribution, so it adapts to the
instrument and the period rather than being a number in basis points that means
something different for BTCUSDT than for XRPUSDT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketGate:
    """Conditions the market must meet before a trade is considered.

    ``min_volatility_quantile``
        Trade only when recent realised volatility is above this quantile of
        the training window. 0.0 keeps everything; 0.5 keeps the busier half.
        A quantile rather than a level, because a level in basis points is a
        different filter on every instrument.
    ``max_spread_quantile``
        Skip observations where the spread is unusually wide. Wide spreads are
        both more expensive to cross and a sign that the book is thin.
    ``volatility_column``
        Which feature carries recent volatility. Defaults to a fifty-row
        window, four minutes on the 5 s grid — long enough to be a regime and
        short enough to react.
    """

    min_volatility_quantile: float = 0.0
    max_spread_quantile: float = 1.0
    volatility_column: str = "log_mid_vol50"
    spread_column: str = "spread_bp_now"

    def __post_init__(self) -> None:
        for name in ("min_volatility_quantile", "max_spread_quantile"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def is_open(self) -> bool:
        """True when the gate lets everything through."""
        return self.min_volatility_quantile <= 0.0 and self.max_spread_quantile >= 1.0

    @property
    def label(self) -> str:
        if self.is_open:
            return "no gate"
        parts = []
        if self.min_volatility_quantile > 0:
            parts.append(f"vol>q{self.min_volatility_quantile:g}")
        if self.max_spread_quantile < 1:
            parts.append(f"spread<q{self.max_spread_quantile:g}")
        return "/".join(parts)

    def thresholds(self, frame: pd.DataFrame) -> dict[str, float]:
        """Turn the quantiles into levels, using this frame's distribution.

        Always called on the training window and then applied unchanged to
        whatever comes next. Recomputing them on the block being traded would
        make the gate depend on the period it is judging.
        """
        out: dict[str, float] = {}
        if self.min_volatility_quantile > 0 and self.volatility_column in frame.columns:
            column = frame[self.volatility_column].dropna()
            if not column.empty:
                out["min_volatility"] = float(column.quantile(self.min_volatility_quantile))
        if self.max_spread_quantile < 1 and self.spread_column in frame.columns:
            column = frame[self.spread_column].dropna()
            if not column.empty:
                out["max_spread"] = float(column.quantile(self.max_spread_quantile))
        return out

    def mask(self, frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
        """Rows the gate lets through, given levels fitted elsewhere."""
        allowed = pd.Series(True, index=frame.index)
        if "min_volatility" in thresholds and self.volatility_column in frame.columns:
            allowed &= frame[self.volatility_column] >= thresholds["min_volatility"]
        if "max_spread" in thresholds and self.spread_column in frame.columns:
            allowed &= frame[self.spread_column] <= thresholds["max_spread"]
        return allowed.fillna(False)


#: Quantiles worth trying. Above about 0.8 the sample gets too thin to fit on,
#: which is itself the trade-off the gate is making: a stricter gate trades in
#: better conditions and has less to learn from.
GATE_GRID: tuple[MarketGate, ...] = (
    MarketGate(),
    MarketGate(min_volatility_quantile=0.3),
    MarketGate(min_volatility_quantile=0.5),
    MarketGate(min_volatility_quantile=0.7),
    MarketGate(max_spread_quantile=0.7),
    MarketGate(min_volatility_quantile=0.5, max_spread_quantile=0.9),
    MarketGate(min_volatility_quantile=0.7, max_spread_quantile=0.9),
)


def realised_volatility_bp(mid: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation of log returns, in basis points.

    Provided for callers whose frames do not carry a volatility feature — the
    gate needs one, and computing it from the price path is preferable to
    silently letting everything through.
    """
    returns = np.full(len(mid), np.nan)
    returns[1:] = np.log(mid[1:] / mid[:-1]) * 1e4
    return pd.Series(returns).rolling(window, min_periods=window // 2).std().to_numpy()
