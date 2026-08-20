"""Trading an instrument against its peers instead of against the future.

Every signal in this project so far asked the same question of one instrument:
where is its own price going. §26 settled what that is worth — a few basis
points of gross edge against a round trip several times larger.

A cross-sectional strategy asks a different question, and it is the question
most systematic equity trading is built on. Not "will this go up" but "has this
moved further than its peers, and will the difference close". The signal is a
deviation from the cross-section, and two properties make it worth trying where
direction failed:

* **Reversion beats prediction.** A short-horizon dispersion across correlated
  instruments is a far stronger statistical effect than the direction of any one
  of them. The cross-section supplies a reference point that a single series
  does not have.
* **The market drops out.** Going long the laggards and short the leaders in
  equal weight removes the common move. What is left is the relative part,
  which is the only part the signal claimed to know about.

What it costs
-------------
Both legs cross the spread, so a cross-sectional trade pays for *every* name it
touches. A basket of ten pays ten round trips to express one view, and the
per-name edge has to clear the per-name cost. Breadth does not make a small edge
affordable — it makes a small edge repeatable, which is a different thing and
only helps if the edge is positive to begin with.

The construction
----------------
At each row, for each instrument, the recent return over a lookback is compared
against the cross-sectional mean of those returns. The deviation is ranked, the
extremes are traded against each other, and the whole basket is held for a fixed
horizon. Weights sum to zero by construction; the gross exposure is normalised
so results across basket sizes are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class RelativeError(ValueError):
    """The panel cannot support the strategy as configured."""


@dataclass(frozen=True)
class CrossSection:
    """How the relative signal is formed and traded.

    ``lookback``
        Rows the recent return is measured over. This is the window whose
        dispersion the strategy bets will close.
    ``basket``
        How many names are taken on each side. Larger baskets dilute the signal
        and diversify the noise; the cost is linear in the number of names, so
        this is not free breadth.
    ``hold``
        Rows the basket is held before being closed.
    ``demean``
        Subtract the cross-sectional mean return, making the position neutral to
        the common move. Turning it off is the control that shows how much of
        any result is simply market direction.
    """

    lookback: int = 24
    basket: int = 3
    hold: int = 24
    demean: bool = True

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise RelativeError(f"lookback must be at least 2, got {self.lookback}")
        if self.basket < 1:
            raise RelativeError(f"basket must be at least 1, got {self.basket}")


def deviations(log_prices: pd.DataFrame, config: CrossSection) -> pd.DataFrame:
    """Each instrument's recent return, relative to the cross-section.

    Causal by construction: the return at row *t* spans *t - lookback* to *t*,
    and the cross-sectional mean subtracted from it is taken over the same row.
    Nothing here reads forward.
    """
    returns = log_prices.diff(config.lookback) * 1e4
    if not config.demean:
        return returns
    # Subtracting the row mean is what makes the signal relative rather than
    # directional: a day when everything rose contributes nothing.
    return returns.sub(returns.mean(axis=1), axis=0)


def weights(deviation: pd.DataFrame, config: CrossSection) -> pd.DataFrame:
    """Long the most depressed names, short the most extended, equally weighted.

    The sign is reversion: an instrument that has *fallen* relative to its peers
    is bought. Weights on each side sum to 0.5 in absolute value, so the gross
    exposure is 1 whatever the basket size and results are comparable across
    configurations.
    """
    usable = deviation.notna().sum(axis=1)
    if (usable >= 2 * config.basket).sum() == 0:
        raise RelativeError(
            f"need {2 * config.basket} instruments per row; the panel has at most "
            f"{int(usable.max())}"
        )

    ranks = deviation.rank(axis=1, method="first")
    count = deviation.notna().sum(axis=1)
    out = pd.DataFrame(0.0, index=deviation.index, columns=deviation.columns)
    long_side = ranks.le(config.basket, axis=0)
    short_side = ranks.gt(count - config.basket, axis=0)
    enough = count >= 2 * config.basket
    out = out.mask(long_side.where(enough, other=False), 0.5 / config.basket)
    out = out.mask(short_side.where(enough, other=False), -0.5 / config.basket)
    return out.fillna(0.0)


def backtest(
    log_prices: pd.DataFrame,
    config: CrossSection,
    *,
    cost_bp_per_name: float | pd.Series,
    stride: int | None = None,
) -> pd.DataFrame:
    """Form the basket every ``stride`` rows, hold it, and count every leg.

    ``cost_bp_per_name`` is charged on each name entered *and* exited, because a
    cross-sectional position is not one trade — it is one trade per name, and a
    backtest that charges it once is measuring a strategy nobody can run.
    """
    stride = stride or config.hold
    deviation = deviations(log_prices, config)
    basket = weights(deviation, config)
    values = log_prices.to_numpy()

    rows = []
    for start in range(config.lookback, len(log_prices) - config.hold, stride):
        held = basket.iloc[start].to_numpy()
        if not np.any(held):
            continue
        forward = (values[start + config.hold] - values[start]) * 1e4
        ok = np.isfinite(forward) & (held != 0.0)
        if not ok.any():
            continue

        gross = float(np.sum(held[ok] * forward[ok]))
        names = int(ok.sum())
        cost = cost_bp_per_name
        if isinstance(cost, pd.Series):
            cost = float(cost.iloc[start])
        # Every name pays its own round trip, weighted the same way the position
        # is weighted.
        charged = float(np.sum(np.abs(held[ok])) * cost)
        rows.append(
            {
                "at": start,
                "names": names,
                "gross_bp": gross,
                "cost_bp": charged,
                "net_bp": gross - charged,
            }
        )
    return pd.DataFrame(rows)
