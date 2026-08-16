"""Horizon diagnostics: what can be predicted, and what is worth trading.

These two measurements answer the question a short-horizon project has to
answer before it fits anything, and they answer it without fitting anything.

:func:`information_coefficient_by_horizon`
    How strongly each feature correlates with the future return, as a function
    of how far ahead you look. This is predictability.

:func:`breakeven_by_horizon`
    What share of moments have a future move larger than the round-trip cost.
    This is tradeability, and it is an upper bound: it already assumes the
    direction is predicted perfectly.

Run together they expose the central tension of the problem. On BTCUSDT
futures in early 2024, queue imbalance correlates with the next second's return
at about 0.27 and with the next five minutes' at about 0.04 — while the share
of moments whose move clears an 11 bp taker round trip runs the other way, from
0.01% at one second to 25% at five minutes.

**The horizon where prediction works and the horizon where trading pays do not
overlap.** Where there is signal the moves are too small to cover the fee;
where the moves are large enough the signal has decayed to noise. No model
choice changes that, which is why these diagnostics belong before the modelling
rather than after it.

One caveat travels with the sub-second numbers. Part of the correlation there
is mechanical: the mid oscillates between bid and ask, and the microprice is
simply a better estimate of the underlying price. That component is not
tradeable — capturing it means crossing the spread that creates it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from lobml.data.schema import mid_price
from lobml.features.registry import REGISTRY, Registry

#: Horizons in observations. On a 100 ms grid these span 0.1 s to 10 minutes.
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 30, 50, 100, 300, 600, 1200, 3000, 6000)


def forward_log_return(book: pd.DataFrame, horizon: int) -> np.ndarray:
    """Log return of mid over ``horizon`` observations, NaN where unknown."""
    mid = mid_price(book).to_numpy(dtype="float64")
    out = np.full(len(mid), np.nan)
    if horizon < len(mid):
        out[:-horizon] = np.log(mid[horizon:] / mid[:-horizon])
    return out


def information_coefficient_by_horizon(
    book: pd.DataFrame,
    features: Sequence[str],
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """Correlation of each feature with the future return, per horizon.

    Plain Pearson correlation on the raw values. Deliberately simple: the point
    is the *shape* of the decay, and a more elaborate statistic would obscure it
    without changing it.

    Returned with horizons as rows so the decay reads top to bottom.
    """
    reg = registry if registry is not None else REGISTRY
    values = {name: reg.get(name)(book).to_numpy(dtype="float64") for name in features}

    rows = []
    for horizon in horizons:
        forward = forward_log_return(book, horizon)
        row: dict[str, float] = {"horizon": horizon, "seconds": np.nan}
        for name, series in values.items():
            usable = ~np.isnan(series) & ~np.isnan(forward)
            row[name] = (
                float(np.corrcoef(series[usable], forward[usable])[0, 1])
                if usable.sum() > 2
                else np.nan
            )
        rows.append(row)

    table = pd.DataFrame(rows).set_index("horizon")
    return table.drop(columns=["seconds"])


def breakeven_by_horizon(
    book: pd.DataFrame,
    cost_bp: float,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.Series:
    """Share of moments whose future move exceeds ``cost_bp``, per horizon.

    An upper bound under perfect foresight. If this is small at a horizon, no
    classifier makes that horizon tradeable — the moves simply are not large
    enough to pay the fee.
    """
    shares = {}
    for horizon in horizons:
        move = np.abs(forward_log_return(book, horizon)) * 1e4
        usable = ~np.isnan(move)
        shares[horizon] = float((move[usable] > cost_bp).mean()) if usable.any() else np.nan
    return pd.Series(shares, name="breakeven_share").rename_axis("horizon")


def horizon_summary(
    book: pd.DataFrame,
    features: Sequence[str],
    cost_bp: float,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    grid_ms: int = 100,
    registry: Registry | None = None,
) -> pd.DataFrame:
    """Predictability and tradeability side by side, per horizon.

    Putting the two columns in one table is the point. Read separately, each
    looks like an ordinary decay curve; read together, they show that the two
    curves move in opposite directions and that their useful ranges do not
    meet.
    """
    ic = information_coefficient_by_horizon(book, features, horizons=horizons, registry=registry)
    breakeven = breakeven_by_horizon(book, cost_bp, horizons=horizons)

    out = ic.copy()
    out.insert(0, "seconds", [h * grid_ms / 1000.0 for h in out.index])
    out["breakeven_share"] = breakeven
    return out


def format_horizon_summary(summary: pd.DataFrame) -> str:
    """Render the summary as a readable table."""
    display = summary.copy()
    display["seconds"] = display["seconds"].map(
        lambda s: f"{s:.1f}s" if s < 60 else f"{s / 60:.0f}min"
    )
    display["breakeven_share"] = display["breakeven_share"].map(lambda v: f"{v:.2%}")
    for column in display.columns:
        if column not in ("seconds", "breakeven_share"):
            display[column] = display[column].map(lambda v: f"{v:+.4f}")
    return display.to_string()
