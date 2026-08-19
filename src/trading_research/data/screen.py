"""Ranking instruments before any model is fitted.

Step 0 of the pipeline, and until now the only one done by hand. It is not a
cosmetic choice: the instrument fixes the cost floor, and every result in this
project is a comparison against that floor.

What to rank on
---------------
Not the cost. A cheap instrument that never moves is worse than an expensive
one that does, and ranking on the fee alone would put a one-tick major at the
top of a list it belongs at the bottom of.

The right quantity follows from the arithmetic the rest of the project
established. Expected edge per trade is roughly

    edge = information coefficient * volatility of the move

and it has to clear the round trip. **Both halves are measured here**, because
an early version of this screen measured only the second and ranked XMRUSDT
third of forty-four. XMRUSDT moves nine times as far as BTCUSDT per two
minutes; its queue imbalance correlates with the next two minutes at 0.0002
against BTCUSDT's 0.049. It has enormous room and nothing to fill it with, and
every model tried on it did worse than on the instrument it was supposed to
replace.

So two numbers, and the product of them:

**headroom** — the share of moments whose move over the horizon exceeds the cost
of trading it. An upper bound assuming the direction is predicted perfectly, so
no model can beat it and none is needed to compute it.

**predictability** — the correlation between queue imbalance now and the move
over the horizon. One feature, no model, no fitting. It is not the best
predictor available, and that is the point: it is a cheap proxy for whether the
book says anything at all about this instrument.

Volatility without predictability is a casino, and predictability without
volatility cannot pay for the round trip. Ranking on either alone finds the
wrong instrument, and the screen now says which half is missing rather than
producing one number that hides it.

An instrument scoring low on both is definitely not tradeable, which is what a
screen is for: it eliminates, it does not select.

Deliberately model-free and direction-free
------------------------------------------
The screen does not know what will be traded or how. That is the point — it
bounds every strategy at once, so it can be run before deciding on any of them.
Ranking instruments by how well one particular model did on them would be
choosing the instrument to suit the model, which is the wrong way round and
much more expensive.

Cheap by construction
---------------------
Two or three days of best bid and ask per instrument, resampled to a second.
Nothing here needs the full pipeline, and a screen that costs as much as a
study is not a screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.backtest.costs import TakerCosts
from trading_research.data.schema import (
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
    mid_price,
)

BP: float = 1e4


@dataclass(frozen=True)
class ScreenResult:
    """What one instrument looks like before anything is fitted."""

    symbol: str
    rows: int
    days: int
    median_spread_bp: float
    round_trip_bp: float
    volatility_bp: float
    headroom: float
    moves_per_day: float
    #: Correlation of queue imbalance with the forward move. The other half of
    #: the edge identity, and the half that varies most between instruments.
    information_coefficient: float = float("nan")

    def as_row(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "days": self.days,
            "median_spread_bp": self.median_spread_bp,
            "round_trip_bp": self.round_trip_bp,
            "volatility_bp": self.volatility_bp,
            "headroom": self.headroom,
            "moves_per_day": self.moves_per_day,
            "information_coefficient": self.information_coefficient,
            # How many round trips a typical move covers. Below one, a
            # perfectly predicted average move still does not pay.
            "move_over_cost": self.volatility_bp / self.round_trip_bp,
            # Both halves together: what a trade is worth if the one cheap
            # feature is all the edge there is. This is the column to rank on.
            "edge_over_cost": abs(self.information_coefficient)
            * self.volatility_bp
            / self.round_trip_bp,
        }


def screen_frame(
    book: pd.DataFrame,
    symbol: str,
    *,
    horizon_s: float,
    costs: TakerCosts | None = None,
) -> ScreenResult:
    """Measure one instrument's headroom at one horizon."""
    costs = costs or TakerCosts()
    mid = mid_price(book).to_numpy(dtype="float64")
    spread_bp = (
        (book[ask_price_col(0)] - book[bid_price_col(0)]).to_numpy(dtype="float64") / mid * BP
    )
    median_spread = float(np.nanmedian(spread_bp))
    round_trip = (
        2 * costs.fee_bp_per_side
        + costs.half_spread_multiplier * median_spread
        + 2 * costs.slippage_bp
    )

    seconds = float(pd.Series(book["timestamp"]).diff().dt.total_seconds().median())
    step = max(1, round(horizon_s / max(seconds, 1e-9)))
    if step >= len(mid):
        raise ValueError(f"{symbol}: {len(mid)} rows cannot support a {horizon_s}s horizon")

    signed_move = np.log(mid[step:] / mid[:-step]) * BP
    move = np.abs(signed_move)
    days = int(pd.Series(book["timestamp"]).dt.date.nunique())
    clears = move > round_trip

    # The other half of the identity, from the one feature that carries most of
    # the signal. Computed here rather than left to the modelling stage because
    # an instrument the book says nothing about should be eliminated before
    # anything is fitted, not after.
    bid_size = book[bid_size_col(0)].to_numpy(dtype="float64")[:-step]
    ask_size = book[ask_size_col(0)].to_numpy(dtype="float64")[:-step]
    total = bid_size + ask_size
    imbalance = np.divide(
        bid_size - ask_size, total, out=np.full_like(total, np.nan), where=total > 0
    )
    usable = np.isfinite(imbalance) & np.isfinite(signed_move)
    ic = (
        float(np.corrcoef(imbalance[usable], signed_move[usable])[0, 1])
        if usable.sum() > 100
        else float("nan")
    )

    return ScreenResult(
        symbol=symbol,
        rows=len(book),
        days=days,
        median_spread_bp=median_spread,
        round_trip_bp=round_trip,
        volatility_bp=float(np.nanstd(move)),
        headroom=float(np.nanmean(clears)),
        information_coefficient=ic,
        # Non-overlapping equivalent: how many times a day a move of that size
        # is available, if each one could be traded once.
        moves_per_day=float(clears.sum() / max(days, 1) / step),
    )


def screen_directory(
    raw_root: Path | str,
    *,
    horizon_s: float = 120.0,
    kind: str = "bookTicker",
    max_days: int | None = None,
    costs: TakerCosts | None = None,
) -> pd.DataFrame:
    """Screen every instrument present under ``raw_root``.

    Ordered by ``edge_over_cost``, which multiplies the two halves. Ranking on
    headroom alone put an instrument with no predictability at the top; ranking
    on the coefficient alone would put a still one there. Instruments that cannot be read or have
    too little data are reported with the reason rather than dropped, because a
    screen that silently omits candidates is worse than one that says why.
    """
    root = Path(raw_root)
    rows: list[dict[str, object]] = []

    for directory in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        files = sorted((directory / kind).glob("*.parquet"))
        if max_days is not None:
            files = files[:max_days]
        if not files:
            rows.append({"symbol": directory.name, "skipped": f"no {kind} data"})
            continue
        try:
            book = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            result = screen_frame(book, directory.name, horizon_s=horizon_s, costs=costs)
        except Exception as exc:  # a bad instrument must not end the screen
            rows.append({"symbol": directory.name, "skipped": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(result.as_row())

    table = pd.DataFrame(rows)
    if "edge_over_cost" in table.columns:
        table = table.sort_values("edge_over_cost", ascending=False, na_position="last")
    return table.reset_index(drop=True)
