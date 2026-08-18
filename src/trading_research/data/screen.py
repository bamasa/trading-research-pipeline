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
established. Expected edge per trade is roughly the information coefficient
times the volatility of the move over the horizon, and it has to clear the
round trip. So the screen measures the second half of that directly:

**headroom** — the share of moments whose move over the horizon exceeds the
cost of trading it. That is an upper bound assuming the direction is predicted
perfectly, so no model can beat it, and it needs no model to compute.

An instrument with 5% headroom is not necessarily tradeable. An instrument with
0.1% is definitely not, and that is what a screen is for: it eliminates, it does
not select.

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
from trading_research.data.schema import ask_price_col, bid_price_col, mid_price

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

    def as_row(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "days": self.days,
            "median_spread_bp": self.median_spread_bp,
            "round_trip_bp": self.round_trip_bp,
            "volatility_bp": self.volatility_bp,
            "headroom": self.headroom,
            "moves_per_day": self.moves_per_day,
            # How many round trips a typical move covers. Below one, a
            # perfectly predicted average move still does not pay.
            "move_over_cost": self.volatility_bp / self.round_trip_bp,
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

    move = np.abs(np.log(mid[step:] / mid[:-step]) * BP)
    days = int(pd.Series(book["timestamp"]).dt.date.nunique())
    clears = move > round_trip

    return ScreenResult(
        symbol=symbol,
        rows=len(book),
        days=days,
        median_spread_bp=median_spread,
        round_trip_bp=round_trip,
        volatility_bp=float(np.nanstd(move)),
        headroom=float(np.nanmean(clears)),
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

    Ordered by headroom, best first. Instruments that cannot be read or have
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
    if "headroom" in table.columns:
        table = table.sort_values("headroom", ascending=False, na_position="last")
    return table.reset_index(drop=True)
