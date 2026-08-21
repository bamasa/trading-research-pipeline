"""The end-to-end pipeline: its honesty properties, not its numbers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.data.grid import to_grid
from trading_research.pipeline.discovery import _with_neighbourhood

# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def _book(timestamps) -> pd.DataFrame:
    stamps = pd.to_datetime(timestamps)
    n = len(stamps)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "bid_price_0": np.linspace(100.0, 101.0, n),
            "ask_price_0": np.linspace(100.1, 101.1, n),
        }
    )


def test_grid_requires_the_columns_it_reads() -> None:
    with pytest.raises(KeyError, match="bid_price_0"):
        to_grid(pd.DataFrame({"timestamp": pd.to_datetime(["2024-02-01"])}))


def test_grid_refuses_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty"):
        to_grid(_book([]).iloc[:0])


def test_a_missing_day_stays_missing() -> None:
    """The bug this module records: a whole-span resample invents days."""
    book = pd.concat(
        [
            _book([f"2024-02-01 00:00:{s:02d}" for s in range(0, 50, 5)]),
            _book([f"2024-02-04 00:00:{s:02d}" for s in range(0, 50, 5)]),
        ],
        ignore_index=True,
    )
    days = set(to_grid(book)["timestamp"].dt.date.astype(str))
    assert days == {"2024-02-01", "2024-02-04"}


# ---------------------------------------------------------------------------
# Neighbourhood selection
# ---------------------------------------------------------------------------


def _surface() -> pd.DataFrame:
    rows = []
    for lookback in (100, 200, 300):
        for hold in (100, 200, 300):
            rows.append(
                {
                    "lookback_s": lookback,
                    "hold_s": hold,
                    "rate": 20.0,
                    "net_per_trade_bp": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_an_isolated_peak_does_not_win() -> None:
    """The maximum of a noisy surface is whichever cell noise favoured.

    A lone spike surrounded by losses must lose to a flat profitable region --
    that is the whole point of scoring neighbourhoods, and it is the selection
    mistake behind two of this project's killed findings.
    """
    surface = _surface()
    # A spike in the corner, surrounded by deep losses.
    surface.loc[(surface.lookback_s == 300) & (surface.hold_s == 300), "net_per_trade_bp"] = 60.0
    surface.loc[(surface.lookback_s == 300) & (surface.hold_s == 200), "net_per_trade_bp"] = -30.0
    surface.loc[(surface.lookback_s == 200) & (surface.hold_s == 300), "net_per_trade_bp"] = -30.0

    scored = _with_neighbourhood(surface)
    winner = scored.loc[scored["neighbourhood_bp"].idxmax()]
    assert not (winner["lookback_s"] == 300 and winner["hold_s"] == 300)


def test_a_smooth_region_beats_its_own_members_alone() -> None:
    surface = _surface()
    # A genuine region: the centre and all its neighbours are good.
    for lookback in (100, 200, 300):
        for hold in (100, 200, 300):
            if lookback == 200 or hold == 200:
                surface.loc[
                    (surface.lookback_s == lookback) & (surface.hold_s == hold),
                    "net_per_trade_bp",
                ] = 20.0
    scored = _with_neighbourhood(surface)
    # Neighbourhood scores tie broadly on a small grid, so selection breaks
    # ties by the cell's own value -- mirroring what run() does. Without that
    # tie-break a corner outside the region can win on row order, which is how
    # this test caught a real defect in the first version of the selection.
    winner = scored.sort_values(["neighbourhood_bp", "net_per_trade_bp"]).iloc[-1]
    assert winner["lookback_s"] == 200 or winner["hold_s"] == 200
    assert winner["net_per_trade_bp"] == 20.0


def test_rates_are_not_mixed_across_neighbourhoods() -> None:
    """Cells at different trade rates are different strategies, not neighbours."""
    surface = pd.concat([_surface(), _surface().assign(rate=60.0)], ignore_index=True)
    surface.loc[surface.rate == 60.0, "net_per_trade_bp"] = -50.0
    scored = _with_neighbourhood(surface)
    slow = scored[scored.rate == 20.0]
    assert (slow["neighbourhood_bp"] == 5.0).all(), "the losing rate must not bleed in"
