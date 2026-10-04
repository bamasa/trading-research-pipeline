"""The end-to-end pipeline: its honesty properties, not its numbers."""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# The execution stage: taker against passive entry where there are prints
# ---------------------------------------------------------------------------


def _panel(
    tmp_path: Path, with_prints: tuple[str, ...] = ("S0", "S1")
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray, int]:
    """Six instruments, three days on the 5 s grid, prints on disk for some."""
    from trading_research.pipeline.discovery import ROWS_PER_DAY

    rng = np.random.default_rng(11)
    n = 3 * ROWS_PER_DAY
    index = pd.date_range("2020-01-06", periods=n, freq="5s", tz="UTC")
    common = np.cumsum(rng.normal(0, 2e-4, n))
    panel, books = {}, {}
    for k in range(6):
        symbol = f"S{k}"
        mid = 100.0 * np.exp(common + np.cumsum(rng.normal(0, 3e-4, n)))
        tick = 0.01
        bid = np.floor(mid / tick) * tick
        ask = bid + tick * rng.integers(1, 3, n)
        panel[symbol] = pd.Series(np.log((bid + ask) / 2), index=index)
        books[symbol] = pd.DataFrame(
            {
                "bid_price_0": bid,
                "ask_price_0": ask,
                "bid_size_0": rng.uniform(1, 10, n),
                "ask_size_0": rng.uniform(1, 10, n),
            },
            index=index,
        )
        if symbol not in with_prints:
            continue
        rows = rng.choice(n, size=n // 2, replace=False)
        side = rng.choice([-1, 1], len(rows))
        prints = pd.DataFrame(
            {
                "timestamp": index[rows] + pd.to_timedelta(rng.uniform(0, 4.9, len(rows)), "s"),
                "price": np.where(side < 0, bid[rows], ask[rows]),
                "size": rng.exponential(4.0, len(rows)),
                "aggressor": side,
            }
        ).sort_values("timestamp")
        for day, part in prints.groupby(prints["timestamp"].dt.date):
            target = tmp_path / "trades" / symbol / f"{day.isoformat()}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(target, index=False)
    frame = pd.DataFrame(panel)
    return frame, books, frame.to_numpy(), 2 * ROWS_PER_DAY


def test_the_execution_stage_compares_modes_only_where_there_are_prints(tmp_path: Path) -> None:
    from trading_research.backtest.costs import TakerCosts
    from trading_research.pipeline.discovery import _trade_all, execution_stage

    panel, books, values, cut = _panel(tmp_path)
    costs = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
    _, thresholds = _trade_all(panel, books, values, costs, 24, 24, 60.0, slice(0, cut))
    stage = execution_stage(
        panel, books, values, costs, (24, 24, 60.0), cut, thresholds, tmp_path / "trades"
    )
    assert "2 of 6 instruments" in stage.decision and "chosen on the search block" in stage.decision
    table = stage.table.set_index(["block", "mode"])
    assert (table["instruments"] == 2).all()
    assert table.loc[("held out", "taker"), "fill_rate"] == 1.0
    assert 0.0 < table.loc[("held out", "passive_entry"), "fill_rate"] < 1.0
    # The held-out block chose nothing: its figures are the same whichever mode won.
    taken = execution_stage(
        panel, books, values, costs, (24, 24, 60.0), cut, thresholds, tmp_path / "trades",
        take_passive=True,
    )  # fmt: skip
    assert taken.decision.startswith("passive_entry (taken as asked)")
    pd.testing.assert_frame_equal(taken.table, stage.table)


def test_without_prints_the_stage_says_so(tmp_path: Path) -> None:
    from trading_research.backtest.costs import TakerCosts
    from trading_research.pipeline.discovery import execution_stage

    panel, books, values, cut = _panel(tmp_path, with_prints=())
    stage = execution_stage(
        panel, books, values, TakerCosts(), (24, 24, 60.0), cut, {}, tmp_path / "trades"
    )
    assert stage.decision == "taker only: none of 6 instruments has prints"
    with pytest.raises(ValueError, match="execution must be one of"):
        from trading_research.pipeline.discovery import run

        run(execution="sometimes", fetch=False)
