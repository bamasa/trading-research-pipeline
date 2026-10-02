"""Determinism: the same inputs give the same outputs, however the work is split."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.market_making.events import DayEvents
from trading_research.market_making.quoters import TouchQuoter
from trading_research.market_making.simulator import SimConfig, run_days, simulate_day
from trading_research.market_making.synthetic import SYNTHETIC_DAY, random_market

CONFIG = SimConfig(clip_notional=250.0, clip_touch_share=1e9, warmup_s=0.0)


def test_the_same_inputs_give_identical_results() -> None:
    market = random_market(4, n_snapshots=800, prints_per_snapshot=1.5, funding=True)
    first = simulate_day(market, TouchQuoter(), CONFIG)
    second = simulate_day(market, TouchQuoter(), CONFIG)
    for name in ("fills", "orders", "equity", "crossed"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert first.counters == second.counters
    assert first.decomposition == second.decomposition
    assert len(first.fills) > 50


def _write(events: DayEvents, books: Path, trades: Path) -> None:
    """Put a synthetic day on disk in the layout the downloaders write."""
    tick = events.spec.tick
    symbol = events.spec.symbol
    name = f"{events.day.isoformat()}.parquet"
    book: dict[str, object] = {"timestamp": pd.to_datetime(events.book_ts, unit="ns", utc=True)}
    for level in range(events.depth):
        book[f"bid_price_{level}"] = np.round(events.bid_px[:, level] * tick, 8)
        book[f"bid_size_{level}"] = events.bid_sz[:, level]
        book[f"ask_price_{level}"] = np.round(events.ask_px[:, level] * tick, 8)
        book[f"ask_size_{level}"] = events.ask_sz[:, level]
    (books / symbol).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(book).to_parquet(books / symbol / name, index=False)
    prints = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(events.trade_ts, unit="ns", utc=True),
            "price": np.round(events.trade_px * tick, 8),
            "size": events.trade_sz,
            "aggressor": events.trade_aggressor.astype(np.int64),
        }
    )
    (trades / symbol).mkdir(parents=True, exist_ok=True)
    prints.to_parquet(trades / symbol / name, index=False)


def test_worker_count_does_not_change_results(tmp_path: Path) -> None:
    books, trades = tmp_path / "book", tmp_path / "trades"
    days = [SYNTHETIC_DAY + timedelta(days=i) for i in range(3)]
    for i, day in enumerate(days):
        _write(random_market(i, n_snapshots=500, day=day), books, trades)
    missing = SYNTHETIC_DAY + timedelta(days=3)  # no files at all

    def run(workers: int, cache: Path | None = None) -> pd.DataFrame:
        return run_days(
            "FUZZUSDT",
            [*reversed(days), missing],
            TouchQuoter,
            CONFIG,
            book_roots=[books],
            trades_root=trades,
            funding_root=None,
            workers=workers,
            cache=cache,
        )

    one = run(1)
    assert one["day"].tolist() == [d.isoformat() for d in [*days, missing]]
    assert (one["status"].iloc[:3] == "ok").all()
    assert (one["fills"].iloc[:3] > 0).all()
    pd.testing.assert_frame_equal(one, run(2))

    cached = run(1, tmp_path / "cache")
    pd.testing.assert_frame_equal(one, cached)
    written = sorted(p.name for p in (tmp_path / "cache").rglob("*.parquet"))
    assert written.count("fills.parquet") == 3
    pd.testing.assert_frame_equal(one, run(1, tmp_path / "cache"))  # read back, not re-run


def test_a_day_missing_either_plane_is_reported_not_simulated(tmp_path: Path) -> None:
    books, trades = tmp_path / "book", tmp_path / "trades"
    _write(random_market(0, n_snapshots=100), books, trades)
    (trades / "FUZZUSDT" / f"{SYNTHETIC_DAY.isoformat()}.parquet").unlink()
    frame = run_days(
        "FUZZUSDT",
        [SYNTHETIC_DAY],
        TouchQuoter,
        CONFIG,
        book_roots=[books],
        trades_root=trades,
        funding_root=None,
    )
    assert "no trades" in frame["status"].iloc[0]
    assert np.isnan(frame["net"].iloc[0])
