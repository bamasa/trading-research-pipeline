"""Shared fixtures.

The synthetic datasets here are small and module-scoped: generation is cheap but
not free, and every test that needs "a valid book" should get the same one so a
failure points at the code under test rather than at a different random draw.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from trading_research.data import synthetic
from trading_research.market_making.events import DayEvents


@pytest.fixture(scope="session")
def config() -> synthetic.SyntheticConfig:
    """A small, fast configuration exercising every part of the generator."""
    return synthetic.SyntheticConfig(n_steps=3_000, depth=5, seed=7)


@pytest.fixture(scope="session")
def dataset(config: synthetic.SyntheticConfig) -> synthetic.SyntheticDataset:
    return synthetic.generate(config)


@pytest.fixture(scope="session")
def book(dataset: synthetic.SyntheticDataset):
    return dataset.book


@pytest.fixture(scope="session")
def trades(dataset: synthetic.SyntheticDataset):
    return dataset.trades


def _write_market_day(events: DayEvents, root: Path, *, funding: bool = True) -> None:
    """Put a synthetic market-making day on disk in the layout the downloaders write.

    ``root/book``, ``root/trades`` and, with ``funding``, ``root/funding``, one
    parquet file per instrument-day, as :func:`market_making.events.load_day`
    reads them.
    """
    import numpy as np
    import pandas as pd

    tick, symbol = events.spec.tick, events.spec.symbol
    name = f"{events.day.isoformat()}.parquet"
    book: dict[str, object] = {"timestamp": pd.to_datetime(events.book_ts, unit="ns", utc=True)}
    for level in range(events.depth):
        book[f"bid_price_{level}"] = np.round(events.bid_px[:, level] * tick, 8)
        book[f"bid_size_{level}"] = events.bid_sz[:, level]
        book[f"ask_price_{level}"] = np.round(events.ask_px[:, level] * tick, 8)
        book[f"ask_size_{level}"] = events.ask_sz[:, level]
    prints = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(events.trade_ts, unit="ns", utc=True),
            "price": np.round(events.trade_px * tick, 8),
            "size": events.trade_sz,
            "aggressor": events.trade_aggressor.astype(np.int64),
        }
    )
    planes = {"book": pd.DataFrame(book), "trades": prints}
    if funding:
        start = pd.Timestamp(events.day.isoformat(), tz="UTC")
        stamps = pd.date_range(start, periods=3, freq="8h")
        planes["funding"] = pd.DataFrame({"timestamp": stamps, "rate": np.full(3, 1e-4)})
    for plane, frame in planes.items():
        (root / plane / symbol).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(root / plane / symbol / name, index=False)


@pytest.fixture(scope="session")
def write_market_day() -> Callable[..., None]:
    """A function writing one synthetic market-making day under a root."""
    return _write_market_day
