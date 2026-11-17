"""Synthetic print days and a local stand-in for the archive, for the tests and the dry run.

Nothing here touches the network. :func:`random_day` draws a day of prints on a
tick grid around a random-walk touch with a one-tick spread, some prints
sweeping through it; :func:`csv_gz` writes a frame in Bybit's file format
(optionally newest first); :class:`FakeArchive` answers the archive's and the
funding endpoint's URLs from memory, 404 for a day it does not hold.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
from collections.abc import Mapping
from datetime import date
from email.message import Message
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_DAY, day_start_ns
from trading_research.mm_bybit import archive
from trading_research.spot import prints as contract


def frame(
    day: date, ts_ns: np.ndarray, price: np.ndarray, size: np.ndarray, aggressor: np.ndarray
) -> pd.DataFrame:
    """Prints in the contract from arrays, in the given order; every stamp on ``day``."""
    n = len(ts_ns)
    start = day_start_ns(day)
    stamps = np.asarray(ts_ns, dtype=np.int64)
    if n and (stamps.min() < start or stamps.max() >= start + NS_PER_DAY):
        raise ValueError(f"a synthetic print is stamped off {day}")
    return contract.assemble(
        pd.Series(pd.to_datetime(np.asarray(ts_ns, dtype=np.int64), unit="ns", utc=True)),
        pd.Series(np.asarray(price, dtype=np.float64)),
        pd.Series(np.asarray(size, dtype=np.float64)),
        np.asarray(aggressor, dtype=np.int8),
        pd.Series(np.arange(n, dtype=np.int64)),
        np.zeros(n, dtype=bool),
        archive.SOURCE,
    )


def random_day(
    day: date,
    *,
    seed: int = 0,
    n_prints: int = 20_000,
    tick: float = 0.5,
    start_price: float = 30_000.0,
    move_prob: float = 0.05,
    sweep_prob: float = 0.03,
) -> pd.DataFrame:
    """A day of prints at the touch of a random walk, a few sweeping through it."""
    rng = np.random.default_rng(seed)
    start = day_start_ns(day)
    stamps = np.sort(rng.integers(0, NS_PER_DAY // 1_000_000, size=n_prints)) * 1_000_000
    bid = np.round(start_price / tick) + np.cumsum(rng.choice((-1, 0, 1), size=n_prints,
                                                              p=(move_prob, 1 - 2 * move_prob, move_prob)))  # fmt: skip
    side = rng.choice((1, -1), size=n_prints)
    through = rng.random(n_prints) < sweep_prob
    ticks = np.where(side > 0, bid + 1 + through, bid - through)
    size = np.round(rng.exponential(0.05, size=n_prints) + 0.001, 3)
    return frame(day, start + stamps, ticks * tick, size, side)


def funding(day: date, rates: tuple[float, float, float] = (1e-4, 1e-4, 1e-4)) -> pd.DataFrame:
    """Three settlements on the 8-hour grid, in the history's columns."""
    start = day_start_ns(day)
    stamps = start + np.arange(3, dtype=np.int64) * 8 * 3600 * 1_000_000_000
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(stamps, unit="ns", utc=True), "rate": list(rates)}
    )


def csv_gz(prints: pd.DataFrame, *, descending: bool = False) -> bytes:
    """A frame in Bybit's daily file format, gzip-compressed."""
    ns = prints["timestamp"].astype("int64").to_numpy()
    text = pd.DataFrame(
        {
            "timestamp": [f"{v // 1_000_000_000}.{v % 1_000_000_000:09d}"[:-5] for v in ns],
            "symbol": "BTCUSDT",
            "side": np.where(prints["aggressor"].to_numpy() > 0, "Buy", "Sell"),
            "size": prints["size"].to_numpy(),
            "price": prints["price"].to_numpy(),
            "tickDirection": "ZeroPlusTick",
            "trdMatchID": [f"id-{k}" for k in range(len(ns))],
        }
    )
    if descending:
        text = text.iloc[::-1]
    buffer = io.StringIO()
    text.to_csv(buffer, index=False)
    return gzip.compress(buffer.getvalue().encode())


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FakeArchive:
    """Answers archive and funding URLs from memory, like ``urllib.request.urlopen``."""

    def __init__(self, files: Mapping[date, bytes], settlements: pd.DataFrame | None = None):
        self.files = dict(files)
        self.settlements = settlements
        self.requests: list[str] = []

    def __call__(self, url: str, timeout: float = 0.0) -> _Response:  # noqa: ARG002
        self.requests.append(url)
        for day, payload in self.files.items():
            if url == archive.url(day):
                return _Response(payload)
        if "funding/history" in url and self.settlements is not None:
            rows = [
                {"symbol": "BTCUSDT", "fundingRate": str(rate),
                 "fundingRateTimestamp": str(int(ts.value // 1_000_000))}
                for ts, rate in zip(self.settlements["timestamp"], self.settlements["rate"],
                                    strict=True)
            ][::-1]  # fmt: skip
            body: dict[str, Any] = {"retCode": 0, "result": {"list": rows}}
            return _Response(json.dumps(body).encode())
        raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
