"""Synthetic spot prints for the spot study's signal tests, and the guard against the network.

Prints follow the contract of ``spot/prints.py`` (``timestamp``, ``price``,
``size``, ``aggressor`` and ``rpi``), in archive order: several share a stamp
(a taker sweeping levels), sizes are often below the 0.010 BTC clip so sweeps
run out, a few prints are flagged ``rpi``, and the day has a quiet stretch.
Nothing here reads a file or the network; :func:`no_network` makes any attempt
to connect fail in the test modules that use it.
"""

from __future__ import annotations

import socket
from datetime import date

import numpy as np
import pandas as pd
import pytest

DAY = date(2026, 3, 3)
TICK = 0.01


def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every socket connection attempt raise."""

    def refuse(*_: object, **__: object) -> None:
        raise OSError("the spot tests must not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def make_prints(
    n: int = 2_000,
    *,
    seed: int = 0,
    day: date = DAY,
    start_s: float = 30.0,
    mean_gap_s: float = 0.4,
    quiet_at: int | None = 1_200,
    quiet_s: float = 90.0,
    rpi_share: float = 0.03,
    tail_at_s: float | None = None,
) -> pd.DataFrame:
    """``n`` prints from ``start_s`` after midnight, ties and sweeps included.

    ``tail_at_s`` moves the last tenth of the prints to that many seconds after
    midnight, for tests of the day's last minutes.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(mean_gap_s, n) * (rng.random(n) > 0.3)
    if quiet_at is not None and quiet_at < n:
        gaps[quiet_at] += quiet_s
    origin = pd.Timestamp(day, tz="UTC").value + round(start_s * 1e9)
    ms = np.floor(np.cumsum(gaps) * 1e3).astype(np.int64)
    if tail_at_s is not None:
        tail = n - n // 10
        ms[tail:] = round((tail_at_s - start_s) * 1e3) + (ms[tail:] - ms[tail])
    stamps = origin + ms * 1_000_000
    side = np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8)
    # Runs: a print often repeats the previous side, as a sweep does.
    repeat = (rng.random(n) < 0.5) & (np.r_[1, np.diff(stamps)] == 0)
    for i in np.flatnonzero(repeat):
        side[i] = side[i - 1]
    mid = 60_000.0 + np.cumsum(rng.choice([-1, 0, 0, 1], n)) * TICK * 5
    price = np.round(mid + side * TICK * (1 + rng.integers(0, 3, n)), 2)
    size = np.round(rng.exponential(0.004, n) + 0.0001, 4)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps, unit="ns", utc=True),
            "price": price,
            "size": size,
            "aggressor": side,
            "rpi": rng.random(n) < rpi_share,
        }
    )


def test_the_synthetic_prints_have_ties_sweeps_and_flags() -> None:
    prints = make_prints()
    stamps = prints["timestamp"].astype("int64").to_numpy()
    assert (np.diff(stamps) >= 0).all()
    assert (np.diff(stamps) == 0).sum() > 100
    assert prints["rpi"].any() and not prints["rpi"].all()
    assert (prints["size"] < 0.010).mean() > 0.5
    assert set(prints["aggressor"].unique()) == {-1, 1}
    assert prints["timestamp"].dt.date.eq(DAY).all()


def test_the_network_guard_refuses_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)
    with pytest.raises(OSError, match="must not reach the network"):
        socket.create_connection(("127.0.0.1", 9))
