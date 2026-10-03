"""Determinism: the same inputs give the same outputs, however the work is split."""

from __future__ import annotations

from dataclasses import replace
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def _write(events: DayEvents, books: Path, trades: Path, *, gaps: int = 0) -> None:
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
    frame = pd.DataFrame(book)
    frame.attrs["sequence_gaps"] = gaps
    frame.to_parquet(books / symbol / name, index=False)
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


def _run(
    books: Path,
    trades: Path,
    days: list,
    cache: Path | None = None,
    funding: Path | None = None,
    config: SimConfig = CONFIG,
):
    return run_days(
        "FUZZUSDT",
        days,
        TouchQuoter,
        config,
        book_roots=[books],
        trades_root=trades,
        funding_root=funding,
        cache=cache,
    )


def test_the_cache_is_keyed_by_the_data_it_read(tmp_path: Path) -> None:
    """Different data under different roots never shares a cached answer, and
    rewriting a file under the same root invalidates it."""
    cache = tmp_path / "cache"
    a = (tmp_path / "a" / "book", tmp_path / "a" / "trades")
    b = (tmp_path / "b" / "book", tmp_path / "b" / "trades")
    _write(random_market(1, n_snapshots=500), *a)
    _write(random_market(2, n_snapshots=500), *b)

    first = _run(*a, [SYNTHETIC_DAY], cache)
    fresh = _run(*b, [SYNTHETIC_DAY])
    cached = _run(*b, [SYNTHETIC_DAY], cache)
    assert first["net"].iloc[0] != fresh["net"].iloc[0]
    assert cached["net"].iloc[0] == fresh["net"].iloc[0]

    _write(random_market(3, n_snapshots=500), *a)  # same path, new content
    rewritten = _run(*a, [SYNTHETIC_DAY], cache)
    assert rewritten["net"].iloc[0] == _run(*a, [SYNTHETIC_DAY])["net"].iloc[0]
    assert rewritten["net"].iloc[0] != first["net"].iloc[0]


def test_a_change_to_the_simulator_or_the_funding_root_invalidates_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_research.market_making import simulator

    books, trades, cache = tmp_path / "book", tmp_path / "trades", tmp_path / "cache"
    _write(random_market(1, n_snapshots=300), books, trades)

    def keys() -> int:
        return len([p for p in cache.iterdir() if p.is_dir()])

    _run(books, trades, [SYNTHETIC_DAY], cache)
    _run(books, trades, [SYNTHETIC_DAY], cache)
    assert keys() == 1
    monkeypatch.setattr(simulator, "source_fingerprint", lambda: "a different simulator")
    _run(books, trades, [SYNTHETIC_DAY], cache)
    assert keys() == 2
    funding = tmp_path / "funding"
    (funding / "FUZZUSDT").mkdir(parents=True)
    start = pd.Timestamp(SYNTHETIC_DAY.isoformat(), tz="UTC").value
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [start + h * 3600 * 10**9 for h in (0, 8, 16)], unit="ns", utc=True
            ),
            "rate": [1e-4, 1e-4, 1e-4],
        }
    ).to_parquet(funding / "FUZZUSDT" / f"{SYNTHETIC_DAY.isoformat()}.parquet", index=False)
    with_funding = _run(books, trades, [SYNTHETIC_DAY], cache, funding)
    assert keys() == 3
    assert "no_funding" not in with_funding["flags"].iloc[0]


def test_a_day_with_bad_data_is_skipped_with_its_reason(tmp_path: Path) -> None:
    """An off-grid print or an unexpected funding interval skips that day, says
    why, and the rest of the run goes on."""
    books, trades, funding = tmp_path / "book", tmp_path / "trades", tmp_path / "funding"
    days = [SYNTHETIC_DAY + timedelta(days=i) for i in range(3)]
    for i, day in enumerate(days):
        _write(random_market(i, n_snapshots=300, day=day), books, trades)
    off_grid = trades / "FUZZUSDT" / f"{days[1].isoformat()}.parquet"
    prints = pd.read_parquet(off_grid)
    prints.loc[0, "price"] = prints.loc[0, "price"] + 0.005  # half a tick
    prints.to_parquet(off_grid, index=False)
    (funding / "FUZZUSDT").mkdir(parents=True)
    for i, day in enumerate(days):
        start = pd.Timestamp(day.isoformat(), tz="UTC").value
        hours = (0, 4, 8, 12, 16, 20) if i == 2 else (0, 8, 16)
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [start + h * 3600 * 10**9 for h in hours], unit="ns", utc=True
                ),
                "rate": [1e-4] * len(hours),
            }
        ).to_parquet(funding / "FUZZUSDT" / f"{day.isoformat()}.parquet", index=False)

    # The synthetic days last half a minute: stop and flatten inside them.
    inside = CONFIG.with_(stop_quoting_at=time(0, 0, 20), flatten_at=time(0, 0, 25))
    frame = _run(books, trades, days, funding=funding, config=inside)
    assert frame["status"].iloc[0] == "ok"
    assert "tick" in frame["status"].iloc[1]
    assert "every 4 h" in frame["status"].iloc[2]
    assert frame["excluded"].tolist() == [False, True, True]


def test_a_day_with_sequence_gaps_is_flagged_and_excluded(tmp_path: Path) -> None:
    """The stored book counts its sequence gaps but cannot say where they fell,
    so a day with more than the allowed number is kept out of verdicts."""
    market = random_market(0, n_snapshots=200)
    gapped = replace(market, sequence_gaps=2)
    assert "sequence_gaps" in simulate_day(gapped, TouchQuoter(), CONFIG).flags
    assert "sequence_gaps" not in simulate_day(market, TouchQuoter(), CONFIG).flags
    tolerant = CONFIG.with_(max_sequence_gaps=5)
    assert "sequence_gaps" not in simulate_day(gapped, TouchQuoter(), tolerant).flags

    books, trades = tmp_path / "book", tmp_path / "trades"
    _write(market, books, trades, gaps=3)
    frame = _run(books, trades, [SYNTHETIC_DAY])
    assert frame["excluded"].iloc[0]
    assert "sequence_gaps" in frame["flags"].iloc[0]
    assert frame["sequence_gaps"].iloc[0] == 3


def _cells() -> list:
    from functools import partial

    from trading_research.market_making.quoters import SkewQuoter
    from trading_research.market_making.simulator import Cell

    skew = SkewQuoter(skew_bp=5.0, k=0.5, min_edge_bp=0.0, sigma_ref=5.0, maker_bp=0.0)
    return [
        Cell("S0", TouchQuoter, CONFIG),
        Cell("S1", partial(SkewQuoter, **{**skew.__dict__}), CONFIG.with_(soft_limit_clips=3.0)),
    ]


def test_cells_run_like_days_whatever_the_split(tmp_path: Path) -> None:
    from trading_research.market_making.simulator import SUMMARY_COLUMNS, run_cells

    books, trades = tmp_path / "book", tmp_path / "trades"
    days = [SYNTHETIC_DAY + timedelta(days=i) for i in range(3)]
    for i, day in enumerate(days):
        _write(random_market(10 + i, n_snapshots=400, day=day), books, trades)

    def run(workers: int, chunk: int, cache: Path | None = None) -> pd.DataFrame:
        frame = run_cells(
            "FUZZUSDT",
            list(reversed(days)),
            _cells(),
            book_roots=[books],
            trades_root=trades,
            funding_root=None,
            workers=workers,
            cache=cache,
            cells_per_job=chunk,
        )
        return frame.drop(columns="worker_peak_rss_mb")

    one = run(1, 1)
    assert one["cell"].tolist() == ["S0"] * 3 + ["S1"] * 3
    assert one["day"].tolist() == [d.isoformat() for d in days] * 2
    pd.testing.assert_frame_equal(one, run(2, 2))
    cached = run(1, 2, tmp_path / "cache")
    pd.testing.assert_frame_equal(one, cached)
    pd.testing.assert_frame_equal(one, run(2, 1, tmp_path / "cache"))
    alone = run_days(
        "FUZZUSDT",
        days,
        TouchQuoter,
        CONFIG,
        book_roots=[books],
        trades_root=trades,
        funding_root=None,
    )
    columns = list(SUMMARY_COLUMNS)
    pd.testing.assert_frame_equal(
        one[one["cell"] == "S0"][columns].reset_index(drop=True), alone[columns]
    )


def test_the_signals_a_day_read_are_part_of_its_cache_key(tmp_path: Path) -> None:
    from trading_research.market_making.quoters import (
        INDEX_SIGNAL,
        LEAN_TRIGGER_SIGNAL,
        ReversionLean,
        SkewQuoter,
    )
    from trading_research.market_making.signals import SignalTape
    from trading_research.market_making.simulator import Cell, day_start_ns, run_cells

    books, trades = tmp_path / "book", tmp_path / "trades"
    _write(random_market(3, n_snapshots=600, prints_per_snapshot=1.5), books, trades)
    start = day_start_ns(SYNTHETIC_DAY)
    stamps = start + np.arange(1, 70) * 1_000_000_000

    def tapes(level: float) -> dict[str, SignalTape]:
        return {
            INDEX_SIGNAL: SignalTape(INDEX_SIGNAL, stamps, np.full(len(stamps), level)),
            LEAN_TRIGGER_SIGNAL: SignalTape(LEAN_TRIGGER_SIGNAL, stamps, stamps / 1e9),
        }

    lean = ReversionLean(
        SkewQuoter(0.0, 0.0, 0.0, 5.0, maker_bp=0.0), beta=-1.0, lam=1.0, one_sided=True
    )

    def run(level: float) -> pd.DataFrame:
        return run_cells(
            "FUZZUSDT",
            [SYNTHETIC_DAY],
            [Cell("S4", lambda: lean, CONFIG)],
            book_roots=[books],
            trades_root=trades,
            funding_root=None,
            tapes=tapes(level),
            cache=tmp_path / "cache",
        )

    up, down = run(50.0), run(-50.0)
    assert up["net"].iloc[0] != down["net"].iloc[0]
    assert len(list((tmp_path / "cache").iterdir())) == 2
    again = run(50.0)
    pd.testing.assert_frame_equal(
        up.drop(columns="worker_peak_rss_mb"), again.drop(columns="worker_peak_rss_mb")
    )
