"""Every reader and fetcher of the 2025 study asks its access before it opens anything."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_research.data import fetch_log as fl
from trading_research.data import fetch_stage as fs
from trading_research.data import study_access as sa
from trading_research.data import study_fetch as sf
from trading_research.data import study_loader as sl
from trading_research.data.disk import StageBudget
from trading_research.market_making.prereg import HeldOutLocked

H_DAY = date(2025, 7, 8)
E_DAY = date(2025, 7, 3)


def opened(block: str) -> sa.StudyAccess:
    session = (block, "commit", "config", "env")
    return sa.StudyAccess(sa.grants_for(sa.opened_pairs(block)), "first read", session=session,
                          issuer=sa._ISSUER)  # fmt: skip


def garbage(path: Path) -> Path:
    """A file that would fail to parse: a refusal must come before any read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a parquet file")
    return path


def test_the_book_loader_refuses_before_opening_the_file(tmp_path: Path) -> None:
    garbage(sf.book_path("BTCUSDT", H_DAY, tmp_path))
    with pytest.raises(HeldOutLocked, match="cannot load BTCUSDT book on 2025-07-08"):
        sl.load_book_day("BTCUSDT", H_DAY, tmp_path, max_sequence_gaps=0)
    with pytest.raises(HeldOutLocked):
        sl.load_book_day(
            "BTCUSDT", H_DAY, tmp_path, max_sequence_gaps=0, access=sa.development_access()
        )
    with pytest.raises(ValueError, match=r"(?i)parquet|magic"):
        sl.load_book_day("BTCUSDT", H_DAY, tmp_path, max_sequence_gaps=0, access=opened("H"))


def test_prints_and_funding_are_refused_outside_the_access(tmp_path: Path) -> None:
    with pytest.raises(HeldOutLocked, match="prints"):
        sl.load_prints_day("BTCUSDT", H_DAY, tmp_path)
    with pytest.raises(HeldOutLocked, match="funding"):
        sl.load_funding_day("BTCUSDT", H_DAY, tmp_path)
    with pytest.raises(HeldOutLocked, match="funding"):
        sl.load_funding_day("BTCUSDT", E_DAY, tmp_path, access=opened("H"))


def test_the_simulator_day_needs_every_plane_it_reads(tmp_path: Path) -> None:
    roots: dict[str, Any] = {"book_roots": [tmp_path], "trades_root": tmp_path}
    with pytest.raises(HeldOutLocked, match="book"):
        sl.load_events_day("BTCUSDT", H_DAY, funding_root=tmp_path, **roots)
    with pytest.raises(HeldOutLocked, match="book"):
        sl.load_events_day("BTCUSDT", E_DAY, funding_root=None, access=opened("H"), **roots)


def bars_day(root: Path, day: date, interval: str = "1h") -> None:
    stamps = pd.date_range(datetime(day.year, day.month, day.day, tzinfo=UTC), periods=24, freq="h")
    frame = pd.DataFrame({"timestamp": stamps, "close": range(24)})
    path = sl.bars_path("BTCUSDT", day, root, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_bars_are_read_day_by_day_inside_the_access(tmp_path: Path) -> None:
    days = [date(2025, 6, 29), date(2025, 6, 30)]
    for day in [*days, date(2025, 7, 1)]:
        bars_day(tmp_path, day)
        bars_day(tmp_path, day, "1d")
    assert len(sl.load_bars("BTCUSDT", days, tmp_path)) == 48
    with pytest.raises(HeldOutLocked, match="bars on 2025-07-01"):
        sl.load_bars("BTCUSDT", [*days, date(2025, 7, 1)], tmp_path)
    with pytest.raises(HeldOutLocked):
        sl.load_bars("BTCUSDT", [date(2025, 7, 1)], tmp_path, interval="1d")
    assert len(sl.load_bars("BTCUSDT", [date(2025, 7, 1)], tmp_path, access=opened("H"))) == 24
    with pytest.raises(FileNotFoundError):
        sl.load_bars("BTCUSDT", [date(2025, 7, 2)], tmp_path, access=opened("H"))
    cache = tmp_path / "_archives" / "BTCUSDT" / "klines-1h"
    cache.mkdir(parents=True)
    (cache / "BTCUSDT-1h-2025-07.zip").write_bytes(b"")
    with pytest.raises(HeldOutLocked, match="monthly"):
        sl.load_bars("BTCUSDT", days, tmp_path)


def no_request(url: str, timeout: float) -> Any:
    raise AssertionError(f"a refused day made a request: {url}")


def test_the_fetchers_refuse_before_any_request(tmp_path: Path) -> None:
    with pytest.raises(HeldOutLocked, match="cannot fetch BTCUSDT book"):
        sf.fetch_book_day("BTCUSDT", H_DAY, tmp_path, tmp_path / "a", opener=no_request)
    with pytest.raises(HeldOutLocked, match="prints"):
        sf.fetch_prints_day("BTCUSDT", H_DAY, tmp_path, tmp_path / "a", opener=no_request)
    with pytest.raises(HeldOutLocked, match="bars"):
        sf.fetch_bars_day("BTCUSDT", E_DAY, tmp_path)
    span = [date(2025, 6, 30), date(2025, 7, 8)]
    with pytest.raises(HeldOutLocked, match="2025-07-01"):
        sf.fetch_funding_span("BTCUSDT", span, tmp_path, opener=no_request, access=opened("H"))
    with pytest.raises(HeldOutLocked):
        sf.fetch_book_day("ETHUSDT", date(2024, 12, 3), tmp_path, tmp_path / "a", opener=no_request)


def test_a_bars_fetch_is_refused_while_a_late_monthly_archive_is_cached(tmp_path: Path) -> None:
    cache = tmp_path / "_archives" / "BTCUSDT" / "klines-1h"
    cache.mkdir(parents=True)
    (cache / "BTCUSDT-1h-2025-09.zip").write_bytes(b"")
    with pytest.raises(HeldOutLocked, match="monthly"):
        sf.fetch_bars_day("BTCUSDT", date(2024, 12, 3), tmp_path)


def test_a_stage_reaching_past_its_access_stops_before_fetching(tmp_path: Path) -> None:
    roots = fs.StudyRoots.under(tmp_path)
    plan = fs.StagePlan(book=(H_DAY,))
    budget = StageBudget(archives_in_flight_gb=1e9, books_gb=0.0, prints_gb=0.0)
    log = fl.FetchLog(tmp_path / "log.json")
    with pytest.raises(HeldOutLocked):
        fs.run_stage("BTCUSDT", plan, roots, log, budget=budget, book_workers=1, opener=no_request)
