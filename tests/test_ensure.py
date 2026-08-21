"""Data that fetches itself: what it fetches, and what it refuses to."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_research.data.ensure import (
    DataUnavailable,
    FetchPlan,
    ensure_book,
    plan,
)


def _write(root: Path, symbol: str, days: list[str]) -> None:
    directory = root / symbol
    directory.mkdir(parents=True, exist_ok=True)
    for day in days:
        (directory / f"{day}.parquet").write_bytes(b"not empty")


def test_a_plan_separates_what_is_here_from_what_is_not(tmp_path: Path) -> None:
    _write(tmp_path, "BTCUSDT", ["2024-02-01", "2024-02-03"])
    result = plan("BTCUSDT", date(2024, 2, 1), date(2024, 2, 4), tmp_path)

    assert [d.isoformat() for d in result.present] == ["2024-02-01", "2024-02-03"]
    assert [d.isoformat() for d in result.missing] == ["2024-02-02", "2024-02-04"]
    assert not result.complete


def test_an_empty_file_counts_as_missing(tmp_path: Path) -> None:
    """A zero-byte file is an interrupted download, not a day of data."""
    directory = tmp_path / "BTCUSDT"
    directory.mkdir(parents=True)
    (directory / "2024-02-01.parquet").write_bytes(b"")

    result = plan("BTCUSDT", date(2024, 2, 1), date(2024, 2, 1), tmp_path)
    assert result.missing and not result.present


def test_a_complete_span_needs_no_fetching(tmp_path: Path) -> None:
    _write(tmp_path, "BTCUSDT", ["2024-02-01", "2024-02-02"])
    result = plan("BTCUSDT", date(2024, 2, 1), date(2024, 2, 2), tmp_path)
    assert result.complete
    assert not result.missing


def test_nothing_is_fetched_when_the_span_is_already_here(tmp_path: Path) -> None:
    """The property that makes re-running an experiment free.

    ``ensure_book`` would import the downloader and hit the network; a complete
    span must return before it gets there.
    """
    _write(tmp_path, "BTCUSDT", ["2024-02-01"])
    result = ensure_book(
        "BTCUSDT", date(2024, 2, 1), date(2024, 2, 1), root=tmp_path, on_progress=None
    )
    assert result.complete


def test_a_dry_run_reports_without_fetching(tmp_path: Path) -> None:
    said: list[str] = []
    result = ensure_book(
        "BTCUSDT",
        date(2024, 2, 1),
        date(2024, 2, 3),
        root=tmp_path,
        dry_run=True,
        on_progress=said.append,
    )
    assert len(result.missing) == 3
    assert said and "3 to fetch" in said[0]
    assert not (tmp_path / "BTCUSDT").exists()


def test_a_backwards_span_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="before start"):
        plan("BTCUSDT", date(2024, 3, 1), date(2024, 2, 1), tmp_path)


def test_a_plan_describes_itself_in_one_line() -> None:
    described = FetchPlan(
        "XRPUSDT", present=[date(2024, 2, 1)], missing=[date(2024, 2, 2)]
    ).describe()
    assert "XRPUSDT" in described
    assert "1 day(s) present" in described
    assert "1 to fetch" in described


def test_a_venue_with_nothing_raises_rather_than_returning_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silence would let an experiment run on no data and report a result."""
    import trading_research.data.bybit as bybit

    def refuse(*args: object, **kwargs: object) -> None:
        raise bybit.BybitArchiveError("404")

    monkeypatch.setattr(bybit, "download_range", refuse)
    with pytest.raises(DataUnavailable, match="no order-book data"):
        ensure_book("NOPEUSDT", date(2024, 2, 1), date(2024, 2, 2), root=tmp_path, on_progress=None)


def test_days_the_venue_lacks_are_skipped_when_others_arrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An instrument listed mid-span has no archive before it existed."""
    import trading_research.data.bybit as bybit

    def sometimes(symbol: str, start: date, end: date, out_dir: Path, **kwargs: object) -> None:
        if start == date(2024, 2, 1):
            raise bybit.BybitArchiveError("404")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{start.isoformat()}.parquet").write_bytes(b"data")

    monkeypatch.setattr(bybit, "download_range", sometimes)
    result = ensure_book(
        "LATEUSDT", date(2024, 2, 1), date(2024, 2, 3), root=tmp_path, on_progress=None
    )
    assert len(result.present) == 2
    assert [d.isoformat() for d in result.missing] == ["2024-02-01"]
