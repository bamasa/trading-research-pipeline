"""The 2025 study's blocks and its access: per instrument, block and plane.

The constants are checked against the registered YAML, and every reader is
shown to refuse a day before it opens anything: a refused day costs nothing
and reveals nothing.
"""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from trading_research.data import study_access as sa
from trading_research.market_making.prereg import HeldOutLocked

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"


def registered() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def opened(block: str) -> sa.StudyAccess:
    """An access as the ledger issues it (the token is the ledger's)."""
    pairs = sa.opened_pairs(block)
    return sa.StudyAccess(sa.grants_for(pairs), "first read", issuer=sa._ISSUER)


def test_the_blocks_are_the_registered_ones() -> None:
    tree = registered()
    for name, (start, end) in sa.BLOCKS.items():
        block = tree["blocks"][name]
        assert (block["start"], block["end"], block["days"]) == (start, end, sa.BLOCK_DAYS[name])
        assert len(sa.block_days(name)) == sa.BLOCK_DAYS[name]
    o = tree["blocks"]["O"]
    assert (o["start"], o["end"]) == sa.O_WINDOW
    assert tuple(tree["data"]["book"]["ob200_reader_test_days"]) == sa.READER_TEST_DAYS
    assert tree["data"]["book"]["archive_change"] == sa.ARCHIVE_CHANGE
    assert tuple(tree["blocks"]["cold_start_days"]) == sa.COLD_START_DAYS
    assert tuple(tree["instruments"]) == sa.INSTRUMENTS
    stage0 = tree["search"]["stages"][0]["what"]
    assert all(day.isoformat() in stage0 for day in sa.TIMING_DAYS)
    ends = [end for _, end in sa.BLOCKS.values()]
    starts = [start for start, _ in sa.BLOCKS.values()]
    assert all(b - a == timedelta(days=1) for a, b in zip(ends, starts[1:], strict=False))


def test_every_day_has_one_block() -> None:
    assert sa.block_of(date(2024, 11, 30)) is None
    assert sa.block_of(date(2024, 12, 1)) == "W"
    assert sa.block_of(date(2025, 6, 30)) == "D"
    assert sa.block_of(date(2025, 7, 7)) == "E"
    assert sa.block_of(date(2025, 8, 20)) == "H"
    assert sa.block_of(date(2025, 8, 21)) == "F"
    assert sa.block_of(date(2025, 12, 31)) == "P"
    assert sa.block_of(date(2026, 1, 1)) is None
    assert sa.block_of(date(2026, 1, 6)) == sa.READER_TEST
    assert sa.block_of(date(2024, 3, 5)) == sa.TIMING
    assert sa.block_days(sa.READER_TEST) == list(sa.READER_TEST_DAYS)


def test_the_development_access_stops_at_the_end_of_d_in_every_plane() -> None:
    access = sa.development_access()
    for plane in sa.PLANES:
        for day in (date(2024, 12, 1), date(2025, 6, 30), date(2026, 1, 5), date(2024, 2, 11)):
            assert access.allows("BTCUSDT", day, plane)
        for day in (date(2025, 7, 1), date(2025, 7, 8), date(2025, 9, 1), date(2025, 12, 31)):
            assert not access.allows("BTCUSDT", day, plane)
            with pytest.raises(HeldOutLocked, match="permits BTCUSDT"):
                access.require("BTCUSDT", [day], plane)
    assert not access.allows("ETHUSDT", date(2024, 12, 1), sa.BOOK)
    assert not access.allows("BTCUSDT", date(2024, 1, 1), sa.BOOK)
    assert access.open_through == date(2025, 6, 30)
    with pytest.raises(ValueError, match="unknown plane"):
        access.require("BTCUSDT", [date(2024, 12, 1)], "ticks")


def test_only_the_ledger_issues_an_access_beyond_development() -> None:
    with pytest.raises(HeldOutLocked, match="issued only by the study's ledger"):
        sa.StudyAccess(sa.grants_for(sa.opened_pairs("H")), "first read")
    with pytest.raises(ValueError, match="unknown"):
        sa.StudyAccess(frozenset({("ETHUSDT", "D", sa.BOOK)}), "development")
    timing_only = sa.StudyAccess(sa.grants_for({(sa.TIMING, sa.BOOK)}), "development")
    assert timing_only.open_through == date(2024, 11, 30)


def test_each_opening_adds_its_block_and_keeps_the_earlier_ones() -> None:
    h = opened("H")
    assert h.allows("BTCUSDT", date(2025, 7, 3), sa.BARS)
    assert not h.allows("BTCUSDT", date(2025, 7, 3), sa.BOOK)
    assert h.allows("BTCUSDT", date(2025, 7, 8), sa.FUNDING)
    assert not h.allows("BTCUSDT", date(2025, 8, 21), sa.BOOK)
    assert h.open_through == date(2025, 8, 20)
    f = opened("F")
    assert f.allows("BTCUSDT", date(2025, 7, 8), sa.BOOK)
    assert f.allows("BTCUSDT", date(2025, 11, 16), sa.PRINTS)
    assert not f.allows("BTCUSDT", date(2025, 11, 17), sa.BARS)
    assert opened("P").open_through == date(2025, 12, 31)
    with pytest.raises(ValueError, match="opens"):
        sa.opened_pairs("D")


def test_the_days_an_opening_adds_are_its_block_and_for_h_e_bars() -> None:
    days = sa.opening_days("H")
    assert days[sa.BOOK] == tuple(sa.block_days("H"))
    assert days[sa.BARS] == tuple(sa.block_days("E") + sa.block_days("H"))
    assert days[sa.FUNDING] == days[sa.PRINTS] == days[sa.BOOK]
    assert sa.opening_days("F")[sa.BARS] == tuple(sa.block_days("F"))


def test_an_issued_access_survives_a_worker_process() -> None:
    copy = pickle.loads(pickle.dumps(opened("H")))
    assert copy.issuer is sa._ISSUER
    assert copy.allows("BTCUSDT", date(2025, 7, 8), sa.BOOK)
    sa.StudyAccess(copy.grants, copy.stamp, issuer=copy.issuer)


def monthly(root: Path, month: str, interval: str = "1h") -> Path:
    folder = root / "_archives" / "BTCUSDT" / f"klines-{interval}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"BTCUSDT-{interval}-{month}.zip"
    path.write_bytes(b"")
    return path


def test_a_monthly_kline_file_past_the_opened_blocks_is_refused(tmp_path: Path) -> None:
    access = sa.development_access()
    monthly(tmp_path, "2025-06")
    monthly(tmp_path, "2025-07-03")  # a daily archive: its own day only
    access.refuse_monthly_bars("BTCUSDT", tmp_path)
    late = monthly(tmp_path, "2025-07")
    with pytest.raises(HeldOutLocked, match=late.name):
        access.refuse_monthly_bars("BTCUSDT", tmp_path)
    access.refuse_monthly_bars("BTCUSDT", tmp_path, interval="1d")
    opened("H").refuse_monthly_bars("BTCUSDT", tmp_path)
    monthly(tmp_path, "2025-08")
    with pytest.raises(HeldOutLocked):
        opened("H").refuse_monthly_bars("BTCUSDT", tmp_path)
