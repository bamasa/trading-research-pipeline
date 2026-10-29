"""The spot study's access: its blocks, planes, instruments and who may read what."""

from __future__ import annotations

import pickle
import socket
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import access as sa

ROOT = Path(__file__).resolve().parents[1]
CONFIG: dict[str, Any] = yaml.safe_load((ROOT / "configs" / "spot_2026.yaml").read_text("utf-8"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("tests do not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_blocks_match_the_configuration() -> None:
    blocks = CONFIG["blocks"]
    for name, (start, end) in sa.BLOCKS.items():
        assert (blocks[name]["start"], blocks[name]["end"]) == (start, end)
        assert len(sa.block_days(name)) == sa.BLOCK_DAYS[name] == blocks[name]["days"]
    assert tuple(blocks["reader_test"]["days"]) == sa.READER_TEST_DAYS
    assert tuple(blocks["P_embargo"]) == sa.P_EMBARGO
    assert tuple(blocks["never_in_a_block"]) == sa.NEVER_READ_DAYS
    for key, spec in CONFIG["instruments"].items():
        found = sa.instrument(spec["id"])
        assert (found.key, found.venue, found.symbol, found.quote) == (
            key, spec["venue"], spec["symbol"], spec["quote"],
        )  # fmt: skip
        assert (sa.BOOK in found.planes) == (spec["book"] != "none")


def test_development_access_permits_reader_test_w_and_d_only() -> None:
    access = sa.development_access()
    for name in sa.INSTRUMENTS:
        for plane in sa.instrument(name).planes:
            for block in sa.DEVELOPMENT_BLOCKS:
                access.require(name, sa.block_days(block), plane)
            for block in ("E", "H", "P"):
                for day in (sa.block_days(block)[0], sa.block_days(block)[-1]):
                    assert not access.allows(name, day, plane)
                    with pytest.raises(HeldOutLocked, match="permits"):
                        access.require(name, [day], plane)
    outside = [date(2026, 1, 28), date(2026, 3, 31), sa.P_EMBARGO[0], date(2027, 1, 1)]
    for day in [*outside, *sa.NEVER_READ_DAYS]:
        assert not access.allows(sa.BYBIT, day, sa.BOOK)


def test_no_day_of_2025_is_ever_read() -> None:
    opened = sa.SpotAccess(sa.grants_for(sa.opened_blocks("P")), "first read",
                           session=("P", "c", "s", "e"), seal=sa._SEAL)  # fmt: skip
    for day in (date(2025, 1, 1), date(2025, 6, 30), date(2025, 12, 31)):
        with pytest.raises(HeldOutLocked, match="no day of 2025"):
            opened.require(sa.BINANCE, [day], sa.PRINTS)
    for day in sa.NEVER_READ_DAYS:
        assert not opened.allows(sa.BYBIT, day, sa.PRINTS)


def test_held_out_access_only_from_the_ledger() -> None:
    with pytest.raises(HeldOutLocked, match="ledger"):
        sa.SpotAccess(sa.grants_for(("H",)), "first read")
    with pytest.raises(ValueError, match="unknown"):
        sa.SpotAccess(frozenset({(sa.BYBIT, "E", sa.BOOK)}), "development")
    opened = sa.SpotAccess(sa.grants_for(sa.opened_blocks("H")), "first read",
                           session=("H", "c", "s", "e"), seal=sa._SEAL)  # fmt: skip
    opened.require(sa.BYBIT, sa.block_days("H"), sa.BOOK)
    assert not opened.allows(sa.BYBIT, sa.block_days("P")[0], sa.BOOK)
    assert pickle.loads(pickle.dumps(opened)).seal is sa._SEAL
    assert sa.opened_blocks("P") == ("reader_test", "W", "D", "H", "P")
    with pytest.raises(ValueError):
        sa.opened_blocks("D")


def test_instruments_are_venue_qualified_and_planes_checked() -> None:
    access = sa.development_access()
    with pytest.raises(ValueError, match="never by symbol alone"):
        access.require("BTCUSDT", [sa.READER_TEST_DAYS[0]], sa.BOOK)
    with pytest.raises(ValueError, match="publishes"):
        access.require(sa.BINANCE, [sa.READER_TEST_DAYS[0]], sa.BOOK)  # no spot book on Binance
    with pytest.raises(ValueError):
        sa.SpotAccess(frozenset({(sa.BINANCE, "D", sa.BOOK)}), "development")


def test_reader_test_access_carries_its_purpose() -> None:
    access = sa.reader_test_access("dry run")
    assert access.stamp == "reader test (dry run)"
    access.require(sa.BYBIT, sa.READER_TEST_DAYS, sa.PRINTS)
    with pytest.raises(HeldOutLocked):
        access.require(sa.BYBIT, sa.block_days("W")[:1], sa.PRINTS)
    with pytest.raises(ValueError, match="serve"):
        sa.reader_test_access("scoring")
    assert sa.require_day(None, sa.BYBIT, sa.READER_TEST_DAYS[1], sa.BOOK).stamp == "development"


def test_reader_test_days_lie_in_no_block() -> None:
    days = {d for name in sa.BLOCKS for d in sa.block_days(name)}
    assert not days & set(sa.READER_TEST_DAYS) and not days & set(sa.NEVER_READ_DAYS)
    assert all(sa.block_of(d) == sa.READER_TEST for d in sa.READER_TEST_DAYS)
    assert sa.block_of(sa.BLOCKS["H"][0] - timedelta(days=1)) == "E"


def test_data_roots_and_files_of_days(tmp_path: Path) -> None:
    base = tmp_path / "data"
    root = sa.data_root(sa.BYBIT, sa.BOOK, base)
    assert root == base / "spot" / "bybit" / "book" / "BTCUSDT"
    assert sa.data_root(sa.BINANCE, sa.PRINTS, base).parts[-3:] == ("binance", "prints", "BTCFDUSD")
    with pytest.raises(HeldOutLocked, match="outside"):
        sa.require_spot_path(base / "book" / "BTCUSDT" / "x.parquet", base)
    sa.require_spot_path(root / "x.parquet", base)
    assert sa.files_of_days(sa.block_days("H"), base) == []
    root.mkdir(parents=True)
    (root / "2026-07-31.parquet").write_bytes(b"")
    assert sa.files_of_days(sa.block_days("H"), base) == []
    named = root / "2026-08-08_BTCUSDT_ob200.data.zip"
    named.write_bytes(b"")
    monthly = sa.data_root(sa.BINANCE, sa.PRINTS, base) / "BTCFDUSD-aggTrades-2026-10.zip"
    monthly.parent.mkdir(parents=True)
    monthly.write_bytes(b"")
    assert sa.files_of_days(sa.block_days("H"), base) == sorted([monthly, named])
    assert sa.files_of_days(sa.block_days("P"), base) == []


def test_the_writers_keep_to_the_study_data_layout(tmp_path: Path) -> None:
    from trading_research.spot.fetch import write_parquet
    from trading_research.spot.fetch_stage import SpotRoots

    frame = pd.DataFrame({"x": [1]})
    good = tmp_path / "spot" / "bybit" / "book" / "BTCUSDT" / "2026-05-01.parquet"
    write_parquet(frame, good)
    assert good.exists()
    for bad in (tmp_path / "book" / "BTCUSDT" / "2026-05-01.parquet",
                tmp_path / "spot" / "binance" / "book" / "BTCFDUSD" / "2026-05-01.parquet"):  # fmt: skip
        with pytest.raises(HeldOutLocked, match="data layout"):
            write_parquet(frame, bad)
    assert SpotRoots.under(tmp_path / "spot").data == tmp_path / "spot"
    with pytest.raises(HeldOutLocked, match="outside the spot study's data root"):
        SpotRoots.under(tmp_path / "data")
