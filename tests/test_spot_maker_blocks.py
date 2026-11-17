"""The maker study's access: development days only, H and P only through the ledger's seal."""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import pytest

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot_maker import blocks as bk


def test_development_access_reads_reader_test_w_and_d_only() -> None:
    access = bk.development_access()
    access.require(
        bk.INSTRUMENT, [date(2026, 3, 3), date(2026, 4, 30), date(2026, 7, 31)], "prints"
    )
    for refused in (date(2026, 8, 1), date(2026, 8, 8), date(2025, 6, 1), date(2026, 1, 5)):
        with pytest.raises(HeldOutLocked):
            access.require(bk.INSTRUMENT, [refused], "prints")


def test_other_instrument_or_plane_is_refused() -> None:
    access = bk.development_access()
    with pytest.raises(ValueError):
        access.require("bybit-spot:BTCUSDT", [date(2026, 5, 1)], "prints")
    with pytest.raises(ValueError):
        access.require(bk.INSTRUMENT, [date(2026, 5, 1)], "book")


def test_held_out_needs_the_ledger_seal() -> None:
    with pytest.raises(HeldOutLocked):
        bk.MakerAccess(frozenset({"H"}), "first read")
    with pytest.raises(HeldOutLocked):
        bk.MakerAccess(bk.opened_blocks("H"), "first read", ("H",), seal=object())
    sealed = bk.MakerAccess(bk.opened_blocks("H"), "first read", ("H", "abc"), seal=bk._SEAL)
    sealed.require(bk.INSTRUMENT, [date(2026, 8, 8), date(2026, 10, 21)], "prints")
    with pytest.raises(HeldOutLocked):
        sealed.require(bk.INSTRUMENT, [date(2026, 8, 7)], "prints")  # E is never granted
    with pytest.raises(HeldOutLocked):
        bk.MakerAccess(frozenset({"H"}), "first read", ("H", "abc"), seal=bk._SEAL)
    assert pickle.loads(pickle.dumps(sealed)).seal is bk._SEAL


def test_reader_test_access_and_paths(tmp_path: Path) -> None:
    access = bk.reader_test_access("readers' proof")
    assert access.stamp == "reader test (readers' proof)"
    with pytest.raises(HeldOutLocked):
        access.require(bk.INSTRUMENT, [date(2026, 4, 1)], "prints")
    with pytest.raises(ValueError):
        bk.reader_test_access("scoring")
    path = bk.day_file(tmp_path, date(2026, 8, 9))
    assert path == tmp_path / "data/spot/binance/prints/BTCFDUSD/2026-08-09.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert bk.files_of_days(tmp_path, [date(2026, 8, 9)]) == [path]
    assert bk.files_of_days(tmp_path, [date(2026, 8, 10)]) == []
