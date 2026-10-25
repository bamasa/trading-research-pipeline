"""Binance BTCFDUSD aggregated prints: the side flag, both units of time, the checksum."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Iterator
from datetime import date

import pandas as pd
import pytest

from trading_research.spot import binance as bn
from trading_research.spot import prints
from trading_research.spot import synthetic as syn

DAY = date(2026, 3, 3)
OLD = date(2024, 6, 1)


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


def _archive(rows: list[str], day: date = DAY) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as packed:
        packed.writestr(f"BTCFDUSD-aggTrades-{day}.csv", "\n".join(rows) + "\n" if rows else "")
    return buffer.getvalue()


def test_is_buyer_maker_true_is_a_seller_initiated_print() -> None:
    rows = [
        "100,68830.80,0.00100,1,1,1772496001361582,True,True",
        "101,68830.90,0.00200,2,3,1772496003621643,False,True",
    ]
    frame, checks = bn.read(_archive(rows), DAY)
    assert frame["aggressor"].tolist() == [-1, 1]
    assert frame["rpi"].tolist() == [False, False]
    assert frame["trade_id"].tolist() == [100, 101]
    assert frame["source"].iloc[0] == prints.BINANCE_SOURCE
    assert list(frame.columns) == list(prints.COLUMNS)
    assert checks == bn.AggChecks(rows=2, unit="us", not_best_match=0, ids_consecutive=True)


@pytest.mark.parametrize(("stamp", "unit"), [("1772496001361582", "us"), ("1772496001361", "ms")])
def test_both_units_of_time_give_the_same_instant(stamp: str, unit: str) -> None:
    frame, checks = bn.read(_archive([f"7,68830.80,0.00100,1,1,{stamp},False,True"]), DAY)
    assert checks.unit == unit
    assert frame["timestamp"].iloc[0].floor("ms") == pd.Timestamp(
        "2026-03-03 00:00:01.361", tz="UTC"
    )


def test_the_synthetic_venue_writes_milliseconds_before_2025_and_microseconds_after() -> None:
    market = syn.SpotMarket(frozenset({DAY, OLD}), step_ms=60_000)
    for day, unit in ((OLD, "ms"), (DAY, "us")):
        frame, checks = bn.read(market.binance_archive(day), day)
        assert checks.unit == unit and len(frame) > 0
        assert frame["timestamp"].dt.date.eq(day).all()


def test_a_file_mixing_units_or_off_its_day_raises() -> None:
    mixed = [
        "1,10.00,1.0,1,1,1772496001361,False,True",
        "2,10.00,1.0,2,2,1772496002361582,False,True",
    ]
    with pytest.raises(bn.BinancePrintsError, match="more than one unit"):
        bn.read(_archive(mixed), DAY)
    with pytest.raises(bn.BinancePrintsError, match="not around"):
        bn.read(_archive(["1,10.00,1.0,1,1,1772496001361582,False,True"]), date(2026, 3, 10))


def test_an_archive_without_rows_is_a_day_without_a_print() -> None:
    frame, checks = bn.read(_archive([]), DAY)
    assert frame.empty and checks.rows == 0


def test_the_published_checksum_is_checked() -> None:
    payload = _archive(["1,10.00,1.0,1,1,1772496001361582,False,True"])
    digest = hashlib.sha256(payload).hexdigest()
    assert bn.verify(payload, f"{digest}  BTCFDUSD-aggTrades-2026-03-03.zip\n".encode()) == digest
    with pytest.raises(bn.BinanceChecksumError, match="not the published"):
        bn.verify(payload + b"x", digest)
    with pytest.raises(bn.BinanceChecksumError, match="not a checksum"):
        bn.verify(payload, b"<html>")


def test_best_match_false_is_counted_and_sets_nothing() -> None:
    rows = [
        "1,10.00,1.0,1,1,1772496001361582,False,False",
        "3,10.00,1.0,2,2,1772496002361582,True,True",
    ]
    frame, checks = bn.read(_archive(rows), DAY)
    assert checks.not_best_match == 1 and not checks.ids_consecutive
    assert len(frame) == 2 and not frame["rpi"].any()
