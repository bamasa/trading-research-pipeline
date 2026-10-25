"""Bybit's spot print parser, held to the probed contract; and the print contract itself."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_research.spot import bybit_prints as bp
from trading_research.spot import prints
from trading_research.spot import synthetic as syn

DAY = date(2026, 3, 3)
HEADER = b"id,timestamp,price,volume,side,rpi\n"


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


def test_sides_units_ids_and_rpi_are_read_as_the_archive_writes_them() -> None:
    raw = HEADER + b"1,1772496000082,68830.8,0.000073,sell,0\n2,1772496000082,68830.9,0.5,buy,1\n"
    frame, checks = bp.read(raw)
    assert list(frame.columns) == list(prints.COLUMNS)
    assert frame["aggressor"].tolist() == [-1, 1]  # "sell" a seller crossed, "buy" a buyer
    assert frame["rpi"].tolist() == [False, True]
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-03-03 00:00:00.082", tz="UTC")
    assert frame["size"].tolist() == [0.000073, 0.5]
    assert frame["trade_id"].tolist() == [1, 2]
    assert frame["source"].iloc[0] == prints.BYBIT_SOURCE
    assert checks == bp.PrintChecks(2, {"buy": 1, "sell": 1}, 1, True, True)
    assert checks.rpi_share == 0.5


def test_prints_sharing_a_stamp_keep_the_archive_order() -> None:
    raw = (
        HEADER
        + b"1,1772496000100,10.0,1,buy,0\n2,1772496000090,9.0,1,sell,0\n3,1772496000100,11.0,1,buy,0\n"
    )
    frame, checks = bp.read(raw)
    assert frame["trade_id"].tolist() == [2, 1, 3]
    assert not checks.stamps_ordered


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"timestamp,symbol,side,size,price\n1.5,BTCUSDT,Buy,1,10\n", "header"),
        (HEADER + b"1,1772496000082,10.0,1,Buy,0\n", "side strings"),
        (HEADER + b"1,1772496000.082,10.0,1,buy,0\n", "does not parse"),
        (HEADER + b"1,1772496000082000,10.0,1,buy,0\n", "milliseconds"),
        (HEADER + b"1,1772496000,10.0,1,buy,0\n", "milliseconds"),
        (HEADER + b"1,1772496000082,10.0,1,buy,2\n", "rpi"),
        (HEADER + b"1,1772496000082,-10.0,1,buy,0\n", "positive"),
        (HEADER + b"1,1772496000082,10.0,0,buy,0\n", "positive"),
    ],
)
def test_anything_off_the_contract_raises(raw: bytes, match: str) -> None:
    with pytest.raises(prints.PrintContractError, match=match):
        bp.read(raw)


def test_a_header_without_rows_is_a_day_without_a_print() -> None:
    frame, checks = bp.read(HEADER)
    assert frame.empty and checks.rows == 0
    assert prints.check(frame) is frame


def test_the_synthetic_day_parses_whole() -> None:
    market = syn.SpotMarket(frozenset({DAY}), step_ms=60_000, rpi_every=10)
    frame, checks = bp.read(market.bybit_prints_csv(DAY))
    assert checks.rows == len(frame) > 0 and checks.ids_consecutive
    assert checks.rpi == len(frame) // 10
    assert frame["timestamp"].is_monotonic_increasing


def test_the_contract_check_refuses_wrong_columns_types_and_sides() -> None:
    good = prints.empty(prints.BINANCE_SOURCE)
    with pytest.raises(prints.PrintContractError, match="columns"):
        prints.check(good.drop(columns="rpi"))
    with pytest.raises(prints.PrintContractError, match="types"):
        prints.check(good.astype({"aggressor": "int64"}))
    one = prints.assemble(
        pd.Series(pd.to_datetime([1], unit="s", utc=True)),
        pd.Series([1.0]),
        pd.Series([1.0]),
        np.array([1]),
        pd.Series([7]),
        np.array([False]),
        "x",
    )
    with pytest.raises(prints.PrintContractError, match="aggressor"):
        prints.check(one.assign(aggressor=np.array([0], dtype=np.int8)))
