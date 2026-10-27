"""The public-stream parsers on a recorded fixture, and Bybit's book against the archive replay.

The fixture is six seconds of each public stream (Bybit spot and linear,
Binance spot and USD-M, BTCUSDT), recorded on 2026-10-27 with
``trading-research paper record``: every message as it arrived, with its
receipt time. No test here opens a connection.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.data import bybit
from trading_research.live.events import BookRow, Gap, Print, rows_to_frame
from trading_research.live.record import read_raw_log
from trading_research.live.replay import open_day
from trading_research.live.stream import BinanceParser, BybitParser, connections

FIXTURES = Path(__file__).parent / "fixtures" / "paper"
PAIRS = [("bybit", "spot"), ("bybit", "linear"), ("binance", "spot"), ("binance", "linear")]


def fixture(venue: str, market: str) -> Path:
    return FIXTURES / f"{venue}-{market}-BTCUSDT-2026-10-27.jsonl"


@pytest.mark.parametrize(("venue", "market"), PAIRS)
def test_the_recorded_stream_parses_into_rows_and_prints(venue: str, market: str) -> None:
    day = open_day(fixture(venue, market), venue=venue, market=market, symbol="BTCUSDT")
    rows, prints = day.rows, day.prints
    assert len(rows) > 40 and len(prints) > 30
    assert not [e for e in day.events if isinstance(e, Gap)]
    for row in rows:
        assert len(row.bid_px) == len(row.ask_px) == 10
        assert list(row.bid_px) == sorted(row.bid_px, reverse=True)
        assert list(row.ask_px) == sorted(row.ask_px)
        assert row.bid_px[0] < row.ask_px[0]
        assert min(row.bid_sz) > 0 and min(row.ask_sz) > 0
    # At most one row per 100 ms interval, stamps increasing.
    buckets = [r.ts_ns // 100_000_000 for r in rows]
    assert buckets == sorted(set(buckets))
    assert {p.aggressor for p in prints} <= {-1, 1}
    assert all(p.size > 0 and p.price > 0 for p in prints)
    # Prints trade at or near the touch the rows show.
    mids = np.array([r.mid for r in rows])
    assert abs(np.median([p.price for p in prints]) / np.median(mids) - 1) < 1e-3


@pytest.fixture
def bybit_archive(tmp_path: Path) -> Path:
    lines = [
        text
        for _, connection, text in read_raw_log(fixture("bybit", "linear"))
        if not connection.startswith("#") and '"orderbook.' in text[:60]
    ]
    archive = tmp_path / "BTCUSDT-2026-10-27-ob200.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        packed.writestr("2026-10-27_BTCUSDT_ob200.data", "\n".join(lines) + "\n")
    return archive


def test_bybit_parser_matches_reconstruct_row_for_row(bybit_archive: Path) -> None:
    batch = bybit.reconstruct(bybit_archive, symbol="BTCUSDT", source="x")
    with zipfile.ZipFile(bybit_archive) as packed:
        lines = packed.read(packed.namelist()[0]).decode().splitlines()
    parser = BybitParser()
    rows = [e for line in lines for e in parser.parse(line, 0) if isinstance(e, BookRow)]
    live = rows_to_frame(rows, symbol="BTCUSDT", source="x")
    assert len(live) == len(batch) > 50
    pd.testing.assert_frame_equal(
        live.reset_index(drop=True), batch[live.columns].reset_index(drop=True)
    )


def test_bybit_sequence_gap_is_reported_and_asks_for_a_snapshot() -> None:
    parser = BybitParser()
    snapshot = {
        "topic": "orderbook.200.X",
        "type": "snapshot",
        "ts": 1_000,
        "data": {
            "s": "X",
            "b": [[str(100 - i), "1"] for i in range(10)],
            "a": [[str(101 + i), "1"] for i in range(10)],
            "u": 5,
        },
    }
    delta = {
        "topic": "orderbook.200.X",
        "type": "delta",
        "ts": 1_200,
        "data": {"s": "X", "b": [["100", "2"]], "a": [], "u": 7},
    }
    first = parser.parse(json.dumps(snapshot), 1)
    assert isinstance(first[0], BookRow) and first[0].bid_sz[0] == 1.0
    events = parser.parse(json.dumps(delta), 2)
    gaps = [e for e in events if isinstance(e, Gap)]
    assert gaps and gaps[0].kind == "sequence" and "7 after 5" in gaps[0].detail
    assert parser.needs_resync
    row = next(e for e in events if isinstance(e, BookRow))
    assert row.bid_sz[0] == 2.0  # applied anyway, as the archive replay applies it


def test_bybit_print_side_is_the_takers() -> None:
    message = {
        "topic": "publicTrade.X",
        "type": "snapshot",
        "ts": 5,
        "data": [
            {"T": 5, "s": "X", "S": "Buy", "v": "0.5", "p": "100.5", "i": "a", "seq": 9},
            {
                "T": 5,
                "s": "X",
                "S": "Sell",
                "v": "0.2",
                "p": "100.0",
                "i": "b",
                "seq": 10,
                "RPI": True,
            },
        ],
    }
    prints = [e for e in BybitParser().parse(json.dumps(message), 6) if isinstance(e, Print)]
    assert [(p.aggressor, p.size, p.rpi) for p in prints] == [(1, 0.5, False), (-1, 0.2, True)]
    assert prints[0].ts_ns == 5_000_000


def test_binance_buyer_maker_means_the_seller_crossed_and_ids_must_run_on() -> None:
    parser = BinanceParser(market="spot")
    first = {
        "stream": "x@aggTrade",
        "data": {"e": "aggTrade", "a": 10, "p": "1", "q": "2", "T": 1, "m": True},
    }
    second = {
        "stream": "x@aggTrade",
        "data": {"e": "aggTrade", "a": 12, "p": "1", "q": "2", "T": 2, "m": False},
    }
    events = parser.parse(json.dumps(first), 0) + parser.parse(json.dumps(second), 0)
    prints = [e for e in events if isinstance(e, Print)]
    gaps = [e for e in events if isinstance(e, Gap)]
    assert [p.aggressor for p in prints] == [-1, 1]
    assert len(gaps) == 1 and gaps[0].kind == "trade_sequence"


def test_binance_futures_depth_checks_the_previous_update_id() -> None:
    parser = BinanceParser(market="linear")
    levels = {
        "b": [[str(100 - i), "1"] for i in range(10)],
        "a": [[str(101 + i), "1"] for i in range(10)],
    }
    push = {"e": "depthUpdate", "T": 1_000, "U": 1, "u": 10, "pu": 0, **levels}
    good = {"e": "depthUpdate", "T": 1_100, "U": 11, "u": 20, "pu": 10, **levels}
    broken = {"e": "depthUpdate", "T": 1_200, "U": 31, "u": 40, "pu": 30, **levels}
    events = []
    for message in (push, good, broken):
        events += parser.parse(json.dumps({"stream": "x@depth10@100ms", "data": message}), 0)
    assert len([e for e in events if isinstance(e, BookRow)]) == 3
    gaps = [e for e in events if isinstance(e, Gap)]
    assert len(gaps) == 1 and "pu 30 after u 20" in gaps[0].detail


def test_binance_spot_rows_are_stamped_at_receipt() -> None:
    parser = BinanceParser(market="spot")
    message = {
        "lastUpdateId": 7,
        "bids": [[str(100 - i), "1"] for i in range(10)],
        "asks": [[str(101 + i), "1"] for i in range(10)],
    }
    (row,) = parser.parse(json.dumps({"stream": "x@depth10@100ms", "data": message}), 123_456_789)
    assert isinstance(row, BookRow) and row.ts_ns == row.recv_ns == 123_456_789


def test_every_connection_is_a_public_market_data_stream() -> None:
    for venue, market in PAIRS:
        for connection in connections(venue, market, "BTCUSDT"):
            assert connection.url.startswith("wss://")
            assert (
                "public" in connection.url
                or "market" in connection.url
                or "stream?" in connection.url
            )
            assert "private" not in connection.url and "listenKey" not in connection.url
            assert "order" not in connection.url
