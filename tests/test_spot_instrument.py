"""Spot tick and lot steps: both venues' specifications, the clip's lot grid, the frozen record."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from trading_research.spot import instrument as ins
from trading_research.spot import synthetic as syn
from trading_research.spot.archives import BINANCE_SPOT, BYBIT_SPOT


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


def test_each_venue_specification_gives_its_tick_and_lot() -> None:
    bybit = ins.parse_spec(BYBIT_SPOT, json.loads(syn.bybit_spec()))
    binance = ins.parse_spec(BINANCE_SPOT, json.loads(syn.binance_spec()))
    assert (bybit.tick, bybit.lot) == (syn.BYBIT_TICK, syn.BYBIT_LOT)
    assert (binance.tick, binance.lot) == (syn.BINANCE_TICK, syn.BINANCE_LOT)
    assert bybit.clip == binance.clip == 0.010
    assert (bybit.venue, bybit.symbol, bybit.quote) == ("bybit", "BTCUSDT", "USDT")
    assert (binance.venue, binance.symbol, binance.quote) == ("binance", "BTCFDUSD", "FDUSD")


def test_the_spot_category_and_the_spot_lot_field_are_required() -> None:
    linear = {"retCode": 0, "result": {"category": "linear", "list": []}}
    with pytest.raises(ins.InstrumentSpecError, match="not spot"):
        ins.parse_bybit(linear)
    no_base = json.loads(syn.bybit_spec())
    del no_base["result"]["list"][0]["lotSizeFilter"]["basePrecision"]
    with pytest.raises(KeyError):
        ins.parse_bybit(no_base)
    with pytest.raises(ins.InstrumentSpecError, match="retCode"):
        ins.parse_bybit({"retCode": 10001})
    with pytest.raises(ins.InstrumentSpecError, match="not in Binance"):
        ins.parse_binance({"symbols": [{"symbol": "BTCUSDT", "filters": []}]})
    no_lot = {"symbols": [{"symbol": "BTCFDUSD", "filters": [{"filterType": "PRICE_FILTER"}]}]}
    with pytest.raises(ins.InstrumentSpecError, match="LOT_SIZE"):
        ins.parse_binance(no_lot)


def test_the_clip_must_sit_on_the_lot_grid_and_the_instrument_be_known() -> None:
    with pytest.raises(ins.InstrumentSpecError, match="not a multiple of the lot"):
        ins.SpotInstrument(BYBIT_SPOT, tick=0.1, lot=0.003)
    with pytest.raises(ins.InstrumentSpecError, match="unknown instrument"):
        ins.SpotInstrument("BTCUSDT", tick=0.1, lot=0.001)
    with pytest.raises(ins.InstrumentSpecError, match="positive"):
        ins.SpotInstrument(BINANCE_SPOT, tick=0.0, lot=0.001)


def test_the_specification_is_fetched_from_the_registered_endpoint() -> None:
    venue = syn.SpotVenue(syn.SpotMarket(frozenset()))
    spec = ins.fetch_spec(BINANCE_SPOT, opener=venue)
    assert venue.asked == [("GET", ins.SPEC_URLS[BINANCE_SPOT])]
    assert "category=spot" in ins.SPEC_URLS[BYBIT_SPOT]
    assert spec.tick == syn.BINANCE_TICK


def test_the_frozen_record_round_trips_and_null_is_refused() -> None:
    specs = [
        ins.SpotInstrument(BYBIT_SPOT, 0.1, 0.000001),
        ins.SpotInstrument(BINANCE_SPOT, 0.01, 0.00001),
    ]
    record = ins.to_record(specs)
    assert [ins.from_registration(record, s.instrument) for s in specs] == specs
    with pytest.raises(ins.InstrumentSpecError, match="not frozen"):
        ins.from_registration(None, BYBIT_SPOT)
    with pytest.raises(ins.InstrumentSpecError, match="has no"):
        ins.from_registration({}, BYBIT_SPOT)
