"""Spot tick and lot steps, from each venue's public instrument endpoint, spot category.

Registered (``data.instrument_spec``, frozen by the D amendment): each
instrument's tick and lot come from its venue's spot specification and are
recorded in the amendment; the off-grid exclusion uses each day's own tick
(:mod:`trading_research.spot.loader`), and the specification's tick is set
beside it as a note. The clip, 0.010 BTC for every slot, must sit on the lot
grid.

- **Bybit spot**: ``/v5/market/instruments-info?category=spot``; the tick is
  ``priceFilter.tickSize`` and the lot ``lotSizeFilter.basePrecision`` (spot
  has no ``qtyStep``).
- **Binance spot**: ``/api/v3/exchangeInfo``; the tick is the
  ``PRICE_FILTER``'s ``tickSize`` and the lot the ``LOT_SIZE``'s ``stepSize``.

Both endpoints are public and need no credential.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from trading_research.data.study_loader import on_grid
from trading_research.spot.archives import BINANCE_SPOT, BYBIT_SPOT, INSTRUMENTS, Opener

#: Base-currency size of one trade, every slot, no sizing.
CLIP_BTC = 0.010

SPEC_URLS: dict[str, str] = {
    BYBIT_SPOT: "https://api.bybit.com/v5/market/instruments-info?category=spot&symbol=BTCUSDT",
    BINANCE_SPOT: "https://api.binance.com/api/v3/exchangeInfo?symbol=BTCFDUSD",
}
QUOTES: dict[str, str] = {BYBIT_SPOT: "USDT", BINANCE_SPOT: "FDUSD"}


class InstrumentSpecError(ValueError):
    """A specification is missing, malformed or not yet frozen."""


@dataclass(frozen=True)
class SpotInstrument:
    """A spot instrument as the study trades it: its tick, its lot and its clip."""

    instrument: str
    tick: float
    lot: float
    clip: float = CLIP_BTC

    def __post_init__(self) -> None:
        if self.instrument not in INSTRUMENTS:
            raise InstrumentSpecError(f"unknown instrument {self.instrument!r}")
        if not (self.tick > 0 and self.lot > 0 and self.clip > 0):
            raise InstrumentSpecError(f"tick, lot and clip must be positive: {self}")
        if not on_grid(self.clip, self.lot):
            raise InstrumentSpecError(
                f"the clip {self.clip} is not a multiple of the lot {self.lot}"
            )

    @property
    def venue(self) -> str:
        return self.instrument.split("-", 1)[0]

    @property
    def symbol(self) -> str:
        return self.instrument.split(":", 1)[1]

    @property
    def quote(self) -> str:
        return QUOTES[self.instrument]


def parse_bybit(payload: Mapping[str, Any], symbol: str = "BTCUSDT") -> tuple[float, float]:
    """Tick and lot from Bybit's spot instrument-info answer."""
    if payload.get("retCode") != 0:
        raise InstrumentSpecError(f"Bybit answered retCode={payload.get('retCode')!r}")
    result = payload.get("result", {})
    if result.get("category", "spot") != "spot":
        raise InstrumentSpecError(f"category {result.get('category')!r} is not spot")
    for row in result.get("list", []):
        if row.get("symbol") == symbol:
            return float(row["priceFilter"]["tickSize"]), float(
                row["lotSizeFilter"]["basePrecision"]
            )
    raise InstrumentSpecError(f"{symbol} is not in Bybit's spot list")


def parse_binance(payload: Mapping[str, Any], symbol: str = "BTCFDUSD") -> tuple[float, float]:
    """Tick and lot from Binance's spot exchange-info answer."""
    for row in payload.get("symbols", []):
        if row.get("symbol") != symbol:
            continue
        filters = {f.get("filterType"): f for f in row.get("filters", [])}
        if "PRICE_FILTER" not in filters or "LOT_SIZE" not in filters:
            raise InstrumentSpecError(f"{symbol} lacks a PRICE_FILTER or a LOT_SIZE filter")
        return float(filters["PRICE_FILTER"]["tickSize"]), float(filters["LOT_SIZE"]["stepSize"])
    raise InstrumentSpecError(f"{symbol} is not in Binance's exchange info")


def parse_spec(instrument: str, payload: Mapping[str, Any]) -> SpotInstrument:
    """The instrument from its venue's answer."""
    parse = parse_bybit if instrument == BYBIT_SPOT else parse_binance
    if instrument not in SPEC_URLS:
        raise InstrumentSpecError(f"unknown instrument {instrument!r}")
    tick, lot = parse(payload, instrument.split(":", 1)[1])
    return SpotInstrument(instrument, tick, lot)


def fetch_spec(
    instrument: str, *, opener: Opener = urllib.request.urlopen, timeout: float = 30.0
) -> SpotInstrument:
    """Today's spot specification of ``instrument``, from its venue."""
    if instrument not in SPEC_URLS:
        raise InstrumentSpecError(f"unknown instrument {instrument!r}")
    with opener(SPEC_URLS[instrument], timeout=timeout) as response:
        payload = json.loads(response.read())
    return parse_spec(instrument, payload)


def from_registration(frozen: Mapping[str, Any] | None, instrument: str) -> SpotInstrument:
    """The instrument as ``data.instrument_spec`` records it; refused while it is null."""
    if frozen is None:
        raise InstrumentSpecError(
            "data.instrument_spec is not frozen yet (the D amendment sets it)"
        )
    if instrument not in frozen:
        raise InstrumentSpecError(f"data.instrument_spec has no {instrument!r}")
    entry = frozen[instrument]
    return SpotInstrument(instrument, float(entry["tick"]), float(entry["lot"]))


def to_record(instruments: list[SpotInstrument]) -> dict[str, dict[str, float]]:
    """``{instrument: {tick, lot}}``, the shape ``data.instrument_spec`` takes."""
    return {i.instrument: {"tick": i.tick, "lot": i.lot} for i in instruments}
