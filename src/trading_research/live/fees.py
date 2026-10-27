"""The fee a paper fill pays: the repository's dated tables, looked up at the fill's instant.

Every paper fill pays, in basis points of its notional, the maker or taker fee
of the chosen tier in force at the fill's instant. The rows come from the
tables the backtests already use, never restated here:

==================  ==============================================================  ========
instrument          table                                                           default
==================  ==============================================================  ========
bybit spot BTCUSDT  ``configs/spot_2026.yaml`` ``fees``, 2025-01-31..2026-10-22        non_vip
binance spot        ``configs/spot_2026.yaml`` ``fees`` (BTCFDUSD only)                 regular
BTCFDUSD
bybit linear        ``backtest/costs.py`` ``BYBIT_LINEAR_TIERS``, the page dated         base
                    2026-09-02 as transcribed 2026-10-02
binance linear      ``backtest/costs.py``: the standard taker ``BINANCE_UM_TAKER_BP``   standard
                    and the 2 bp maker its notes state; undated
==================  ==============================================================  ========

One row is added here, for the venue the live smoke test runs on and no table
covered: Binance's spot schedule as read on 2026-10-27 (UTC) from the public
endpoint the spot registration lists as B8 (``commission/trade-level/get``:
regular 10/10 bp) with BTCUSDT absent from its zero-fee list (B5), recorded as
holding on the day it was read.

Each quote says what it rests on:

* ``sourced``: the instant lies inside a dated interval;
* ``carried``: the instant lies after the last interval and the run allowed the
  last interval to be carried forward (the default for a paper run, whose
  instants are always later than any transcription); the page flags it;
* ``undated``: the table has no dates (Binance USD-M);
* ``given``: ``--fee-bp`` on the command line, not sourced at all.

An instrument with no table and no ``--fee-bp`` is refused: a paper run does
not invent a fee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trading_research.backtest.costs import (
    BINANCE_UM_TAKER_BP,
    BYBIT_FEE_RETRIEVED,
    BYBIT_FEE_SOURCE,
    BYBIT_LINEAR_TIERS,
)

NS_PER_S = 1_000_000_000
#: The spot registration's configuration, found from the package so any working directory works.
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "spot_2026.yaml"


class FeeNotSourced(LookupError):
    """No table holds this instrument, tier or instant, and nothing was given."""


@dataclass(frozen=True)
class FeeQuote:
    maker_bp: float
    taker_bp: float
    status: str
    note: str


@dataclass(frozen=True)
class Interval:
    """A tier table holding from ``start_ns`` (included) to ``end_ns`` (excluded); None: undated."""

    start_ns: int | None
    end_ns: int | None
    tiers: dict[str, tuple[float, float]]
    source: str


def _iso(ns: int) -> str:
    import pandas as pd

    return pd.Timestamp(ns, tz="UTC").isoformat()


def _ns(stamp: str) -> int:
    import pandas as pd

    return int(pd.Timestamp(stamp).tz_convert("UTC").value)


#: Binance spot, read on 2026-10-27 (UTC) for the venue the smoke test runs on.
BINANCE_SPOT_READ_2026_10_27 = Interval(
    _ns("2026-10-27T00:00:00Z"),
    _ns("2026-10-28T00:00:00Z"),
    {"regular": (10.0, 10.0), "vip1": (9.0, 10.0), "vip2": (8.0, 10.0), "vip3": (4.0, 6.0)},
    "https://www.binance.com/bapi/composite/v1/public/commission/trade-level/get (levels 0-3), "
    "read 2026-10-27; BTCUSDT absent from "
    "https://www.binance.com/bapi/accounts/v1/public/commission/symbol-activity-list, read "
    "2026-10-27; held to cover the UTC day it was read",
)


def _config_intervals(key: str, config: Path) -> list[Interval]:
    """The spot registration's dated intervals for ``key``, through its own reader."""
    from trading_research.spot.costs import FeeTable

    fees: dict[str, Any] = yaml.safe_load(config.read_text(encoding="utf-8"))["fees"]
    table = FeeTable.from_config(fees)
    out = []
    for interval in sorted(
        (iv for iv in table.intervals if iv.instrument == key), key=lambda iv: iv.start_ns
    ):
        out.append(
            Interval(
                interval.start_ns,
                interval.end_ns,
                {name: (maker, taker) for name, maker, taker in interval.tiers},
                f"configs/{config.name} fees.intervals.{key} (sources {', '.join(interval.sources)})",
            )
        )
    return out


def tables(
    venue: str, market: str, symbol: str, config: Path = CONFIG_PATH
) -> tuple[list[Interval], str]:
    """The intervals that hold for an instrument, and its default tier."""
    symbol = symbol.upper()
    if (venue, market, symbol) == ("bybit", "spot", "BTCUSDT"):
        return _config_intervals("bybit_spot_BTCUSDT", config), "non_vip"
    if (venue, market, symbol) == ("binance", "spot", "BTCFDUSD"):
        return _config_intervals("binance_spot_BTCFDUSD", config), "regular"
    if (venue, market, symbol) == ("binance", "spot", "BTCUSDT"):
        return [BINANCE_SPOT_READ_2026_10_27], "regular"
    if (venue, market) == ("bybit", "linear"):
        tiers = {t.name: (t.maker_bp, t.taker_bp) for t in BYBIT_LINEAR_TIERS}
        return [
            Interval(
                _ns("2026-09-02T00:00:00Z"),
                _ns(f"{BYBIT_FEE_RETRIEVED}T00:00:00Z") + 86_400 * NS_PER_S,
                tiers,
                f"backtest/costs.py BYBIT_LINEAR_TIERS from {BYBIT_FEE_SOURCE}, page dated "
                f"2026-09-02, transcribed {BYBIT_FEE_RETRIEVED}",
            )
        ], "base"
    if (venue, market) == ("binance", "linear"):
        return [
            Interval(
                None,
                None,
                {"standard": (2.0, BINANCE_UM_TAKER_BP)},
                "backtest/costs.py BINANCE_UM_TAKER_BP and the 2 bp maker its notes state; undated",
            )
        ], "standard"
    return [], ""


@dataclass
class FeeSchedule:
    """The fees one paper run pays, looked up per fill."""

    venue: str
    market: str
    symbol: str
    tier: str
    intervals: list[Interval] = field(default_factory=list)
    given: tuple[float, float] | None = None
    carry_forward: bool = True
    #: Look every fill up at this instant instead of its own (a replay of a day the
    #: table does not reach, a synthetic day); the quote's status says so.
    as_of_ns: int | None = None

    def quote(self, ts_ns: int) -> FeeQuote:
        if self.as_of_ns is not None:
            inner = self._quote(self.as_of_ns)
            stamp = _iso(self.as_of_ns)
            return FeeQuote(
                inner.maker_bp,
                inner.taker_bp,
                f"{inner.status} as of {stamp}",
                f"looked up at {stamp}, not at the fill's instant: {inner.note}",
            )
        return self._quote(ts_ns)

    def _quote(self, ts_ns: int) -> FeeQuote:
        if self.given is not None:
            maker, taker = self.given
            return FeeQuote(maker, taker, "given", "--fee-bp on the command line; not sourced")
        listed = [iv for iv in self.intervals if self.tier in iv.tiers]
        for iv in listed:
            if iv.start_ns is None:
                maker, taker = iv.tiers[self.tier]
                return FeeQuote(maker, taker, "undated", iv.source)
        holding = [
            iv
            for iv in listed
            if iv.start_ns is not None
            and iv.end_ns is not None
            and iv.start_ns <= ts_ns < iv.end_ns
        ]
        if holding:
            maker = max(iv.tiers[self.tier][0] for iv in holding)
            taker = max(iv.tiers[self.tier][1] for iv in holding)
            return FeeQuote(maker, taker, "sourced", "; ".join(iv.source for iv in holding))
        earlier = [iv for iv in listed if iv.end_ns is not None and iv.end_ns <= ts_ns]
        if earlier and self.carry_forward:
            last = max(earlier, key=lambda iv: iv.end_ns or 0)
            maker, taker = last.tiers[self.tier]
            return FeeQuote(maker, taker, "carried", f"carried forward past its end: {last.source}")
        raise FeeNotSourced(
            f"{self.venue} {self.market} {self.symbol} tier {self.tier!r}: no sourced interval "
            f"holds instant {ts_ns}; pass --fee-bp MAKER,TAKER to run at a fee you state"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "market": self.market,
            "symbol": self.symbol,
            "tier": self.tier,
            "given_bp": list(self.given) if self.given else None,
            "carry_forward": self.carry_forward,
            "as_of": _iso(self.as_of_ns) if self.as_of_ns is not None else None,
            "intervals": [
                {
                    "start_ns": iv.start_ns,
                    "end_ns": iv.end_ns,
                    "maker_taker_bp": list(iv.tiers.get(self.tier, (float("nan"),) * 2)),
                    "source": iv.source,
                }
                for iv in self.intervals
                if self.tier in iv.tiers
            ],
        }


def schedule_for(
    venue: str,
    market: str,
    symbol: str,
    tier: str | None = None,
    *,
    given: tuple[float, float] | None = None,
    carry_forward: bool = True,
    as_of_ns: int | None = None,
    config: Path = CONFIG_PATH,
) -> FeeSchedule:
    """The schedule a run pays; refuses an instrument or tier no table lists, unless given."""
    intervals, default = tables(venue, market, symbol, config)
    chosen = tier or default or "given"
    if given is None:
        if not intervals:
            raise FeeNotSourced(
                f"no fee table covers {venue} {market} {symbol.upper()}; pass --fee-bp MAKER,TAKER"
            )
        known = sorted({name for iv in intervals for name in iv.tiers})
        if chosen not in known:
            raise FeeNotSourced(f"tier {chosen!r} is not in the table; tiers listed: {known}")
    schedule = FeeSchedule(
        venue, market, symbol.upper(), chosen, intervals, given, carry_forward, as_of_ns
    )
    if as_of_ns is not None:
        schedule.quote(as_of_ns)  # refuses now, not at the first fill
    return schedule
