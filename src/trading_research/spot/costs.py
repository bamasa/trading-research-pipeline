"""The spot study's fee table: sourced, dated, and the only place a spot fee comes from.

The registration's section "Fees" (``docs/preregistration/spot_2026.md``) and
the configuration's ``fees`` block (``configs/spot_2026.yaml``) fix it. The
table is read from that block, never restated here, so that the fee
amendment, which may only add intervals there for P's days, changes no code.

The rules, as registered:

* **Every fill pays the fee in force at its instant**, of the sourced interval
  of its venue that holds that instant, in basis points of the fill's notional,
  booked in the quote at the fill price. An interval runs from its ``start``
  (included) to its ``end`` (excluded), UTC.
* **Two intervals at one instant** (sources that disagree) give each side's
  higher fee, the rule the registration fixes for P's transcription.
* **A day is covered at a tier** when every instant of it, midnight to the next
  midnight, lies in an interval that lists the tier. A day not covered at its
  venue's verdict tier (Binance's ``regular``, Bybit's ``non_vip``) is excluded
  on that venue, reason ``fee not sourced``; an instant no interval covers
  raises :class:`FeeNotSourced`.
* **The book executor charges one fee a day**: the day's higher maker and higher
  taker fee of the tier over the intervals the day meets. It differs from the
  instant's fee only on a day an interval boundary splits, possible only in P.
* **The book executor's settings are built here only.** :func:`book_settings`
  makes them from the table; :func:`execute_book_day` and
  :func:`book_hold_table`, the two ways a book slot is run, refuse any other
  settings, the executor's default (the perpetual's 5.5 bp taker) included, and
  check every fill's fee against the table (K-units).

The undated cells (Bybit's Supreme VIP and Pro 6, read without a capture) are
kept for reporting only; they cover no instant and no fill ever pays them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from trading_research.backtest.costs import FeeTier
from trading_research.backtest.taker import TakerBook, TakerDay, TakerSettings, execute_day
from trading_research.market_making.events import NS_PER_S
from trading_research.search_engine.walk_tables import HoldTable, WalkTable, hold_table

CONFIG_PATH = Path("configs/spot_2026.yaml")
#: The exclusion reason of a day the table does not cover at the verdict tier.
FEE_NOT_SOURCED = "fee not sourced"
#: Venue-qualified identifiers and the configuration's instrument keys.
INSTRUMENT_KEYS: dict[str, str] = {
    "binance-spot:BTCFDUSD": "binance_spot_BTCFDUSD",
    "bybit-spot:BTCUSDT": "bybit_spot_BTCUSDT",
}
NS_PER_DAY = 86_400 * NS_PER_S


class FeeNotSourced(LookupError):
    """An instant or a day that no sourced interval covers at the tier asked for."""


class FeeUnitsError(ValueError):
    """A fill's fee differs from the one the table assigns it (K-units)."""


class SettingsRefused(ValueError):
    """Book executor settings that were not built from the fee table."""


def instrument_key(name: str) -> str:
    """The configuration's key for ``name``, a key or a venue-qualified identifier."""
    if name in INSTRUMENT_KEYS.values():
        return name
    if name in INSTRUMENT_KEYS:
        return INSTRUMENT_KEYS[name]
    raise KeyError(f"unknown spot instrument {name!r}; known: {sorted(INSTRUMENT_KEYS)}")


def instant_ns(stamp: str | datetime) -> int:
    """An ISO-8601 UTC instant (``...Z``) or an aware datetime as epoch nanoseconds."""
    moment = datetime.fromisoformat(stamp) if isinstance(stamp, str) else stamp
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise ValueError(f"an interval bound must be given in UTC, got {stamp!r}")
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = moment - epoch
    return (delta.days * 86_400 + delta.seconds) * NS_PER_S + delta.microseconds * 1000


def day_bounds_ns(day: date) -> tuple[int, int]:
    """Midnight of ``day`` and the next midnight, UTC, in epoch nanoseconds."""
    start = instant_ns(datetime(day.year, day.month, day.day, tzinfo=UTC))
    return start, start + NS_PER_DAY


@dataclass(frozen=True)
class FeeInterval:
    """One row of the table: a venue's tiers from ``start_ns`` to ``end_ns`` (excluded)."""

    instrument: str
    start_ns: int
    end_ns: int
    sources: tuple[str, ...]
    #: ``(tier, maker_bp, taker_bp)``, in the order the configuration lists them.
    tiers: tuple[tuple[str, float, float], ...]

    def __post_init__(self) -> None:
        if not self.start_ns < self.end_ns:
            raise ValueError(f"an interval must end after it starts: {self}")
        if not self.sources:
            raise ValueError("an interval without a source is not a sourced interval")
        for tier, maker, taker in self.tiers:
            if not (np.isfinite(maker) and np.isfinite(taker) and maker >= 0 and taker >= 0):
                raise ValueError(f"tier {tier!r} has a fee that is negative or not finite")

    def fee(self, tier: str) -> tuple[float, float] | None:
        """``(maker_bp, taker_bp)`` of ``tier``, or None when the interval does not list it."""
        for name, maker, taker in self.tiers:
            if name == tier:
                return maker, taker
        return None


@dataclass(frozen=True)
class FeeTable:
    """Every sourced interval of both venues, and each venue's verdict tier."""

    intervals: tuple[FeeInterval, ...]
    #: ``{instrument key: verdict tier}``.
    verdict: Mapping[str, str]
    #: The undated cells, ``{instrument key: {tier: (maker_bp, taker_bp)}}``; reported only.
    undated: Mapping[str, Mapping[str, tuple[float, float]]]

    @classmethod
    def from_config(cls, fees: Mapping[str, Any]) -> FeeTable:
        """The table from the configuration's ``fees`` block."""
        known_sources = set(fees.get("sources", {}))
        intervals = []
        for instrument, rows in fees["intervals"].items():
            for row in rows:
                sources = tuple(str(s) for s in row["sources"])
                unknown = [s for s in sources if known_sources and s not in known_sources]
                if unknown:
                    raise ValueError(f"{instrument}: sources {unknown} are not in fees.sources")
                tiers = tuple((str(t), float(v[0]), float(v[1])) for t, v in row["tiers"].items())
                start, end = instant_ns(row["start"]), instant_ns(row["end"])
                intervals.append(
                    FeeInterval(instrument_key(instrument), start, end, sources, tiers)
                )
        verdict = {instrument_key(k): str(v) for k, v in fees["verdict_tier"].items()}
        undated = {
            instrument_key(k): {
                str(t): (float(v[0]), float(v[1])) for t, v in cell["tiers"].items()
            }
            for k, cell in (fees.get("undated_cells") or {}).items()
        }
        return cls(tuple(intervals), verdict, undated)

    def verdict_tier(self, instrument: str) -> str:
        return self.verdict[instrument_key(instrument)]

    def listing(self, instrument: str, tier: str) -> list[FeeInterval]:
        """The intervals of ``instrument`` that list ``tier``, by start."""
        key = instrument_key(instrument)
        rows = [iv for iv in self.intervals if iv.instrument == key and iv.fee(tier) is not None]
        return sorted(rows, key=lambda iv: (iv.start_ns, iv.end_ns))

    def fees_bp(self, instrument: str, tier: str, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(maker_bp, taker_bp)`` at each instant of ``ns`` (epoch nanoseconds).

        Raises :class:`FeeNotSourced` if any instant lies in no interval listing ``tier``.
        """
        stamps = np.asarray(ns, dtype=np.int64)
        maker = np.full(stamps.shape, np.nan)
        taker = np.full(stamps.shape, np.nan)
        for iv in self.listing(instrument, tier):
            inside = (stamps >= iv.start_ns) & (stamps < iv.end_ns)
            pair = iv.fee(tier)
            assert pair is not None
            maker = np.where(inside, np.fmax(maker, pair[0]), maker)
            taker = np.where(inside, np.fmax(taker, pair[1]), taker)
        if np.isnan(taker).any():
            first = int(stamps[np.isnan(taker)].flat[0])
            raise FeeNotSourced(f"{instrument} {tier}: no sourced interval holds instant {first}")
        return maker, taker

    def taker_bp(self, instrument: str, tier: str, ns: np.ndarray) -> np.ndarray:
        """The taker fee at each instant: what every fill of this study pays."""
        return self.fees_bp(instrument, tier, ns)[1]

    def covered(self, instrument: str, day: date, tier: str | None = None) -> bool:
        """Whether every instant of ``day`` lies in an interval listing ``tier``
        (the venue's verdict tier when ``tier`` is None)."""
        tier = tier or self.verdict_tier(instrument)
        cursor, end = day_bounds_ns(day)
        moved = True
        while cursor < end and moved:
            moved = False
            for iv in self.listing(instrument, tier):
                if iv.start_ns <= cursor < iv.end_ns:
                    cursor, moved = iv.end_ns, True
        return cursor >= end

    def exclusion(self, instrument: str, day: date) -> str | None:
        """``fee not sourced`` for a day not covered at the venue's verdict tier, else None."""
        return None if self.covered(instrument, day) else FEE_NOT_SOURCED

    def day_fees(self, instrument: str, day: date, tier: str | None = None) -> FeeTier:
        """The book executor's one fee for ``day``: the higher maker and the higher
        taker fee of ``tier`` over the intervals the day meets."""
        tier = tier or self.verdict_tier(instrument)
        if not self.covered(instrument, day, tier):
            raise FeeNotSourced(f"{instrument} {tier}: {day} is not covered ({FEE_NOT_SOURCED})")
        start, end = day_bounds_ns(day)
        met = [iv.fee(tier) for iv in self.listing(instrument, tier)
               if iv.start_ns < end and iv.end_ns > start]  # fmt: skip
        maker = max(pair[0] for pair in met if pair is not None)
        taker = max(pair[1] for pair in met if pair is not None)
        return FeeTier(tier, maker, taker, requires=f"{instrument_key(instrument)} on {day}")

    def tiers(self, instrument: str) -> tuple[str, ...]:
        """Every tier the table lists for ``instrument``, in first-listed order."""
        key, seen = instrument_key(instrument), list[str]()
        for iv in self.intervals:
            if iv.instrument == key:
                seen.extend(t for t, _, _ in iv.tiers if t not in seen)
        return tuple(seen)


def load_fee_table(path: Path = CONFIG_PATH) -> FeeTable:
    """The registered table, read from the configuration's ``fees`` block."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return FeeTable.from_config(config["fees"])


@dataclass(frozen=True, kw_only=True)
class BookSettings(TakerSettings):
    """The book executor's settings for one venue, tier and day, fees from the table."""

    instrument: str
    tier: str
    day: date


def book_settings(
    table: FeeTable,
    instrument: str,
    day: date,
    tier: str | None = None,
    *,
    latency_ns: int = 10_000_000,
) -> BookSettings:
    """Settings for ``backtest.taker.execute_day`` with the table's fee for ``day``;
    every other value is the executor's registered default."""
    key = instrument_key(instrument)
    tier = tier or table.verdict_tier(key)
    fees = table.day_fees(key, day, tier)
    return BookSettings(fees=fees, instrument=key, tier=tier, day=day, latency_ns=latency_ns)


def require_book_settings(settings: object, table: FeeTable, day: date) -> BookSettings:
    """``settings`` if they are the table's for ``day``; :class:`SettingsRefused` otherwise."""
    if not isinstance(settings, BookSettings):
        raise SettingsRefused(
            "the book executor runs only on settings built by spot.costs.book_settings; "
            f"refused {type(settings).__name__} (the executor's default charges the "
            "perpetual's 5.5 bp taker fee)"
        )
    if settings.day != day:
        raise SettingsRefused(f"settings built for {settings.day} given a book of {day}")
    expected = table.day_fees(settings.instrument, day, settings.tier)
    if settings.fees != expected:
        raise SettingsRefused(f"settings charge {settings.fees}, the table {expected}")
    return settings


def fill_fee(notional: np.ndarray | float, fee_bp: np.ndarray | float) -> np.ndarray:
    """A fill's fee in quote, ``notional x fee_bp x 1e-4``: the accounting's arithmetic."""
    return np.asarray(notional, dtype=np.float64) * np.asarray(fee_bp, dtype=np.float64) * 1e-4


def check_fill_fees(expected: np.ndarray, charged: np.ndarray, *, tol: float = 1e-12) -> None:
    """K-units on fills: each fee charged, in quote, against the table's.

    ``expected`` is :func:`fill_fee` of each fill's notional at the fee the table
    assigns its venue, tier and instant. Raises :class:`FeeUnitsError` at the
    first fill that differs by more than ``tol`` of its fee.
    """
    want = np.asarray(expected, dtype=np.float64)
    got = np.asarray(charged, dtype=np.float64)
    if want.shape != got.shape:
        raise FeeUnitsError(f"{got.shape} fees charged against {want.shape} expected")
    bad = ~(np.abs(got - want) <= tol * np.maximum(np.abs(want), 1e-300))
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise FeeUnitsError(f"fill {k}: fee {got[k]!r} charged, the table's is {want[k]!r}")


def execute_book_day(
    book: TakerBook,
    decision_ns: np.ndarray,
    side: np.ndarray,
    hold_ns: int,
    settings: object,
    table: FeeTable,
) -> TakerDay:
    """``backtest.taker.execute_day`` unchanged, on settings the table built for the
    book's day, with every fill's fee checked against the table (K-units)."""
    checked = require_book_settings(settings, table, book.day)
    day = execute_day(book, decision_ns, side, hold_ns, checked)
    trades = day.trades
    size = trades["size"].to_numpy(dtype=np.float64)
    entry = trades["entry_price"].to_numpy(dtype=np.float64) * size
    exit_ = trades["exit_price"].to_numpy(dtype=np.float64) * size
    taker = checked.fees.taker_bp
    expected = fill_fee(entry, taker) + fill_fee(exit_, taker)
    check_fill_fees(expected, trades["fees"].to_numpy(dtype=np.float64))
    return day


def book_hold_table(walk: WalkTable, hold_ns: int, settings: object, table: FeeTable) -> HoldTable:
    """``search_engine.walk_tables.hold_table`` on settings the table built for the day."""
    return hold_table(walk, hold_ns, require_book_settings(settings, table, walk.day))
