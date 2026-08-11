"""Tests for the Binance archive downloader.

Everything here runs offline against archives built in-memory. That is
deliberate: the parsing quirks these tests pin down — shifting headers, changing
epoch units, an inverted boolean — are exactly the failures that produce
plausible numbers instead of errors, and a test that needs the network is a test
that gets skipped when it matters.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile
from datetime import date

import pandas as pd
import pytest

from lobml.data.binance import (
    ArchiveSpec,
    BinanceArchiveError,
    epoch_unit,
    iter_days,
    parse_agg_trades,
    parse_book_ticker,
    read_archive_csv,
    resample_book,
)
from lobml.data.schema import TRADE_SCHEMA, book_schema
from lobml.data.validate import validate_book, validate_trades


def make_archive(csv: str, name: str = "data.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, csv)
    return buffer.getvalue()


TRADES_SPEC = ArchiveSpec(market="futures-um", kind="aggTrades", symbol="BTCUSDT")
BOOK_SPEC = ArchiveSpec(market="futures-um", kind="bookTicker", symbol="BTCUSDT")

# Real layouts. Futures aggTrades carry a header and lower-case booleans;
# older spot exports have neither, and an extra is_best_match column.
FUTURES_AGG_TRADES = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    "2008134055,42570.3,0.008,4557527089,4557527090,1707091200181,true\n"
    "2008134056,42570.4,0.143,4557527091,4557527092,1707091200290,false\n"
    "2008134057,42570.1,0.500,4557527093,4557527094,1707091200999,true\n"
)
SPOT_AGG_TRADES_HEADERLESS = (
    "3599977417,105594.02,0.00047,5013856486,5013856486,1750032000153116,False,True\n"
    "3599977418,105594.05,0.00093,5013856487,5013856487,1750032000387884,True,True\n"
)
BOOK_TICKER = (
    "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
    "3932256034651,42570.3,0.821,42570.4,7.207,1707091200006,1707091200012\n"
    "3932256034700,42570.2,1.500,42570.5,2.000,1707091200150,1707091200160\n"
    "3932256034800,42570.1,3.000,42570.6,1.000,1707091200280,1707091200290\n"
)


# ---------------------------------------------------------------------------
# Archive plumbing
# ---------------------------------------------------------------------------


def test_url_layout_matches_the_published_archive() -> None:
    url = BOOK_SPEC.url(date(2024, 2, 5), "daily")
    assert url.endswith(
        "/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-02-05.zip"
    )


def test_monthly_and_daily_names_differ() -> None:
    assert TRADES_SPEC.filename(date(2024, 2, 5), "monthly").endswith("2024-02.zip")
    assert TRADES_SPEC.filename(date(2024, 2, 5), "daily").endswith("2024-02-05.zip")


def test_unknown_market_is_rejected() -> None:
    with pytest.raises(ValueError, match="market"):
        ArchiveSpec(market="derivatives", kind="aggTrades", symbol="BTCUSDT")  # type: ignore[arg-type]


def test_unsupported_kind_is_rejected() -> None:
    """A dataset with no parser must fail at construction, not mid-download."""
    with pytest.raises(ValueError, match="kind"):
        ArchiveSpec(market="spot", kind="klines", symbol="BTCUSDT")


def test_source_tag_records_market_and_dataset() -> None:
    assert BOOK_SPEC.source_tag == "binance-futures-um-bookticker"


def test_archive_with_several_csvs_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.csv", "1\n")
        archive.writestr("b.csv", "2\n")
    with pytest.raises(BinanceArchiveError, match="exactly one CSV"):
        read_archive_csv(buffer.getvalue())


def test_empty_archive_is_rejected() -> None:
    with pytest.raises(BinanceArchiveError, match="no rows"):
        read_archive_csv(make_archive("a,b,c\n"))


def test_checksum_shape_is_what_the_archive_publishes() -> None:
    """Guards the parse of '<sha256>  <filename>' used to verify a download."""
    payload = make_archive(FUTURES_AGG_TRADES)
    published = f"{hashlib.sha256(payload).hexdigest()}  BTCUSDT-aggTrades-2024-02-05.zip"
    assert published.split()[0] == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Epoch units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1707091200, "s"),
        (1707091200181, "ms"),
        (1750032000153116, "us"),
        (1750032000153116000, "ns"),
    ],
)
def test_epoch_unit_is_inferred_from_magnitude(value: int, expected: str) -> None:
    """Binance changed units between exports; both are bare integers."""
    assert epoch_unit(pd.Series([value, value + 1])) == expected


def test_wrong_epoch_unit_fails_silently_in_the_dangerous_direction() -> None:
    """Shows why the unit is inferred rather than assumed.

    Reading microseconds as milliseconds overflows, which is loud and harmless.
    Reading milliseconds as microseconds does not: it yields January 1970 — a
    real date, in order, that parses and validates. Every timestamp would be
    wrong by decades and nothing would raise.
    """
    millis = 1707091200181  # 2024-02-05
    assert pd.to_datetime(millis, unit="ms", utc=True).year == 2024
    assert pd.to_datetime(millis, unit="us", utc=True).year == 1970  # silent

    micros = 1750032000153116  # 2025-06-16
    assert pd.to_datetime(micros, unit="us", utc=True).year == 2025
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        pd.to_datetime(micros, unit="ms", utc=True)


# ---------------------------------------------------------------------------
# aggTrades
# ---------------------------------------------------------------------------


def test_futures_agg_trades_parse_to_the_contract() -> None:
    frame = parse_agg_trades(make_archive(FUTURES_AGG_TRADES), TRADES_SPEC)
    TRADE_SCHEMA.validate(frame)
    assert len(frame) == 3
    assert validate_trades(frame).ok


def test_headerless_export_keeps_its_first_row() -> None:
    """Reading a headerless file as headed silently drops one trade."""
    spot = ArchiveSpec(market="spot", kind="aggTrades", symbol="BTCUSDT")
    frame = parse_agg_trades(make_archive(SPOT_AGG_TRADES_HEADERLESS), spot)
    assert len(frame) == 2
    assert frame["trade_id"].iloc[0] == 3599977417


def test_microsecond_timestamps_are_read_correctly() -> None:
    spot = ArchiveSpec(market="spot", kind="aggTrades", symbol="BTCUSDT")
    frame = parse_agg_trades(make_archive(SPOT_AGG_TRADES_HEADERLESS), spot)
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2025-06-16 00:00:00.153116", tz="UTC")


def test_lowercase_booleans_are_parsed_not_coerced() -> None:
    """Every non-empty string is truthy; a naive cast makes all trades buys."""
    frame = parse_agg_trades(make_archive(FUTURES_AGG_TRADES), TRADES_SPEC)
    assert list(frame["is_buyer_maker"]) == [True, False, True]


def test_agg_trades_are_stamped_with_their_source() -> None:
    frame = parse_agg_trades(make_archive(FUTURES_AGG_TRADES), TRADES_SPEC)
    assert (frame["source"] == "binance-futures-um-aggtrades").all()
    assert (frame["symbol"] == "BTCUSDT").all()


def test_missing_column_is_reported_rather_than_guessed() -> None:
    broken = "agg_trade_id,price,quantity\n1,2.0,3.0\n"
    with pytest.raises(BinanceArchiveError, match="missing column"):
        parse_agg_trades(make_archive(broken), TRADES_SPEC)


# ---------------------------------------------------------------------------
# bookTicker
# ---------------------------------------------------------------------------


def test_book_ticker_parses_to_a_one_level_book() -> None:
    frame = parse_book_ticker(make_archive(BOOK_TICKER), BOOK_SPEC)
    book_schema(1).validate(frame)
    assert validate_book(frame, sampled=False).ok


def test_book_ticker_uses_transaction_time_not_event_time() -> None:
    """Features must be stamped when the book changed, not when it was pushed."""
    frame = parse_book_ticker(make_archive(BOOK_TICKER), BOOK_SPEC)
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2024-02-05 00:00:00.006", tz="UTC")


def test_book_ticker_keeps_the_update_id_as_sequence() -> None:
    frame = parse_book_ticker(make_archive(BOOK_TICKER), BOOK_SPEC)
    assert frame["sequence_id"].iloc[0] == 3932256034651


def test_book_ticker_missing_column_is_reported() -> None:
    broken = "update_id,best_bid_price\n1,2.0\n"
    with pytest.raises(BinanceArchiveError, match="missing column"):
        parse_book_ticker(make_archive(broken), BOOK_SPEC)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def test_resampling_keeps_the_last_state_of_each_interval() -> None:
    """The state at the end of an interval is what a decision then would see."""
    frame = parse_book_ticker(make_archive(BOOK_TICKER), BOOK_SPEC)
    reduced = resample_book(frame, "100ms")
    # Updates at .006, .150 and .280 fall into the .000, .100 and .200 buckets.
    assert len(reduced) == 3
    assert reduced["sequence_id"].iloc[-1] == 3932256034800


def test_resampling_collapses_several_updates_in_one_interval() -> None:
    frame = parse_book_ticker(make_archive(BOOK_TICKER), BOOK_SPEC)
    reduced = resample_book(frame, "1s")
    assert len(reduced) == 1
    assert reduced["sequence_id"].iloc[0] == 3932256034800


def test_resampling_drops_empty_intervals_rather_than_filling_them() -> None:
    """Filling would repeat a sequence_id and forge an exchange update."""
    frame = parse_book_ticker(make_archive(BOOK_TICKER), BOOK_SPEC)
    reduced = resample_book(frame, "10ms")
    assert not reduced["sequence_id"].duplicated().any()
    assert len(reduced) == 3


def test_resampled_book_still_satisfies_the_contract() -> None:
    frame = parse_book_ticker(make_archive(BOOK_TICKER), BOOK_SPEC)
    reduced = resample_book(frame, "100ms")
    book_schema(1).validate(reduced)
    assert validate_book(reduced, sampled=True).ok


# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------


def test_iter_days_is_inclusive_at_both_ends() -> None:
    days = list(iter_days(date(2024, 2, 27), date(2024, 3, 1)))
    assert days == [date(2024, 2, 27), date(2024, 2, 28), date(2024, 2, 29), date(2024, 3, 1)]


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="before start"):
        list(iter_days(date(2024, 3, 1), date(2024, 2, 1)))


# ---------------------------------------------------------------------------
# Network resilience
# ---------------------------------------------------------------------------


def test_dropped_connection_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 60-day download used to die on the first blip; it died at day 38."""
    import http.client

    from lobml.data import binance

    calls = {"n": 0}

    def flaky(url, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.client.RemoteDisconnected("Remote end closed connection")
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(binance.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(binance.time, "sleep", lambda _: None)

    # Retries exhaust the transient failures, then the real 404 surfaces.
    with pytest.raises(BinanceArchiveError, match="404"):
        binance._fetch("https://example.invalid/x.zip")
    assert calls["n"] == 3


def test_missing_archive_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 means the day does not exist; asking again cannot help."""
    from lobml.data import binance

    calls = {"n": 0}

    def missing(url, timeout=0):
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(binance.urllib.request, "urlopen", missing)
    monkeypatch.setattr(binance.time, "sleep", lambda _: None)

    with pytest.raises(BinanceArchiveError, match="404"):
        binance._fetch("https://example.invalid/x.zip")
    assert calls["n"] == 1
