"""Downloader for Binance's public market-data archives.

Source: https://data.binance.vision — documented at
https://github.com/binance/binance-public-data. No account, key or credential
is involved; these are static files served over HTTPS.

What is actually published
--------------------------
This matters more than it sounds, because it bounds what the project can do
with free data:

===================  =========================================  ==============
Market               Datasets                                   Book depth
===================  =========================================  ==============
``spot``             aggTrades, trades, klines                  none at all
``futures-um``       + bookTicker, bookDepth, funding, …        one level
``futures-cm``       + bookTicker, …                            one level
===================  =========================================  ==============

**No exchange publishes multi-level order-book history for free.** ``bookTicker``
carries the best bid and ask with their resting sizes — the touch, and nothing
behind it. ``bookDepth`` aggregates liquidity within ±1…5% of mid roughly twice
a minute, which is a slow liquidity context rather than a book.

So features that need the touch (spread, microprice, queue imbalance, order flow
imbalance at level 0) work on real data today. Features that need depth (level
imbalance, book slope, concentration) need the collector, which records the
websocket stream going forward. :func:`lobml.data.schema.book_schema` is
parameterised by depth precisely so both produce the same contract and the same
feature code runs over either.

Volume
------
``bookTicker`` is heavy: BTCUSDT emits ~13.5M updates a day, ~200 MB compressed.
Two months of two symbols is roughly 16 GB of downloading. Three things keep
that manageable, and none of them is optional at this scale:

1. One file is fetched, converted and released before the next is touched, so
   peak memory is one day rather than one range.
2. Output is resampled to a grid on the way in (see :func:`resample_book`), which
   is what turns 6.5 GB of book into 0.6 GB.
3. Work is partitioned by day and already-converted days are skipped, so an
   interrupted download resumes instead of restarting.

Raw archives are kept by default, so a reconversion — a different grid, a fixed
parser — costs no bandwidth. Pass ``keep_raw=False`` to discard them.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from lobml.data.schema import TRADE_SCHEMA, book_schema

BASE_URL: Final = "https://data.binance.vision"
LISTING_URL: Final = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_S3_NS: Final = "{http://s3.amazonaws.com/doc/2006-03-01/}"

Market = Literal["spot", "futures-um", "futures-cm"]
Frequency = Literal["daily", "monthly"]

#: Path fragment for each market in the archive layout.
_MARKET_PREFIX: Final[dict[str, str]] = {
    "spot": "spot",
    "futures-um": "futures/um",
    "futures-cm": "futures/cm",
}

#: Datasets this module knows how to convert into a project contract.
SUPPORTED_KINDS: Final = ("aggTrades", "bookTicker")


class BinanceArchiveError(RuntimeError):
    """A requested archive is missing, corrupt, or fails its checksum."""


@dataclass(frozen=True)
class ArchiveSpec:
    """Identifies one dataset for one instrument."""

    market: Market
    kind: str
    symbol: str

    def __post_init__(self) -> None:
        if self.market not in _MARKET_PREFIX:
            raise ValueError(
                f"unknown market {self.market!r}; expected one of {list(_MARKET_PREFIX)}"
            )
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError(
                f"unsupported kind {self.kind!r}; expected one of {list(SUPPORTED_KINDS)}"
            )

    @property
    def source_tag(self) -> str:
        """Value written into the ``source`` column of every produced row."""
        return f"binance-{self.market}-{self.kind.lower()}"

    def prefix(self, frequency: Frequency) -> str:
        return f"data/{_MARKET_PREFIX[self.market]}/{frequency}/{self.kind}/{self.symbol}/"

    def filename(self, period: date, frequency: Frequency) -> str:
        stamp = period.strftime("%Y-%m-%d") if frequency == "daily" else period.strftime("%Y-%m")
        return f"{self.symbol}-{self.kind}-{stamp}.zip"

    def url(self, period: date, frequency: Frequency) -> str:
        return f"{BASE_URL}/{self.prefix(frequency)}{self.filename(period, frequency)}"


# ---------------------------------------------------------------------------
# Listing and fetching
# ---------------------------------------------------------------------------


def list_available(spec: ArchiveSpec, frequency: Frequency) -> list[str]:
    """Return the archive filenames published for ``spec``, oldest first.

    The bucket listing is paginated at 1000 keys and each archive contributes
    two (the zip and its checksum), so a single request silently truncates at
    500 files — long enough to look complete and short enough to be wrong.
    Following the continuation marker is what makes "which months exist?" a
    reliable answer rather than a plausible one.
    """
    names: list[str] = []
    marker = ""
    while True:
        query = urllib.parse.urlencode(
            {
                "delimiter": "/",
                "prefix": spec.prefix(frequency),
                "max-keys": "1000",
                "marker": marker,
            }
        )
        with urllib.request.urlopen(f"{LISTING_URL}?{query}", timeout=60) as response:
            tree = ET.fromstring(response.read())

        keys = [node.text or "" for node in tree.iter(f"{_S3_NS}Key")]
        if not keys:
            break
        names += [k.rsplit("/", 1)[-1] for k in keys if k.endswith(".zip")]
        marker = keys[-1]
        if (tree.findtext(f"{_S3_NS}IsTruncated") or "false").lower() != "true":
            break

    return sorted(names)


#: Transient failures worth retrying. A dropped connection is not an
#: application error and says nothing about whether the archive exists.
_TRANSIENT: Final = (
    http.client.HTTPException,  # includes RemoteDisconnected, BadStatusLine
    ConnectionError,
    TimeoutError,
    OSError,
)


def _fetch(url: str, timeout: int = 900, *, attempts: int = 4) -> bytes:
    """Download one URL, retrying transient network failures.

    Retries are not optional at this scale. Fetching two months of book data is
    hours of transfer over hundreds of requests, and a single dropped
    connection used to escape as ``RemoteDisconnected`` and end the whole run —
    which is how a 60-day download stopped at day 38.

    An HTTP error is *not* retried: a 404 means the archive does not exist for
    that day, and asking again four times will not change that.
    """
    delay = 2.0
    last: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload: bytes = response.read()
                return payload
        except urllib.error.HTTPError as exc:
            raise BinanceArchiveError(f"{url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            # URLError wraps both name-resolution failures and dropped sockets;
            # only the latter is worth another try.
            if not isinstance(exc.reason, _TRANSIENT) or attempt == attempts:
                raise BinanceArchiveError(f"{url}: {exc.reason}") from exc
            last = exc
        except _TRANSIENT as exc:
            if attempt == attempts:
                raise BinanceArchiveError(f"{url}: {exc!r} after {attempts} attempt(s)") from exc
            last = exc

        time.sleep(delay)
        delay *= 2

    raise BinanceArchiveError(f"{url}: {last!r} after {attempts} attempt(s)")


def fetch_archive(url: str, *, verify: bool = True) -> bytes:
    """Download one archive and check it against the published SHA-256.

    The checksum is verified by default. A truncated download produces a zip
    that still opens and still parses — just with a missing tail — so the
    failure would surface as a silently short dataset rather than an error.
    """
    payload = _fetch(url)
    if not verify:
        return payload

    try:
        expected = _fetch(f"{url}.CHECKSUM", timeout=120).decode().split()[0].strip()
    except BinanceArchiveError:
        # Not every archive ships one; absence is not corruption.
        return payload

    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise BinanceArchiveError(
            f"{url}: checksum mismatch (expected {expected[:12]}…, got {actual[:12]}…); "
            f"the download is incomplete or corrupt"
        )
    return payload


def read_archive_csv(payload: bytes, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read the single CSV inside a Binance archive.

    Two format quirks are handled here rather than at each call site, because
    both change values silently rather than raising:

    **Headers come and go.** Older exports have no header row; newer ones do.
    Reading a headerless file as though it had one drops the first trade;
    reading a headed file as though it did not turns the header into a data row
    and forces every column to ``object``.

    **Epoch units changed.** Binance moved some datasets from milliseconds to
    microseconds. Both are plain integers, so the wrong assumption yields
    timestamps in 1970 or 56000 — plausible-looking numbers that only fail much
    later. Units are therefore inferred per column, by magnitude.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise BinanceArchiveError(f"expected exactly one CSV in the archive, found {members}")
        raw = archive.read(members[0])

    first_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    has_header = not _looks_numeric(first_line.split(",")[0])

    frame = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else _POSITIONAL_COLUMNS.get(len(first_line.split(",")), None),
        usecols=usecols,
    )
    if frame.empty:
        raise BinanceArchiveError("archive contains no rows")
    return frame


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


#: Column names for headerless exports, keyed by field count. Only the layouts
#: this module converts are listed; anything else is rejected loudly rather
#: than guessed at.
_POSITIONAL_COLUMNS: Final[dict[int, list[str]]] = {
    # aggTrades, spot: the trailing is_best_match field is spot-only.
    8: [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
        "is_best_match",
    ],
    # aggTrades, futures.
    7: [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    ],
}


def epoch_unit(values: pd.Series) -> str:
    """Infer whether integer timestamps are in s, ms, µs or ns.

    Decided by magnitude against a fixed reference, which is unambiguous for any
    date this project can encounter: the four candidates differ by three orders
    of magnitude each, and no plausible trading timestamp sits near a boundary.
    """
    median = float(np.nanmedian(values.astype("float64")))
    for unit, upper in (("s", 1e11), ("ms", 1e14), ("us", 1e17)):
        if median < upper:
            return unit
    return "ns"


#: The units :func:`epoch_unit` can return, as pandas accepts them.
EpochUnit = Literal["s", "ms", "us", "ns"]


def to_utc(values: pd.Series) -> pd.Series:
    """Convert integer epoch timestamps to timezone-aware UTC."""
    unit: EpochUnit = epoch_unit(values)  # type: ignore[assignment]
    return pd.to_datetime(values.astype("int64"), unit=unit, utc=True)


def _to_bool(values: pd.Series) -> pd.Series:
    """Parse a boolean column that may arrive as bool, 'true'/'false' or 'True'."""
    if values.dtype == bool:
        return values
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .astype("bool")
    )


# ---------------------------------------------------------------------------
# Conversion into project contracts
# ---------------------------------------------------------------------------


def parse_agg_trades(payload: bytes, spec: ArchiveSpec) -> pd.DataFrame:
    """Convert an ``aggTrades`` archive into a frame satisfying ``TRADE_SCHEMA``.

    An aggregated trade collapses consecutive fills of one aggressing order at
    one price into a single row. That is a coarser view than the raw tape, and
    it is the right default here: it is an order of magnitude smaller, and order
    flow measured per aggressing order is closer to what the features are meant
    to describe than order flow measured per fill.
    """
    frame = read_archive_csv(payload)
    frame = _rename_agg_trade_columns(frame)

    out = pd.DataFrame(
        {
            "timestamp": to_utc(frame["transact_time"]),
            "symbol": pd.array([spec.symbol] * len(frame), dtype="string"),
            "price": frame["price"].astype("float64"),
            "quantity": frame["quantity"].astype("float64"),
            "is_buyer_maker": _to_bool(frame["is_buyer_maker"]),
            "trade_id": frame["agg_trade_id"].astype("int64"),
            "source": pd.array([spec.source_tag] * len(frame), dtype="string"),
        }
    )
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    TRADE_SCHEMA.validate(out)
    return out


def _rename_agg_trade_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise the column names Binance has used for aggTrades over the years."""
    aliases = {
        "agg_trade_id": "agg_trade_id",
        "aggregate_trade_id": "agg_trade_id",
        "a": "agg_trade_id",
        "transact_time": "transact_time",
        "transaction_time": "transact_time",
        "timestamp": "transact_time",
        "time": "transact_time",
        "t": "transact_time",
        "price": "price",
        "p": "price",
        "quantity": "quantity",
        "qty": "quantity",
        "q": "quantity",
        "is_buyer_maker": "is_buyer_maker",
        "m": "is_buyer_maker",
    }
    renamed = frame.rename(columns={c: aliases[c] for c in frame.columns if c in aliases})
    required = {"agg_trade_id", "transact_time", "price", "quantity", "is_buyer_maker"}
    missing = required - set(renamed.columns)
    if missing:
        raise BinanceArchiveError(f"aggTrades archive is missing column(s): {sorted(missing)}")
    return renamed


def parse_book_ticker(payload: bytes, spec: ArchiveSpec) -> pd.DataFrame:
    """Convert a ``bookTicker`` archive into a one-level book frame.

    ``transaction_time`` is used rather than ``event_time``: the former is when
    the book changed, the latter when Binance pushed the message. Features must
    be stamped with the moment information became true, not the moment it was
    delivered, or every latency-sensitive comparison is quietly shifted.
    """
    frame = read_archive_csv(payload)
    required = {
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "transaction_time",
    }
    missing = required - set(frame.columns)
    if missing:
        raise BinanceArchiveError(f"bookTicker archive is missing column(s): {sorted(missing)}")

    out = pd.DataFrame(
        {
            "timestamp": to_utc(frame["transaction_time"]),
            "symbol": pd.array([spec.symbol] * len(frame), dtype="string"),
            "sequence_id": frame["update_id"].astype("int64"),
            "source": pd.array([spec.source_tag] * len(frame), dtype="string"),
            "bid_price_0": frame["best_bid_price"].astype("float64"),
            "bid_size_0": frame["best_bid_qty"].astype("float64"),
            "ask_price_0": frame["best_ask_price"].astype("float64"),
            "ask_size_0": frame["best_ask_qty"].astype("float64"),
        }
    )
    out = out.sort_values(["timestamp", "sequence_id"], kind="stable").reset_index(drop=True)
    book_schema(1).validate(out)
    return out


def resample_book(book: pd.DataFrame, grid: str) -> pd.DataFrame:
    """Reduce a book frame to the last state observed in each grid interval.

    Why resample at all: the raw stream is ~13.5M updates a day for BTCUSDT, and
    a model on a seconds-to-minutes horizon cannot use that resolution. Taking
    the last update in each interval keeps the state that was true at the end of
    it, which is the state a decision made at that moment would have seen.

    Empty intervals are **dropped, not forward-filled**. Filling would repeat a
    ``sequence_id``, which is the exchange's own statement that an update
    happened — and the validator would then be unable to distinguish a real
    duplicate, which means a broken feed, from one manufactured here. Genuine
    pauses instead surface as time gaps, which the validator reports and any
    window-based feature must handle explicitly.

    This discards intra-interval updates. That is a modelling decision with
    consequences — it caps how fast a strategy can be assumed to react — and it
    is recorded in the manifest so a result is never read as though it were
    computed at full resolution.
    """
    if book.empty:
        return book
    reduced = (
        book.set_index("timestamp")
        .resample(grid)
        .last()
        .dropna(subset=["bid_price_0"])
        .reset_index()
    )
    reduced["sequence_id"] = reduced["sequence_id"].astype("int64")
    reduced["symbol"] = reduced["symbol"].astype("string")
    reduced["source"] = reduced["source"].astype("string")
    book_schema(1).validate(reduced)
    return reduced


PARSERS: Final = {"aggTrades": parse_agg_trades, "bookTicker": parse_book_ticker}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def iter_days(start: date, end: date) -> Iterator[date]:
    """Yield each day in ``[start, end]`` inclusive."""
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@dataclass
class DayResult:
    """Outcome of processing one day, for the run summary."""

    day: date
    rows: int
    bytes_downloaded: int
    output: Path | None
    skipped: str | None = None


ProgressCallback = "Callable[[DayResult], None]"


def download_range(
    spec: ArchiveSpec,
    start: date,
    end: date,
    output: Path | str,
    *,
    grid: str | None = None,
    keep_raw: bool = True,
    verify: bool = True,
    overwrite: bool = False,
    on_day: object = None,
) -> list[DayResult]:
    """Download a date range one day at a time, converting as it goes.

    Daily archives are used rather than monthly ones even for long ranges. A
    month of ``bookTicker`` is a single multi-gigabyte object: an interruption
    loses all of it, and nothing can be inspected until the whole thing lands.
    Per-day files make the job resumable, and resumability is what makes a
    16 GB download practical.

    Already-converted days are skipped unless ``overwrite``, so re-running after
    a failure costs only the days that are actually missing.
    """
    out_dir = Path(output) / spec.symbol / spec.kind
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(output) / "_archives" / spec.symbol / spec.kind
    if keep_raw:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[DayResult] = []
    for day in iter_days(start, end):
        target = out_dir / f"{day:%Y-%m-%d}.parquet"
        if target.exists() and not overwrite:
            result = DayResult(
                day=day, rows=0, bytes_downloaded=0, output=target, skipped="already converted"
            )
            results.append(result)
            _notify(on_day, result)
            continue

        try:
            payload, downloaded = _payload_for(
                spec, day, cache_dir, keep_raw=keep_raw, verify=verify
            )
        except BinanceArchiveError as exc:
            result = DayResult(day=day, rows=0, bytes_downloaded=0, output=None, skipped=str(exc))
            results.append(result)
            _notify(on_day, result)
            continue

        frame = PARSERS[spec.kind](payload, spec)
        del payload
        if spec.kind == "bookTicker" and grid:
            frame = resample_book(frame, grid)

        frame.to_parquet(target, compression="zstd", index=False)
        result = DayResult(day=day, rows=len(frame), bytes_downloaded=downloaded, output=target)
        results.append(result)
        _notify(on_day, result)
        del frame

        # Rewritten after every day, not once at the end. The manifest is what
        # makes a directory discoverable, so writing it last would leave an
        # interrupted download as parquet files the rest of the pipeline cannot
        # see — the exact situation resumability is meant to avoid. It is a few
        # hundred bytes against a day of parsing, so the cost is irrelevant.
        _write_manifest(spec, out_dir, results, grid=grid, start=start, end=end)

    _write_manifest(spec, out_dir, results, grid=grid, start=start, end=end)
    return results


def _notify(callback: object, result: DayResult) -> None:
    if callable(callback):
        callback(result)


def _payload_for(
    spec: ArchiveSpec,
    day: date,
    cache_dir: Path,
    *,
    keep_raw: bool,
    verify: bool,
) -> tuple[bytes, int]:
    """Return the raw archive for one day, using the cache when it is present."""
    cached = cache_dir / spec.filename(day, "daily")
    if keep_raw and cached.exists():
        return cached.read_bytes(), 0

    payload = fetch_archive(spec.url(day, "daily"), verify=verify)
    if keep_raw:
        cached.write_bytes(payload)
    return payload, len(payload)


def _write_manifest(
    spec: ArchiveSpec,
    out_dir: Path,
    results: list[DayResult],
    *,
    grid: str | None,
    start: date,
    end: date,
) -> None:
    """Record what was fetched and how it was transformed.

    The grid is the important entry. A resampled book is not the raw feed, and a
    result computed on it must never be read as though it were: the manifest is
    what keeps that distinction attached to the data rather than to someone's
    memory of how it was produced.
    """
    import json

    from lobml import __version__
    from lobml.data.schema import SCHEMA_VERSION

    converted = [r for r in results if r.output is not None and r.skipped is None]
    payload = {
        "source": spec.source_tag,
        "market": spec.market,
        "kind": spec.kind,
        "symbol": spec.symbol,
        "plane": "book" if spec.kind == "bookTicker" else "trades",
        "schema_version": SCHEMA_VERSION,
        "lobml_version": __version__,
        "requested_range": [start.isoformat(), end.isoformat()],
        "days_converted": len(converted),
        "days_missing": [r.day.isoformat() for r in results if r.output is None],
        "rows": sum(r.rows for r in converted),
        "resample_grid": grid,
        "depth": 1 if spec.kind == "bookTicker" else None,
        "attribution": "Binance public market data, https://data.binance.vision",
        "notes": (
            "bookTicker carries only the best bid and ask. Depth beyond the touch "
            "is not published and requires the collector."
            if spec.kind == "bookTicker"
            else "Aggregated trades: consecutive fills of one aggressing order at one price are one row."
        ),
    }
    if grid:
        payload["resample_note"] = (
            f"Book reduced to the last update in each {grid} interval. "
            f"Intra-interval updates are discarded; empty intervals are dropped, not filled."
        )

    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
