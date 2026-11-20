"""Binance USD-M monthly archives of daily klines and funding, for many symbols at once.

``data.binance.vision`` publishes, per symbol and month, one zip of daily
klines and one of funding settlements; a delisted symbol keeps its archives,
so listing the bucket gives the universe without survivorship. Each month is
written once as ``<root>/<kind>/<SYMBOL>/<YYYY-MM>.parquet``; a month the
archive does not have (before listing, after delisting) is recorded as an
empty ``.missing`` file so it is not asked for again.
"""

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision/data/futures/um/monthly"
#: Tries of a request that fails other than by a 404, with 1, 2, 4 ... s between.
ATTEMPTS = 5
KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore")  # fmt: skip


def _get(address: str, timeout: float) -> bytes:
    """GET with :data:`ATTEMPTS` tries on network errors; an HTTP error other than these raises."""
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(address, timeout=timeout) as r:
                return bytes(r.read())
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError):
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def list_symbols(timeout: float = 30.0) -> list[str]:
    """Every symbol with a monthly kline archive, delisted ones included."""
    prefix = "data/futures/um/monthly/klines/"
    out: list[str] = []
    marker = ""
    while True:
        url = f"{BUCKET}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
        body = _get(url, timeout).decode()
        out += re.findall(rf"<Prefix>{prefix}([^/]+)/</Prefix>", body)
        if "<IsTruncated>true" not in body:
            return out
        marker = re.findall(r"<NextMarker>([^<]+)", body)[0]


def listed_months(symbol: str, timeout: float = 30.0) -> list[str]:
    """The months the archive has daily klines for, as ``YYYY-MM``."""
    prefix = f"data/futures/um/monthly/klines/{symbol}/1d/"
    body = _get(f"{BUCKET}?prefix={prefix}", timeout).decode()
    return sorted(set(re.findall(rf"{symbol}-1d-(\d{{4}}-\d{{2}})\.zip<", body)))


def url(kind: str, symbol: str, month: str) -> str:
    if kind == "klines":
        return f"{FILES}/klines/{symbol}/1d/{symbol}-1d-{month}.zip"
    return f"{FILES}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def parse(kind: str, raw: bytes) -> pd.DataFrame:
    """A month's zip as a frame: klines (``open_time``, ``open``, ``close``, ``quote_volume``)
    or funding (``time``, ``rate``), times in UTC."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode()
    first = text.split("\n", 1)[0]
    header = 0 if first[:1].isalpha() else None
    frame = pd.read_csv(io.StringIO(text), header=header)
    if kind == "klines":
        if header is None:
            frame.columns = list(KLINE_COLUMNS)[: frame.shape[1]]
        return pd.DataFrame({
            "open_time": pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True),
            "open": frame["open"].astype(float), "close": frame["close"].astype(float),
            "quote_volume": frame["quote_volume"].astype(float),
        })  # fmt: skip
    if header is None:
        frame.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][
            : frame.shape[1]
        ]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(frame["calc_time"].astype("int64"), unit="ms", utc=True),
            "rate": frame["last_funding_rate"].astype(float),
        }
    )


def path(root: Path, kind: str, symbol: str, month: str) -> Path:
    return root / kind / symbol / f"{month}.parquet"


def fetch_month(root: Path, kind: str, symbol: str, month: str, timeout: float = 30.0) -> str:
    """Fetch one month if not on disk; returns 'have', 'got' or 'missing'."""
    target = path(root, kind, symbol, month)
    missing = target.with_suffix(".missing")
    if target.exists() or missing.exists():
        return "have"
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url(kind, symbol, month), timeout=timeout) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.touch()
                return "missing"
            if attempt == ATTEMPTS - 1:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == ATTEMPTS - 1:
                raise
        time.sleep(2**attempt)
    parse(kind, raw).to_parquet(target, index=False)
    return "got"


def download(
    root: Path, symbols: Iterable[str], months: Iterable[str], kinds: Iterable[str] = ("klines", "fundingRate"),
    workers: int = 8,
) -> dict[str, int]:  # fmt: skip
    jobs = [(k, s, m) for s in symbols for m in months for k in kinds]
    counts: dict[str, int] = {"have": 0, "got": 0, "missing": 0}
    with ThreadPoolExecutor(workers) as pool:
        for status in pool.map(lambda j: fetch_month(root, *j), jobs):
            counts[status] += 1
    return counts


def load(root: Path, kind: str, symbol: str, months: Iterable[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for m in months if (p := path(root, kind, symbol, m)).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
