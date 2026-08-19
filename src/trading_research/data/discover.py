"""Finding the candidates, rather than being handed them.

The screen in :mod:`trading_research.data.screen` ranks whatever has been
downloaded. Deciding what to download was the remaining manual step, and it is
the one that decides what the screen can possibly find: a candidate nobody
thought of is a candidate the screen never sees. The instrument that came top
of the first screen was there because it occurred to someone to include it,
which is not a process.

This enumerates instead. Binance publishes every listed contract and its
traded volume on public endpoints — no credentials, the same class of source as
the historical archives.

Two filters, and both matter
----------------------------
**Listed before the period.** Volume figures come from today and the data being
screened does not. A contract listed last month has no February 2024 history,
and ranking it highly wastes a download and, worse, quietly shortens the
candidate list by displacing something that does.

**Enough volume to be worth the round trip.** Not because volume predicts
returns — it does not — but because a thin contract has a wide spread and a
fill nobody can rely on, and the cost model here assumes neither.

Ranking by volume is a proxy for liquidity and nothing more. It decides what
gets *measured*; the screen decides what is *good*, and the two answer different
questions. A high-volume instrument with no headroom is exactly what §19 found
at the top of the volume table.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

#: Public endpoints. No credentials, no rate limits that matter at this volume.
EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
TICKER_24H = "https://fapi.binance.com/fapi/v1/ticker/24hr"

USER_AGENT = "trading-research-pipeline (public data screen)"


@dataclass(frozen=True)
class Candidate:
    """One contract, and why it is or is not worth downloading."""

    symbol: str
    quote_volume_usd: float
    onboard_date: date
    tick_size: float
    status: str

    def as_row(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "quote_volume_usd": self.quote_volume_usd,
            "onboard_date": self.onboard_date,
            "tick_size": self.tick_size,
            "status": self.status,
        }


def _fetch(url: str, timeout: float = 30.0) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def list_perpetuals(*, listed_before: date | None = None) -> list[Candidate]:
    """Every USDT perpetual, with its volume and when it was listed.

    ``listed_before`` drops contracts that did not exist during the period
    being screened. Without it the top of the list fills with recent listings
    whose archives return nothing, and the candidate set silently shrinks.
    """
    info = _fetch(EXCHANGE_INFO)
    tickers = _fetch(TICKER_24H)
    assert isinstance(info, dict) and isinstance(tickers, list)

    volume = {t["symbol"]: float(t.get("quoteVolume", 0.0)) for t in tickers}
    out: list[Candidate] = []

    for entry in info.get("symbols", []):
        if (
            entry.get("contractType") != "PERPETUAL"
            or entry.get("quoteAsset") != "USDT"
            or entry.get("status") != "TRADING"
        ):
            continue
        onboard = datetime.fromtimestamp(entry.get("onboardDate", 0) / 1000, tz=UTC).date()
        if listed_before is not None and onboard >= listed_before:
            continue

        tick = 0.0
        for f in entry.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                tick = float(f.get("tickSize", 0.0))
        out.append(
            Candidate(
                symbol=entry["symbol"],
                quote_volume_usd=volume.get(entry["symbol"], 0.0),
                onboard_date=onboard,
                tick_size=tick,
                status=entry["status"],
            )
        )

    out.sort(key=lambda c: c.quote_volume_usd, reverse=True)
    return out


def candidate_table(
    *,
    listed_before: date | None = None,
    top: int | None = None,
    minimum_volume_usd: float = 0.0,
) -> pd.DataFrame:
    """The candidate list as a table, ranked by traded volume."""
    candidates = [
        c
        for c in list_perpetuals(listed_before=listed_before)
        if c.quote_volume_usd >= minimum_volume_usd
    ]
    if top is not None:
        candidates = candidates[:top]
    return pd.DataFrame([c.as_row() for c in candidates])
