"""The carry search's choice, written down before each month and scored after it.

``trp search carry`` chose its rule on 2021-2023 and read it once on
2023-2025; both are the past. This journal is the rule on the future: at the
start of each month it writes which coin the rule holds and when it entered,
from the funding of the previous three months alone, and a commit timestamps
that before the month happens. After the month ends it scores the month: the
carry's return after costs, and how the coin it held did against every coin
of the universe over the same hours, which says whether choosing by funding
added anything over holding any carry.

The rule is the search's choice, unchanged: one coin, the highest funding a
day over three months, no floor (``top1-lb3-floor_none``), on the search's
seventeen coins and Binance Regular's costs. The registration is
``docs/preregistration/carry_live_2026_11.md``.

A month's pick is ``YYYY-MM.pick.json`` and its score ``YYYY-MM.score.json``;
neither is rewritten once written. :func:`step` does whatever is due: it
scores every picked month that is over and whose data Binance has published,
then picks the current month if it has no pick yet. Funding archives appear a
few hours into the next month, so a pick is made then, and the coin enters at
the next full hour after the pick; until then the previous coin is held.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.strategies.carry import (
    UNIVERSE,
    Market,
    Rule,
    _price,
    choose,
    cost,
    hourly_pnl,
    load,
    scores,
)

RULE = Rule(slots=1, lookback=3, threshold_bp_a_day=0.0)
#: A coin is a candidate only with at least this many funding settlements a day in each month read.
MIN_SETTLEMENTS_A_DAY = 2.7
#: Picking waits until this many of the seventeen coins have the last month's funding.
MIN_COINS = 15
#: The first month in the journal; it is entered mid-month, when the journal began.
FIRST = pd.Period("2026-11", "M")
HOUR = timedelta(hours=1)

Loader = Callable[[str, Sequence[pd.Period]], Market]


def _readable(_coin: str, _month: pd.Period) -> bool:
    return True


def _iso(t: pd.Timestamp | datetime) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _start(month: pd.Period) -> pd.Timestamp:
    return pd.Timestamp(month.start_time).tz_localize("UTC")


def _last_hour(month: pd.Period) -> pd.Timestamp:
    return pd.Timestamp(month.end_time).tz_localize("UTC").floor("1h")


def funded(mk: Market, month: pd.Period) -> bool:
    """Whether a coin's funding of ``month`` is all there."""
    a, b = _start(month), _last_hour(month) + HOUR
    n = int(((mk.funding.index >= a) & (mk.funding.index < b)).sum())
    return n >= MIN_SETTLEMENTS_A_DAY * month.days_in_month


def priced(mk: Market, month: pd.Period) -> bool:
    """Whether a coin's spot and perpetual closes reach ``month``'s last hour."""
    last = _last_hour(month)
    return bool(len(mk.spot) and len(mk.perp) and mk.spot.index[-1] >= last
                and mk.perp.index[-1] >= last)  # fmt: skip


def source_sha256() -> str:
    """The sha256 of the journal's and the carry engine's sources, written into every pick."""
    from trading_research.strategies import carry

    h = hashlib.sha256()
    for path in (Path(__file__), Path(carry.__file__)):
        h.update(path.read_bytes())
    return h.hexdigest()


def pick(markets: dict[str, Market], month: pd.Period, now: pd.Timestamp) -> dict[str, Any] | None:
    """The rule's coin for ``month``, or None while the funding it reads is not all published."""
    window = [month - k for k in range(1, RULE.lookback + 1)]
    ready = {c: mk for c, mk in markets.items() if all(funded(mk, m) for m in window)}
    if len(ready) < MIN_COINS:
        return None
    chosen = choose(ready, month, RULE, _readable)
    s = scores(ready, month, RULE, _readable)
    enters = max(now.floor("1h") + HOUR, _start(month))
    return {
        "month": str(month),
        "rule": RULE.name,
        "coin": chosen[0] if chosen else None,
        "made_at": _iso(now),
        "enters_at": _iso(enters),
        "funding_months": [str(m) for m in sorted(window)],
        "funding_bp_a_day": {c: round(v, 4) for c, v in sorted(s.items(), key=lambda kv: -kv[1])},
        "coins_read": sorted(ready),
        "source_sha256": source_sha256(),
    }


@dataclass
class Held:
    coin: str | None = None
    quantity: float = 0.0
    equity: float = 1.0


def carry_bp(mk: Market, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """One unit of notional a leg held as the carry from ``start`` to ``end``, gross, in bp."""
    q = 1.0 / _price(mk, start)
    return float(hourly_pnl(mk, q, start, end).sum()) * 1e4


def score(
    markets: dict[str, Market], chosen: dict[str, Any], held: Held
) -> tuple[dict[str, Any], Held]:
    """``chosen``'s month after it ended: the account's return and the coin against the universe."""
    month = pd.Period(chosen["month"], "M")
    start, end = _start(month), _last_hour(month)
    enters = pd.Timestamp(chosen["enters_at"])
    coin = chosen["coin"]
    equity, pnl = held.equity, 0.0
    quantity = held.quantity
    if coin != held.coin:
        if held.coin is not None:
            old = markets[held.coin]
            if enters > start:
                pnl += float(hourly_pnl(old, quantity, start, enters - HOUR).sum())
            pnl -= cost(quantity * _price(old, enters))
            quantity = 0.0
        if coin is not None:
            mk = markets[coin]
            quantity = (equity + pnl) / RULE.slots / 2 / _price(mk, enters)
            pnl -= cost(quantity * _price(mk, enters))
            pnl += float(hourly_pnl(mk, quantity, enters, end).sum())
    elif coin is not None:
        pnl += float(hourly_pnl(markets[coin], quantity, start, end).sum())
    universe = {c: round(carry_bp(mk, enters, end), 3) for c, mk in markets.items()
                if priced(mk, month) and _price(mk, enters) == _price(mk, enters)}  # fmt: skip
    average = float(np.mean(list(universe.values())))
    out: dict[str, Any] = {
        "month": chosen["month"],
        "coin": coin,
        "from": chosen["enters_at"],
        "to": _iso(end),
        "return_pct": pnl / equity * 100,
        "equity_after": equity + pnl,
        "coin_carry_bp": universe.get(coin) if coin else 0.0,
        "all_coins_carry_bp": average,
        "coins_compared": len(universe),
        "rank_among_coins": (sum(v > universe[coin] for v in universe.values()) + 1)
        if coin
        else None,
        "source_sha256": source_sha256(),
    }
    out["beats_all_coins"] = coin is not None and out["coin_carry_bp"] > average
    return out, Held(coin, quantity, equity + pnl)


def _read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _write_once(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists: the journal does not rewrite an entry")
    path.write_text(json.dumps(data, indent=2) + "\n")


def state(journal: Path) -> Held:
    """The account after the last scored month."""
    held = Held()
    for path in sorted(journal.glob("*.score.json")):
        s = _read(path)
        held = Held(s["coin"], float(s["held_quantity"]), float(s["equity_after"]))
    return held


def step(journal: Path, loader: Loader, now: datetime | pd.Timestamp | None = None) -> list[str]:
    """Score what is over, then pick the current month. Returns what it did, one line each."""
    journal.mkdir(parents=True, exist_ok=True)
    at = pd.Timestamp(now if now is not None else datetime.now(UTC))
    at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
    current = pd.Period(at.tz_localize(None), "M")
    did: list[str] = []
    for path in sorted(journal.glob("*.pick.json")):
        month = pd.Period(path.name[:7], "M")
        target = journal / f"{month}.score.json"
        if target.exists() or month >= current:
            continue
        chosen = _read(path)
        held = state(journal)
        coins = {c for c in (chosen["coin"], held.coin) if c} | set(UNIVERSE)
        markets = {c: loader(c, [month]) for c in sorted(coins)}
        needed = [c for c in (chosen["coin"], held.coin) if c]
        if not all(priced(markets[c], month) and funded(markets[c], month) for c in needed):
            did.append(f"{month}: over, but Binance has not published all of it yet; not scored")
            break
        result, after = score(markets, chosen, held)
        result["held_quantity"] = after.quantity
        _write_once(target, result)
        did.append(f"{month}: scored, {result['coin']} {result['return_pct']:+.3f}%, "
                   f"rank {result['rank_among_coins']} of {result['coins_compared']}")  # fmt: skip
    target = journal / f"{current}.pick.json"
    if current >= FIRST and not target.exists():
        window = [current - k for k in range(1, RULE.lookback + 1)]
        markets = {c: loader(c, sorted(window)) for c in UNIVERSE}
        made = pick(markets, current, at)
        if made is None:
            did.append(f"{current}: the last month's funding is not published yet; no pick")
        else:
            _write_once(target, made)
            did.append(f"{current}: {made['coin']} from {made['enters_at']}")
    (journal / "README.md").write_text(page(journal), encoding="utf-8")
    return did


def page(journal: Path) -> str:
    """The journal as a table: each month's pick and, once it is over, its score."""
    rows = []
    for path in sorted(journal.glob("*.pick.json")):
        p = _read(path)
        s_path = journal / f"{p['month']}.score.json"
        if s_path.exists():
            s = _read(s_path)
            rows.append(f"| {p['month']} | {p['coin']} | {p['made_at']} | {s['return_pct']:+.3f}% | "
                        f"{(s['equity_after'] - 1) * 100:+.2f}% | {s['coin_carry_bp']:+.1f} | "
                        f"{s['all_coins_carry_bp']:+.1f} | {s['rank_among_coins']} of "
                        f"{s['coins_compared']} |")  # fmt: skip
        else:
            rows.append(f"| {p['month']} | {p['coin']} | {p['made_at']} | not over | | | | |")
    return "\n".join([
        "# The carry journal",
        "",
        "Written by `trp search carry-live`. Each month's coin is chosen from the funding of the",
        "previous three months and committed before the month is over; its score is added after.",
        "Return after Binance Regular's costs on the whole account; carry in bp of one leg's",
        "notional, gross, from the entry hour to the month's end, for the coin held and for the",
        "average of all seventeen coins over the same hours. Registered in",
        "`docs/preregistration/carry_live_2026_11.md` on the main branch.",
        "",
        "| month | coin | chosen at (UTC) | return | account | coin's carry, bp | all coins, bp | rank |",
        "|---|---|---|--:|--:|--:|--:|--:|",
        *rows,
        "",
    ])  # fmt: skip


def binance_loader(root: Path) -> Loader:
    """Months of a coin's spot and perpetual hourly closes and funding, from Binance's archives."""

    def read(coin: str, months: Sequence[pd.Period]) -> Market:
        try:
            return load(root, coin, months)
        except ValueError:  # nothing published for these months: an empty market
            empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
            return Market(empty, empty, empty)

    return read


__all__ = ["FIRST", "RULE", "Held", "binance_loader", "page", "pick", "score", "state", "step"]
