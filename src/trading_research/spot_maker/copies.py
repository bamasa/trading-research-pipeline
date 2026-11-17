"""The study's data: the spot study's converted day files copied in, and a block's days loaded.

**Copies, not fetches** (the registration, "Data"). :func:`copy_days` asks the
access for each day before it touches a file, reads the spot study's converted
file read-only, writes it to this checkout's ``data/spot/binance/prints/BTCFDUSD/``
with the same sha256 (a copy already there must have it), and records each
day in ``data/spot_maker_2026/copies.json``. A day without a file at the source
is recorded ``absent`` and the loader will exclude it (``no_prints_file``).

**Loading** (:func:`walk`): each day through ``spot/loader.load_prints_day``
inside the access, its reasons the loader's and the fee table's (``fee not
sourced``); a day is **warm** when the previous calendar day is usable and in
the same block, or W before D (``spot/days.follows``): its features read that
day's prints and its touch proxies carry that day's last prints; any other day
is cold, its first 61 minutes not scored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.market_making.prereg import HeldOutLocked, sha256_bytes
from trading_research.spot import days as sd
from trading_research.spot import loader as ld
from trading_research.spot.archives import BINANCE_PRINTS
from trading_research.spot.costs import FeeTable
from trading_research.spot.fills import PrintTape
from trading_research.spot.state import utc_now, write_json
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.touch import last_prices

COLD_MINUTES = 61


def copy_log(repo: Path) -> dict[str, Any]:
    path = Path(repo) / rg.COPY_LOG_PATH
    loaded: dict[str, Any] = json.loads(path.read_text("utf-8")) if path.exists() else {"files": {}}
    return loaded


def copy_days(source: Path, repo: Path, days: Iterable[date], access: Any) -> dict[str, Any]:
    """Copy each day's file inside ``access``; the records of these days."""
    if Path(source).resolve() == Path(repo).resolve():
        raise HeldOutLocked("the source of the copies is this checkout itself")
    log, out = copy_log(repo), {}
    for day in days:
        access.require(bk.INSTRUMENT, [day], bk.PLANE, what="copy")
        src, dst = bk.day_file(source, day), bk.day_file(repo, day)
        if not src.exists():
            out[day.isoformat()] = {"status": "absent", "at": utc_now(), "stamp": access.stamp}
            continue
        data = src.read_bytes()
        digest = sha256_bytes(data)
        if dst.exists():
            if sha256_bytes(dst.read_bytes()) != digest:
                raise HeldOutLocked(f"{dst} differs from its source")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            partial = dst.with_name(dst.name + ".part")
            partial.write_bytes(data)
            partial.replace(dst)
        out[day.isoformat()] = {"status": "copied", "sha256": digest, "bytes": len(data),
                                "at": utc_now(), "stamp": access.stamp}  # fmt: skip
    log["files"].update(out)
    write_json(Path(repo) / rg.COPY_LOG_PATH, log)
    return out


@dataclass
class DayInput:
    """One day as loaded, with its reasons and warm state."""

    day: date
    loaded: ld.PrintsDay
    reasons: tuple[str, ...]
    warm: bool = False
    history: pd.DataFrame | None = None
    carried: tuple[float, float] | None = None

    @property
    def usable(self) -> bool:
        return not self.reasons

    @property
    def frame(self) -> pd.DataFrame:
        if self.loaded.frame is None or not self.usable:
            raise ValueError(f"{self.day} is excluded: {self.reasons}")
        return self.loaded.frame

    def tape(self) -> PrintTape:
        return PrintTape.from_frame(self.frame, self.day)

    @property
    def scored_from(self) -> int:
        start = int(pd.Timestamp(self.day, tz="UTC").value)
        return start if self.warm else start + COLD_MINUTES * 60 * rg.NS_PER_S


def load_day(repo: Path, day: date, access: Any, table: FeeTable) -> DayInput:
    loaded = ld.load_prints_day(BINANCE_PRINTS, day, bk.data_root(repo), access=access,
                                spec_tick=rg.TICK)  # fmt: skip
    reasons = set(loaded.exclusions)
    fee = table.exclusion(bk.INSTRUMENT, day)
    if fee is not None:
        reasons.add(fee)
    return DayInput(day, loaded, tuple(sorted(reasons)))


def walk(repo: Path, days: Iterable[date], access: Any, table: FeeTable) -> Iterator[DayInput]:
    """Each day in order with its warm state; a previous day not listed is loaded for it."""
    previous: DayInput | None = None
    for day in days:
        current = load_day(repo, day, access, table)
        before = day - timedelta(days=1)
        if sd.follows(day, before) and (previous is None or previous.day != before):
            previous = load_day(repo, before, access, table)
        if (
            previous is not None
            and previous.day == before
            and previous.usable
            and sd.follows(day, before)
        ):
            current.warm = True
            current.history = previous.frame
            current.carried = last_prices(previous.tape())
        yield current
        previous = current
