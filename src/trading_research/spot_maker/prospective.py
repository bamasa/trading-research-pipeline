"""P, the prospective block: its deadline, its last day published, what may change since H, its fetch.

As registered under "Status rules" and "Data": P is read once, only for H's
candidates, after H's results are committed, 2026-12-31 is published on
Binance's archive and the fee amendment covers P's days, by its deadline (the
later of 2027-01-31 and 30 days after the committer date, UTC, of the commit
that first adds H's results file). Since H's read only ``README.md``, paths
under ``experiments/results/`` and paths under ``docs/`` outside
``docs/preregistration/`` may change, but for the fee amendment: the
configuration's P intervals (without them it is H's file) and a dated entry
appended to the document.

P's days are not on disk: the first process of P's read fetches them into this
checkout by the spot study's fetch code and terms (``spot/binance.fetch_day``,
``data/fetch_stage.run_plane``: the archive and its ``.CHECKSUM`` verified,
converted by the strict parser, a 404 a missing plane), each day asked of the
read's sealed access first.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from trading_research.data import fetch_log as fl
from trading_research.data.fetch_stage import run_plane
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import archives as ar
from trading_research.spot import binance
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import registration as rg

FETCH_LOG_PATH = Path("data/spot_maker_2026/fetch_P.json")
DEADLINE_FLOOR = date(2027, 1, 31)
DEADLINE_DAYS = 30


def deadline(h_results_committed: date) -> date:
    return max(DEADLINE_FLOOR, h_results_committed + timedelta(days=DEADLINE_DAYS))


def allowed_after_h(path: str) -> bool:
    if path == "README.md" or path.startswith("experiments/results/"):
        return True
    return path.startswith("docs/") and not path.startswith("docs/preregistration/")


def check_since_h(changed: Sequence[str], base_sha256: str, h_config_sha256: str,
                  document: str, h_document: str) -> None:  # fmt: skip
    """Only results, docs and the README changed since H's read, but for the fee
    amendment's intervals and its entry appended to the document."""
    fee = {rg.CONFIG_PATH.as_posix(), rg.DOCUMENT_PATH.as_posix()}
    outside = [p for p in changed if not allowed_after_h(p) and p not in fee]
    if outside:
        raise HeldOutLocked(f"{outside[:3]} changed since H's read; only results and docs may")
    if base_sha256 != h_config_sha256:
        raise HeldOutLocked(
            "the configuration without P's intervals is not the one H was read with"
        )
    if not document.startswith(h_document.rstrip("\n")):
        raise HeldOutLocked("the document changed since H's read other than by appending")


def published(day: date, opener: Any = None) -> bool:
    """Binance's archive answers 200 for the day's aggregated prints."""
    kwargs = {} if opener is None else {"opener": opener}
    return ar.head(ar.BINANCE_PRINTS.url(day), **kwargs).status == 200


def check_open(today: date, h_committed: date, last_published: Callable[[], bool]) -> None:
    if today > deadline(h_committed):
        raise HeldOutLocked(f"P's deadline {deadline(h_committed)} has passed: no P read opens")
    if not last_published():
        raise HeldOutLocked(
            f"P's last day {bk.BLOCKS['P'][1]} is not published on Binance's archive"
        )


def fetch_days(repo: Path, days: Sequence[date], access: Any, *, opener: Any = None,
               clock: Callable[[], datetime] = lambda: datetime.now(UTC),
               sleep: Callable[[float], None] | None = None) -> dict[str, Any]:  # fmt: skip
    """Fetch each of P's days inside ``access``; a record per day, as the copies'."""
    access.require(bk.INSTRUMENT, days, bk.PLANE, what="fetch")
    root = bk.data_root(repo)
    archives = root / "_archives_in_flight"
    archives.mkdir(parents=True, exist_ok=True)
    log = fl.FetchLog(Path(repo) / FETCH_LOG_PATH)
    fetch_one = functools.partial(binance.fetch_day, root=root, archive_dir=archives,
                                  access=access, opener=opener)  # fmt: skip
    if sleep is None:
        run_plane(ar.BINANCE_PRINTS.name, list(days), fetch_one, log, clock=clock)
    else:
        run_plane(ar.BINANCE_PRINTS.name, list(days), fetch_one, log, clock=clock, sleep=sleep)
    out: dict[str, Any] = {}
    for day in days:
        record = log.get(ar.BINANCE_PRINTS.name, day)
        on_disk = record.status == fl.ON_DISK and bk.day_file(repo, day).exists()
        out[day.isoformat()] = {"status": "fetched" if on_disk else "absent",
                                "fetch_status": record.status, "sha256": record.sha256,
                                "stamp": access.stamp}  # fmt: skip
    return out
