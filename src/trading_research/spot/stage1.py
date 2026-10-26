"""Stage 1: W and D fetched, replayed and stored, after the code-freeze note.

Registered (the procedure's step 1, "Data budget", "Blocks"):

* **The code freeze.** ``docs/spot_2026_code_freeze.md`` names, on a line
  ``Code-freeze commit: `<sha>```, a commit that HEAD descends from; no study
  code (``src/``, ``experiments/spot_2026.py``, the configuration,
  ``pyproject.toml``, ``uv.lock``) changed after it or is changed in the tree.
* **The fetch** of W and D on every plane under the development access, after
  the disk check, Bybit's book replayed at the walk depth stage 0 set.
* **The gap threshold** ``max_sequence_gaps``: the 95th percentile of D's
  per-day gap counts, rounded up, over every D day whose book replayed
  (the fetch log's counts, the counts the loader reads).
* **Exclusions** per day and venue, by reason, under that threshold.
* **The leakage checks** (K-leak) of every print member, touch proxy and book
  input on the first W day with a row interval holding prints at two distinct
  stamps past the longest declared lookback: truncation inside that interval and
  streaming parity at positions past that lookback, so each buffer drops history
  (``validation/leakage_timed``); the touch proxies are checked as features of the
  prints, and the grid's proxies must be theirs; a failure stops the study.
* **The fitted values**: the median touch size of Bybit's spot book over W's
  usable days (``data/feature_store.median_touch_size``).
* **The stores**: every D day usable on a venue, its inputs at every tenth row
  (float16; the book's raw sizes over the touch size) with the targets of every
  hold and label fee, ``reads_until_ns`` and whether the row is scored, one zstd
  parquet per day and store; a day already stored is not computed again.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import fetch_log as fl
from trading_research.data.feature_store import median_touch_size, to_stored
from trading_research.data.study_loader import max_sequence_gaps
from trading_research.features.timed import stamps_ns
from trading_research.spot import access as sa
from trading_research.spot import archives as ar
from trading_research.spot import code_freeze as cf
from trading_research.spot import days as dy
from trading_research.spot import registration as sr
from trading_research.spot import runlog
from trading_research.spot.book_inputs import book_targets
from trading_research.spot.code_freeze import CODE_FREEZE_LINE, CODE_FREEZE_NOTE, STUDY_CODE
from trading_research.spot.context import SpotStudy, git
from trading_research.spot.fits import store_path
from trading_research.spot.plan import HOLDS_S, LABEL_FEES_BP, STORES, TARGET_STRIDE
from trading_research.spot.print_label import print_targets
from trading_research.spot.stage0 import fetch

__all__ = ["CODE_FREEZE_LINE", "CODE_FREEZE_NOTE", "STUDY_CODE", "code_freeze", "frozen_under"]


class LeakError(AssertionError):
    """A feature failed its leakage check (K-leak): the study stops."""


def code_freeze(study: SpotStudy, expected: str | None = None) -> dict[str, str]:
    """The note's commit, refused unless no study code (the configuration included)
    changed after it and, with ``expected``, it is the commit stage 1 ran under."""
    from trading_research.spot.registration import amendments_text

    document = study.repo / sr.DOCUMENT_PATH
    amendments = amendments_text(document.read_text(encoding="utf-8")) if document.exists() else ""
    return {"commit": cf.check(study.repo, git, amendments, expected=expected)}


def frozen_under(study: SpotStudy) -> dict[str, str]:
    """D's and the amendment's check: the code freeze stage 1 recorded, unchanged."""
    return code_freeze(study, expected=str(study.state.record("1")["code_freeze"]["commit"]))


def gap_threshold(log: fl.FetchLog, d_days: list[date]) -> tuple[int, dict[str, int]]:
    """``max_sequence_gaps`` over every D day whose book replayed, and the counts."""
    counts = {}
    for day in d_days:
        record = log.records.get(log.key(ar.BYBIT_BOOK.name, day))
        if record is not None and record.status == fl.ON_DISK:
            counts[day.isoformat()] = int(record.stats.get("sequence_gaps", 0))
    if not counts:
        raise RuntimeError("no D day's book replayed: the gap threshold has no population")
    return max_sequence_gaps(list(counts.values())), counts


#: How far past the longest declared lookback the streaming positions start, and
#: how much longer than that the slice runs: the parity must drop history.
PARITY_MARGIN_NS = 1_000_000_000
SLICE_MARGIN_NS = 120_000_000_000


def cut_slice(
    rows: pd.DataFrame, prints: pd.DataFrame, start: int, *, min_span_ns: int = 0,
    min_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, float] | None:  # fmt: skip
    """The day's first row interval from ``start`` that holds prints at two distinct
    stamps strictly inside it (where ``validation/leakage_timed`` cuts), the rows up
    to a little after it, their prints, and the cut as a share of those rows. The
    interval lies at least ``min_span_ns`` and ``min_rows`` after the first row."""
    stamps, ps = stamps_ns(rows), stamps_ns(prints)
    if len(stamps) < 2 or not len(ps):
        return None
    lo = np.searchsorted(ps, stamps[:-1], side="right")
    hi = np.searchsorted(ps, stamps[1:], side="left")
    distinct = ps[np.clip(hi - 1, 0, len(ps) - 1)] > ps[np.clip(lo, 0, len(ps) - 1)]
    eligible = np.flatnonzero((hi > lo) & distinct) + 1
    eligible = eligible[eligible >= max(start, 100, min_rows + 1)]
    eligible = eligible[stamps[eligible] >= stamps[0] + min_span_ns]
    if not len(eligible):
        return None
    k = int(eligible[0])
    end = min(len(rows), k + 50)
    kept = rows.iloc[:end].reset_index(drop=True)
    return kept, prints[ps <= stamps[end - 1]], (k + 0.5) / end


def leakage(study: SpotStudy, inputs: dy.DayInputs, store: str) -> dict[str, Any] | None:
    """Truncation inside a row interval and streaming parity of every input of
    ``store`` on a W day (None when the day has no interval to cut inside)."""
    from trading_research.spot.book_inputs import book_registry
    from trading_research.spot.grid import print_grid

    name, columns = STORES[store]
    prints: pd.DataFrame | None
    if store == "book":
        assert inputs.book.frame is not None and inputs.bybit.frame is not None
        rows, prints, registry = (
            dy.ten_levels(inputs.book.frame),
            inputs.bybit.frame,
            book_registry(),
        )
    else:
        prints = inputs.prints(name).frame
        assert prints is not None
        rows, registry = print_grid(prints), dy.registry()
    assert prints is not None
    try:
        found = check_store(store, registry, list(columns), rows, prints,
                            study.scale.leakage_rows, study.scale.leakage_positions)  # fmt: skip
    except LeakError as exc:
        raise LeakError(f"{exc} (on {inputs.day})") from exc
    if found is None:
        return None
    return {"day": inputs.day.isoformat(), **found}


def check_store(
    store: str, registry: Any, columns: list[str], rows: pd.DataFrame, prints: pd.DataFrame,
    start: int, count: int,
) -> dict[str, Any] | None:  # fmt: skip
    """One store's K-leak on one day's rows and prints (None: no interval to cut in)."""
    from trading_research.features.timed import seconds_ns
    from trading_research.spot.grid import PROXY_COLUMNS, proxy_registry, touch_proxies
    from trading_research.validation.leakage_timed import LeakageError, assert_timed_honest

    lookback_ns = seconds_ns(registry.longest_lookback_s(list(columns)))
    lookback_rows = registry.longest_lookback_rows(list(columns))
    found = cut_slice(rows.reset_index(drop=True), prints, start,
                      min_span_ns=lookback_ns + SLICE_MARGIN_NS, min_rows=lookback_rows)  # fmt: skip
    if found is None:
        return None
    rows, prints, cut = found
    positions = parity_positions(rows, lookback_ns, lookback_rows, count)
    checks = [(registry, list(columns))]
    if store != "book":  # the touch proxies are row state of the print grid
        checks.append((proxy_registry(), list(PROXY_COLUMNS)))
        rebuilt = touch_proxies(prints, stamps_ns(rows))
        if not np.array_equal(rows[list(PROXY_COLUMNS)].to_numpy(), rebuilt.to_numpy(),
                              equal_nan=True):  # fmt: skip
            raise LeakError(f"K-leak: {store}: the grid's touch proxies are not the proxies "
                            "of the prints at the row stamps")  # fmt: skip
    try:
        for reg, names in checks:
            assert_timed_honest(reg, rows, prints, names=names, positions=positions, cut=cut)
    except LeakageError as exc:
        raise LeakError(f"K-leak: {store}: {exc}") from exc
    return {"rows": len(rows), "features": len(columns), "proxies": len(checks) > 1, "cut": cut,
            "positions": len(positions), "lookback_s": lookback_ns / 1e9, "passed": True}  # fmt: skip


def parity_positions(
    rows: pd.DataFrame, lookback_ns: int, lookback_rows: int, count: int
) -> list[int]:
    """Streaming positions far enough into the slice that each declared buffer drops
    history: past the longest lookback in time and in rows from the first row."""
    stamps = stamps_ns(rows)
    first = int(np.searchsorted(stamps, stamps[0] + lookback_ns + PARITY_MARGIN_NS, side="left"))
    first = min(max(first, lookback_rows + 1), len(rows) - 1)
    return sorted(set(np.linspace(first, len(rows) - 1, count).astype(int).tolist()))


def write_store(
    study: SpotStudy, store: str, inputs: dy.DayInputs, touch: float, depth: int
) -> dict[str, Any]:
    """One day of one store: inputs and targets at every tenth row, and the scored flag."""
    path = store_path(study, store, inputs.day)
    if path.exists():
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        return {"rows": int(pq.ParquetFile(path).metadata.num_rows), "bytes": path.stat().st_size}
    name, _ = STORES[store]
    frame = dy.store_frame(store, inputs)
    stamps = stamps_ns(frame)
    at = np.arange(0, len(frame), TARGET_STRIDE, dtype=np.int64)
    if store == "book":
        assert inputs.book.frame is not None
        targets = book_targets(inputs.book.frame, HOLDS_S, LABEL_FEES_BP, depth=depth, at=at)
    else:
        targets = print_targets(dy.tape(inputs, name), stamps, HOLDS_S, LABEL_FEES_BP, at=at)
    scale = dy.layout(store, touch if store == "book" else None)
    stored = to_stored(frame.iloc[at].reset_index(drop=True), scale)
    stored["scored"] = stamps[at] >= inputs.scored_from(name)
    out = pd.concat([stored, targets.drop(columns=["timestamp"]).reset_index(drop=True)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    out.to_parquet(partial, compression="zstd", index=False)
    partial.replace(path)
    return {"rows": len(out), "bytes": path.stat().st_size}


def spec_of(record0: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """Stage 0's tick and lot, keyed by instrument id."""
    ids = {i.key: i.id for i in sa.INSTRUMENTS.values()}
    return {
        ids[k]: (float(v["tick"]), float(v["lot"])) for k, v in record0["instrument_spec"].items()
    }


def run(study: SpotStudy) -> dict[str, Any]:
    """Stage 1's record (module docstring)."""
    from trading_research.spot.stage0 import _disk, timed

    freeze = code_freeze(study)
    record0 = study.state.record("0")
    walk = int(record0["walk"]["walk"])
    w_days, d_days = list(study.scale.days("W")), list(study.scale.days("D"))
    access = sa.development_access()
    with study.serving():
        fetch_s, log = timed(
            lambda: fetch(study, 1, tuple(sorted({*w_days, *d_days})), access, walk)
        )
    threshold, counts = gap_threshold(log, d_days)
    spec = spec_of(record0)

    def load(day: date) -> dy.DayInputs:
        return dy.load_day(day, study.spot_root, access=access, logs=[log], table=study.fee_table,
                           max_sequence_gaps=threshold, depth=walk, spec=spec)  # fmt: skip

    # Logged before any D day is loaded, so a stage that stops still has its row.
    runlog.append(study.repo, run=f"D{study.run}", stage="1", days=d_days,
                  note="stage 1: features and targets of D's days")  # fmt: skip
    days: dict[str, Any] = {}
    stores: dict[str, dict[str, Any]] = {s: {} for s in STORES}
    w_books: list[pd.DataFrame] = []
    leak: dict[str, Any] = {}
    touch: float | None = None
    for inputs in dy.walk_days(sorted({*w_days, *d_days}), load):
        days[inputs.day.isoformat()] = {name: {"reasons": list(inputs.reasons[name]),
                                               "warm": inputs.warm[name]} for name in dy.VENUE_PLANES}  # fmt: skip
        if sa.block_of(inputs.day) == "W":
            for store, (name, _) in STORES.items():
                if store not in leak and inputs.usable(name):
                    found = leakage(study, inputs, store)
                    if found is not None:
                        leak[store] = found
            if inputs.usable(sa.BYBIT) and inputs.book.frame is not None:
                w_books.append(inputs.book.frame[["bid_size_0", "ask_size_0"]].copy())
            continue
        if touch is None:
            if not w_books:
                raise RuntimeError("W has no usable Bybit day: no touch size to scale the book by")
            touch = median_touch_size(w_books)
            w_books = []
        for store, (name, _) in STORES.items():
            if inputs.usable(name):
                stores[store][inputs.day.isoformat()] = write_store(
                    study, store, inputs, touch, walk
                )
    missing = sorted(set(STORES) - set(leak))
    if missing:
        raise LeakError(f"no usable W day has a row interval to cut inside for {missing}: "
                        "the leakage checks could not run")  # fmt: skip
    if touch is None:
        raise RuntimeError("D has no day: nothing to store")
    excluded = {sa.instrument(n).key: [[d, r] for d, v in days.items() for r in v[n]["reasons"]]
                for n in dy.VENUE_PLANES}  # fmt: skip
    return {"code_freeze": freeze, "walk": walk, "fetch_s": fetch_s,
            "fetch": {p: dict(log.counts(p)) for p in ar.PLANES},
            "max_sequence_gaps": threshold, "gap_counts_D": counts, "days": days,
            "excluded": excluded, "touch_size": touch, "leakage": leak,
            "stores": {s: {"days": len(v), "rows": sum(x["rows"] for x in v.values()),
                           "bytes": sum(x["bytes"] for x in v.values())} for s, v in stores.items()},
            "instrument_spec": record0["instrument_spec"], "disk": _disk(study)}  # fmt: skip
