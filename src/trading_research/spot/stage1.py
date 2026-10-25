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
  input on a slice of W's first usable day: truncation inside a row interval and
  streaming parity (``validation/leakage_timed``); a failure stops the study.
* **The fitted values**: the median touch size of Bybit's spot book over W's
  usable days (``data/feature_store.median_touch_size``).
* **The stores**: every D day usable on a venue, its inputs at every tenth row
  (float16; the book's raw sizes over the touch size) with the targets of every
  hold and label fee, ``reads_until_ns`` and whether the row is scored, one zstd
  parquet per day and store; a day already stored is not computed again.
"""

from __future__ import annotations

import re
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
from trading_research.spot import days as dy
from trading_research.spot import runlog
from trading_research.spot.book_inputs import book_targets
from trading_research.spot.context import SpotStudy, git
from trading_research.spot.fits import store_path
from trading_research.spot.plan import HOLDS_S, LABEL_FEES_BP, STORES, TARGET_STRIDE
from trading_research.spot.print_label import print_targets
from trading_research.spot.stage0 import fetch

CODE_FREEZE_NOTE = "docs/spot_2026_code_freeze.md"
CODE_FREEZE_LINE = re.compile(r"Code-freeze commit: `([0-9a-f]{7,40})`")
STUDY_CODE = ("src/", "experiments/spot_2026.py", "configs/spot_2026.yaml", "pyproject.toml",
              "uv.lock")  # fmt: skip


class LeakError(AssertionError):
    """A feature failed its leakage check (K-leak): the study stops."""


def code_freeze(study: SpotStudy) -> dict[str, str]:
    """The note's commit, refused unless no study code changed after it."""
    path = study.repo / CODE_FREEZE_NOTE
    if not path.exists():
        raise RuntimeError(f"no {CODE_FREEZE_NOTE}: the code freeze is recorded, with a line "
                           "'Code-freeze commit: `<sha>`', before the stage 1 fetch")  # fmt: skip
    git(study.repo, "ls-files", "--error-unmatch", CODE_FREEZE_NOTE)
    found = CODE_FREEZE_LINE.findall(path.read_text(encoding="utf-8"))
    if not found:
        raise RuntimeError(f"{CODE_FREEZE_NOTE} names no 'Code-freeze commit: `<sha>`'")
    commit = git(study.repo, "rev-parse", f"{found[-1]}^{{commit}}").strip()
    git(study.repo, "merge-base", "--is-ancestor", commit, "HEAD")
    changed = git(study.repo, "diff", "--name-only", f"{commit}..HEAD", "--", *STUDY_CODE).split()
    dirty = git(study.repo, "status", "--porcelain", "--", *STUDY_CODE).splitlines()
    if changed or dirty:
        raise RuntimeError(f"study code changed after the code-freeze commit {commit[:7]}: "
                           f"{(changed + [d[3:] for d in dirty])[:5]}; a change is a correction "
                           "under the amendment protocol")  # fmt: skip
    return {"commit": commit}


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


def leakage(study: SpotStudy, inputs: dy.DayInputs) -> dict[str, Any]:
    """Truncation and streaming parity of every input on a slice of a W day."""
    from trading_research.spot.book_inputs import book_registry
    from trading_research.spot.grid import print_grid
    from trading_research.validation.leakage_timed import LeakageError, assert_timed_honest

    n, k = study.scale.leakage_rows, study.scale.leakage_positions
    out: dict[str, Any] = {}
    for store, (name, columns) in STORES.items():
        if store == "book":
            assert inputs.book.frame is not None and inputs.bybit.frame is not None
            rows = dy.ten_levels(inputs.book.frame).iloc[:n].reset_index(drop=True)
            prints = inputs.bybit.frame[stamps_ns(inputs.bybit.frame) <= stamps_ns(rows)[-1]]
            registry = book_registry()
        else:
            prints = inputs.prints(name).frame
            assert prints is not None
            rows = print_grid(prints).iloc[:n].reset_index(drop=True)
            prints = prints[stamps_ns(prints) <= stamps_ns(rows)[-1]]
            registry = dy.registry()
        positions = np.linspace(len(rows) // 2, len(rows) - 1, k).astype(int).tolist()
        try:
            assert_timed_honest(registry, rows, prints, names=list(columns), positions=positions)
        except LeakageError as exc:
            raise LeakError(f"K-leak: {store} on {inputs.day}: {exc}") from exc
        out[store] = {"day": inputs.day.isoformat(), "rows": len(rows), "features": len(columns),
                      "positions": len(positions), "passed": True}  # fmt: skip
    return out


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


def spec_of(record0: dict[str, Any]) -> dict[str, tuple[float, float]]:
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

    days: dict[str, Any] = {}
    stores: dict[str, dict[str, Any]] = {s: {} for s in STORES}
    w_books: list[pd.DataFrame] = []
    leak: dict[str, Any] | None = None
    touch: float | None = None
    for inputs in dy.walk_days(sorted({*w_days, *d_days}), load):
        days[inputs.day.isoformat()] = {name: {"reasons": list(inputs.reasons[name]),
                                               "warm": inputs.warm[name]} for name in dy.VENUE_PLANES}  # fmt: skip
        if sa.block_of(inputs.day) == "W":
            if leak is None and inputs.usable(sa.BINANCE) and inputs.usable(sa.BYBIT):
                leak = leakage(study, inputs)
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
    if leak is None:
        raise LeakError("no W day is usable on both venues: the leakage checks could not run")
    if touch is None:
        raise RuntimeError("D has no day: nothing to store")
    usable_d = [
        d for d in d_days if any(not days[d.isoformat()][n]["reasons"] for n in dy.VENUE_PLANES)
    ]
    runlog.append(study.repo, run=f"D{study.run}", stage="1", days=usable_d,
                  note="stage 1: features and targets of D's days")  # fmt: skip
    excluded = {sa.instrument(n).key: [[d, r] for d, v in days.items() for r in v[n]["reasons"]]
                for n in dy.VENUE_PLANES}  # fmt: skip
    return {"code_freeze": freeze, "walk": walk, "fetch_s": fetch_s,
            "fetch": {p: dict(log.counts(p)) for p in ar.PLANES},
            "max_sequence_gaps": threshold, "gap_counts_D": counts, "days": days,
            "excluded": excluded, "touch_size": touch, "leakage": leak,
            "stores": {s: {"days": len(v), "rows": sum(x["rows"] for x in v.values()),
                           "bytes": sum(x["bytes"] for x in v.values())} for s, v in stores.items()},
            "instrument_spec": record0["instrument_spec"], "disk": _disk(study)}  # fmt: skip
