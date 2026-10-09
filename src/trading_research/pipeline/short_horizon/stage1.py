"""Stage 1: fetch, replay and features for W and D, the leakage tests and selection on January.

In order, under the development access (W, D, the reader-test days; nothing
of E, H, F or P):

1. **Gates.** The YAML validated against the registration, the catalogue's
   hash, any cut stage 0 needed registered by an amendment, and the code-freeze
   note (``docs/short_horizon_2025_code_freeze.md``) naming a commit after which
   no study code has changed.
2. **The fetch** of W's and D's books, prints and funding, the three ``ob200``
   reader-test days' books, and the hourly bars from 2024-12-01 to 2025-06-30,
   through :func:`trading_research.data.fetch_stage.run_stage`, which checks the
   disk first, retries every failure but a 404, and returns only when each day
   is on disk or terminally excluded. The fetch log is kept, so a stopped fetch
   resumes.
3. **The instrument's** tick and lot from the venue's specification.
4. **Every day validated** (:func:`validate`): fetch exclusions, the loader's
   (non-positive or off-grid prices), the simulator's day loader's refusals,
   a missing funding history, and the sequence-gap threshold, the 95th
   percentile of D's per-day gap counts, rounded up, with the days above it.
   The reader-test days' stream statistics are recorded against it.
5. **The fitted values** on W, the feature stores, selection on January and the
   leakage tests (:mod:`.leakage`).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, timedelta
from functools import partial
from typing import Any

from trading_research.data import fetch_log as fl
from trading_research.data import study_access as sa
from trading_research.data import study_loader as sl
from trading_research.data.disk import stage_budget
from trading_research.data.fetch_stage import StagePlan, head_check_present, run_stage
from trading_research.pipeline.short_horizon.context import Study, git

CODE_FREEZE_NOTE = "docs/short_horizon_2025_code_freeze.md"
CODE_FREEZE_LINE = re.compile(r"Code-freeze commit: `([0-9a-f]{7,40})`")
#: What the code freeze covers: a change here after the note is a correction.
STUDY_CODE = ("src/", "experiments/short_horizon_2025.py", "configs/short_horizon_2025.yaml",
              "pyproject.toml", "uv.lock")  # fmt: skip
HEAD_CHECK = "experiments/results/short_horizon_2025_head_check.csv"
BARS_SPAN = (date(2024, 12, 1), date(2025, 6, 30))
STAGE_1_PRINT_GB = 15.3


def code_freeze(study: Study) -> dict[str, str]:
    """The note's commit, refused unless no study code changed after it."""
    path = study.repo / CODE_FREEZE_NOTE
    if not path.exists():
        raise RuntimeError(
            f"no {CODE_FREEZE_NOTE}: the code freeze is recorded, with a line "
            "'Code-freeze commit: `<sha>`', before the stage 1 fetch"
        )
    found = CODE_FREEZE_LINE.findall(path.read_text(encoding="utf-8"))
    if not found:
        raise RuntimeError(f"{CODE_FREEZE_NOTE} names no 'Code-freeze commit: `<sha>`'")
    commit = git(study.repo, "rev-parse", f"{found[-1]}^{{commit}}").strip()
    git(study.repo, "merge-base", "--is-ancestor", commit, "HEAD")
    changed = git(study.repo, "diff", "--name-only", f"{commit}..HEAD", "--", *STUDY_CODE).split()
    dirty = git(study.repo, "status", "--porcelain", "--", *STUDY_CODE).splitlines()
    if changed or dirty:
        raise RuntimeError(
            f"study code changed after the code-freeze commit {commit[:7]}: "
            f"{(changed + [d[3:] for d in dirty])[:5]}; a change is a correction under the "
            "amendment protocol"
        )
    return {"commit": commit}


def development_plan(study: Study) -> StagePlan:
    """Stage 1's days per plane: W and D, the reader-test books, D's hourly bars."""
    days = tuple(sorted({*study.scale.days("W"), *study.scale.days("D")}))
    bars = tuple(
        BARS_SPAN[0] + timedelta(days=i) for i in range((BARS_SPAN[1] - BARS_SPAN[0]).days + 1)
    )
    reader = study.scale.days(sa.READER_TEST)
    return StagePlan(book=days + reader, prints=days, bars=bars, funding=days)


def fetch(study: Study, book_workers: int) -> fl.FetchLog:
    """Every planned day on disk or terminally excluded, under the development access.

    A dry run fetches from its synthetic venue only: the venue is the opener of
    every archive, and every plane is fetched in this process (no worker
    process), where the venue also answers ``urlopen`` and sockets are refused."""
    plan = development_plan(study)
    opener = study.venue
    if opener is not None:
        present = opener.market.present()
        present["ob200"] = present["ob200"] | opener.reader_test_days
        book_workers, print_workers = 1, 1
    else:
        present = head_check_present(study.repo / HEAD_CHECK)
        print_workers = max(1, study.workers)
    share = len(plan.prints) / (sa.BLOCK_DAYS["W"] + sa.BLOCK_DAYS["D"])
    budget = stage_budget(book_days=len(plan.book), print_archives_gb=STAGE_1_PRINT_GB * share,
                          workers=book_workers)  # fmt: skip
    log = fl.FetchLog(study.dir("1") / "fetch_log.json")
    run_stage(study.symbol, plan, study.roots, log, budget=budget, book_workers=book_workers,
              print_workers=print_workers, expected_present=present, opener=opener,
              access=sa.development_access())  # fmt: skip
    if not log.complete():
        raise RuntimeError("the fetch returned with days neither on disk nor excluded")
    return log


def instrument(study: Study) -> sl.StudyInstrument:
    """Tick and lot from the venue's specification (the venue the run reads)."""
    return sl.fetch_instrument(study.symbol)


def _events(study: Study, day: date) -> Any:
    return sl.load_events_day(
        study.symbol, day, book_roots=[study.roots.book], trades_root=study.roots.trades,
        funding_root=study.roots.funding, access=sa.development_access(),
    )  # fmt: skip


def validate(study: Study, log: fl.FetchLog, spec: sl.StudyInstrument) -> dict[str, Any]:
    """Every W and D day's exclusions, the gap threshold, and the reader-test days."""
    from trading_research.market_making.events import DayUnavailable

    access = sa.development_access()
    excluded = log.excluded()
    days = sorted({*study.scale.days("W"), *study.scale.days("D")})
    gaps: dict[date, int] = {}
    found: dict[str, dict[str, Any]] = {}
    for day in days:
        reasons = [
            status if plane == sa.BOOK else f"{plane}: {status}"
            for plane in (sa.BOOK, sa.PRINTS, sa.FUNDING)
            if (status := excluded.get(f"{plane}/{day}")) is not None
        ]
        notes: list[str] = []
        if f"{sa.BOOK}/{day}" not in excluded:
            book = sl.load_book_day(study.symbol, day, study.roots.book, max_sequence_gaps=None,
                                    spec_tick=spec.tick, access=access)  # fmt: skip
            reasons += list(book.exclusions)
            notes += list(book.notes)
            if book.sequence_gaps is not None:
                gaps[day] = book.sequence_gaps
        if not reasons:
            try:
                _events(study, day)
            except DayUnavailable as exc:
                reasons.append(f"simulator day loader: {exc}")
        found[day.isoformat()] = {"block": sa.block_of(day), "reasons": reasons, "notes": notes}
    d_gaps = [n for day, n in gaps.items() if sa.block_of(day) == "D"]
    threshold = sl.max_sequence_gaps(d_gaps)
    for day, n in gaps.items():
        if n > threshold:
            found[day.isoformat()]["reasons"].append(sl.SEQUENCE_GAPS)
    usable = [d for d in days if not found[d.isoformat()]["reasons"]]
    from trading_research.pipeline.short_horizon.features import previous_day

    for day in days:
        entry = found[day.isoformat()]
        entry["usable"] = not entry["reasons"]
        entry["cold"] = previous_day(day, usable) is None
        entry["sequence_gaps"] = gaps.get(day)
    by_reason = Counter(r for e in found.values() for r in e["reasons"])
    reader = {}
    for day in study.scale.days(sa.READER_TEST):
        record = log.get(sa.BOOK, day)
        stats = dict(record.stats)
        stats["status"] = record.status
        stats["within_threshold"] = (
            record.status == fl.ON_DISK and int(stats.get("sequence_gaps", 0)) <= threshold
        )
        reader[day.isoformat()] = stats
    return {"days": found, "max_sequence_gaps": threshold, "excluded_by_reason": dict(by_reason),
            "reader_test": reader}  # fmt: skip


def _book(study: Study, day: date, threshold: int | None) -> Any:
    return sl.load_book_day(study.symbol, day, study.roots.book, max_sequence_gaps=threshold,
                            access=sa.development_access()).frame  # fmt: skip


def _prints(study: Study, day: date) -> Any:
    return sl.load_prints_day(study.symbol, day, study.roots.trades, access=sa.development_access())


def fitted_values(study: Study, w_days: list[date], threshold: int) -> dict[str, Any]:
    """W's constants: touch size, the tensor's calibration and S1's ``sigma_ref``."""
    from trading_research.data.feature_store import median_touch_size
    from trading_research.features.lob_tensor import TensorCalibrator
    from trading_research.pipeline.short_horizon_execution import sigma_ref_from_days, study_config

    if not w_days:
        raise RuntimeError("W has no usable day: nothing to fit the warm-up values on")
    calibrator = TensorCalibrator()
    for day in w_days:
        calibrator.add(_book(study, day, threshold))
    touch = median_touch_size(_book(study, day, threshold) for day in w_days)
    sigma_ref = sigma_ref_from_days((_events(study, d) for d in w_days), study_config(threshold))
    return {"touch_size": touch, "tensor": calibrator.result().to_dict(), "sigma_ref": sigma_ref}


def build_stores(
    study: Study, usable: list[date], fitted: dict[str, Any], threshold: int
) -> dict[str, Any]:
    """Both float16 stores and the targets of every usable W and D day, one day at a
    time in calendar order; a day already stored is not built again."""
    from trading_research.data.feature_store import fit_layout, open_stores
    from trading_research.features.lob_tensor import TensorCalibration
    from trading_research.pipeline.short_horizon import features as ft

    calibration = TensorCalibration.from_dict(fitted["tensor"])
    root = study.dir("1", "features")
    holds = sorted({*(study.scale.holds_s or (5.0, 20.0, 50.0, 120.0)), 20.0})

    def frames(only: list[date]) -> Any:
        previous: tuple[Any, Any] | None = None
        last: date | None = None
        for day in only:
            rows, prints = _book(study, day, threshold), _prints(study, day)
            history = (
                previous if last is not None and ft.previous_day(day, usable) == last else None
            )
            yield day, rows, ft.day_frame(rows, prints, calibration, history)
            previous, last = (rows, prints), day

    w_days = [d for d in usable if sa.block_of(d) == "W"]
    layout, finiteness = fit_layout((f for _, _, f in frames(w_days)), touch_size=fitted["touch_size"],
                                    raw_size_columns=ft.RAW_SIZES)  # fmt: skip
    stores = open_stores(root, layout)
    overflow: dict[str, list[str]] = {}
    for day, rows, frame in frames(usable):
        targets = study.dir("1", "targets") / f"{day}.parquet"
        clip = sl.clip_for(study.symbol)
        ft.read_or_build(targets, partial(ft.day_targets, rows, holds, (0.0, 5.5), clip))
        if all(store.path(day).exists() for store in stores.values()):
            continue
        for store in stores.values():
            written = store.write_day(day, frame)
            if written.overflow:
                overflow[day.isoformat()] = list(written.overflow)
    sizes = {name: store.size_bytes() for name, store in stores.items()}
    return {"layout": layout.to_dict(), "finiteness": {"overflow": list(finiteness.overflow),
            "infinite_at_source": list(finiteness.infinite_at_source)}, "overflow_after_W": overflow,
            "store_bytes": sizes, "holds_with_targets": holds}  # fmt: skip


def select_features(inputs: Any, january: list[date]) -> dict[str, Any]:
    """Selection on January's every-tenth rows: the walked label and the forward move at 20 s."""
    import numpy as np
    import pandas as pd

    from trading_research.features.selection import FeatureSelector
    from trading_research.pipeline.short_horizon import features as ft

    if not january:
        raise RuntimeError("January has no usable day to select features on")
    xs, ys, fs = [], [], []
    for day in january:
        targets = inputs.targets(day)
        xs.append(inputs.features(day, ft.micro_columns(), every_row=False).astype("float64"))
        ys.append(targets[ft.label_column(20.0, 0.0)])
        fs.append(targets[ft.forward_column(20.0)])
    x = pd.concat(xs, ignore_index=True)
    # A column with no finite value in January has no variance to select on, and
    # would leave the selector no complete row; it goes before the selector runs.
    empty = [c for c in x.columns if not np.isfinite(x[c].to_numpy()).any()]
    x = x.drop(columns=empty)
    selector = FeatureSelector().fit(
        x, pd.concat(ys, ignore_index=True), pd.concat(fs, ignore_index=True)
    )
    report = selector.report_.to_dict() if selector.report_ is not None else {}
    report["dropped_without_a_value"] = empty
    return {"selected": list(selector.selected_), "selection_report": report, "rows": len(x)}


def run(study: Study) -> dict[str, Any]:
    """Stage 1's record; the leakage tests raise and stop the study on a leak."""
    from trading_research.pipeline.short_horizon import leakage
    from trading_research.pipeline.short_horizon.inputs import Inputs
    from trading_research.pipeline.short_horizon.stage0 import require_no_pending_cuts
    from trading_research.search_engine.catalogue import parse_key

    registration = study.registration
    catalogue = study.catalogue()
    record0 = study.record("0")
    require_no_pending_cuts(record0, catalogue.sha256)
    freeze = code_freeze(study)
    log = fetch(study, int(record0["replay_memory"]["workers"]))
    spec = instrument(study)
    record: dict[str, Any] = {"commit": study.commit(), "config_sha256": registration.sha256,
                              "catalogue_sha256": catalogue.sha256, "code_freeze": freeze,
                              "instrument_spec": {"tick": spec.tick, "lot": spec.lot},
                              "fetch_counts": {p: dict(log.counts(p)) for p in sa.PLANES}}  # fmt: skip
    record |= validate(study, log, spec)
    threshold = int(record["max_sequence_gaps"])
    usable = [date.fromisoformat(d) for d, e in record["days"].items() if e["usable"]]
    w_days = [d for d in usable if sa.block_of(d) == "W"]
    record["fitted"] = fitted_values(study, w_days, threshold)
    record["stores"] = build_stores(study, usable, record["fitted"], threshold)
    inputs = Inputs(study, given=record)
    january = [d for d in inputs.usable("D") if d.month == 1]
    record |= select_features(inputs, january)
    network = any(parse_key(k).model == "tcn" for k in study.keys)
    record["leakage"] = leakage.run(inputs, network)
    return record
