"""The fold-level leakage test runs the study's own fit path, and catches a fit that reads late.

A small stage-1 work root is built on synthetic days (book, both feature stores,
targets) for one fold; the test fits it twice through ``fits.FoldData``,
``fits.fit_main_process`` and ``fits.fit_networks``, once on that root and once
on a scratch root with every row after the cutoff mutated.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from trading_research.data.feature_store import StoreLayout, open_stores
from trading_research.data.study_fetch import book_path
from trading_research.pipeline.short_horizon import features as ft
from trading_research.pipeline.short_horizon import fits
from trading_research.pipeline.short_horizon import leakage as lk
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.inputs import Inputs
from trading_research.pipeline.short_horizon.plan import DRY_RUN, Fold, midnight_ns

NS = 1_000_000_000
DAYS = [date(2025, 1, 1) + timedelta(days=i) for i in range(8)]
FOLD = Fold("2025-01-test", tuple(DAYS[:4]), tuple(DAYS[4:6]), tuple(DAYS[6:]), DAYS[4], DAYS[6])
MICRO = ["f0", "f1", "f2"]
STEP_S = 4


def _book(day: date, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 86_400 // STEP_S
    stamps = midnight_ns(day) + np.arange(n) * STEP_S * NS + rng.integers(0, 50_000_000, n)
    bid0 = 950_000 + np.cumsum(rng.choice([-1, 0, 1], size=n))
    frame: dict[str, Any] = {"timestamp": pd.to_datetime(stamps, unit="ns", utc=True)}
    for level in range(10):
        frame[f"bid_price_{level}"] = np.round((bid0 - level) * 0.1, 1)
        frame[f"bid_size_{level}"] = np.full(n, 0.5)
        frame[f"ask_price_{level}"] = np.round((bid0 + 1 + level) * 0.1, 1)
        frame[f"ask_size_{level}"] = np.full(n, 0.5)
    return pd.DataFrame(frame)


def _features(book: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    mid = (book["bid_price_0"] + book["ask_price_0"]).to_numpy() / 2
    out = {ft.TIME: book["timestamp"]}
    for k, name in enumerate([*MICRO, *ft.TENSOR]):
        trailing = np.r_[0.0, np.diff(mid)] if k == 0 else 0.0  # f0 sees the last move only
        out[name] = rng.normal(size=len(book)) + trailing
    return pd.DataFrame(out)


@pytest.fixture(scope="module")
def stage1(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    root = tmp_path_factory.mktemp("fold_level")
    study = Study(root / "repo", DRY_RUN, work_root=root / "work")
    layout = StoreLayout(1.0, ())
    stores = open_stores(study.work / "1" / "features", layout)
    for k, day in enumerate(DAYS):
        book = _book(day, k)
        path = book_path(study.symbol, day, study.roots.book)
        path.parent.mkdir(parents=True, exist_ok=True)
        book.to_parquet(path, index=False)
        for store in stores.values():
            store.write_day(day, _features(book, k))
        targets = ft.day_targets(book, (20.0,), (0.0, 5.5), 0.01)
        (study.work / "1" / "targets").mkdir(parents=True, exist_ok=True)
        targets.to_parquet(study.work / "1" / "targets" / f"{day}.parquet", index=False)
    record = {"max_sequence_gaps": 0, "selected": MICRO,
              "stores": {"layout": layout.to_dict(), "holds_with_targets": [20.0]},
              "days": {d.isoformat(): {"usable": True, "cold": i == 0} for i, d in enumerate(DAYS)}}  # fmt: skip
    return Inputs(study, given=record)


def test_the_cutoff_reaches_the_last_exit_snapshot_an_inner_row_reads() -> None:
    end = FOLD.inner_end_ns(20.0)
    before = end - np.arange(30, 0, -1) * 100_000_000  # rows to 100 ms before the end
    after = end + 45 * NS + np.arange(0, 400) * 100_000_000  # a 45 s gap, then rows again
    stamps = np.r_[midnight_ns(DAYS[5]), before, after]
    frame = pd.DataFrame({"timestamp": pd.to_datetime(stamps, unit="ns", utc=True)})
    inputs = SimpleNamespace(book=lambda _: frame)
    cutoff = lk.cutoff_ns(inputs, FOLD)  # type: ignore[arg-type]
    assert cutoff > end + 20 * NS  # the registered end plus the hold is not enough here
    rows = ft.target_rows(len(stamps))
    entry = np.searchsorted(stamps, stamps[rows] + 10_000_000)
    exit_ = np.searchsorted(stamps, stamps[np.minimum(entry, len(stamps) - 1)] + 20 * NS)
    reads = exit_[(stamps[rows] < end) & (exit_ < len(stamps))]
    assert cutoff == stamps[reads].max() < midnight_ns(DAYS[6])


needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs 'deep'")


def _late(inputs: Any, model: Any, columns: Any, days: Any, before_ns: int) -> np.ndarray:
    """Thresholds read three hours of rows meant to be purged."""
    return _INNER_PROBA(inputs, model, columns, days, before_ns + 3 * 3600 * NS)


_INNER_PROBA = fits.inner_proba


def _in_a_process_of_its_own(repo: Path, work_root: Path, record: dict[str, Any], network: bool,
                             late_thresholds: bool) -> dict[str, Any]:  # fmt: skip
    """The fold-level test on the stage-1 root at ``repo``, in a process torch was never
    loaded into, as the study's own main process is (its networks fit in processes of
    their own): the xgboost and torch wheels bring different OpenMP runtimes, and a
    process that has loaded torch crashes in xgboost's first fit. Returns what was
    called and found, or the leak the test raised."""
    inputs = Inputs(Study(repo, DRY_RUN, work_root=work_root), given=record)
    with (
        mock.patch.object(lk, "fit_main_process", wraps=fits.fit_main_process) as main,
        mock.patch.object(fits.FoldData, "load", wraps=fits.FoldData.load) as load,
        mock.patch.object(
            fits.FoldData, "gate_levels", autospec=True, side_effect=fits.FoldData.gate_levels
        ) as gates,  # fmt: skip
        mock.patch.object(fits, "inner_proba", _late if late_thresholds else _INNER_PROBA),
    ):
        try:
            found = lk.fold_level(inputs, [FOLD], network=network)
        except lk.LeakError as exc:
            return {"leak": str(exc)}
    scratch = inputs.study.work / "1" / "fold_leakage" / FOLD.name
    return {"found": found, "calls": (main.call_count, load.call_count, gates.call_count),
            "scratch_left": scratch.exists()}  # fmt: skip


def _own_process(stage1: Inputs, *, network: bool, late_thresholds: bool = False) -> dict[str, Any]:
    from trading_research.pipeline.short_horizon.models import in_fresh_process

    study = stage1.study
    return in_fresh_process(_in_a_process_of_its_own, study.repo, study.work_root, stage1.record,
                            network, late_thresholds)  # fmt: skip


@needs_torch
def test_the_fold_level_test_fits_through_the_production_path_twice(stage1: Inputs) -> None:
    found = _own_process(stage1, network=True)
    assert "leak" not in found, found
    assert found["calls"] == (6, 2, 2)  # three models twice; the rows and the gate levels twice
    artefacts = found["found"][0]["artefacts"]
    for name in ("gbm/weights", "logistic/weights", "bound_ridge/weights", "gbm/thresholds",
                 "logistic/thresholds", "gate_levels", "tcn/weights_sha256", "tcn/state_sha256",
                 "tcn/inner_ic", "tcn/checkpoint_epoch"):  # fmt: skip
        assert name in artefacts
    assert found["found"][0]["past_inner_end_plus_hold_ns"] >= 0
    assert not found["scratch_left"]


def test_thresholds_set_on_rows_meant_to_be_purged_are_caught(stage1: Inputs) -> None:
    found = _own_process(stage1, network=False, late_thresholds=True)
    assert "thresh" in found.get("leak", ""), found
