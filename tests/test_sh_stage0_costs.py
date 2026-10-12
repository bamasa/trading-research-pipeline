"""Stage 0: the network's timing over a whole day and on the CPU, the projection, the archives."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from _sh_catalogue import catalogue_keys
from trading_research.pipeline.short_horizon import stage0
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.models import in_fresh_process
from trading_research.pipeline.short_horizon.plan import DRY_RUN
from trading_research.pipeline.short_horizon.stage0 import network_costs


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="the network needs 'deep'")
def test_the_network_is_timed_over_a_whole_day_with_a_sample_beside_it() -> None:
    params = DRY_RUN.params("tcn")
    costs = in_fresh_process(network_costs, params, 600, 4, 5_000)
    assert set(costs) == {
        "batch_s",
        "inference_row_s",
        "inference_day_s",
        "batch_sample_s",
        "inference_sample_row_s",
    }
    assert costs["inference_row_s"] == pytest.approx(costs["inference_day_s"] / 5_000)
    assert all(value > 0 for value in costs.values())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="the network needs 'deep'")
def test_the_fold_level_tests_network_is_timed_on_the_cpu() -> None:
    from trading_research.pipeline.short_horizon.stage0 import cpu_network_costs

    costs = in_fresh_process(cpu_network_costs, DRY_RUN.params("tcn"), 600, 4)
    assert set(costs) == {"cpu_batch_s", "cpu_inference_row_s"}
    assert all(value > 0 for value in costs.values())


def test_a_store_day_is_timed_and_its_file_removed(tmp_path: Path) -> None:
    seconds = stage0.store_day_cost(tmp_path / "timing", 2_000, 5)
    assert seconds > 0 and not (tmp_path / "timing").exists()


def _costs() -> dict[str, float]:
    names = ["walk_table_s", "hold_table_s", "cell_day_s", "null_b_cell_day_s", "features_s",
             "labels_s", "null_a_draw_s", "simulator_cell_day_s", "network_batch_s",
             "network_inference_row_s", "network_cpu_batch_s", "network_cpu_inference_row_s",
             "store_day_s", "replay_day_s"]  # fmt: skip
    costs = dict.fromkeys(names, 0.01)
    for model in ("gbm", "logistic", "bound_ridge"):
        costs[f"{model}_fit_row_s"] = costs[f"{model}_predict_row_s"] = 1e-6
    return costs | {"network_batch_size": 512.0, "replay_workers": 4.0}


def test_the_projection_times_the_fold_level_test_at_the_registered_workers() -> None:
    keys = catalogue_keys()
    size = stage0.Sizes(150, 212, 864_000, 1_800_000, 20, 7, 137, 5)
    costs = _costs()
    hours = stage0.projection(costs, stage0.counts(keys), size, 8, 999)
    assert hours["1_fold_leakage"] > 0
    slower = stage0.projection(
        costs | {"network_cpu_batch_s": 1.0}, stage0.counts(keys), size, 8, 999
    )
    assert slower["1_fold_leakage"] - hours["1_fold_leakage"] == pytest.approx(
        5 * 2 * 1_800_000 / 512 * (1.0 - 0.01) / 3600
    )
    _, _, uncut = stage0.projected(keys, costs, size, 999)
    assert uncut == stage0.projection(costs, stage0.counts(keys), size, stage0.GENERAL_WORKERS, 999)
    assert uncut != stage0.projection(costs, stage0.counts(keys), size, 1, 999)


def _study(tmp_path: Path) -> Study:
    return Study(tmp_path / "repo", DRY_RUN)


def test_stage_0_refuses_without_both_registered_timing_archives(tmp_path: Path) -> None:
    study = _study(tmp_path)
    folder = study.roots.archives.parent / "timing_archives"
    folder.mkdir(parents=True)
    first, second = study.scale.timing_days
    (folder / f"{study.symbol}-{first}-ob500.zip").write_bytes(b"x")
    (folder / f"{study.symbol}-2024-01-01-ob500.zip").write_bytes(b"x")  # not registered
    with pytest.raises(FileNotFoundError, match=f"{study.symbol}-{second}-ob500.zip"):
        stage0.timing_archives(study)
    (folder / f"{study.symbol}-{second}-ob500.zip").write_bytes(b"x")
    found = stage0.timing_archives(study)
    assert [p.name for p in found] == [f"{study.symbol}-{d}-ob500.zip" for d in (first, second)]
