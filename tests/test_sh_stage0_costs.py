"""Stage 0's network timing: a training batch and inference per row, each over a whole day."""

from __future__ import annotations

import importlib.util

import pytest

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
