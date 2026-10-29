"""The search takes a model plugin by name, pins its file, and says what a result is short of."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_research.pipeline import stages
from trading_research.plugins import registry
from trading_research.walkthrough import steps
from trading_research.walkthrough import verdict as vd
from trading_research.walkthrough.layout import Dataset, SyntheticInstrument, write_synthetic
from trading_research.walkthrough.record import SearchError, SearchRun

FIRST = date(2031, 3, 3)
NAME = "headroom_test_logistic"


def _tiny_logistic(**params: Any) -> Any:
    """A model plugin for the test: the built-in logistic regression under another name."""
    return stages.build_model("logistic", **params)


@pytest.fixture
def plugin() -> Any:
    registry.discover()
    spec = registry.add("model", NAME, _tiny_logistic)
    yield spec
    registry.unregister("model", NAME)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("walk")
    write_synthetic(root, [SyntheticInstrument("SYNAUSDT", 1, 0.3)], first_day=FIRST, days=4,
                    overrides={"step_ms": 5000})  # fmt: skip
    return root


def _attempts(gross: list[float], cost: float, filled: list[bool] | None = None) -> pd.DataFrame:
    filled = filled or [True] * len(gross)
    net = [g - cost if f else 0.0 for g, f in zip(gross, filled, strict=True)]
    return pd.DataFrame({"day": ["d"] * len(gross), "gross_bp": gross, "cost_bp": cost,
                         "net_bp": net, "filled": filled})  # fmt: skip


def test_a_model_plugin_is_built_by_name_where_a_built_in_would_be(plugin: Any) -> None:
    built = stages.build_model(NAME, C=0.5)
    assert type(built) is type(stages.build_model("logistic"))
    with pytest.raises(stages.StageError, match="model plugin"):
        stages.build_model("no_such_model_anywhere")


def test_the_headroom_restates_the_same_attempts() -> None:
    # Six filled trades: three right (+4 each), three wrong (-2 each), cost 3 each.
    found = vd.headroom(_attempts([4, 4, 4, -2, -2, -2], 3.0), "taker")
    assert found["net_per_attempt_bp"] == pytest.approx(-2.0)
    assert found["gross_per_attempt_bp"] == pytest.approx(1.0)
    assert found["cost_cut_to_break_even_bp"] == pytest.approx(2.0)
    direction = found["direction"]
    assert direction["hit_rate"] == pytest.approx(0.5) and direction[
        "mean_loss_bp"
    ] == pytest.approx(2.0)
    # A further share d right turns a 2 bp loss into a 2 bp gain: +4d a trade.
    assert direction["hit_gain_to_break_even"] == pytest.approx(0.5)
    assert [r["net_per_attempt_bp"] for r in direction["rungs"]] == pytest.approx(
        [-2.0 + 4 * d for d in vd.HIT_RUNGS]
    )
    # Misses dilute the lever by the fill rate.
    half = vd.headroom(_attempts([4, -2, 0, 0], 3.0, [True, True, False, False]), "passive")
    assert half["direction"]["fill_rate"] == pytest.approx(0.5)
    assert half["direction"]["rungs"][0]["net_per_attempt_bp"] == pytest.approx(
        half["net_per_attempt_bp"] + 0.5 * 2 * 0.01 * 2.0
    )


def test_no_cost_rescues_a_non_positive_gross_and_the_maker_gets_no_direction() -> None:
    found = vd.headroom(_attempts([-1, -1], 2.0), "taker")
    assert math.isnan(found["cost_to_break_even_bp"])
    assert "no fee cut" in steps.headroom_line(found)
    maker = vd.headroom(_attempts([1, -3], 1.0), "maker")
    assert maker["direction"] is None and "not directional" in maker["note"]


def test_an_unknown_rule_or_unbuildable_model_is_refused_before_registration(
    data: Path, tmp_path: Path
) -> None:
    run = SearchRun(tmp_path / "run")
    with pytest.raises(SearchError, match="rules must be among"):
        steps.register(run, Dataset.under(data), rules=["astrology"])
    with pytest.raises(SearchError, match="cannot be built"):
        steps.register(run, Dataset.under(data), models=["no_such_model_anywhere"])
    assert not run.registration_path.exists()


def test_the_search_tries_a_model_plugin_and_pins_its_file(
    plugin: Any, data: Path, tmp_path: Path
) -> None:
    run = SearchRun(tmp_path / "run")
    steps.register(run, Dataset.under(data), family=("taker",), rules=["momentum"],
                   models=["logistic", NAME], holds_s=(30,))  # fmt: skip
    registration = json.loads(run.registration_path.read_text())
    assert registration["methods"] == {"rules": ["momentum"], "models": ["logistic", NAME]}
    assert registration["plugins"][NAME]["sha256"] == plugin.fingerprint() != ""
    assert registration["parameters"][NAME] == [{}]
    steps.instruments(run)
    steps.frequency(run)
    steps.method(run)
    table = pd.read_csv(run.step_dir("method") / "method.csv")
    assert set(table["model"]) == {"momentum", "logistic", NAME}
    # A plugin file that changed after registration stops every later step.
    steps.check_plugins(registration)
    moved = {"plugins": {NAME: {**registration["plugins"][NAME], "sha256": "0" * 64}}}
    with pytest.raises(SearchError, match="changed after the search was registered"):
        steps.check_plugins(moved)
