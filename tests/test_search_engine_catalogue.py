"""The catalogue of taker cells: generated from the registration, checked by hash."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from trading_research.pipeline.regime import window_hours
from trading_research.search_engine.catalogue import (
    REGIME_WINDOW_H,
    Catalogue,
    CatalogueError,
    catalogue_keys,
    catalogue_sha256,
    parse_key,
    serialise,
    written,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"
S25 = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none"


@pytest.fixture(scope="module")
def search() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["search"]


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return Catalogue.load(CONFIG)


def test_the_registered_hash_is_reproduced(catalogue: Catalogue, search: dict[str, Any]) -> None:
    assert catalogue.sha256 == search["catalogue"]["sha256"]
    assert len(catalogue) == search["catalogue"]["taker_cells"] == 2400
    assert {k: len(v) for k, v in catalogue.strata().items()} == {"C": 1536, "R": 864}
    assert search["named"]["S25"] == S25 and S25 in catalogue


def test_the_counts_of_the_registration(catalogue: Catalogue, search: dict[str, Any]) -> None:
    configurations = catalogue.configurations()
    assert len(configurations) == search["catalogue"]["signal_configurations"] == 40
    sizes = {parse_key(cells[0]).stratum: len(cells) for cells in configurations.values()}
    assert sizes == {"C": 48, "R": 108}  # gates x rates x regimes; gates x k x beta x regimes
    groups = catalogue.table().groupby(["stratum", "model", "inputs", "gate"]).ngroups
    assert groups == search["catalogue"]["groups"] == 18


def test_the_serialisation_is_byte_order_with_a_newline_after_each() -> None:
    keys = ["b|x", "B|x", "a+b", "a_b"]
    data = serialise(keys)
    assert data == b"B|x\na+b\na_b\nb|x\n"
    assert catalogue_sha256(keys) == catalogue_sha256(reversed(keys))


def test_a_changed_axis_changes_the_hash(search: dict[str, Any]) -> None:
    changed = copy.deepcopy(search)
    changed["axes"]["hold_s"]["values"] = [5, 20, 50]
    with pytest.raises(CatalogueError, match="differs from the registered"):
        Catalogue.from_search(changed).verify()
    renamed = copy.deepcopy(search)
    renamed["catalogue"]["key_R"] = renamed["catalogue"]["key_R"].replace("beta_bp", "beta")
    with pytest.raises(CatalogueError):
        Catalogue.from_search(renamed).verify()


def test_numbers_are_written_as_registered(search: dict[str, Any]) -> None:
    allowed = [written(v) for v in search["catalogue"]["numbers_written_as"]]
    assert [written(v, allowed) for v in (0, 0.5, 1, 5.5, 240)] == ["0", "0.5", "1", "5.5", "240"]
    with pytest.raises(CatalogueError, match="not as registered"):
        written(1.0, allowed)
    with pytest.raises(CatalogueError):
        written(True)
    bad = copy.deepcopy(search)
    bad["axes"]["k"]["values"] = [0.0, 0.5, 1]
    with pytest.raises(CatalogueError, match="not as registered"):
        catalogue_keys(bad)


def test_keys_outside_the_catalogue_are_refused(catalogue: Catalogue) -> None:
    catalogue.require([S25])
    with pytest.raises(CatalogueError, match="outside the catalogue"):
        catalogue.require([S25, S25.replace("hold_s=50", "hold_s=300")])
    with pytest.raises(CatalogueError, match="byte order"):
        Catalogue(tuple(reversed(catalogue.keys)), catalogue.registered_sha256)


def test_keys_parse_to_numeric_axes() -> None:
    cell = parse_key(
        "BTCUSDT|R|tcn|tensor+micro|quiet_out|hold_s=5|k=0.5|beta_bp=2|regime=pull_120h"
    )
    assert (cell.stratum, cell.model, cell.inputs, cell.gate) == (
        "R",
        "tcn",
        "tensor+micro",
        "quiet_out",
    )
    assert (cell.hold_s, cell.k, cell.beta_bp, cell.regime_h) == (5.0, 0.5, 2.0, 120)
    assert np.isnan(cell.trades_per_day) and np.isnan(cell.label_fee_bp)
    assert cell.configuration == "BTCUSDT|R|tcn|tensor+micro|hold_s=5"
    s25 = parse_key(S25)
    assert (s25.hold_s, s25.trades_per_day, s25.label_fee_bp, s25.regime_h) == (50, 240, 0, 0)
    assert s25.configuration == "BTCUSDT|C|gbm|micro|hold_s=50|label_fee_bp=0"
    for name, hours in REGIME_WINDOW_H.items():
        assert window_hours(name) == hours
    assert sorted(REGIME_WINDOW_H, key=REGIME_WINDOW_H.get)[-1] == "pull_120h"


@pytest.mark.parametrize(
    "key",
    [
        "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0",
        "BTCUSDT|X|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none",
        "BTCUSDT|C|gbm|micro|open|hold_s=50|k=0|label_fee_bp=0|regime=none",
        "BTCUSDT|C|gbm|micro|open|hold_s=fifty|trades_per_day=240|label_fee_bp=0|regime=none",
        "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=pull_8h",
    ],
)
def test_malformed_keys_are_refused(key: str) -> None:
    with pytest.raises(CatalogueError):
        parse_key(key)


def test_s25_has_its_registered_neighbourhood(catalogue: Catalogue) -> None:
    index = catalogue.neighbourhoods()
    table = catalogue.table()
    near = table.loc[index[catalogue.keys.index(S25)][index[catalogue.keys.index(S25)] >= 0]]
    assert len(near) == 24
    assert set(near.hold_s) == {20, 50, 120} and set(near.trades_per_day) == {100, 240}
    assert set(near.label_fee_bp) == {0, 5.5} and set(near.regime_h) == {0, 4}
    assert set(near.gate) == {"open"} and set(near.inputs) == {"micro"}
    r_cell = catalogue.keys.index(
        "BTCUSDT|R|bound_ridge|micro|open|hold_s=20|k=0.5|beta_bp=1|regime=pull_24h"
    )
    assert (index[r_cell] >= 0).sum() == 3 * 3 * 3 * 3
