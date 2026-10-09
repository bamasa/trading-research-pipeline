"""Catalogue keys as a table: parsing, regions and neighbourhoods."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from _sh_catalogue import REGISTERED_SHA256, catalogue_keys
from trading_research.reporting.search_keys import (
    REGIONS,
    S25_KEY,
    CatalogueKeyError,
    catalogue_table,
    in_region,
    neighbourhoods,
    parse_key,
    region_tags,
)


def test_the_generated_catalogue_is_the_registered_one() -> None:
    keys = catalogue_keys()
    text = "".join(f"{k}\n" for k in sorted(keys, key=str.encode))
    assert len(keys) == 2400
    assert hashlib.sha256(text.encode()).hexdigest() == REGISTERED_SHA256
    assert S25_KEY in keys


def test_a_key_parses_into_numbers_and_the_regime_window() -> None:
    c = parse_key(S25_KEY)
    assert (c["stratum"], c["model"], c["inputs"], c["gate"]) == ("C", "gbm", "micro", "open")
    assert (c["hold_s"], c["trades_per_day"], c["label_fee_bp"], c["regime_h"]) == (50, 240, 0, 0)
    assert np.isnan(c["k"]) and np.isnan(c["beta_bp"])
    r = parse_key("BTCUSDT|R|tcn|tensor+micro|open|hold_s=5|k=0.5|beta_bp=1|regime=pull_120h")
    assert (r["k"], r["beta_bp"], r["regime"], r["regime_h"]) == (0.5, 1.0, "pull_120h", 120.0)
    assert np.isnan(r["trades_per_day"]) and np.isnan(r["label_fee_bp"])


@pytest.mark.parametrize(
    "bad",
    [
        "BTCUSDT|X|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none",
        "BTCUSDT|C|gbm|micro|open|hold_s=50|k=240|label_fee_bp=0|regime=none",
        "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0",
        "BTCUSDT|C|gbm|micro|open|hold_s=|trades_per_day=240|label_fee_bp=0|regime=none",
    ],
)
def test_a_string_that_is_not_a_key_is_refused(bad: str) -> None:
    with pytest.raises(CatalogueKeyError):
        parse_key(bad)


def test_a_key_listed_twice_is_refused() -> None:
    with pytest.raises(CatalogueKeyError):
        catalogue_table([S25_KEY, S25_KEY])


def test_the_regions_have_their_registered_sizes() -> None:
    table = catalogue_table(catalogue_keys())
    assert in_region(table, REGIONS["T1"]).sum() == 4 * 4  # hold x rate
    assert in_region(table, REGIONS["T2"]).sum() == 4 * 3 * 3  # hold x k x beta
    tags = region_tags(table)
    assert tags.count("T1") == 16 and tags.count("T2") == 36
    assert tags[table.index[table.key == S25_KEY][0]] == ""


def test_s25s_neighbourhood_from_the_keys_has_twenty_four_cells() -> None:
    table = catalogue_table(catalogue_keys())
    index = neighbourhoods(table)
    row = int(np.flatnonzero(table.key == S25_KEY)[0])
    near = table.loc[index[row][index[row] >= 0]]
    assert len(near) == 24
    assert set(near.hold_s) == {20, 50, 120} and set(near.trades_per_day) == {100, 240}
    assert set(near.regime) == {"none", "pull_4h"} and set(near.label_fee_bp) == {0, 5.5}
    r_row = int(np.flatnonzero(table.stratum == "R")[0])
    assert set(table.loc[index[r_row][index[r_row] >= 0], "stratum"]) == {"R"}
