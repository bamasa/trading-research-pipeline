"""The nulls file: every replication, the counts of (iv) and p_search."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.reporting.nulls import (
    NULLS_COLUMNS,
    nulls_table,
    read_nulls,
    s25_nulls_table,
    write_nulls,
)
from trading_research.validation.choice import ChoiceError, StratumOutcome
from trading_research.validation.placebo_search import (
    NullDraws,
    admission_iv,
    exceedances,
    p_search,
    placebo_p,
)

KEYS = [f"cell{i}" for i in range(10)]


def _draws(replications: int, seed: int, strata: tuple[str, ...] = ("C", "R")) -> NullDraws:
    return NullDraws(
        strata, np.random.default_rng(seed).normal(1.0, 0.5, (replications, len(strata)))
    )


@pytest.fixture
def search() -> tuple[dict, NullDraws, NullDraws]:
    a_null, b_null = _draws(999, 0), _draws(137, 1)
    observed = {"C": StratumOutcome(3, 2.4), "R": StratumOutcome(None, 0.0)}
    results = admission_iv(observed, a_null, b_null, runs=2, refit={"C": 1.1})
    return results, a_null, b_null


def test_every_replication_is_written_with_its_maximum(search: tuple) -> None:
    results, a_null, b_null = search
    table = nulls_table(results, a_null, b_null, KEYS, runs=2)
    assert tuple(table.columns) == NULLS_COLUMNS
    reps = table.loc[table["row"] == "replication"]
    assert len(reps) == (999 + 137) * 3
    b_max = reps.loc[(reps["null"] == "B") & (reps["stratum"] == "max")]
    assert list(b_max["replication"]) == list(range(7, 144))
    np.testing.assert_allclose(b_max["value"], b_null.maximum)
    a_c = reps.loc[(reps["null"] == "A") & (reps["stratum"] == "C")]
    assert list(a_c["replication"]) == list(range(999))
    q = table.loc[(table["row"] == "quantile") & (table["null"] == "A")].set_index("level")
    assert q.loc[0.95, "value"] == pytest.approx(np.quantile(a_null.maximum, 0.95))


def test_the_counts_and_p_values_are_those_of_admission(search: tuple) -> None:
    results, a_null, b_null = search
    table = nulls_table(results, a_null, b_null, KEYS, runs=2)
    p = table.loc[(table["row"] == "p") & (table["stratum"] == "C")].set_index("null")
    assert p.loc["A", "count"] == exceedances(a_null.maximum, 2.4)
    assert p.loc["B", "value"] == pytest.approx(placebo_p(int(p.loc["B", "count"]), 137))
    assert set(p["key"]) == {"cell3"} and set(p["runs"]) == {2}
    observed = table.loc[table["row"] == "observed"].set_index("stratum")
    assert observed.loc["C", "key"] == "cell3" and observed.loc["R", "key"] == ""
    assert table.loc[table["row"] == "refit", "value"].tolist() == [1.1]
    assert table.loc[table["row"] == "p_search", "value"].iloc[0] == p_search(results, 2)
    premium = table.loc[(table["row"] == "premium") & (table["null"] == "A"), "value"].iloc[0]
    assert premium == pytest.approx(2.4 - np.median(a_null.maximum))
    r = table.loc[(table["row"] == "p") & (table["stratum"] == "R")]
    assert list(r["value"]) == [1.0, 1.0]
    assert not ((table["row"] == "premium") & (table["stratum"] == "R")).any()


def test_counts_that_disagree_with_the_replications_are_refused(search: tuple) -> None:
    results, a_null, b_null = search
    wrong = dict(results) | {"C": replace(results["C"], count_a=results["C"].count_a + 1)}
    with pytest.raises(ChoiceError, match="counts"):
        nulls_table(wrong, a_null, b_null, KEYS)
    with pytest.raises(ChoiceError):
        nulls_table(results, a_null, b_null, KEYS, shifts=range(10))
    with pytest.raises(ChoiceError):
        nulls_table(results, a_null, _draws(137, 1, ("C",)), KEYS)


def test_the_global_flip_is_a_diagnostic_row(search: tuple) -> None:
    results, a_null, b_null = search
    flip = {"C": StratumOutcome(5, -1.5), "R": StratumOutcome(None, 0.0)}
    table = nulls_table(results, a_null, b_null, KEYS, global_flip=flip)
    rows = table.loc[table["row"] == "global_flip"].set_index("stratum")
    assert rows.loc["C", "key"] == "cell5" and rows.loc["C", "value"] == -1.5


def test_s25_is_set_against_its_own_nulls() -> None:
    a_null, b_null = _draws(999, 2, ("C",)), _draws(137, 3, ("C",))
    table = s25_nulls_table(1.7, a_null, b_null)
    assert set(table["subject"]) == {"S25"}
    assert not (table["stratum"] == "max").any()
    p = table.loc[table["row"] == "p"].set_index("null")
    assert p.loc["A", "count"] == exceedances(a_null.a[:, 0], 1.7)
    assert p.loc["B", "replications"] == 137
    with pytest.raises(ChoiceError):
        s25_nulls_table(1.7, _draws(999, 2), b_null)


def test_the_file_round_trips(search: tuple, tmp_path: Path) -> None:
    results, a_null, b_null = search
    tables = [
        nulls_table(results, a_null, b_null, KEYS, runs=2),
        s25_nulls_table(1.7, _draws(999, 2, ("C",)), _draws(137, 3, ("C",))),
    ]
    path = write_nulls(tables, tmp_path / "nulls.csv")
    assert "only p_search is quoted" in path.read_text().splitlines()[0]
    back = read_nulls(path)
    both = pd.concat(tables, ignore_index=True)
    assert list(back["row"]) == list(both["row"]) and list(back["key"]) == list(both["key"])
    np.testing.assert_allclose(back["value"], both["value"], rtol=1e-9)
