"""The nulls file: every replication of the placebo search, in one long table.

Each row has a ``subject`` (``search``, the whole choice re-run, or ``S25``,
the named cell against the same nulls with no selection) and a ``row`` kind:

* ``replication``: one replication's A per stratum, and its maximum over
  strata (stratum ``max``), the number every observed A is counted against;
  ``replication`` is Null A's seed b or Null B's shift s;
* ``observed``: each stratum's chosen cell (``key``) and its A;
* ``refit``: stratum C's refit A, which C* must beat strictly;
* ``quantile``: a quantile (``level``) of the null maximum;
* ``p``: each stratum's count of replications at or above its A, the
  replications and p = (1 + count) / (1 + replications), per null;
* ``premium``: the selection premium, A minus the median null maximum;
* ``global_flip``: every trade flipped at once, a diagnostic only;
* ``p_search``: max(p_A, p_B) of the stratum with the larger A, times R_D
  (``runs``), capped at 1: the one D number quoted, "on development data,
  selected".

The counts are taken from the admission results and checked against the
replications written beside them, so the file cannot disagree with (iv).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.reporting.leaderboard import SELECTED
from trading_research.validation.choice import ChoiceError, StratumOutcome
from trading_research.validation.placebo_search import (
    NullDraws,
    SearchNullResult,
    exceedances,
    null_b_shifts,
    p_search,
    placebo_p,
)

NULLS_COLUMNS: tuple[str, ...] = (
    "subject",
    "row",
    "null",
    "stratum",
    "replication",
    "level",
    "key",
    "value",
    "count",
    "replications",
    "runs",
)
NULL_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)
#: Where the entry point writes it (``artefacts.files``).
NULLS_FILE = "experiments/results/short_horizon_2025_nulls.csv"
_NAN = float("nan")


def _row(subject: str, row: str, **values: object) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(NULLS_COLUMNS, _NAN)
    out.update({"subject": subject, "row": row, "null": "", "stratum": "", "key": ""})
    out.update(values)
    return out


def _draw_rows(
    subject: str, name: str, draws: NullDraws, labels: Sequence[int], quantiles: Sequence[float]
) -> list[dict[str, object]]:
    if len(labels) != draws.replications:
        raise ChoiceError(f"{len(labels)} labels for {draws.replications} replications of {name}")
    rows = []
    columns = [*draws.strata, "max"] if len(draws.strata) > 1 else list(draws.strata)
    values = np.column_stack([draws.a, draws.maximum]) if len(draws.strata) > 1 else draws.a
    for label, line in zip(labels, values, strict=True):
        for stratum, value in zip(columns, line, strict=True):
            rows.append(
                _row(
                    subject,
                    "replication",
                    null=name,
                    stratum=stratum,
                    replication=int(label),
                    value=float(value),
                )
            )
    for level in quantiles:
        q = float(np.quantile(draws.maximum, level))
        rows.append(_row(subject, "quantile", null=name, stratum=columns[-1], level=level, value=q))
    return rows


def _checked(result: SearchNullResult, a_null: NullDraws, b_null: NullDraws) -> None:
    if result.cell is None:
        return
    found = (exceedances(a_null.maximum, result.a), exceedances(b_null.maximum, result.a))
    if found != (result.count_a, result.count_b):
        raise ChoiceError(
            f"stratum {result.stratum}: counts {(result.count_a, result.count_b)} in the results, "
            f"{found} in the replications"
        )


def nulls_table(
    results: Mapping[str, SearchNullResult],
    a_null: NullDraws,
    b_null: NullDraws,
    keys: Sequence[str],
    *,
    runs: int = 1,
    shifts: Sequence[int] | None = None,
    global_flip: Mapping[str, StratumOutcome] | None = None,
    quantiles: Sequence[float] = NULL_QUANTILES,
) -> pd.DataFrame:
    """The search's rows: ``results`` from ``placebo_search.admission_iv``,
    the two nulls' draws, the catalogue ``keys`` in the choice's row order,
    Null B's shifts (the registered 7 to 143 by default) and R_D as ``runs``."""
    if a_null.strata != b_null.strata or set(results) != set(a_null.strata):
        raise ChoiceError("the results and both nulls must cover the same strata")
    labels_b = list(null_b_shifts()) if shifts is None else list(shifts)
    rows = _draw_rows("search", "A", a_null, range(a_null.replications), quantiles)
    rows += _draw_rows("search", "B", b_null, labels_b, quantiles)
    for name, result in results.items():
        _checked(result, a_null, b_null)
        key = "" if result.cell is None else keys[result.cell]
        rows.append(_row("search", "observed", stratum=name, key=key, value=result.a))
        if result.a_refit is not None:
            rows.append(_row("search", "refit", stratum=name, value=result.a_refit))
        for null, draws, count, p in (
            ("A", a_null, result.count_a, result.p_a),
            ("B", b_null, result.count_b, result.p_b),
        ):
            rows.append(
                _row("search", "p", null=null, stratum=name, key=key, value=p, count=count,
                     replications=draws.replications, runs=runs)
            )  # fmt: skip
            if result.cell is not None:
                premium = result.a - float(np.median(draws.maximum))
                rows.append(
                    _row("search", "premium", null=null, stratum=name, key=key, value=premium)
                )
    for name, outcome in (global_flip or {}).items():
        key = "" if outcome.cell is None else keys[outcome.cell]
        rows.append(_row("search", "global_flip", stratum=name, key=key, value=outcome.a))
    rows.append(_row("search", "p_search", value=p_search(results, runs), runs=runs))
    return pd.DataFrame(rows, columns=list(NULLS_COLUMNS))


def s25_nulls_table(
    observed_a: float,
    a_null: NullDraws,
    b_null: NullDraws,
    *,
    shifts: Sequence[int] | None = None,
    quantiles: Sequence[float] = NULL_QUANTILES,
) -> pd.DataFrame:
    """S25's D value against its own Null A and Null B: the nulls re-run with
    its cell fixed (``fixed={"C": row}``, one stratum), so no selection. The
    p-values are reported and gate nothing."""
    if len(a_null.strata) != 1 or len(b_null.strata) != 1:
        raise ChoiceError("S25's nulls have one stratum, its fixed cell")
    labels_b = list(null_b_shifts()) if shifts is None else list(shifts)
    rows = _draw_rows("S25", "A", a_null, range(a_null.replications), quantiles)
    rows += _draw_rows("S25", "B", b_null, labels_b, quantiles)
    stratum = a_null.strata[0]
    rows.append(_row("S25", "observed", stratum=stratum, value=observed_a))
    for null, draws in (("A", a_null), ("B", b_null)):
        count = exceedances(draws.maximum, observed_a)
        p = placebo_p(count, draws.replications)
        rows.append(
            _row("S25", "p", null=null, stratum=stratum, value=p, count=count,
                 replications=draws.replications)
        )  # fmt: skip
    return pd.DataFrame(rows, columns=list(NULLS_COLUMNS))


def write_nulls(tables: Sequence[pd.DataFrame], path: Path | str) -> Path:
    """The nulls file: the search's rows and S25's, under the D label."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(list(tables), ignore_index=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# on development data, {SELECTED}; only p_search is quoted, labelled so\n")
        table.loc[:, list(NULLS_COLUMNS)].to_csv(handle, index=False, float_format="%.10g")
    return target


def read_nulls(path: Path | str) -> pd.DataFrame:
    """The nulls file as written."""
    table = pd.read_csv(path, comment="#", keep_default_na=False, na_values=["", "nan", "NaN"])
    for column in ("null", "stratum", "key"):
        table[column] = table[column].fillna("").astype(str)
    return table
