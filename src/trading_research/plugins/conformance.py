"""The conformance suite: what a plugin must pass before a study may use it.

Four checks, each a pass, a fail, or a reason it did not apply:

(a) **protocol**
    The declarations exist and are of the right kind (a look-back of zero or
    more rows, a horizon of one or more, inputs the frame carries), the methods
    run, and what they return has the promised shape. Running twice gives the
    same answer, and nothing the plugin is handed is written to.
(b) **no look-ahead**
    For a strategy: the repository's mutation test. Every input after a row is
    changed (prices by :func:`trading_research.validation.fold_leakage.mutate_after_cutoff`,
    everything else reversed in time, see
    :func:`trading_research.plugins.market.mutate_after`) and the decisions up
    to that row, digested by :func:`trading_research.validation.fold_leakage.digest`,
    must not move, at twenty rows of the trading span. Then the declared
    look-back: the decision at a row from only the ``lookback + 1`` rows ending
    there must equal the decision from the whole history (skipped for a
    strategy that declares itself stateful). For a model: the fold-level test
    of the studies (:func:`~trading_research.validation.fold_leakage.check_fold`)
    on its fit, and the same mutation on the rows it predicts. For a feature
    set: truncation invariance and the declared look-back, the two checks of
    :mod:`trading_research.validation.leakage`. For a label: it reads no
    further than its horizon. For an execution: truncation, so that no fill
    reads the market after it.
(c) **execution**
    A strategy, or a model made into one
    (:class:`~trading_research.plugins.model_strategy.ModelStrategy`), is fitted
    on the front of the demo market and its decisions after that are run
    through each execution plugin with the fee table's row applied: one result
    row per execution. An execution plugin runs a fixed reference stream.
(d) **synthetic**
    The planted-edge recovery, the null and the placebo checks of
    ``trading_research.synthetic``, when that package is installed and offers
    ``plugin_checks(kind=, name=, build=)``; otherwise skipped, and said so.

Every number here is measured on synthetic data and is evidence about the
code, never about a market.
"""

from __future__ import annotations

import importlib
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from trading_research.plugins.market import MarketData, demo_market, matches, mutate_after
from trading_research.plugins.protocols import DECLARED, METHODS, PROTOCOLS
from trading_research.plugins.registry import PluginSpec, names, resolve

PASS, FAIL, SKIPPED, NA = "pass", "fail", "skipped", "n/a"

#: Rows at which the mutation and the look-back tests look.
CUTOFFS = 20
#: The share of the demo market a plugin is fitted on before it trades.
TRAIN_SHARE = 0.4
#: The hold, in rows, of a model's decisions and of the executions' reference stream.
CHECK_HOLD = 30


@dataclass
class CheckResult:
    check: str
    status: str
    detail: str = ""
    numbers: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conformance:
    """Every check of one plugin, and the execution table when there is one."""

    spec: PluginSpec
    params: dict[str, Any]
    market: str
    fee_tier: str
    results: list[CheckResult] = field(default_factory=list)
    table: pd.DataFrame | None = None
    when: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))

    @property
    def passed(self) -> bool:
        return all(r.status != FAIL for r in self.results)

    def status(self, check: str) -> str:
        found = [r.status for r in self.results if r.check.startswith(check)]
        if not found:
            return NA
        if FAIL in found:
            return FAIL
        return PASS if PASS in found else found[0]


@dataclass
class Setup:
    """What every check of one run shares: the market and where trading starts."""

    market: MarketData
    fee_tier: str = "base"
    executions: tuple[str, ...] = ()
    train_share: float = TRAIN_SHARE

    @property
    def frame(self) -> pd.DataFrame:
        return self.market.frame

    @property
    def train_end(self) -> int:
        return int(len(self.frame) * self.train_share)

    @property
    def history(self) -> pd.DataFrame:
        return self.frame.iloc[: self.train_end]

    def cutoff_rows(self, start: int, stop: int, count: int = CUTOFFS) -> list[int]:
        start, stop = max(start, 1), min(stop, len(self.frame) - 2)
        if stop <= start:
            return []
        return sorted({int(r) for r in np.linspace(start, stop, count)})


def _error(exc: BaseException) -> str:
    last = traceback.extract_tb(exc.__traceback__)[-1:]
    where = f" ({last[0].name}, line {last[0].lineno})" if last else ""
    return f"{type(exc).__name__}: {exc}{where}"


def _guard(check: str, run: Callable[[], CheckResult]) -> CheckResult:
    try:
        return run()
    except Exception as exc:  # a crash is a failure of the check, with its reason
        return CheckResult(check, FAIL, f"raised {_error(exc)}")


def _frame_digest(frame: pd.DataFrame) -> str:
    from trading_research.validation.fold_leakage import digest

    return digest(frame)


# ---------------------------------------------------------------------------
# (a) protocol, shared part
# ---------------------------------------------------------------------------


def declared_problems(kind: str, instance: Any) -> list[str]:
    """What is missing or malformed in an instance's declarations."""
    problems = []
    subject: Any = instance  # isinstance below would narrow it to ``object``
    if not isinstance(subject, PROTOCOLS[kind]):
        missing = [a for a in (*DECLARED[kind], *METHODS[kind]) if not hasattr(instance, a)]
        problems.append(f"does not meet the {kind} protocol; missing {missing}")
        return problems
    for name in METHODS[kind]:
        if not callable(getattr(instance, name)):
            problems.append(f"{name} is not callable")
    if "lookback" in DECLARED[kind]:
        value = instance.lookback
        if not isinstance(value, int | np.integer) or isinstance(value, bool) or value < 0:
            problems.append(f"lookback must be a whole number of rows >= 0, got {value!r}")
    if "horizon" in DECLARED[kind]:
        value = instance.horizon
        if not isinstance(value, int | np.integer) or isinstance(value, bool) or value < 1:
            problems.append(f"horizon must be a whole number of rows >= 1, got {value!r}")
    if "inputs" in DECLARED[kind]:
        inputs = instance.inputs
        if not isinstance(inputs, tuple) or not all(isinstance(c, str) for c in inputs):
            problems.append(f"inputs must be a tuple of column names, got {inputs!r}")
    return problems


def _decisions_problems(decisions: Any, n: int) -> list[str]:
    values = np.asarray(decisions)
    if values.shape != (n,):
        return [f"decide returned shape {values.shape}, expected ({n},)"]
    if not np.isin(values, (-1, 0, 1)).all():
        return ["decide returned values other than -1, 0 and +1"]
    return []


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _fitted(build: Callable[[], Any], history: pd.DataFrame) -> Any:
    instance = build()
    instance.fit(history.copy())
    return instance


def strategy_protocol(build: Callable[[], Any], setup: Setup) -> CheckResult:
    instance = build()
    problems = declared_problems("strategy", instance)
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems))
    found = matches(instance.inputs, list(setup.frame.columns))
    absent = [p for p, cols in found.items() if not cols]
    if absent:
        problems.append(f"declared inputs the frame does not carry: {absent}")
    history = setup.history.copy()
    before = _frame_digest(history)
    instance.fit(history)
    if _frame_digest(history) != before:
        problems.append("fit wrote into the history it was given")
    frame = setup.frame.copy()
    before = _frame_digest(frame)
    first = np.asarray(instance.decide(frame))
    if _frame_digest(frame) != before:
        problems.append("decide wrote into the frame it was given")
    problems += _decisions_problems(first, len(frame))
    if not problems and not np.array_equal(first, np.asarray(instance.decide(frame))):
        problems.append("decide gave different decisions on the same frame")
    trading = first[setup.train_end :] if not problems else np.zeros(0)
    numbers = {
        "lookback": int(instance.lookback),
        "horizon": int(instance.horizon),
        "stateful": bool(getattr(instance, "stateful", False)),
        "inputs": ", ".join(instance.inputs),
        "long_rows": int((trading == 1).sum()),
        "short_rows": int((trading == -1).sum()),
    }
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems), numbers)
    return CheckResult("a. protocol", PASS, "declarations, fit and decide as promised", numbers)


def mutation_lookahead(
    decide: Callable[[pd.DataFrame], np.ndarray], setup: Setup, start: int, check: str
) -> CheckResult:
    """Decisions up to each cutoff row, before and after every later input changed."""
    from trading_research.validation.fold_leakage import FoldLeakResult, digest

    frame = setup.frame
    full = np.asarray(decide(frame))
    rows = setup.cutoff_rows(start, len(frame) - 2)
    leaks: list[str] = []
    changed_rows = 0
    for k, row in enumerate(rows):
        mutated = np.asarray(decide(mutate_after(frame, row, seed=k)))
        result = FoldLeakResult(
            f"row {row}",
            pd.Timestamp(frame["timestamp"].iloc[row]),
            {"decisions": digest(full[: row + 1])},
            {"decisions": digest(mutated[: row + 1])},
        )
        if result.leaks:
            moved = np.flatnonzero(full[: row + 1] != mutated[: row + 1])
            changed_rows += len(moved)
            leaks.append(
                f"cutoff row {row}: {len(moved)} decision(s) moved, first {row - moved[0]} row(s) before it"
            )
    numbers = {"cutoffs": len(rows), "cutoffs_moved": len(leaks), "decisions_moved": changed_rows}
    if not rows:
        return CheckResult(check, FAIL, "the frame is too short to place a cutoff", numbers)
    if leaks:
        return CheckResult(
            check,
            FAIL,
            "decisions changed when only later inputs changed: " + "; ".join(leaks[:3]),
            numbers,
        )
    return CheckResult(
        check,
        PASS,
        f"decisions up to {len(rows)} cutoff rows unchanged when every later input was changed",
        numbers,
    )


def declared_lookback(
    decide: Callable[[pd.DataFrame], np.ndarray], lookback: int, setup: Setup, check: str
) -> CheckResult:
    """The decision at a row from only its declared window equals the batch decision."""
    frame = setup.frame
    full = np.asarray(decide(frame))
    start = max(setup.train_end, lookback)
    active = [r for r in np.flatnonzero(full) if r >= start]
    rows = setup.cutoff_rows(start, len(frame) - 1, CUTOFFS // 2)
    if active:
        picks = np.linspace(0, len(active) - 1, min(len(active), CUTOFFS // 2)).astype(int)
        rows = sorted(set(rows) | {int(active[i]) for i in picks})
    wrong = []
    for row in rows:
        window = frame.iloc[row - lookback : row + 1].reset_index(drop=True)
        value = int(np.asarray(decide(window))[-1])
        if value != int(full[row]):
            wrong.append(
                f"row {row}: {int(full[row])} from the history, {value} from {lookback + 1} rows"
            )
    numbers = {
        "rows_tested": len(rows),
        "rows_with_a_decision": len([r for r in rows if full[r]]),
        "mismatches": len(wrong),
    }
    if wrong:
        return CheckResult(
            check,
            FAIL,
            f"needs more history than the declared {lookback} rows: " + "; ".join(wrong[:3]),
            numbers,
        )
    return CheckResult(
        check,
        PASS,
        f"decisions from the declared {lookback + 1}-row window match at {len(rows)} rows",
        numbers,
    )


def run_executions(
    decisions: np.ndarray, hold: int, setup: Setup, executions: Sequence[str]
) -> tuple[CheckResult, pd.DataFrame]:
    """Each execution plugin on the same decisions; one summary row each."""
    rows, failures = [], []
    for name in executions:
        try:
            execution = resolve("execution", name, fees=setup.fee_tier)
            result = execution.execute(decisions, setup.market, hold=hold)
            missing = [c for c in _attempt_columns() if c not in result.attempts.columns]
            if missing:
                raise ValueError(f"attempts lack {missing}")
            rows.append({"execution": name, "fee_tier": setup.fee_tier, **result.summary})
        except Exception as exc:
            failures.append(f"{name}: {_error(exc)}")
            rows.append({"execution": name, "fee_tier": setup.fee_tier, "error": _error(exc)})
    table = pd.DataFrame(rows)
    decided = int((np.asarray(decisions) != 0).sum())
    numbers = {"decision_rows": decided, "hold_rows": hold, "executions": len(executions)}
    if failures:
        return CheckResult("c. execution", FAIL, "; ".join(failures), numbers), table
    note = "" if decided else "; no decision in the trading span, so every mode ran on nothing"
    return (
        CheckResult(
            "c. execution",
            PASS,
            f"ran through {', '.join(executions)} with the {setup.fee_tier} fee tier{note}",
            numbers,
        ),
        table,
    )


def _attempt_columns() -> tuple[str, ...]:
    from trading_research.pipeline.execution import ATTEMPT_COLUMNS

    return ATTEMPT_COLUMNS


def synthetic_checks(spec: PluginSpec, build: Callable[[], Any]) -> list[CheckResult]:
    """The planted-edge, null and placebo checks, from ``trading_research.synthetic``."""
    try:
        module = importlib.import_module("trading_research.synthetic")
    except ModuleNotFoundError as exc:
        if exc.name in ("trading_research.synthetic", "trading_research"):
            return [
                CheckResult("d. synthetic", SKIPPED, "skipped: synthetic checks not on this branch")
            ]
        raise
    hook = getattr(module, "plugin_checks", None)
    if hook is None:
        return [
            CheckResult(
                "d. synthetic",
                SKIPPED,
                "skipped: trading_research.synthetic offers no plugin_checks(kind=, name=, build=) hook",
            )
        ]
    out = []
    for row in hook(kind=spec.kind, name=spec.name, build=build):
        out.append(
            CheckResult(
                f"d. synthetic: {row.get('check', 'check')}",
                str(row.get("status", FAIL)),
                str(row.get("detail", "")),
                dict(row.get("numbers", {})),
            )
        )
    return out or [CheckResult("d. synthetic", SKIPPED, "skipped: the hook returned no check")]


def check_strategy(spec: PluginSpec, params: dict[str, Any], setup: Setup) -> Conformance:
    out = Conformance(spec, params, setup.market.source, setup.fee_tier)

    def build() -> Any:
        return spec.build(**params)

    protocol = _guard("a. protocol", lambda: strategy_protocol(build, setup))
    out.results.append(protocol)
    if protocol.status == FAIL:
        out.results.append(
            CheckResult("b. no look-ahead", SKIPPED, "skipped: the protocol is not met")
        )
        out.results.append(CheckResult("c. execution", SKIPPED, "skipped: the protocol is not met"))
        out.results += synthetic_checks(spec, build)
        return out
    fitted = _fitted(build, setup.history)
    out.results.append(
        _guard(
            "b. no look-ahead",
            lambda: mutation_lookahead(fitted.decide, setup, setup.train_end, "b. no look-ahead"),
        )
    )
    if getattr(fitted, "stateful", False):
        out.results.append(
            CheckResult(
                "b. declared look-back",
                NA,
                "n/a: the strategy declares itself stateful; its decisions read the path since the frame began",
            )
        )
    else:
        out.results.append(
            _guard(
                "b. declared look-back",
                lambda: declared_lookback(
                    fitted.decide, int(fitted.lookback), setup, "b. declared look-back"
                ),
            )
        )
    decisions = np.asarray(fitted.decide(setup.frame)).copy()
    decisions[: setup.train_end] = 0
    result, table = run_executions(decisions, int(fitted.horizon), setup, setup.executions)
    out.results.append(result)
    out.table = table
    out.results += _guard_list(lambda: synthetic_checks(spec, build))
    return out


def _guard_list(run: Callable[[], list[CheckResult]]) -> list[CheckResult]:
    try:
        return run()
    except Exception as exc:
        return [CheckResult("d. synthetic", FAIL, f"raised {_error(exc)}")]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _model_strategy(build: Callable[[], Any], setup: Setup) -> Any:
    from trading_research.backtest.costs import fee_tier
    from trading_research.plugins.model_strategy import ModelStrategy

    return ModelStrategy(build(), horizon=CHECK_HOLD, fees=fee_tier(setup.fee_tier))


def model_protocol(build: Callable[[], Any], setup: Setup) -> CheckResult:
    from trading_research.plugins.model_strategy import model_features, model_label

    instance = build()
    problems = declared_problems("model", instance)
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems))
    x, _ = model_features(setup.history, tuple(instance.inputs))
    absent = [c for c in instance.inputs if c not in x.columns]
    if absent:
        problems.append(f"declared inputs no feature set provides: {absent}")
    target = getattr(instance, "target", "classification")
    y = model_label(setup.history, target, CHECK_HOLD)
    ok = np.isfinite(x.to_numpy(dtype="float64")).all(axis=1) & np.isfinite(y.to_numpy())
    train_x, train_y = x.loc[ok], y.loc[ok]
    before = _frame_digest(train_x)
    instance.fit(train_x, train_y)
    if _frame_digest(train_x) != before:
        problems.append("fit wrote into the inputs it was given")
    proba = np.asarray(instance.predict_proba(train_x), dtype="float64")
    if proba.shape != (len(train_x), 3):
        problems.append(f"predict_proba returned shape {proba.shape}, expected ({len(train_x)}, 3)")
    else:
        rows = proba[np.isfinite(proba).all(axis=1)]
        if len(rows) == 0:
            problems.append("predict_proba returned no finite row")
        elif (
            (rows < -1e-9).any()
            or (rows > 1 + 1e-9).any()
            or not np.allclose(rows.sum(axis=1), 1.0, atol=1e-6)
        ):
            problems.append("predict_proba rows are not probabilities summing to one")
        again = np.asarray(instance.predict_proba(train_x), dtype="float64")
        # Equal up to the order of a sum: a threaded forest adds its trees in
        # whatever order its threads finish.
        same = np.allclose(
            np.nan_to_num(proba, nan=-1), np.nan_to_num(again, nan=-1), rtol=1e-9, atol=1e-12
        )
        if not same:
            problems.append("predict_proba gave different answers on the same rows")
    numbers = {
        "inputs": ", ".join(instance.inputs) or "every column given",
        "lookback": int(instance.lookback),
        "target": target,
        "training_rows": len(train_x),
        "features": int(x.shape[1]),
    }
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems), numbers)
    return CheckResult(
        "a. protocol", PASS, "declarations, fit and predict_proba as promised", numbers
    )


def model_fold_test(build: Callable[[], Any], setup: Setup) -> CheckResult:
    """The studies' fold-level test: the fit, digested, ignores data after its cutoff."""
    from trading_research.plugins.model_strategy import model_features, model_label
    from trading_research.validation.fold_leakage import check_fold

    end = setup.train_end - CHECK_HOLD
    instance_inputs = tuple(build().inputs)
    target = getattr(build(), "target", "classification")

    def fit(frame: pd.DataFrame) -> dict[str, Any]:
        model = build()
        history = frame.iloc[: setup.train_end]
        x, _ = model_features(history, instance_inputs)
        y = model_label(history, target, CHECK_HOLD)
        rows = np.zeros(len(history), dtype=bool)
        rows[:end] = True
        ok = rows & np.isfinite(x.to_numpy(dtype="float64")).all(axis=1) & np.isfinite(y.to_numpy())
        model.fit(x.loc[ok], y.loc[ok])
        return {"predictions_on_training_rows": np.asarray(model.predict_proba(x.loc[ok]))}

    prices = [
        c
        for c in setup.frame.columns
        if "_price_" in c or (c.startswith("peer_") and c.endswith("_mid"))
    ]
    cutoff = setup.frame["timestamp"].iloc[setup.train_end - 1]
    result = check_fold(fit, setup.frame, cutoff, price_columns=prices, fold="training")
    numbers: dict[str, Any] = {"cutoff": str(result.cutoff), "artefacts": len(result.digests)}
    if result.leaks:
        # Say which failure it is: the fold test counts a fit that cannot be
        # recorded by sha256 as failing, whether or not anything leaked.
        from trading_research.validation.fold_leakage import digest, mutate_after_cutoff

        first, second = fit(setup.frame), fit(setup.frame)
        name = "predictions_on_training_rows"
        reproducible = digest(first[name]) == digest(second[name])
        changed = fit(mutate_after_cutoff(setup.frame, cutoff, price_columns=prices))[name]
        moved = float(np.nanmax(np.abs(first[name] - changed)))
        numbers.update(reproducible_to_the_bit=reproducible, largest_change=moved)
        why = (
            "the fit moved when only data after its cutoff changed"
            if reproducible
            else "the fit is not reproducible to the bit on the same rows, so its artefacts "
            f"cannot be recorded by sha256 (largest change after the mutation {moved:.1e}, "
            "the same order as between two fits of the same rows)"
        )
        return CheckResult("b. no look-ahead: fit", FAIL, f"{result}: {why}", numbers)
    return CheckResult(
        "b. no look-ahead: fit",
        PASS,
        "fitted artefacts unchanged when prices after the training cutoff changed (check_fold)",
        numbers,
    )


def _fitted_model(build: Callable[[], Any], setup: Setup) -> tuple[Any, pd.DataFrame, np.ndarray]:
    """A model fitted on the history, every row's inputs, and its predictions on them."""
    from trading_research.plugins.common import predict_rows
    from trading_research.plugins.model_strategy import model_features, model_label

    model = build()
    x_all, _ = model_features(setup.frame, tuple(model.inputs))
    y = model_label(setup.history, getattr(model, "target", "classification"), CHECK_HOLD)
    x = x_all.iloc[: setup.train_end]
    ok = np.isfinite(x.to_numpy(dtype="float64")).all(axis=1) & np.isfinite(y.to_numpy())
    model.fit(x.loc[ok], y.loc[ok])
    return model, x_all, predict_rows(model, x_all)


def model_lookback(build: Callable[[], Any], setup: Setup) -> CheckResult:
    """A prediction from only the declared ``lookback + 1`` input rows equals the batch one."""
    from trading_research.plugins.common import predict_rows

    model, x_all, full = _fitted_model(build, setup)
    lookback = int(model.lookback)
    finite = np.isfinite(x_all.to_numpy(dtype="float64")).all(axis=1)
    first = int(np.argmax(finite)) + lookback
    rows = setup.cutoff_rows(max(setup.train_end, first), len(x_all) - 1, CUTOFFS // 2)
    wrong = []
    for row in rows:
        part = predict_rows(model, x_all.iloc[row - lookback : row + 1])[-1]
        if not np.allclose(part, full[row], rtol=RTOL, atol=1e-9, equal_nan=True):
            wrong.append(f"row {row}")
    numbers = {"rows_tested": len(rows), "mismatches": len(wrong)}
    if wrong:
        return CheckResult(
            "b. declared look-back",
            FAIL,
            f"predictions need more than the declared {lookback} earlier rows: {wrong[:3]}",
            numbers,
        )
    return CheckResult(
        "b. declared look-back",
        PASS,
        f"predictions from the declared {lookback + 1}-row window match at {len(rows)} rows",
        numbers,
    )


def model_prediction_test(build: Callable[[], Any], setup: Setup) -> CheckResult:
    """Predictions at a row ignore every input row after it (reversed and tripled)."""
    from trading_research.plugins.common import predict_rows

    model, x_all, full = _fitted_model(build, setup)
    rows = setup.cutoff_rows(setup.train_end, len(x_all) - 2, CUTOFFS // 2)
    leaks = []
    for row in rows:
        mutated = x_all.copy()
        after = slice(row + 1, len(mutated))
        mutated.iloc[after] = mutated.iloc[after].to_numpy()[::-1] * 3.0
        part = predict_rows(model, mutated)
        # Compared as values, not bits: a threaded forest sums its trees in
        # whatever order its threads finish, and that is not look-ahead.
        same = np.isclose(full[: row + 1], part[: row + 1], rtol=1e-9, atol=1e-12, equal_nan=True)
        moved = np.flatnonzero(~same.all(axis=1))
        if len(moved):
            leaks.append(f"cutoff row {row}: {len(moved)} prediction(s) moved")
    numbers = {"cutoffs": len(rows), "cutoffs_moved": len(leaks)}
    if leaks:
        return CheckResult(
            "b. no look-ahead: predict",
            FAIL,
            "predictions changed when only later rows changed: " + "; ".join(leaks[:3]),
            numbers,
        )
    return CheckResult(
        "b. no look-ahead: predict",
        PASS,
        f"predictions up to {len(rows)} cutoff rows unchanged when every later input row changed",
        numbers,
    )


def check_model(spec: PluginSpec, params: dict[str, Any], setup: Setup) -> Conformance:
    out = Conformance(spec, params, setup.market.source, setup.fee_tier)

    def build() -> Any:
        return spec.build(**params)

    protocol = _guard("a. protocol", lambda: model_protocol(build, setup))
    out.results.append(protocol)
    if protocol.status == FAIL:
        out.results.append(
            CheckResult("b. no look-ahead", SKIPPED, "skipped: the protocol is not met")
        )
        out.results.append(CheckResult("c. execution", SKIPPED, "skipped: the protocol is not met"))
        out.results += _guard_list(lambda: synthetic_checks(spec, build))
        return out
    out.results.append(_guard("b. no look-ahead: fit", lambda: model_fold_test(build, setup)))
    out.results.append(
        _guard("b. no look-ahead: predict", lambda: model_prediction_test(build, setup))
    )
    out.results.append(_guard("b. declared look-back", lambda: model_lookback(build, setup)))

    def execution() -> CheckResult:
        strategy = _model_strategy(build, setup)
        strategy.fit(setup.history)
        decisions = np.asarray(strategy.decide(setup.frame)).copy()
        decisions[: setup.train_end] = 0
        result, table = run_executions(decisions, CHECK_HOLD, setup, setup.executions)
        result.numbers["confidence_chosen_on_validation"] = round(float(strategy.confidence), 3)
        out.table = table
        return result

    out.results.append(_guard("c. execution", execution))
    out.results += _guard_list(lambda: synthetic_checks(spec, build))
    return out


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

RTOL, ATOL = 1e-7, 1e-10


def _diverged(full: pd.DataFrame, part: pd.DataFrame) -> list[str]:
    """Columns whose values differ, NaN in both counting as equal (validation.leakage's rule)."""
    out = []
    for column in full.columns:
        a = full[column].to_numpy(dtype="float64")
        b = (
            part[column].to_numpy(dtype="float64")
            if column in part.columns
            else np.full(len(a), np.nan)
        )
        same = np.isclose(a, b, rtol=RTOL, atol=ATOL) | (np.isnan(a) & np.isnan(b))
        if not same.all():
            out.append(column)
    return out


def features_protocol(build: Callable[[], Any], setup: Setup) -> CheckResult:
    instance = build()
    problems = declared_problems("features", instance)
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems))
    book, trades = setup.market.rows.copy(), setup.market.trades.copy()
    before = (_frame_digest(book), _frame_digest(trades))
    out = instance.compute(book, trades)
    if (_frame_digest(book), _frame_digest(trades)) != before:
        problems.append("compute wrote into the book or the trades it was given")
    if not isinstance(out, pd.DataFrame):
        problems.append(f"compute returned {type(out).__name__}, not a frame")
    else:
        if len(out) != len(book) or not out.index.equals(book.index):
            problems.append("compute's frame is not on the book's index")
        bad = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
        if bad:
            problems.append(f"columns that are not numbers: {bad[:5]}")
        if not problems and _frame_digest(out) != _frame_digest(instance.compute(book, trades)):
            problems.append("compute gave a different frame on the same data")
    numbers = {
        "lookback": int(instance.lookback),
        "columns": int(out.shape[1]) if isinstance(out, pd.DataFrame) else 0,
    }
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems), numbers)
    return CheckResult("a. protocol", PASS, "declarations and compute as promised", numbers)


def _trades_through(trades: pd.DataFrame, stamp: pd.Timestamp) -> pd.DataFrame:
    return trades[pd.DatetimeIndex(trades["timestamp"]) <= stamp].copy()


def features_truncation(build: Callable[[], Any], setup: Setup) -> CheckResult:
    instance = build()
    book, trades = setup.market.rows, setup.market.trades
    full = instance.compute(book, trades)
    cuts = [int(len(book) * share) for share in (0.35, 0.5, 0.6, 0.75, 0.9)]
    leaking: dict[str, int] = {}
    for cut in cuts:
        stamp = pd.Timestamp(book["timestamp"].iloc[cut - 1])
        part = instance.compute(book.iloc[:cut].copy(), _trades_through(trades, stamp))
        for column in _diverged(full.iloc[:cut], part):
            leaking[column] = leaking.get(column, 0) + 1
    numbers = {"cuts": len(cuts), "columns_moved": len(leaking)}
    if leaking:
        return CheckResult(
            "b. no look-ahead",
            FAIL,
            f"values before a cut moved when the data after it was removed: {sorted(leaking)[:6]}",
            numbers,
        )
    return CheckResult(
        "b. no look-ahead",
        PASS,
        f"every column unchanged before each of {len(cuts)} cuts when the data after it was removed",
        numbers,
    )


def features_lookback(build: Callable[[], Any], setup: Setup) -> CheckResult:
    instance = build()
    book, trades = setup.market.rows, setup.market.trades
    lookback = int(instance.lookback)
    full = instance.compute(book, trades)
    stamps = pd.DatetimeIndex(book["timestamp"])
    rows = setup.cutoff_rows(max(lookback + 1, setup.train_end), len(book) - 1, 8)
    wrong: dict[str, int] = {}
    for row in rows:
        start = row - lookback
        window = book.iloc[start : row + 1].copy()
        held = (
            trades[
                (pd.DatetimeIndex(trades["timestamp"]) > stamps[start - 1])
                & (pd.DatetimeIndex(trades["timestamp"]) <= stamps[row])
            ]
            if start > 0
            else _trades_through(trades, stamps[row])
        )
        part = instance.compute(window, held.copy()).iloc[[-1]]
        for column in _diverged(full.iloc[[row]], part.set_axis(full.index[[row]])):
            wrong[column] = wrong.get(column, 0) + 1
    numbers = {"rows_tested": len(rows), "columns_needing_more": len(wrong)}
    if not rows:
        return CheckResult(
            "b. declared look-back",
            FAIL,
            "the data is shorter than the declared look-back",
            numbers,
        )
    if wrong:
        return CheckResult(
            "b. declared look-back",
            FAIL,
            f"columns that need more than the declared {lookback} rows: {sorted(wrong)[:6]}"
            + (f" and {len(wrong) - 6} more" if len(wrong) > 6 else ""),
            numbers,
        )
    return CheckResult(
        "b. declared look-back",
        PASS,
        f"every column from the declared {lookback + 1}-row window matches at {len(rows)} rows",
        numbers,
    )


def check_features(spec: PluginSpec, params: dict[str, Any], setup: Setup) -> Conformance:
    out = Conformance(spec, params, setup.market.source, setup.fee_tier)

    def build() -> Any:
        return spec.build(**params)

    protocol = _guard("a. protocol", lambda: features_protocol(build, setup))
    out.results.append(protocol)
    if protocol.status == FAIL:
        out.results.append(
            CheckResult("b. no look-ahead", SKIPPED, "skipped: the protocol is not met")
        )
    else:
        out.results.append(_guard("b. no look-ahead", lambda: features_truncation(build, setup)))
        out.results.append(_guard("b. declared look-back", lambda: features_lookback(build, setup)))
    out.results.append(CheckResult("c. execution", NA, "n/a: a feature set makes no decision"))
    out.results.append(CheckResult("d. synthetic", NA, "n/a: the synthetic checks test decisions"))
    return out


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def label_protocol(build: Callable[[], Any], setup: Setup) -> CheckResult:
    instance = build()
    problems = declared_problems("label", instance)
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems))
    if instance.kind not in ("classification", "regression"):
        problems.append(f"kind must be classification or regression, got {instance.kind!r}")
    frame = setup.frame.copy()
    before = _frame_digest(frame)
    values = instance.compute(frame)
    if _frame_digest(frame) != before:
        problems.append("compute wrote into the frame it was given")
    if not isinstance(values, pd.Series) or len(values) != len(frame):
        problems.append("compute must return a Series with one value per row")
    else:
        array = values.to_numpy(dtype="float64")
        if np.isfinite(array[-1]):
            problems.append("the last row has a label, though nothing after it is in the data")
        known = array[np.isfinite(array)]
        if instance.kind == "classification" and not np.isin(known, (-1.0, 0.0, 1.0)).all():
            problems.append("a classification label has values other than -1, 0 and +1")
        if len(known) == 0:
            problems.append("no row is labelled")
    numbers = {"horizon": int(instance.horizon), "kind": instance.kind}
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems), numbers)
    numbers["unlabelled_rows"] = int(values.isna().sum())
    return CheckResult(
        "a. protocol",
        PASS,
        "declarations and compute as promised; the last row unlabelled",
        numbers,
    )


def label_reach(build: Callable[[], Any], setup: Setup) -> CheckResult:
    """No further than the horizon: rows after ``t + horizon`` changed, labels up to ``t`` kept."""
    from trading_research.validation.fold_leakage import digest

    instance = build()
    frame = setup.frame
    horizon = int(instance.horizon)
    full = instance.compute(frame).to_numpy(dtype="float64")
    rows = setup.cutoff_rows(setup.train_end, len(frame) - horizon - 3)
    too_far, reaches = [], 0
    for k, row in enumerate(rows):
        later = instance.compute(mutate_after(frame, row + horizon, seed=k)).to_numpy(
            dtype="float64"
        )
        if digest(np.nan_to_num(full[: row + 1], nan=-9)) != digest(
            np.nan_to_num(later[: row + 1], nan=-9)
        ):
            too_far.append(f"row {row}")
        nearer = instance.compute(mutate_after(frame, row, seed=k)).to_numpy(dtype="float64")
        reaches += int(
            not np.array_equal(
                np.nan_to_num(full[: row + 1], nan=-9), np.nan_to_num(nearer[: row + 1], nan=-9)
            )
        )
    numbers = {
        "cutoffs": len(rows),
        "reads_past_horizon": len(too_far),
        "cutoffs_reading_forward": reaches,
    }
    if too_far:
        return CheckResult(
            "b. no look-ahead",
            FAIL,
            f"labels changed when only rows after the horizon changed: {too_far[:3]}",
            numbers,
        )
    return CheckResult(
        "b. no look-ahead",
        PASS,
        f"labels up to each of {len(rows)} rows unchanged when rows beyond the {horizon}-row horizon changed",
        numbers,
    )


def check_label(spec: PluginSpec, params: dict[str, Any], setup: Setup) -> Conformance:
    out = Conformance(spec, params, setup.market.source, setup.fee_tier)

    def build() -> Any:
        return spec.build(**params)

    protocol = _guard("a. protocol", lambda: label_protocol(build, setup))
    out.results.append(protocol)
    if protocol.status == FAIL:
        out.results.append(
            CheckResult("b. no look-ahead", SKIPPED, "skipped: the protocol is not met")
        )
    else:
        out.results.append(_guard("b. no look-ahead", lambda: label_reach(build, setup)))
    out.results.append(CheckResult("c. execution", NA, "n/a: a label makes no decision"))
    out.results.append(CheckResult("d. synthetic", NA, "n/a: the synthetic checks test decisions"))
    return out


# ---------------------------------------------------------------------------
# Executions
# ---------------------------------------------------------------------------


def reference_decisions(setup: Setup, every: int = 97) -> np.ndarray:
    """A fixed stream: a decision every ``every`` rows after training, sides alternating."""
    out = np.zeros(len(setup.frame), dtype=np.int64)
    rows = np.arange(setup.train_end, len(out), every)
    out[rows] = np.where(np.arange(len(rows)) % 2 == 0, 1, -1)
    return out


def execution_protocol(build: Callable[[], Any], setup: Setup) -> CheckResult:
    from trading_research.backtest.costs import FeeTier
    from trading_research.pipeline.execution import ExecutionResult

    instance = build()
    problems = declared_problems("execution", instance)
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems))
    if not isinstance(instance.fill_rule, str) or not instance.fill_rule.strip():
        problems.append("fill_rule must say, in words, how a decision becomes a fill")
    if not isinstance(instance.fees, FeeTier):
        problems.append("fees must be a row of the fee table (backtest.costs.FeeTier)")
    decisions = reference_decisions(setup)
    result = instance.execute(decisions.copy(), setup.market, hold=CHECK_HOLD)
    if not isinstance(result, ExecutionResult):
        problems.append(f"execute returned {type(result).__name__}, not an ExecutionResult")
    else:
        missing = [c for c in _attempt_columns() if c not in result.attempts.columns]
        if missing:
            problems.append(f"attempts lack {missing}")
        for key in ("attempts", "fills", "net_bp"):
            if key not in result.summary:
                problems.append(f"the summary lacks {key}")
    numbers = {
        "fill_rule": getattr(instance, "fill_rule", ""),
        "fee_tier": getattr(instance.fees, "name", ""),
    }
    if problems:
        return CheckResult("a. protocol", FAIL, "; ".join(problems), numbers)
    numbers["reference_attempts"] = len(result.attempts)
    return CheckResult("a. protocol", PASS, "declarations and execute as promised", numbers)


def execution_truncation(build: Callable[[], Any], setup: Setup) -> CheckResult:
    """Attempts settled before a cut are the same with the market after it removed.

    Settled means decided more than the hold, the passive entry's timeout and a
    few rows before the cut: an execution whose fill or exit reads the market
    later than that fails.
    """
    from trading_research.validation.fold_leakage import digest

    instance = build()
    decisions = reference_decisions(setup)
    full = instance.execute(decisions.copy(), setup.market, hold=CHECK_HOLD).attempts
    # An attempt is settled once its entry may have waited out the passive timeout
    # (60 rows) and its hold has run: nothing after that may change it.
    margin = pd.Timedelta(float((CHECK_HOLD + 60 + 5) * setup.market.grid_seconds + 1), unit="s")
    stamps = setup.frame["timestamp"]
    rows = setup.cutoff_rows(setup.train_end + 600, len(stamps) - 2, 6)
    columns = ["ts", "filled", "gross_bp", "cost_bp", "net_bp", "path"]
    moved, compared = [], 0
    for row in rows:
        stamp = pd.Timestamp(stamps.iloc[row])
        cut = setup.market.until(stamp)
        part = instance.execute(decisions[: len(cut.frame)].copy(), cut, hold=CHECK_HOLD).attempts
        limit = (stamp - margin).value

        def settled(table: pd.DataFrame, limit: int = limit) -> pd.DataFrame:
            kept = table[table["ts"].to_numpy(dtype=np.int64) <= limit][columns]
            return kept.sort_values("ts").reset_index(drop=True)

        a, b = settled(full), settled(part)
        compared += len(a)
        if len(a) != len(b) or digest(a.round(9)) != digest(b.round(9)):
            moved.append(f"cut at row {row}")
    numbers = {"cuts": len(rows), "attempts_compared": compared, "cuts_moved": len(moved)}
    if moved:
        return CheckResult(
            "b. no look-ahead",
            FAIL,
            f"settled attempts changed when the market after them was removed: {moved}",
            numbers,
        )
    return CheckResult(
        "b. no look-ahead",
        PASS,
        f"{compared} settled attempts unchanged when the market after each of {len(rows)} cuts was removed",
        numbers,
    )


def check_execution(spec: PluginSpec, params: dict[str, Any], setup: Setup) -> Conformance:
    out = Conformance(spec, params, setup.market.source, setup.fee_tier)
    merged = {"fees": setup.fee_tier, **params}

    def build() -> Any:
        return spec.build(**merged)

    protocol = _guard("a. protocol", lambda: execution_protocol(build, setup))
    out.results.append(protocol)
    if protocol.status == FAIL:
        out.results.append(
            CheckResult("b. no look-ahead", SKIPPED, "skipped: the protocol is not met")
        )
        out.results.append(CheckResult("c. execution", SKIPPED, "skipped: the protocol is not met"))
        return out
    out.results.append(_guard("b. no look-ahead", lambda: execution_truncation(build, setup)))

    def reference() -> CheckResult:
        result = build().execute(reference_decisions(setup), setup.market, hold=CHECK_HOLD)
        out.table = pd.DataFrame(
            [
                {
                    "execution": spec.name,
                    "fee_tier": setup.fee_tier,
                    "decisions": "reference stream",
                    **result.summary,
                }
            ]
        )
        return CheckResult(
            "c. execution",
            PASS,
            f"the reference stream (a decision every 97 rows after training, sides alternating, hold {CHECK_HOLD} rows) ran",
            {"attempts": len(result.attempts)},
        )

    out.results.append(_guard("c. execution", reference))
    out.results.append(CheckResult("d. synthetic", NA, "n/a: the synthetic checks test decisions"))
    return out


# ---------------------------------------------------------------------------
# One entry point
# ---------------------------------------------------------------------------

CHECKS: dict[str, Callable[[PluginSpec, dict[str, Any], Setup], Conformance]] = {
    "strategy": check_strategy,
    "model": check_model,
    "features": check_features,
    "label": check_label,
    "execution": check_execution,
}


def make_setup(
    *,
    executions: Sequence[str] | str = "all",
    fee_tier: str = "base",
    market: MarketData | None = None,
    steps: int = 72_000,
    seed: int = 20260101,
) -> Setup:
    chosen = tuple(names("execution")) if executions == "all" else tuple(executions)
    known = set(names("execution"))
    unknown = [e for e in chosen if e not in known]
    if unknown:
        raise ValueError(f"unknown execution(s) {unknown}; registered: {sorted(known)}")
    return Setup(market or demo_market(steps=steps, seed=seed), fee_tier, chosen)


def check(spec: PluginSpec, setup: Setup, params: dict[str, Any] | None = None) -> Conformance:
    """Every check of one plugin, built with its check parameters and ``params``."""
    merged = {**spec.check_params, **(params or {})}
    return CHECKS[spec.kind](spec, merged, setup)
