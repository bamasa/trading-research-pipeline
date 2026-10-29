"""The planted-edge, null and placebo checks for a plugin, behind ``plugin_checks``.

The plugin conformance suite (:mod:`trading_research.plugins.conformance`, check
*d*) asks this package for ``plugin_checks(kind=, name=, build=)`` and turns
each returned row into a line of the plugin's page. This module is that hook: a
thin adapter that hands a plugin to :func:`trading_research.synthetic.planted_edge`,
:func:`~trading_research.synthetic.null` and
:func:`~trading_research.synthetic.recover` and reads their reports with the
criteria of ``verify edge`` (:mod:`trading_research.synthetic.verify`).

What runs
---------
Book days with a queue-imbalance edge planted (:data:`SPEC`: ten sessions of
half an hour, 16 bp over 30 s, seed fixed) and the same days with nothing
planted. A **strategy** is fitted on the training days' frames (the plugin's
own view: the book on a one-second grid labelled by the end of each bin, with
``mid``, ``log_mid``, ``spread_bp`` and the prints each row owns, as
:func:`trading_research.plugins.market.build_market` builds it) and decides on
each traded day's frame; a decision at a grid label is placed on the book row
stamped at that label, and the taker executor holds it for the generator's
horizon. A **model** is given the feature frame the edge check offers every
model, restricted to its declared inputs. Other kinds make no decisions, and
the suite does not ask.

The rows
--------
``planted edge``
    Found when the day-clustered t is at least 3 on enough trades and days. A
    plugin that does not read the book's imbalance is not expected to find it,
    so a miss is ``n/a``, never a failure.
``no leak``
    Recovered minus planted on the same trades, on the planted days and on the
    null: an absolute day-clustered t of 3 or more means the plugin's trades
    picked favourable noise, which only a plugin reading the future can do.
``null``
    Nothing significant where nothing was planted.
``placebo``
    Each traded day's decisions moved to the other traded days, read only when
    the planted edge was found: the edge must not survive the move.

Every number is measured on generated data at a stated taker fee and says the
code behaved as stated; nothing here is evidence about a market.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.synthetic.book_days import BookDataset, BookDaysSpec

PASS, FAIL, SKIPPED, NA = "pass", "fail", "skipped", "n/a"

#: The book days every plugin is checked on: small enough for seconds a plugin.
SPEC = BookDaysSpec(days=10, session_s=1800, feature="imbalance", size_bp=16.0)

#: Kinds whose plugins make decisions; the suite marks the others n/a itself.
KINDS = ("strategy", "model")


def plugin_checks(
    *,
    kind: str,
    name: str,  # noqa: ARG001 - the hook's signature; the rows do not repeat it
    build: Callable[[], Any],
) -> list[dict[str, Any]]:
    """Rows of ``check``, ``status``, ``detail`` and ``numbers`` for one plugin."""
    if kind not in KINDS:
        return [_row("planted edge", NA, f"n/a: a {kind} plugin makes no decisions")]
    from trading_research.plugins.protocols import DECLARED, METHODS

    instance = build()
    missing = [a for a in (*DECLARED[kind], *METHODS[kind]) if not hasattr(instance, a)]
    if missing:
        return [_row("planted edge", NA, f"n/a: the protocol is not met (no {', '.join(missing)})")]
    skip = _strategy_skip(instance) if kind == "strategy" else _model_skip(instance)
    if skip:
        return [_row("planted edge", NA, f"n/a: {skip}")]
    return _judge(kind, build)


def _candidate(
    kind: str, build: Callable[[], Any], dataset: BookDataset, frames: dict[date, pd.DataFrame]
) -> Any:
    if kind == "strategy":
        return _StrategyRule(build, dataset)
    model = _ModelInputs(build())
    absent = [c for c in model.inputs if c not in next(iter(frames.values())).columns]
    if absent:
        raise _Skip(f"declares inputs the edge check's feature frame does not offer: {absent}")
    return model


class _Skip(Exception):
    """The plugin cannot be put through the planted-edge check, and why."""


def _judge(kind: str, build: Callable[[], Any]) -> list[dict[str, Any]]:
    from trading_research.synthetic import null, planted_edge
    from trading_research.synthetic.edge import EdgeSettings, day_features, recover
    from trading_research.synthetic.verify import MIN_DAYS, MIN_TRADES, T_BAR

    reports = []
    for dataset, placebo in ((planted_edge(SPEC), True), (null(SPEC), False)):
        frames = {day: day_features(dataset, day) for day in dataset.days}
        try:
            candidate = _candidate(kind, build, dataset, frames)
        except _Skip as exc:
            return [_row("planted edge", NA, f"n/a: {exc}")]
        settings = EdgeSettings(placebo=placebo)
        reports.append(recover(candidate, dataset, settings, frames=frames))
    planted, nothing = reports
    s, n, p = planted.summary, nothing.summary, planted.placebo
    fee = f"taker fee {planted.fee_bp:g} bp a side"
    where = (
        f"{SPEC.days} book days of {SPEC.session_s} s, queue-imbalance edge of "
        f"{SPEC.size_bp:g} bp over {SPEC.horizon_s:g} s, seed {SPEC.seed}, {fee}"
    )

    def enough(summary: dict[str, float]) -> bool:
        return summary.get("trades", 0) >= MIN_TRADES and summary.get("days_traded", 0) >= MIN_DAYS

    rows = []
    t = s.get("recovered_t", math.nan)
    found = enough(s) and _finite(t) and t >= T_BAR
    numbers = _pick(s, "trades", "recovered_bp", "recovered_t", "planted_bp", "net_bp")
    measured = (
        f"recovered {_bp(s.get('recovered_bp', math.nan))} a trade, t {_t(t)}, "
        f"{int(s.get('trades', 0))} trades, net {_bp(s.get('net_bp', math.nan))} ({where})"
    )
    if found:
        rows.append(_row("planted edge", PASS, f"found: {measured}", numbers))
    else:
        rows.append(
            _row(
                "planted edge",
                NA,
                f"n/a: not found ({measured}); expected only of a plugin that reads the "
                "book's imbalance",
                numbers,
            )
        )

    for label, summary in (("no leak (planted days)", s), ("no leak (null days)", n)):
        rt = summary.get("residual_t", math.nan)
        detail = (
            f"recovered minus planted on the same trades {_bp(summary.get('residual_bp', math.nan))}, "
            f"t {_t(rt)}; required absolute t below {T_BAR:g}"
        )
        numbers = _pick(summary, "trades", "residual_bp", "residual_t")
        if not enough(summary) or not _finite(rt):
            rows.append(_row(label, NA, f"n/a: too few trades to read ({detail})", numbers))
        else:
            rows.append(_row(label, PASS if abs(rt) < T_BAR else FAIL, detail, numbers))

    nt = n.get("recovered_t", math.nan)
    quiet = not enough(n) or not _finite(nt) or abs(nt) < T_BAR
    rows.append(
        _row(
            "null",
            PASS if quiet else FAIL,
            f"{_bp(n.get('recovered_bp', math.nan))} a trade, t {_t(nt)}, "
            f"{int(n.get('trades', 0))} trades where nothing was planted; required absolute t below "
            f"{T_BAR:g} (or too few trades to read)",
            _pick(n, "trades", "recovered_bp", "recovered_t"),
        )
    )

    if not found:
        rows.append(_row("placebo", NA, "n/a: no planted edge found to move"))
    else:
        pt, worst = p.get("recovered_t", math.nan), p.get("max_shift_t", math.nan)
        shifted, real = p.get("recovered_bp", math.nan), s.get("recovered_bp", math.nan)
        readable = p.get("trades", 0) >= MIN_TRADES and s.get("test_days", 0) >= MIN_DAYS
        killed = (
            (not readable or not _finite(pt) or pt < T_BAR)
            and (not _finite(shifted) or abs(shifted) <= 0.1 * abs(real))
            and (not _finite(worst) or t > worst)
        )
        rows.append(
            _row(
                "placebo",
                PASS if killed else FAIL,
                f"decisions moved to other days: {_bp(shifted)} a trade, pooled t {_t(pt)}, "
                f"max shift t {_t(worst)}; required pooled t < {T_BAR:g}, absolute mean at most 10% of "
                "the real one, the real t above every shift's",
                {
                    k: p.get(k, math.nan)
                    for k in ("trades", "recovered_bp", "recovered_t", "max_shift_t")
                },
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _StrategyRule:
    """A plugin strategy as the callable rule :func:`recover` trades.

    Fitted once, on the training days' frames in the plugin's own layout, then
    called by :func:`recover` with each traded day's feature frame, which only
    names the day: the decisions come from that day's plugin frame.
    """

    def __init__(self, build: Callable[[], Any], dataset: BookDataset) -> None:
        from trading_research.synthetic.edge import EdgeSettings, split_days

        self.strategy = build()
        self.name = str(getattr(self.strategy, "name", type(self.strategy).__name__))
        self.frames = {day: plugin_frame(dataset, day) for day in dataset.days}
        train, _, _ = split_days(dataset.days, EdgeSettings())
        history = pd.concat([self.frames[d] for d in train], ignore_index=True)
        self.strategy.fit(history)

    def __call__(self, visible: pd.DataFrame) -> np.ndarray:
        from trading_research.features.timed import stamps_ns

        day = date.fromisoformat(str(visible["day"].iloc[0]))
        frame = self.frames[day]
        decisions = np.nan_to_num(np.asarray(self.strategy.decide(frame), dtype=np.float64))
        out = np.zeros(len(visible), dtype=np.float64)
        rows = stamps_ns(visible)
        labels = stamps_ns(frame)
        at = np.searchsorted(rows, labels, side="left")
        inside = at < len(rows)
        exact = np.zeros(len(labels), dtype=bool)
        exact[inside] = rows[at[inside]] == labels[inside]
        out[at[exact]] = np.sign(decisions[exact])
        return out


class _ModelInputs:
    """A plugin model given only the columns it declares (all when it declares none)."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.inputs = tuple(getattr(model, "inputs", ()) or ())
        self.name = str(getattr(model, "name", type(model).__name__))

    def _x(self, x: pd.DataFrame) -> pd.DataFrame:
        return x[list(self.inputs)] if self.inputs else x

    def fit(self, x: pd.DataFrame, y: pd.Series) -> _ModelInputs:
        self.model.fit(self._x(x), y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(self._x(x)), dtype=np.float64)


def plugin_frame(dataset: BookDataset, day: date) -> pd.DataFrame:
    """One generated day in the layout a plugin strategy reads, on a one-second grid."""
    from trading_research.plugins.market import build_market

    spec = dataset.spec
    market = build_market(
        spec.symbol, dataset.book[day], dataset.trades[day], 1, "synthetic", spec.tick, spec.lot
    )
    return market.frame


def _strategy_skip(strategy: Any) -> str | None:
    from trading_research.plugins.market import matches

    probe = plugin_frame_columns()
    absent = [p for p, found in matches(tuple(strategy.inputs), probe).items() if not found]
    if absent:
        return (
            f"declares inputs one instrument's book days do not carry ({', '.join(absent)}); "
            "the planted edge is on a single instrument"
        )
    return None


def plugin_frame_columns() -> list[str]:
    """The columns of :func:`plugin_frame`, without generating a day."""
    from trading_research.plugins.market import DERIVED, PRINT_COLUMNS, level_columns

    return ["timestamp", *level_columns(SPEC.depth), *DERIVED, *PRINT_COLUMNS]


def _model_skip(model: Any) -> str | None:
    if getattr(model, "target", "classification") == "regression":
        return "a regression model; the planted-edge check fits three classes"
    if int(getattr(model, "lookback", 0) or 0) > 0:
        return (
            "the model reads a window of earlier rows; the planted-edge check fits on every "
            "fifth row of the training days"
        )
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _row(
    check: str, status: str, detail: str, numbers: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {"check": check, "status": status, "detail": detail, "numbers": dict(numbers or {})}


def _pick(summary: dict[str, float], *keys: str) -> dict[str, float]:
    return {k: round(float(summary.get(k, math.nan)), 4) for k in keys}


def _finite(value: float) -> bool:
    return bool(math.isfinite(value))


def _bp(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:+.2f} bp"


def _t(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:+.2f}"
