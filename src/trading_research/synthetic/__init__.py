"""Synthetic data with planted, known answers, and the checks that read them.

Every generator here has a typed spec, a seed and a ``truth.json`` of what it
planted, and says in its docstring what its data can and cannot prove:

* :mod:`.book_days` — a ten-level book every 100 ms and the prints, with a
  short-horizon edge planted on queue imbalance or on flow, and its null;
* :mod:`.break_days` — the same stream over whole days with one regime break at
  a known time, and the same days without it;
* :mod:`.cross_section` — a universe of instruments on bars with a planted
  reversion or momentum of known size and half-life, and its null;
* :mod:`.order_flow` — an order-level book for the market-making simulator with
  a planted adverse selection, the queue's truth, and its null;
* :mod:`.funding` — a funding-rate series with a planted persistence, and its null.

:mod:`.verify` runs the repository's own components on them (``trading-research
verify``). The small API below is stable, for code that wants to put a strategy
through the same test without the command line::

    from trading_research import synthetic

    data = synthetic.planted_edge(size_bp=12.0)
    report = synthetic.recover("order_flow", data)
    report.summary["recovered_bp"], report.summary["planted_bp"]
    synthetic.recover("order_flow", synthetic.null()).summary["recovered_t"]

Nothing here reads market data, and nothing computed on it is evidence about a
market.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trading_research.synthetic.book_days import BookDataset, BookDaysSpec
    from trading_research.synthetic.edge import EdgeReport, StrategyLike

__all__ = ["null", "planted_edge", "plugin_checks", "recover"]


def planted_edge(spec: BookDaysSpec | None = None, **changes: Any) -> BookDataset:
    """Book days with a planted edge; the planted parameters are in ``.truth``.

    ``changes`` replace fields of ``spec`` (or of the default spec), for example
    ``planted_edge(feature="flow", size_bp=10.0, days=8)``.
    """
    from trading_research.synthetic.book_days import BookDaysSpec, generate
    from trading_research.synthetic.common import with_changes

    base = with_changes(spec or BookDaysSpec(), **changes)
    return generate(with_changes(base, null=False))


def null(spec: BookDaysSpec | None = None, **changes: Any) -> BookDataset:
    """The same book days with nothing to find: the null of :func:`planted_edge`."""
    from trading_research.synthetic.book_days import BookDaysSpec, generate
    from trading_research.synthetic.common import with_changes

    base = with_changes(spec or BookDaysSpec(), **changes)
    return generate(with_changes(base, null=True))


def recover(strategy: StrategyLike, dataset: BookDataset, **settings: Any) -> EdgeReport:
    """Put ``strategy`` through features, labels, fit, threshold and the taker executor.

    ``strategy`` is a registered model or rule name (``"logistic"``,
    ``"order_flow:flow_imbalance_24"``), an object with ``fit`` and
    ``predict_proba``, or a callable from the feature frame to decisions in
    {-1, 0, +1}. ``settings`` are fields of :class:`.edge.EdgeSettings`. The
    report's ``summary`` holds ``recovered_bp`` and ``planted_bp`` per trade with
    their day-clustered t, ``residual_t`` and ``net_bp``; ``placebo`` the shifted
    decisions.
    """
    from trading_research.synthetic.edge import EdgeSettings
    from trading_research.synthetic.edge import recover as run

    return run(strategy, dataset, EdgeSettings(**settings))


def plugin_checks(*, kind: str, name: str, build: Any) -> list[dict[str, Any]]:
    """The planted-edge, null and placebo checks of one plugin, for ``strategies check``.

    The hook the plugin conformance suite looks for (check *d*): ``build``
    makes a fresh instance of the plugin, and each returned row has ``check``,
    ``status`` (pass, fail or n/a), ``detail`` and ``numbers``. A thin adapter
    over :func:`planted_edge`, :func:`null` and :func:`recover`; see
    :mod:`.plugin_hook` (a module of another name, so that importing it never replaces this function on the package).
    """
    from trading_research.synthetic.plugin_hook import plugin_checks as run

    return run(kind=kind, name=name, build=build)
