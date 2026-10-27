"""``trp backtest``: step 6, execution: crossing, resting at the touch, or quoting both sides.

The commands are the flat ``backtest`` (taker, from predictions),
``mm-backtest`` (the market maker in event time), ``exit-search`` and
``retrain-search`` of :mod:`trading_research.cli`, unchanged. A taker, passive
or maker run on a day range from Python is :class:`trading_research.api.Backtest`.
"""

from __future__ import annotations

import typer

from trading_research.cli_common import group

HELP = "6 Execution: taker | passive | maker under one cost model; alone, the taker backtest."
COMMANDS = ("backtest", "mm-backtest", "exit-search", "retrain-search")


def build(flat: typer.Typer) -> typer.Typer:
    """The ``backtest`` group over the flat app's commands."""
    return group(flat, "backtest", HELP, COMMANDS)
