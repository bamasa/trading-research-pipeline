"""``trp signal``: step 4, features, their selection, a fitted model and its predictions.

The commands are the flat ``prepare``, ``select``, ``train`` and ``predict``
of :mod:`trading_research.cli`, unchanged: each reads the window it is given
and writes its files with a manifest.
"""

from __future__ import annotations

import typer

from trading_research.cli_common import group

HELP = "4 Signal: features, their selection on a window, a rule or model fitted, predictions."
COMMANDS = ("prepare", "select", "train", "predict")


def build(flat: typer.Typer) -> typer.Typer:
    """The ``signal`` group over the flat app's commands."""
    return group(flat, "signal", HELP, COMMANDS)
