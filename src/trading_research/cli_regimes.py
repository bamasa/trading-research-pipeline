"""``trp regimes``: step 2, where a return series stopped behaving as before.

The command is the flat ``breaks`` of :mod:`trading_research.cli`, unchanged.
"""

from __future__ import annotations

import typer

from trading_research.cli_common import group

HELP = "2 Regimes: where a price series stopped behaving as before (structural breaks)."
COMMANDS = ("breaks",)


def build(flat: typer.Typer) -> typer.Typer:
    """The ``regimes`` group over the flat app's commands."""
    return group(flat, "regimes", HELP, COMMANDS)
