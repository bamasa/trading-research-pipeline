"""``trp instruments``: step 3, which contracts to look at and which have room to trade.

The commands are the flat ``discover`` and ``screen`` of
:mod:`trading_research.cli`, unchanged.
"""

from __future__ import annotations

import typer

from trading_research.cli_common import group

HELP = "3 Instruments: list the venue's contracts, rank what is on disk by edge over cost."
COMMANDS = ("discover", "screen")


def build(flat: typer.Typer) -> typer.Typer:
    """The ``instruments`` group over the flat app's commands."""
    return group(flat, "instruments", HELP, COMMANDS)
