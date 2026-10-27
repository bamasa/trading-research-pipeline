"""``trp data``: step 1, the data a run reads and the contract it must meet.

The commands are the flat ones of :mod:`trading_research.cli`, unchanged; see
:mod:`trading_research.cli_common` for how a group holds them.
"""

from __future__ import annotations

import typer

from trading_research.cli_common import group

HELP = "1 Data: fetch public archives, generate a synthetic market, check a dataset's contract."
COMMANDS = ("download", "download-book", "generate-demo-data", "validate-data", "describe-schema")


def build(flat: typer.Typer) -> typer.Typer:
    """The ``data`` group over the flat app's commands."""
    return group(flat, "data", HELP, COMMANDS)
