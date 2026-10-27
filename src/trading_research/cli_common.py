"""The command tree's plumbing: groups of existing commands, and help that reads as a map.

The commands in :mod:`trading_research.cli` were written flat, one per step,
and tests and documents call them by those names. The tree groups them by the
step of the pipeline they belong to without renaming anything: a group holds
the same command (the same function, the same options) under the same name,
and the flat name stays as a hidden alias that behaves exactly as before::

    trp data download ...        # the grouped name
    trp download ...             # the same command, hidden from the help

Each group lives in its own ``cli_<area>.py`` and is registered in ``cli.py``
by one ``app.add_typer`` line; :func:`as_map` then hides the flat aliases and
orders the help as the pipeline runs.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Sequence
from typing import Any

import typer
from typer.core import TyperGroup

#: The help panels, in the order the pipeline runs.
STEPS_PANEL = "The pipeline, step by step"
STUDIES_PANEL = "Registered studies and the agent loop"
RUN_PANEL = "End to end, and reference"

#: The top-level commands in the order the help lists them: the map.
ORDER: tuple[str, ...] = (
    "data",
    "regimes",
    "instruments",
    "signal",
    "search",
    "backtest",
    "review",
    "study",
    "agent",
    "demo",
    "reproduce",
    "fees",
)

#: The map, as one line under the help.
MAP = (
    "data → regimes → instruments → signal → search → execution (taker | passive | maker) "
    "→ verdict → study → agent"
)

#: Top-level commands that stay visible: the panel each sits in and its line of help.
VISIBLE: dict[str, tuple[str, str]] = {
    "review": (
        STEPS_PANEL,
        "7 Verdict: the evidence a strategy review needs, from a trades file; it grades nothing.",
    ),
    "reproduce": (
        RUN_PANEL,
        "The finding of §27 end to end on the perpetuals' panel: fetch, audit, search, "
        "the held-out block read once.",
    ),
}


class PipelineMap(TyperGroup):
    """The root group: commands listed in :data:`ORDER`, the rest after them."""

    def list_commands(self, ctx: Any) -> list[str]:  # noqa: ARG002 - click passes it
        names = list(self.commands)
        rank = {name: i for i, name in enumerate(ORDER)}
        return sorted(names, key=lambda name: (rank.get(name, len(ORDER)), names.index(name)))


def group(
    flat: typer.Typer,
    name: str,
    help: str,
    commands: Sequence[str],
    panel: str = STEPS_PANEL,
) -> typer.Typer:
    """A sub-app holding the flat app's ``commands``, unchanged, under their own names."""
    found = {info.name: info for info in flat.registered_commands}
    missing = [c for c in commands if c not in found]
    if missing:
        raise KeyError(f"no flat command {missing}; the flat app has {sorted(map(str, found))}")
    sub = typer.Typer(name=name, help=help, no_args_is_help=name not in found)
    sub.info.rich_help_panel = panel  # read by typer when the group is added
    for command in commands:
        info = copy.copy(found[command])
        info.hidden = False
        info.rich_help_panel = None
        sub.registered_commands.append(info)
    if name in found:
        # A group named as a flat command takes its place, so it runs that
        # command, with its options, whenever no subcommand follows.
        sub.info.invoke_without_command = True
        sub.callback()(_runs_without_subcommand(found[name].callback))
    return sub


def _runs_without_subcommand(command: Any) -> Any:
    """A group callback with ``command``'s parameters that runs it only when
    the group is called without a subcommand."""
    signature = inspect.signature(command, eval_str=True)
    context = inspect.Parameter(
        "ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context
    )

    def callback(ctx: typer.Context, **options: Any) -> Any:
        if ctx.invoked_subcommand is None:
            return command(**options)
        return None

    callback.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[context, *signature.parameters.values()]
    )
    callback.__annotations__ = {
        "ctx": typer.Context,
        **{n: q.annotation for n, q in signature.parameters.items()},
    }
    callback.__doc__ = command.__doc__
    return callback


def single(
    name: str,
    help: str,
    callback: Any,
    panel: str = RUN_PANEL,
) -> typer.Typer:
    """A sub-app that is one command: ``trp <name> [OPTIONS]`` runs ``callback``."""
    sub = typer.Typer(
        name=name,
        help=help,
        invoke_without_command=True,
        no_args_is_help=False,
        subcommand_metavar="",
    )
    sub.info.rich_help_panel = panel  # read by typer when the group is added
    sub.callback()(callback)
    return sub


def as_map(app: typer.Typer) -> None:
    """Hide every flat command a group also holds, and order the help as the map."""
    grouped = {
        info.name
        for sub in app.registered_groups
        if sub.typer_instance is not None
        for info in sub.typer_instance.registered_commands
    }
    for info in app.registered_commands:
        if info.name in VISIBLE:
            info.rich_help_panel, info.short_help = VISIBLE[str(info.name)]
        elif info.name in grouped:
            info.hidden = True
    app.info.cls = PipelineMap
    app.info.epilog = MAP
