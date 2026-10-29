"""``trp study``: run a registered study's stages through its own entry point, or register a new one.

Every step is the entry point's (``python -m experiments.<name> ...``), run
unmodified in a fresh interpreter: a refusal is the entry point's own, with its
exit code, and nothing here opens a block. See :mod:`trading_research.studies`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from trading_research.cli_common import STUDIES_PANEL

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="study",
    help="8 Study: a registered study's stages through its entry point, or a new registration.",
    no_args_is_help=True,
)
app.info.rich_help_panel = STUDIES_PANEL

Repo = Annotated[
    Path | None, typer.Option("--repo", help="The repository the study lives in (default: here).")
]


def _study(name: str, repo: Path | None):  # type: ignore[no-untyped-def]
    from trading_research.studies import get

    try:
        return get(name, repo)
    except KeyError as exc:
        err_console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=2) from exc


def _relay(found) -> None:  # type: ignore[no-untyped-def]
    """The entry point's output as it came, and its exit code."""
    console.print(f"[dim]$ {found.command}[/dim]")
    if found.stdout:
        sys.stdout.write(found.stdout)
    if found.stderr:
        sys.stderr.write(found.stderr)
    if found.returncode:
        raise typer.Exit(code=found.returncode)


@app.command("list")
def list_cmd(repo: Repo = None) -> None:
    """The registered studies: their entry point, registration, configuration and steps."""
    from trading_research.studies import studies

    table = Table(title="Registered studies", show_edge=False)
    for column in ("study", "steps", "registration", "configuration", "what"):
        table.add_column(column, overflow="fold")
    for name, study in studies(repo).items():
        table.add_row(
            name,
            str(len(study.stages())),
            "yes" if study.registration.exists() else "missing",
            "yes" if study.config.exists() else "missing",
            study.description,
        )
    console.print(table)
    console.print("[dim]trp study stages <name> lists a study's steps in order.[/dim]")


@app.command("stages")
def stages_cmd(name: str, repo: Repo = None) -> None:
    """A study's steps in the registered order: its stages, then each block's open and read."""
    study = _study(name, repo)
    for number, stage in enumerate(study.stages(), start=1):
        console.print(f"{number:>2}  python -m experiments.{name} {stage}")
    console.print(
        f"    python -m experiments.{name} status | pages | --dry-run    [dim]any time[/dim]"
    )
    console.print(f"[dim]options the entry point takes: {', '.join(study.options)}[/dim]")


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_cmd(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="The study, as trp study list names it.")],
    stage: Annotated[str | None, typer.Argument(help="A stage, or open / read.")] = None,
    block: Annotated[str | None, typer.Argument(help="The block to open or read.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Every stage on the study's synthetic venue.")
    ] = False,
    repo: Repo = None,
) -> None:
    """Run one step through the entry point; any further option is passed to it unchanged.

    ``trp study run spot_2026 1 --workers 4`` is ``python -m experiments.spot_2026 1
    --workers 4``. ``open`` and ``read`` are refused by the entry point unless its
    ledger allows them; the refusal is printed as it came, with exit code 2.
    """
    study = _study(name, repo)
    words = [w for w in (stage, block) if w]
    if dry_run:
        words.append("--dry-run")
    if not words:
        err_console.print(
            f"[red]name a stage of {name} (trp study stages {name}) or --dry-run[/red]"
        )
        raise typer.Exit(code=2)
    _relay(study.invoke([*words, *ctx.args]))


@app.command("status")
def status_cmd(name: str, repo: Repo = None) -> None:
    """The study's own status: each stage and block, as its entry point reports it."""
    _relay(_study(name, repo).status())


@app.command("pages")
def pages_cmd(name: str, repo: Repo = None) -> None:
    """Write the study's pages again from what has been computed (the entry point's ``pages``)."""
    _relay(_study(name, repo).pages())


@app.command("null-hash")
def null_hash_cmd(name: str, repo: Repo = None) -> None:
    """The YAML's null-placeholder sha256, which the registration document must hold."""
    from trading_research.studies import null_sha256, repository

    path = repository(repo) / "configs" / f"{name}.yaml"
    if not path.exists():
        err_console.print(f"[red]no {path}[/red]")
        raise typer.Exit(code=2)
    console.print(null_sha256(path.read_text(encoding="utf-8")))


@app.command("new")
def new_cmd(
    name: Annotated[str, typer.Argument(help="lowercase, digits and underscores")],
    template: Annotated[
        str, typer.Option("--template", help="taker, passive or maker: the slot it registers.")
    ] = "taker",
    repo: Repo = None,
) -> None:
    """Write a new study's registration skeleton: its document, its YAML and its entry point.

    The entry point refuses every step but ``status`` until each ``<...>`` is
    filled and the document holds the YAML's null-placeholder sha256; nothing
    is committed here.
    """
    from trading_research.studies import create, repository
    from trading_research.studies.template import unfilled

    root = repository(repo)
    try:
        study = create(root, name, template)
    except (ValueError, FileExistsError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    for path in (study.document, study.config, study.entry_point):
        console.print(f"wrote {path.relative_to(root)}")
    left = len(unfilled(study.document)) + len(unfilled(study.config))
    console.print(
        f"\n{left} placeholder(s) to fill. Then: [bold]trp study null-hash {name}[/bold] into the "
        f"document's Amendments, commit both files, write and freeze the code, and run "
        f"[bold]trp study run {name} 0[/bold]."
    )
