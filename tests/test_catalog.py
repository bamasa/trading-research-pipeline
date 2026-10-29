"""The catalogue names only what exists, its commands parse, and the page is current."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
import typer.main

from trading_research import catalog as cat
from trading_research.cli import app
from trading_research.plugins import registry

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "catalog.md"


def _without_tests(text: str) -> str:
    """The page before its test-case counts, which move with every new test."""
    return text.split("\n## Test cases\n")[0]


@pytest.mark.parametrize("path", sorted(set(cat.implemented_paths())))
def test_every_hand_written_row_names_code_that_exists(path: str) -> None:
    assert cat.resolve(path) is not None


@pytest.mark.parametrize(
    "source", [s for s in cat.SOURCES if s.fetch.startswith("trp ")], ids=lambda s: s.fetch
)
def test_every_fetch_command_parses_with_its_options(source: cat.Source) -> None:
    words = shlex.split(source.fetch)[1:]
    command: Any = typer.main.get_command(app)
    while words and words[0] in (getattr(command, "commands", None) or {}):
        command = command.commands[words.pop(0)]
    assert not getattr(command, "commands", None) or not words, f"no command for {source.fetch}"
    command.make_context(source.fetch, list(words))  # raises on an unknown option or value


def test_the_page_is_the_catalogue_as_written_now() -> None:
    assert PAGE.exists(), "run: trp catalog --write docs/catalog.md"
    assert _without_tests(PAGE.read_text(encoding="utf-8")) == _without_tests(cat.page(REPO)), (
        "docs/catalog.md is stale: run trp catalog --write docs/catalog.md"
    )


def test_every_plugin_is_on_the_page() -> None:
    text = PAGE.read_text(encoding="utf-8")
    registry.discover()
    for kind in registry.KINDS:
        for name in registry.names(kind):
            assert f"`{name}`" in text, f"{kind} {name!r} is not in docs/catalog.md"


def test_a_new_model_plugin_appears_in_the_catalogue_with_its_family() -> None:
    registry.discover()
    registry.add("model", "catalog_probe_model", lambda **_: None, summary="A probe.")
    try:
        text = cat.page(REPO)
        assert "| `catalog_probe_model` | plugin (" in text
    finally:
        registry.unregister("model", "catalog_probe_model")


def test_the_readme_summary_counts_match_the_catalogue() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for section, count, _ in cat.summary(REPO):
        assert f"| {section} | {count} |" in readme, (
            f"README's catalogue row for {section!r} is stale"
        )
