"""The basis and pairs registrations quote what their one read wrote."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> tuple[dict, str]:  # type: ignore[type-arg]
    held = json.loads(
        (ROOT / "experiments/results" / f"search_{name}" / "held_out.json").read_text()
    )
    page = (ROOT / "docs/preregistration" / f"{name}_search_2026_11.md").read_text(encoding="utf-8")
    return held, page


def test_the_basis_registration_quotes_its_read() -> None:
    held, page = _read("basis")
    r = held["result"]
    assert held["status"] == "positive" and held["beats_placebos"] == 19
    assert f"+{r['total_pct']:.2f}% over two years, {r['trades']} trades" in page


def test_the_pairs_registration_quotes_its_read() -> None:
    held, page = _read("pairs")
    r = held["result"]
    assert held["status"] == "killed: a placebo earned as much"
    assert f"+{r['total_pct']:.2f}% over two years, {r['trades']} trades" in page
    assert f"it beat {['fifteen'][0]}" in page and held["beats_placebos"] == 15
