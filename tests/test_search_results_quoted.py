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


def test_the_carry_checks_are_quoted_as_written() -> None:
    checks = json.loads((ROOT / "experiments/results/carry_overfitting/checks.json").read_text())
    d, h = checks["development"], checks["held_out"]
    pbo = f"{d['pbo']['pbo']:.2f}"
    holm = f"{d['haircut']['holm']['sharpe_a_year']:.2f}"
    rank = h["rank_among_rules"]
    findings = (ROOT / "docs/findings.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    registration = (ROOT / "docs/preregistration/carry_live_2026_11.md").read_text(encoding="utf-8")
    assert f"PBO is **{pbo}**" in findings and f"{holm} of the {d['sharpe_a_year']:.2f}" in findings
    assert f"{rank}th of 16" in findings and f"ranks\n{rank}th" in readme.replace(
        "ranks ", "ranks\n"
    )
    assert f"(PBO {pbo})" in readme and f"{holm} of {d['sharpe_a_year']:.2f}" in readme.replace(
        "\n", " "
    )
    assert f"PBO {pbo} on D" in registration and f"ranks {rank} of 16" in registration.replace(
        "\n", " "
    )
    assert "need nine" in registration and round(h["min_months_above_zero_if_normal"]) == 9
