"""M1's frozen sources are the ones in the tree, and its family is the registration's."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "m1", ROOT / "experiments/reversion_meta_forward.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_frozen_sources_are_unchanged() -> None:
    assert _script().changed() == []


def test_the_family_is_registered_and_listed() -> None:
    from trading_research.plugins import get

    m1 = _script()
    page = (ROOT / "docs/preregistration/reversion_meta_forward_2026_11.md").read_text()
    for name in m1.FAMILY:
        get(name, "strategy")
        assert f"`{name}`" in page
    assert len(m1.FAMILY) == 9
