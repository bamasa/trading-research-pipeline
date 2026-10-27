"""The plugin registry: built-ins, contrib folders, entry points, and its refusals."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from trading_research.plugins import registry


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Rediscover after the test, once its redirections are undone, so nothing it
    registered outlives it."""
    yield
    monkeypatch.undo()
    registry.discover(force=True)


def test_every_kind_has_its_builtins() -> None:
    assert set(registry.names("strategy")) >= {
        "reversion",
        "relative",
        "spread",
        "hold_first",
        "ev_gate",
    }
    assert set(registry.names("execution")) >= {"taker", "passive", "maker", "event_taker"}
    assert set(registry.names("features")) >= {"book", "depth", "flow", "generated"}
    assert set(registry.names("label")) >= {"directional", "direction", "walked_taker"}
    from trading_research.pipeline import stages

    known = {*stages.MODELS, *stages.ENSEMBLES, *stages.UNCERTAIN, *stages.META, *stages.COMPOSITES}
    assert set(registry.names("model")) >= known
    assert all(s.origin == "built-in" for s in registry.specs() if s.name in known)


def test_the_worked_example_is_found_in_its_contrib_folder() -> None:
    spec = registry.get("book_imbalance", "strategy")
    assert spec.origin == "contrib"
    assert spec.module == "trading_research.strategies.contrib.book_imbalance"


def test_a_built_in_model_is_what_build_model_builds() -> None:
    from trading_research.models.linear import LogisticBaseline

    model = registry.resolve("model", "logistic")
    assert isinstance(model.model, LogisticBaseline)
    assert model.inputs == ()
    assert registry.resolve("model", "order_flow").inputs == ("queue_imbalance",)
    assert registry.resolve("model", "ridge").target == "regression"


def test_the_fingerprint_is_the_defining_file(tmp_path: Path) -> None:
    spec = registry.get("book_imbalance", "strategy")
    assert spec.source is not None
    assert spec.fingerprint() == hashlib.sha256(spec.source.read_bytes()).hexdigest()
    model = registry.get("logistic", "model")
    assert model.source is not None and model.source.name == "linear.py"


def _package(root: Path, name: str, modules: dict[str, str]) -> None:
    folder = root / name
    folder.mkdir()
    (folder / "__init__.py").write_text("")
    for module, text in modules.items():
        (folder / f"{module}.py").write_text(text)


RULE = '''
import numpy as np
from trading_research.plugins import register_strategy


@register_strategy("{name}")
class Rule:
    """Never trades."""

    inputs = ("mid",)
    lookback = 0
    horizon = 5

    def fit(self, history):
        return self

    def decide(self, frame):
        return np.zeros(len(frame), dtype=np.int8)
'''


def test_a_file_in_a_contrib_folder_is_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: None
) -> None:
    _package(tmp_path, "contrib_pkg_a", {"rule": RULE.format(name="dropped_in")})
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry.CONTRIB, "strategy", "contrib_pkg_a")
    registry.discover(force=True)
    spec = registry.get("dropped_in")
    assert (spec.kind, spec.origin, spec.summary) == ("strategy", "contrib", "Never trades.")


def test_a_broken_contrib_file_is_reported_and_hides_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: None
) -> None:
    _package(
        tmp_path,
        "contrib_pkg_b",
        {"broken": "raise RuntimeError('half written')", "good": RULE.format(name="still_here")},
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry.CONTRIB, "strategy", "contrib_pkg_b")
    registry.discover(force=True)
    assert "still_here" in registry.names("strategy")
    errors = dict(registry.load_errors())
    assert "half written" in errors["contrib_pkg_b.broken"]


def test_an_entry_point_registers_from_an_installed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: None
) -> None:
    _package(tmp_path, "forked_rules", {"rules": RULE.format(name="from_a_fork")})
    info = tmp_path / "forked_rules-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: forked-rules\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(
        "[trading_research.strategies]\nfrom_a_fork = forked_rules.rules\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry.discover(force=True)
    spec = registry.get("from_a_fork", "strategy")
    assert spec.origin.startswith("entry point trading_research.strategies:from_a_fork")
    assert "forked-rules" in spec.origin


def test_an_entry_point_naming_a_bare_class_registers_it_under_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh: None
) -> None:
    bare = RULE.format(name="unused").replace('@register_strategy("unused")\n', "")
    _package(tmp_path, "bare_rules", {"rules": bare})
    info = tmp_path / "bare_rules-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: bare-rules\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(
        "[trading_research.strategies]\nbare_rule = bare_rules.rules:Rule\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry.discover(force=True)
    assert registry.get("bare_rule", "strategy").factory.__name__ == "Rule"


def test_a_name_is_taken_once_per_kind(fresh: None) -> None:
    with pytest.raises(registry.PluginError, match="already registered"):
        registry.add("strategy", "reversion", lambda: None)
    with pytest.raises(registry.PluginError, match="a plugin name is"):
        registry.add("strategy", "has space", lambda: None)
    with pytest.raises(registry.PluginError, match="unknown kind"):
        registry.add("sizing", "kelly", lambda: None)


def test_a_name_two_kinds_share_needs_its_kind(fresh: None) -> None:
    registry.add("label", "relative", lambda: None)
    with pytest.raises(registry.PluginError, match="--kind"):
        registry.get("relative")
    assert registry.get("relative", "strategy").kind == "strategy"
    with pytest.raises(registry.PluginError, match="no plugin named"):
        registry.get("nothing_like_it")


def test_importing_the_package_registers_nothing_by_itself() -> None:
    code = (
        "import trading_research.plugins as p, trading_research.plugins.registry as r;"
        "assert not r._STATE.loaded and not any(r._STATE.plugins.values())"
    )
    import subprocess

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
