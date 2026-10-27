"""One registry per kind, a decorator for each, and discovery from three places.

Registering is one decorator on a class or a factory function::

    from trading_research.plugins import register_strategy


    @register_strategy("book_imbalance")
    class BookImbalance: ...

The registries are filled the first time one is read, from:

1. **built-ins**: :mod:`trading_research.plugins.builtins`, adapters that
   register the parts the repository already has without changing them;
2. **contrib folders**: every module in the kind's ``contrib`` package
   (:data:`CONTRIB`), so a file dropped there is seen with no other edit;
3. **entry points**: the groups in :data:`ENTRY_POINT_GROUPS`, so a fork or a
   separate installed package can register from its own ``pyproject.toml``::

       [project.entry-points."trading_research.strategies"]
       my_rule = "my_package.rules"

   Loading an entry point imports its module; a decorator there registers. An
   entry point naming an unregistered class or function registers it under the
   entry point's name.

A module that fails to import, or that registers a name already taken, is not
allowed to break the others: the failure is kept in :func:`load_errors` and
shown by ``trading-research strategies list``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import pkgutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, TypeVar

#: The kinds, in the order they are listed.
KINDS: tuple[str, ...] = ("strategy", "model", "features", "label", "execution")

#: Where each kind's contrib folder lives.
CONTRIB: dict[str, str] = {
    "strategy": "trading_research.strategies.contrib",
    "model": "trading_research.models.contrib",
    "features": "trading_research.features.contrib",
    "label": "trading_research.labels.contrib",
    "execution": "trading_research.backtest.contrib",
}

#: The entry-point group each kind is discovered from.
ENTRY_POINT_GROUPS: dict[str, str] = {
    "strategy": "trading_research.strategies",
    "model": "trading_research.models",
    "features": "trading_research.features",
    "label": "trading_research.labels",
    "execution": "trading_research.executions",
}

BUILTINS = "trading_research.plugins.builtins"

F = TypeVar("F", bound=Callable[..., Any])


class PluginError(LookupError):
    """A plugin is unknown, ambiguous, or registered twice."""


@dataclass(frozen=True)
class PluginSpec:
    """One registered plugin: how to build it and where it came from."""

    kind: str
    name: str
    factory: Callable[..., Any]
    origin: str
    module: str
    summary: str
    #: Parameters the conformance suite builds it with, recorded on its page.
    check_params: dict[str, Any] = field(default_factory=dict)
    #: What the source file and the fingerprint are read from, when the factory
    #: is an adapter around a part defined elsewhere.
    defined_by: Any = None

    def build(self, **params: Any) -> Any:
        """An instance, with ``params`` over the factory's defaults."""
        return self.factory(**params)

    @property
    def source(self) -> Path | None:
        """The file that defines the factory."""
        try:
            found = inspect.getsourcefile(self.defined_by or self.factory)
        except TypeError:
            return None
        return Path(found) if found else None

    def fingerprint(self) -> str:
        """sha256 of the defining file, for a registration to record and pin."""
        path = self.source
        if path is None or not path.exists():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class _State:
    plugins: dict[str, dict[str, PluginSpec]] = field(
        default_factory=lambda: {kind: {} for kind in KINDS}
    )
    errors: list[tuple[str, str]] = field(default_factory=list)
    loaded: bool = False
    origin: str = "registered at run time"


_STATE = _State()


@contextlib.contextmanager
def _origin(label: str) -> Iterator[None]:
    previous, _STATE.origin = _STATE.origin, label
    try:
        yield
    finally:
        _STATE.origin = previous


def _summary(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().split("\n")[0] if doc else ""


def add(
    kind: str,
    name: str,
    factory: Callable[..., Any],
    *,
    summary: str | None = None,
    check_params: dict[str, Any] | None = None,
    defined_by: Any = None,
) -> PluginSpec:
    """Register ``factory`` as ``name`` of ``kind``; refuse a name already taken."""
    if kind not in KINDS:
        raise PluginError(f"unknown kind {kind!r}; kinds: {', '.join(KINDS)}")
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise PluginError(f"a plugin name is letters, digits, '_' and '-', got {name!r}")
    taken = _STATE.plugins[kind].get(name)
    if taken is not None:
        if taken.factory is factory:
            return taken
        raise PluginError(
            f"{kind} {name!r} is already registered by {taken.module} ({taken.origin})"
        )
    spec = PluginSpec(
        kind=kind,
        name=name,
        factory=factory,
        origin=_STATE.origin,
        module=getattr(defined_by or factory, "__module__", "") or "",
        summary=summary if summary is not None else _summary(defined_by or factory),
        check_params=dict(check_params or {}),
        defined_by=defined_by,
    )
    _STATE.plugins[kind][name] = spec
    return spec


def _decorator(kind: str) -> Callable[..., Callable[[F], F]]:
    def register(
        name: str, *, summary: str | None = None, check_params: dict[str, Any] | None = None
    ) -> Callable[[F], F]:
        def wrap(factory: F) -> F:
            add(kind, name, factory, summary=summary, check_params=check_params)
            return factory

        return wrap

    register.__name__ = f"register_{kind}"
    register.__doc__ = f"Register a class or factory as a {kind} plugin under ``name``."
    return register


register_strategy = _decorator("strategy")
register_model = _decorator("model")
register_features = _decorator("features")
register_label = _decorator("label")
register_execution = _decorator("execution")


def _import(module: str, origin: str, *, reload: bool = False) -> None:
    try:
        with _origin(origin):
            if reload and module in sys.modules:
                importlib.reload(sys.modules[module])
            else:
                importlib.import_module(module)
    except Exception as exc:  # a broken plugin must not hide the others
        _STATE.errors.append((module, f"{type(exc).__name__}: {exc}"))


def _contrib_modules(package: str) -> list[str]:
    try:
        found = importlib.import_module(package)
    except Exception as exc:
        _STATE.errors.append((package, f"{type(exc).__name__}: {exc}"))
        return []
    paths = getattr(found, "__path__", None)
    if paths is None:
        return []
    return sorted(
        f"{package}.{info.name}"
        for info in pkgutil.iter_modules(paths)
        if not info.name.startswith("_")
    )


def _entry_points(kind: str, *, reload: bool = False) -> None:
    group = ENTRY_POINT_GROUPS[kind]
    for point in metadata.entry_points(group=group):
        dist = getattr(point.dist, "name", None) or "an installed package"
        origin = f"entry point {group}:{point.name} ({dist})"
        try:
            with _origin(origin):
                if reload and point.module in sys.modules:
                    importlib.reload(sys.modules[point.module])
                loaded = point.load()
                if callable(loaded) and not inspect.ismodule(loaded):
                    known = {s.factory for s in _STATE.plugins[kind].values()}
                    if loaded not in known:
                        add(kind, point.name, loaded)
        except Exception as exc:
            _STATE.errors.append((f"{group}:{point.name}", f"{type(exc).__name__}: {exc}"))


def discover(*, force: bool = False) -> None:
    """Fill every registry: built-ins, then contrib folders, then entry points.

    ``force`` empties the registries and reads all three places again, reloading
    modules already imported, so a file added since is seen.
    """
    if _STATE.loaded and not force:
        return
    _STATE.loaded = True
    if force:
        _STATE.errors.clear()
        for kind in KINDS:
            _STATE.plugins[kind].clear()
    for module in _contrib_modules(BUILTINS):
        _import(module, "built-in", reload=force)
    for kind in KINDS:
        for module in _contrib_modules(CONTRIB[kind]):
            _import(module, "contrib", reload=force)
        _entry_points(kind, reload=force)


def load_errors() -> list[tuple[str, str]]:
    """Modules and entry points that failed to load, with the reason."""
    discover()
    return list(_STATE.errors)


def specs(kind: str | None = None) -> list[PluginSpec]:
    """Every registered plugin, or those of one kind, ordered by kind and name."""
    discover()
    kinds = KINDS if kind is None else (kind,)
    for name in kinds:
        if name not in KINDS:
            raise PluginError(f"unknown kind {name!r}; kinds: {', '.join(KINDS)}")
    return [_STATE.plugins[k][n] for k in kinds for n in sorted(_STATE.plugins[k])]


def names(kind: str) -> list[str]:
    return [spec.name for spec in specs(kind)]


def get(name: str, kind: str | None = None) -> PluginSpec:
    """The plugin called ``name``; ``kind`` settles a name two kinds share."""
    found = [spec for spec in specs(kind) if spec.name == name]
    if not found:
        where = f" {kind}" if kind else ""
        known = ", ".join(spec.name for spec in specs(kind))
        raise PluginError(f"no{where} plugin named {name!r}; registered: {known}")
    if len(found) > 1:
        kinds = ", ".join(spec.kind for spec in found)
        raise PluginError(f"{name!r} is registered as {kinds}; say which with --kind")
    return found[0]


def resolve(kind: str, name: str, **params: Any) -> Any:
    """Build the ``kind`` plugin called ``name``: what a study runner calls."""
    return get(name, kind).build(**params)


def unregister(kind: str, name: str) -> None:
    """Remove a plugin; for tests that register temporary ones."""
    _STATE.plugins[kind].pop(name, None)
