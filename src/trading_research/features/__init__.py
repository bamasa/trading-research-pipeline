"""Feature definitions and the registry that collects them.

Importing this package registers every built-in feature. That is done here, in
one place, rather than left to each caller: a registry populated by import side
effects is a registry whose contents depend on what happened to be imported
first, and a feature silently missing from a run is far worse than one that
fails loudly.
"""

from __future__ import annotations

from trading_research.features import book  # noqa: F401  (import registers the features)
from trading_research.features.registry import (
    REGISTRY,
    Feature,
    FeatureError,
    Registry,
    build,
    feature,
)

__all__ = [
    "REGISTRY",
    "Feature",
    "FeatureError",
    "Registry",
    "build",
    "feature",
]
