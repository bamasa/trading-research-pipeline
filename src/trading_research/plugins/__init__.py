"""Plugins: one registry per kind of part, filled from three places, and the checks a part passes.

Five kinds, each with a protocol (:mod:`.protocols`), a decorator and a
``contrib`` folder that is read without any other edit:

==============  ==========================  ==========================================
kind            decorator                   contrib folder
==============  ==========================  ==========================================
``strategy``    ``@register_strategy``      ``src/trading_research/strategies/contrib/``
``model``       ``@register_model``         ``src/trading_research/models/contrib/``
``features``    ``@register_features``      ``src/trading_research/features/contrib/``
``label``       ``@register_label``         ``src/trading_research/labels/contrib/``
``execution``   ``@register_execution``     ``src/trading_research/backtest/contrib/``
==============  ==========================  ==========================================

Importing this package registers nothing by itself; the registries are filled
the first time one is read (:func:`.registry.discover`). A study runner finds a
part with :func:`resolve`, records :meth:`.registry.PluginSpec.fingerprint`
with its registration, and runs :func:`.conformance.check` before using it.
See ``docs/extending.md``.
"""

from __future__ import annotations

from trading_research.plugins.registry import (
    KINDS,
    PluginError,
    PluginSpec,
    discover,
    get,
    load_errors,
    names,
    register_execution,
    register_features,
    register_label,
    register_model,
    register_strategy,
    resolve,
    specs,
)

__all__ = [
    "KINDS",
    "PluginError",
    "PluginSpec",
    "discover",
    "get",
    "load_errors",
    "names",
    "register_execution",
    "register_features",
    "register_label",
    "register_model",
    "register_strategy",
    "resolve",
    "specs",
]
