"""Contributed execution plugins: every module here is imported when the registry is read.

Add a file with one class or factory decorated with
``@trading_research.plugins.register_execution("name")``, and it is listed by
``trading-research strategies list`` and checked by
``trading-research strategies check <name>``. Modules whose names start with
an underscore are skipped. See ``docs/extending.md``.
"""
