"""The registered studies, run through their own entry points.

Every study registered here has one entry point, ``experiments/<name>.py``,
whose subcommands are its stages, the open and read of its held-out blocks
through its ledger, ``status``, ``pages`` and ``--dry-run``. This package puts
one protocol in front of them (:class:`RegisteredStudy`): a name, the
registration and its YAML, the ordered stages, ``run``, ``status``, ``pages``
and ``dry_run``. The adapter (:class:`EntryPointStudy`) runs the entry point
itself, unmodified, in a fresh interpreter, so a refusal is the entry point's
own and the ledger is never bypassed. :func:`create` writes a new study's
registration skeleton from a template::

    from trading_research.studies import get, studies

    studies()  # {'short_horizon_2025': ..., 'spot_2026': ...}
    spot = get("spot_2026")
    spot.stages()  # ('0', '1', 'D', 'amend', 'open H', 'read H', 'open P', 'read P')
    spot.status().result  # the entry point's own JSON
    spot.run("read H").refused  # True until the entry point allows it
"""

from trading_research.studies.base import EntryPointStudy, RegisteredStudy, StudyRun
from trading_research.studies.registry import get, repository, studies
from trading_research.studies.template import TEMPLATES, create, null_sha256

__all__ = [
    "TEMPLATES",
    "EntryPointStudy",
    "RegisteredStudy",
    "StudyRun",
    "create",
    "get",
    "null_sha256",
    "repository",
    "studies",
]
