"""``python -m trading_research``: the command line, every command and group registered.

The same application as the ``trading-research`` and ``trp`` console scripts.
"""

from __future__ import annotations

import sys

from trading_research.cli import app

if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
