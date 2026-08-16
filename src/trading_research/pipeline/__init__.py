"""The pipeline as separate, individually runnable stages.

Each stage reads files and writes files. That is slower than passing objects in
memory and it is the point: a stage can be re-run without repeating the ones
before it, and every intermediate is there to be inspected. A pipeline that
only emits a final number is one whose middle nobody checks.
"""

from __future__ import annotations

from trading_research.pipeline.stages import (
    StageError,
    StageManifest,
    feature_columns,
    load_prepared,
    prepare,
)

__all__ = [
    "StageError",
    "StageManifest",
    "feature_columns",
    "load_prepared",
    "prepare",
]
