"""The presentation's constants agree with the registered configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from trading_research.reporting.leaderboard import LEADERBOARD_COLUMNS, LEADERBOARD_FILE
from trading_research.reporting.marginals import MARGINALS_FILE
from trading_research.reporting.nulls import NULLS_FILE
from trading_research.reporting.search_keys import REGIONS, S25_KEY

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "configs" / "short_horizon_2025.yaml").read_text())


def test_the_files_and_columns_are_the_registered_ones() -> None:
    artefacts = CONFIG["artefacts"]
    assert artefacts["files"]["leaderboard"] == LEADERBOARD_FILE
    assert artefacts["files"]["marginals"] == MARGINALS_FILE
    assert artefacts["files"]["nulls"] == NULLS_FILE
    assert tuple(artefacts["leaderboard_columns"]) == LEADERBOARD_COLUMNS


def test_the_named_key_and_the_regions_are_the_registered_ones() -> None:
    assert CONFIG["search"]["named"]["S25"] == S25_KEY
    assert CONFIG["slots"]["regions"] == REGIONS
