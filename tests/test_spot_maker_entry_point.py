"""The entry point's commands: a stage, open and read need their arguments; status reads only."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import spot_maker_2026 as entry

REPO = Path(__file__).resolve().parent.parent


def test_a_command_is_required_and_open_needs_a_block() -> None:
    with pytest.raises(SystemExit):
        entry.main([])
    with pytest.raises(SystemExit):
        entry.main(["open"])
    with pytest.raises(SystemExit):
        entry.main(["read", "E"])


def test_the_stages_are_the_registered_ones() -> None:
    from trading_research.spot_maker.context import STAGES

    assert entry.STAGES == STAGES and entry.REPO == REPO
    assert entry.parser().parse_args(["D", "--workers", "4"]).workers == 4
