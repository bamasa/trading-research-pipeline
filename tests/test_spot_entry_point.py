"""The single entry point: its subcommands, and a refusal as a message, not a traceback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments import spot_2026 as entry


def test_the_subcommands_are_the_registered_stages_and_reads() -> None:
    args = entry.parser().parse_args(["read", "H", "--workers", "2"])
    assert (args.command, args.block, args.workers) == ("read", "H", 2)
    assert entry.STAGES == ("0", "1", "D", "amend") and entry.READS == ("H", "P")
    with pytest.raises(SystemExit):
        entry.parser().parse_args(["E"])


def test_status_and_a_stage_out_of_order(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(entry, "REPO", tmp_path)
    assert entry.main(["status"]) == 0
    found = json.loads(capsys.readouterr().out)
    assert found == dict.fromkeys(("0", "1", "D", "amend", "H", "P"), "not run")
    monkeypatch.setattr(sys, "argv", ["spot_2026", "1"])
    assert entry.entry() == 2
    assert "refused: stage 1 needs stage 0" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["spot_2026", "read", "H"])
    assert entry.entry() == 2
    assert not (tmp_path / "artifacts").exists()
