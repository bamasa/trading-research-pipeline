"""One run of the spot study: its paths, its venue, its 2025 binding and its ledger."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from trading_research.spot import registration as sr
from trading_research.spot import synthetic as syn
from trading_research.spot.context import Binding2025, SpotStudy
from trading_research.spot.ledger import files_2025
from trading_research.spot.runner import stage_function
from trading_research.spot.state import STAGES


def test_the_registered_binding_is_the_registrations_own() -> None:
    found = Binding2025()
    assert found.config == sr.CONFIG_2025_PATH and found.config_sha256 == sr.CONFIG_2025_SHA256
    assert dict(found.files) == files_2025() and found.root is None


def test_paths_venue_and_ledger(tmp_path: Path) -> None:
    study = SpotStudy(tmp_path, run=3)
    assert study.work == tmp_path / "artifacts" / "spot_2026" / "run3"
    assert study.spot_root == tmp_path / "data" / "spot"
    assert study.fetch_log(2).path == tmp_path / "data" / "spot_2026" / "fetch_stage2.json"
    assert study.results("H.json") == tmp_path / sr.RESULTS_PATHS["H"]
    assert study.ledger().config_2025_sha256 == sr.CONFIG_2025_SHA256
    venue = syn.SpotVenue(syn.SpotMarket(frozenset({date(2026, 3, 3)})))
    dry = SpotStudy(tmp_path, venue=venue)
    assert dry.opener is venue and dry.present()["bybit_book"] == frozenset({date(2026, 3, 3)})
    assert all(callable(stage_function(s)) for s in STAGES)
