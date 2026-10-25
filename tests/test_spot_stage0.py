"""Stage 0's gates: the readers' proof, the walk depth, the parity test, the print rule."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from _spot_day import DAYS, TABLE, fetch_days, load
from trading_research.data import fetch_log as fl
from trading_research.spot import access as sa
from trading_research.spot import archives as ar
from trading_research.spot import stage0
from trading_research.spot import synthetic as syn


@pytest.fixture(autouse=True)
def _offline():  # type: ignore[no-untyped-def]
    with syn.no_network():
        yield


@pytest.fixture(scope="module")
def fetched(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, fl.FetchLog]:
    base = tmp_path_factory.mktemp("stage0")
    return base, fetch_days(base)


def test_each_reader_is_proven_on_its_day(fetched) -> None:  # type: ignore[no-untyped-def]
    base, log = fetched
    access = sa.reader_test_access("readers' proof")
    book = stage0.book_proof(DAYS[0], log, base / "spot", access)
    assert book["passes"] and book["stream"]["messages"] > 0
    assert book["stream"]["median_interval_ms"] is not None
    prints = stage0.prints_proof(ar.BINANCE_PRINTS, DAYS[0], log, base / "spot", access)
    assert prints["passes"] and prints["stream"]["sides_loaded"]["buy"] > 0
    assert prints["stream"]["unit"] == "us"


def test_the_walk_takes_200_levels_only_above_the_threshold(fetched) -> None:  # type: ignore[no-untyped-def]
    _, log = fetched
    found = stage0.walk_depth(log, DAYS)
    assert found["walk"] == (200 if found["share"] > 0.001 else 10)
    with pytest.raises(stage0.Stage0Failed):
        stage0.walk_depth(log, (date(2026, 3, 9),))


def test_the_parity_gate_and_the_print_rule_records(fetched, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from trading_research.spot.plan import DRY_RUN

    base, log = fetched
    day = load(base, log, DAYS[0])
    study = SimpleNamespace(fee_table=TABLE, scale=DRY_RUN)
    gate = stage0.parity(study, day)  # type: ignore[arg-type]
    assert gate and not any(v["mismatches"] for v in gate.values())
    assert any(v.get("side_dependent_busy") for v in gate.values())
    rule = stage0.print_rule(day, sa.BYBIT)
    assert 0 < rule["unreachable_share"] < 1 and rule["median_sweep_over_clip"] > 0


def test_a_failed_gate_is_named() -> None:
    record = {"reader_test_result": {"x": {"2026-03-03": {"passes": False}}},
              "parity": {"a": {"mismatches": ["net differs"], "side_dependent_busy": 0}}}  # fmt: skip
    found = stage0._failures(record)
    assert any("reader" in f for f in found) and any("parity a" in f for f in found)
    assert any("depends on the side" in f for f in found)
