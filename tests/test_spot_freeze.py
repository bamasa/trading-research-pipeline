"""The D amendment's pieces: excluded days in the frozen shape, frozen numbers, the dry run's text."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from trading_research.market_making.prereg import sha256_bytes, unfrozen_text
from trading_research.spot import access as sa
from trading_research.spot import freeze
from trading_research.spot import registration as sr
from trading_research.spot.dry_run import registered_text

ROOT = Path(__file__).resolve().parents[1]


def test_excluded_days_join_stage_1_and_scoring_in_the_frozen_shape() -> None:
    record1 = {"excluded": {"binance_spot_BTCFDUSD": [["2026-05-03", "no_print"]],
                            "bybit_spot_BTCUSDT": [["2026-04-02", "sequence_gaps"]]}}  # fmt: skip
    record_d = {
        "excluded_on_scoring": {sa.BYBIT: {"2026-06-04": ["flatten_stale"]}, sa.BINANCE: {}}
    }
    found = freeze.excluded_days(record1, record_d)
    assert found["bybit_spot_BTCUSDT"] == [[date(2026, 4, 2), "sequence_gaps"],
                                           [date(2026, 6, 4), "flatten_stale"]]  # fmt: skip
    sr._excluded_days(found, "data.excluded_days_D")
    rendered = yaml.safe_load(yaml.safe_dump(found))
    assert rendered == found


def test_an_undefined_number_is_frozen_as_zero() -> None:
    assert freeze.number(None) == 0.0 and freeze.number(float("nan")) == 0.0
    assert freeze.number(1.5) == 1.5
    values = freeze.FrozenValues({("a",): 1})
    assert values.value(("a",)) == 1


def test_the_dry_run_holds_the_registration_as_registered() -> None:
    config = (ROOT / sr.CONFIG_PATH).read_text(encoding="utf-8")
    assert sha256_bytes(registered_text(sr.CONFIG_PATH, config).encode()) == sha256_bytes(
        unfrozen_text(sr.strip_p_intervals(config)[0]).encode())  # fmt: skip
    document = (ROOT / sr.DOCUMENT_PATH).read_text(encoding="utf-8")
    amended = document + "\n### 2026-11-30: the D amendment\n\nFrozen configuration sha256: `x`\n"
    assert registered_text(sr.DOCUMENT_PATH, amended).rstrip("\n") == document.rstrip("\n")
    assert (
        sr.latest_null_sha256(registered_text(sr.DOCUMENT_PATH, document)) == sr.REGISTERED_SHA256
    )
