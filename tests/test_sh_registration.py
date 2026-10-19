"""The 2025 study's YAML, validated against its registration, and its placeholders."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from trading_research.market_making.prereg import (
    PreRegistrationError,
    placeholder_lines,
    sha256_bytes,
    unfrozen_text,
)
from trading_research.pipeline import short_horizon_registration as shr

ROOT = Path(__file__).resolve().parents[1]
#: The configuration as committed: unfrozen until the D amendment, frozen after it.
LIVE = (ROOT / shr.CONFIG_PATH).read_text(encoding="utf-8")
#: The registered text, every placeholder null, whichever state the file is in.
TEXT = unfrozen_text(LIVE)
LIVE_DOCUMENT = (ROOT / shr.DOCUMENT_PATH).read_text(encoding="utf-8")
#: The registered document: its amendments section as it stood before stage E.
DOCUMENT = (
    LIVE_DOCUMENT.partition("\n" + shr.AMENDMENTS_HEADING + "\n")[0]
    + "\n"
    + shr.AMENDMENTS_HEADING
    + "\n\nNone yet.\n"
)

CELL = "hold_s=50|trades_per_day=240|label_fee_bp=0"
VALUES: dict[tuple[str, ...], Any] = {
    ("data", "max_sequence_gaps"): 3,
    ("data", "excluded_days_D"): [[date(2025, 3, 1), "sequence_gaps"]],
    ("data", "instrument_spec"): {"tick": 0.1, "lot": 0.001},
    ("data", "ob200_reader_test_result"): {
        "max_sequence_gaps": 3,
        "days": {
            d: {"exclusions": [], "passes": True, "fails_on_gaps_alone": False}
            for d in ("2026-01-05", "2026-01-06", "2026-01-07")
        },
        "D": {"stream": {"messages": {"median": 1}}, "findings": {}},
    },
    ("features", "selected"): ["ofi_1", "spread_bp"],
    ("features", "fitted_values_sha256"): "a" * 64,
    ("models", "weights_sha256"): {"gbm_micro_h50_f0": "b" * 64},
    ("models", "checkpoint_epoch"): {"tcn_h20": 7},
    ("models", "fits_lacking_a_class"): [
        {"fold": "2025-02", "configuration": "x", "missing": ["short"], "side": ["short"]}
    ],
    ("search", "result_trials"): {
        "N_trials": 2400,
        "N_eff": 310.5,
        "R_D": 1,
        "ledger_sha256": "c" * 64,
        "ledger_rows": 360000,
        "hand_runs": [],
    },
    ("search", "result_choice"): {"C*": {"cell": "x"}, "R*": {"cell": None}},
    ("search", "result_admission"): {"S-C": {"decision": "not admitted on D (i)"}},
    ("search", "result_nulls"): {"nulls_sha256": "d" * 64},
    ("slots", "S25", "thresholds"): {CELL: {"theta_long": 0.6, "theta_short": 0.61}},
    ("slots", "S-C", "thresholds"): {CELL: {"theta_long": 0.5, "theta_short": 0.52}},
    ("metrics", "mde", "sigma_d"): {"S25": 4.2, "S25-Z": 3.0},
    ("metrics", "mde", "value"): {"S25": 2.4},
    ("measurements", "result_O"): {"A0": "table"},
    ("detector", "flags_D_foldwise"): {"2025-02": ["2025-02-03T05:00:00Z"]},
    ("detector", "hour_profile"): [1.0] * 24,
    ("detector", "alarm_rate_D"): 0.03,
    ("detector", "alarms_D"): [{"timestamp": "2025-03-01T00:00:00Z", "channel": "scale"}],
    ("detector", "r_untestable_declared"): False,
    ("detector", "second_detector_D"): {
        "span": "D",
        "alarms": [{"timestamp": "2025-04-01T00:00:00+00:00", "statistic": "mid_cusum"}],
    },
    ("compute", "record"): {"replay_peak_rss_gb": 3.1},
    ("amendment", "date"): date(2026, 11, 1),
    ("artefacts", "frozen_finalist_files_sha256"): {"S25": "e" * 64},
}


def test_the_registered_file_loads_with_its_hashes() -> None:
    """Before the D amendment the file is the registered one; after it, the file is
    frozen, resets to the registered hash, and the document records its hash."""
    registration = shr.Registration.load(repo=ROOT)
    assert sha256_bytes(TEXT.encode()) == shr.REGISTERED_SHA256
    registration.require_consistent()
    assert tuple(placeholder_lines(TEXT)) == shr.PLACEHOLDERS
    assert set(shr.FROZEN_KINDS) == set(shr.PLACEHOLDERS)
    assert f"`{shr.REGISTERED_SHA256}`" in DOCUMENT
    assert registration.catalogue_sha256 == shr.REGISTERED_CATALOGUE_SHA256
    if registration.frozen:
        registration.require_frozen()
        assert registration.sha256 == sha256_bytes(LIVE.encode())
        assert shr.frozen_hash_line(LIVE) in LIVE_DOCUMENT
        assert shr.amendments_text(LIVE_DOCUMENT).strip() != "None yet."
    else:
        assert registration.sha256 == shr.REGISTERED_SHA256
        assert registration.max_sequence_gaps is None and registration.weights_sha256 == {}
        assert shr.amendments_text(LIVE_DOCUMENT).strip() == "None yet."


def test_a_change_outside_the_placeholders_is_refused() -> None:
    changed = TEXT.replace(
        "latency_ms: 10  # simulator rule R3", "latency_ms: 11  # simulator rule R3"
    )
    assert changed != TEXT
    with pytest.raises(PreRegistrationError, match="outside its placeholders"):
        shr.Registration.from_text(changed)
    with pytest.raises(PreRegistrationError, match="latency"):
        shr.Registration.from_text(changed, null_sha256=sha256_bytes(changed.encode()))


def amended(document: str, *lines: str) -> str:
    return document.replace(
        "\n## Amendments\n\nNone yet.\n", "\n## Amendments\n\n" + "\n".join(lines) + "\n"
    )


def test_a_stage_0_cut_is_accepted_only_with_the_amendments_hashes(tmp_path: Path) -> None:
    cut = TEXT.replace("hold_s: {values: [5, 20, 50, 120]", "hold_s: {values: [5, 20, 50]")
    catalogue = "f" * 64
    cut = cut.replace(shr.REGISTERED_CATALOGUE_SHA256, catalogue)
    null = sha256_bytes(unfrozen_text(cut).encode())
    (tmp_path / "configs").mkdir()
    (tmp_path / shr.CONFIG_PATH).write_text(cut, encoding="utf-8")
    (tmp_path / shr.DOCUMENT_PATH).parent.mkdir(parents=True)
    (tmp_path / shr.DOCUMENT_PATH).write_text(DOCUMENT, encoding="utf-8")
    with pytest.raises(PreRegistrationError, match="outside its placeholders"):
        shr.Registration.load(repo=tmp_path)
    old = amended(
        DOCUMENT, f"Null-placeholder sha256: `{'0' * 64}`", f"Catalogue sha256: `{'1' * 64}`"
    )
    later = old + f"\nNull-placeholder sha256: `{null}`\n\nCatalogue sha256: `{catalogue}`\n"
    (tmp_path / shr.DOCUMENT_PATH).write_text(later, encoding="utf-8")
    loaded = shr.Registration.load(repo=tmp_path)
    assert loaded.catalogue_sha256 == catalogue
    assert shr.latest_null_sha256(later) == null
    assert shr.latest_null_sha256(old) == "0" * 64
    outside = f"Null-placeholder sha256: `{null}`\n" + DOCUMENT
    assert shr.latest_null_sha256(outside) == shr.REGISTERED_SHA256


def test_freezing_writes_every_placeholder_and_nothing_else() -> None:
    frozen = shr.freeze_text(TEXT, VALUES)
    registration = shr.Registration.from_text(frozen).require_frozen()
    assert registration.frozen
    assert sha256_bytes(unfrozen_text(frozen).encode()) == shr.REGISTERED_SHA256
    assert registration.max_sequence_gaps == 3
    assert registration.instrument_spec == (0.1, 0.001)
    assert registration.weights_sha256 == {"gbm_micro_h50_f0": "b" * 64}
    assert registration.finalist_files_sha256 == {"S25": "e" * 64}
    assert registration.result_trials == VALUES[("search", "result_trials")]
    assert shr.frozen_hash_line(frozen) == f"Frozen configuration sha256: `{registration.sha256}`"
    with pytest.raises(PreRegistrationError, match="already frozen"):
        shr.freeze_text(frozen, VALUES)
    partial = {k: v for k, v in VALUES.items() if k != ("amendment", "date")}
    with pytest.raises(PreRegistrationError, match="every placeholder once"):
        shr.freeze_text(TEXT, partial)


def test_a_partly_frozen_file_is_inconsistent() -> None:
    from trading_research.market_making.prereg import _set_placeholders

    one = _set_placeholders(TEXT, {("data", "max_sequence_gaps"): "3"})
    with pytest.raises(PreRegistrationError, match="partly frozen"):
        shr.Registration.from_text(one).require_consistent()


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        (("data", "max_sequence_gaps"), -1),
        (("data", "max_sequence_gaps"), True),
        (("data", "excluded_days_D"), [[date(2025, 7, 9), "sequence_gaps"]]),
        (("data", "instrument_spec"), {"tick": 0.1}),
        (("features", "selected"), ["a", "a"]),
        (("features", "fitted_values_sha256"), "abc"),
        (("models", "weights_sha256"), {}),
        (("models", "checkpoint_epoch"), {"tcn": 21}),
        (("models", "fits_lacking_a_class"), [{"fold": "2025-02"}]),
        (("search", "result_trials"), {"N_trials": 1}),
        (("search", "result_choice"), {"C": {}}),
        (("slots", "S25", "thresholds"), {CELL: {"theta_long": 0.6}}),
        (("metrics", "mde", "value"), {"S9": 1.0}),
        (("detector", "flags_D_foldwise"), {"2025-07": []}),
        (("detector", "hour_profile"), [1.0] * 23),
        (("detector", "alarms_D"), [{"timestamp": "x"}]),
        (("detector", "r_untestable_declared"), "no"),
        (("detector", "second_detector_D"), {"span": "H", "alarms": []}),
        (("amendment", "date"), "2026-11-01"),
        (("artefacts", "frozen_finalist_files_sha256"), {"S25": "e" * 63}),
    ],
)
def test_each_placeholder_refuses_a_value_of_the_wrong_kind(
    path: tuple[str, ...], bad: Any
) -> None:
    with pytest.raises(PreRegistrationError):
        shr.FROZEN_KINDS[path](bad, ".".join(path))
    shr.FROZEN_KINDS[path](VALUES[path], ".".join(path))


def test_nothing_of_a_kind_frozen_is_an_empty_mapping() -> None:
    """No network is frozen when R* is the bound regressor, and stratum C may have no
    C*: the epochs and S-C's thresholds are then empty mappings, not missing values."""
    from trading_research.pipeline import short_horizon_registration as shr

    shr.FROZEN_KINDS[("models", "checkpoint_epoch")]({}, "models.checkpoint_epoch")
    shr.FROZEN_KINDS[("slots", "S-C", "thresholds")]({}, "slots.S-C.thresholds")
    with pytest.raises(shr.PreRegistrationError):
        shr.FROZEN_KINDS[("models", "checkpoint_epoch")]([], "models.checkpoint_epoch")
