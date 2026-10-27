"""Stage E's detector record on D, and the read's check that a later run reproduces it."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pandas as pd

from trading_research.pipeline.short_horizon import flags, freeze, reads
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import DRY_RUN

ALARMS = [
    {"timestamp": "2024-12-20T05:00:00+00:00", "statistic": "scale", "direction": "up"},
    {"timestamp": "2025-03-10T04:00:00+00:00", "statistic": "mean", "direction": "down"},
    {"timestamp": "2025-07-02T01:00:00+00:00", "statistic": "scale", "direction": "up"},
]


def _frozen(*_: Any, **__: Any) -> dict[str, Any]:
    return {"alarms": ALARMS, "profile": [1.0] * 24}


def test_alarms_d_holds_the_registered_span_only(tmp_path: Path) -> None:
    study = Study(tmp_path / "repo", DRY_RUN, work_root=tmp_path / "work")
    with (
        mock.patch.object(flags, "hourly_close", return_value=pd.Series(dtype=float)),
        mock.patch.object(flags, "frozen", _frozen),
        mock.patch.object(flags, "saved_fold_flags", return_value={"guard_flags_ns": []}),
    ):
        record = freeze.detector_record(study)
    assert [a["timestamp"] for a in record["alarms_D"]] == ["2025-03-10T04:00:00+00:00"]
    assert record["alarm_rate_D"] == 1 / 181


def test_a_later_run_reproduces_d_when_its_december_alarms_are_left_aside() -> None:
    recorded = [{"timestamp": "2025-03-10T04:00:00+00:00", "channel": "mean"}]
    registration = SimpleNamespace(value=lambda _: recorded)
    assert not reads.r_void(registration, ALARMS)  # December and July are outside D
    assert reads.r_void(registration, ALARMS[:1])  # D's alarm not reproduced


def _records(with_fix_records: bool) -> dict[str, dict[str, Any]]:
    """The stage records the amendment reads: as this code writes them, or with the
    overfitting, power, lacking-class and second-detector keys the fixed stages add."""
    d: dict[str, Any] = {
        "S25": {"in_scored_cells": True}, "statuses_on_D": {"S-C": "admitted", "S-R": "admitted"},
        "carry_forward": {"C": {}, "R": {}}, "conditions": {"C": None, "R": None},
        "nulls": {"C": {}, "R": {}}, "overfitting": {"pbo": {"C": 0.4}}, "selection_premium": {"C": {}, "R": {}},
        "R_D": 1, "hand_runs": [], "choice": {"C*": {"cell": "c"}, "R*": {"cell": "r"}},
        "files_sha256": {"nulls": "f" * 64}, "p_search": 0.5, "null_A": {}, "null_B": {}, "refit": {},
    }  # fmt: skip
    c: dict[str, Any] = {"admission": {"S-P": {"status_on_D": "admitted", "cell": "p"},
                                       "S-M": {"status_on_D": "admitted", "cell": "m"}},
                         "a0_and_fill": {}}  # fmt: skip
    if with_fix_records:
        d["power_at_bar"] = {"S-C": {"cell": "c", "sigma_D": 4.0, "n_eff": 40.0, "mde": 2.4,
                                     "carry_forward_net": 1.5, "power": 0.31}}  # fmt: skip
        d["fits_lacking_a_class"] = [{"fold": "2025-02", "configuration": "gbm|C|x", "note": "",
                                      "missing": ["short"], "side": ["short"]}]  # fmt: skip
        c["overfitting"] = {"dsr": {"S-P/net": 0.2, "S-M/net": -0.1}}
        c["second_detector_D"] = {"span": "D", "rows": 5, "days": 2, "days_read": 2, "alarms": [
            {"timestamp": "2025-03-01T00:00:00+00:00", "statistic": "mid_cusum", "direction": 1,
             "excursion": 30.0, "label": ""}]}  # fmt: skip
    return {"1": {"days": {}, "max_sequence_gaps": 0, "instrument_spec": {"tick": 0.1, "lot": 0.001},
                  "reader_test": {"days": {}, "max_sequence_gaps": 0, "D": {"findings": {}, "stream": {
                      "messages": {"median": 1}, "median_interval_ms": {"median": 1},
                      "sequence_gaps": {"median": 0}}}}, "selected": ["f"]},
            "B": {"stale_days": []}, "D": d, "C": c}  # fmt: skip


def _values(
    with_fix_records: bool, dsr: dict[str, float]
) -> tuple[Any, dict[tuple[str, ...], Any]]:
    records = _records(with_fix_records)
    study = SimpleNamespace(record=lambda name: records[name], repo=Path("no-repository"))
    parts = {"slots": {"S25": {"neighbourhood": []}, "S-P": {"key": "p"}}, "fits": {
        "weights_sha256": {}, "checkpoint_epoch": {}, "thresholds": {}, "gate_levels": {}},
        "mdes": {}, "trials": {"N_trials": 3, "ledger_sha256": "a" * 64, "ledger_rows": 0,
                               "overfitting": {"N_eff": 2.0, "dsr": dsr}},
        "detector": {"flags_D_foldwise": {}, "hour_profile": [1.0] * 24, "alarm_rate_D": 0.0,
                     "alarms_D": [], "r_untestable_declared": False},
        "fitted_sha256": "b" * 64, "finalists": {}}  # fmt: skip
    with mock.patch.object(freeze, "compute_record", return_value={}):
        return study, freeze.placeholder_values(study, parts)  # type: ignore[arg-type]


def test_the_amendment_carries_the_fixed_stages_overfitting_power_lacking_class_and_second_detector() -> (
    None
):
    from trading_research.pipeline import short_horizon_registration as shr

    dsr = {"C*/G": 0.4, "C*/net": 0.1, "S25/G": 0.5, "S25/net": 0.3}
    study, values = _values(True, dsr)
    admission = values[("search", "result_admission")]
    assert admission["S-C"]["dsr"] == {
        "G": 0.4,
        "net": 0.1,
        "deflated_against": freeze.TAKER_DSR_VARIANCE,
    }
    assert admission["S-R"]["dsr"]["G"] is None and admission["S-R"]["power_at_bar"] is None
    assert admission["S-C"]["power_at_bar"]["power"] == 0.31
    assert admission["S25"]["dsr"]["G"] == 0.5
    assert admission["S-P"]["dsr"] == {"net": 0.2, "deflated_against": freeze.PASSIVE_DSR_VARIANCE}
    assert admission["S-M"]["cell"] == "m" and admission["S-M"]["dsr"]["net"] == -0.1
    assert values[("models", "fits_lacking_a_class")][0]["side"] == ["short"]
    assert values[("detector", "second_detector_D")]["alarms"][0]["statistic"] == "mid_cusum"
    for path in (("models", "fits_lacking_a_class"), ("detector", "second_detector_D")):
        shr.FROZEN_KINDS[path](values[path], ".".join(path))
    text = freeze.amendment_text(study, "frozen", values)  # type: ignore[arg-type]
    assert (
        "- Deflated Sharpe ratios: S25 G 0.5, net 0.3; S-C G 0.4, net 0.1; S-R G not computed"
        in text
    )
    assert "S-P net 0.2; S-M net -0.1 (deflated against " in text
    assert (
        "- Power of the t >= 3 bar: S-C 0.31 against a carry-forward net of 1.5 bp (sigma_D 4, n_eff 40)."
        in text
    )
    assert "- Fits that lacked a class, with the side: 1: 2025-02 gbm|C|x (short)." in text
    assert (
        "- The second detector on D: 1 alarms over 2 days read, the first at 2025-03-01T00:00:00+00:00."
        in text
    )


def test_the_amendment_is_written_when_the_stages_records_lack_the_fixed_keys() -> None:
    from trading_research.pipeline import short_horizon_registration as shr

    study, values = _values(False, {"C*/G": 0.4})
    assert values[("search", "result_admission")]["S-P"]["dsr"]["net"] is None
    assert values[("models", "fits_lacking_a_class")] == []
    assert values[("detector", "second_detector_D")] == freeze.SECOND_DETECTOR_NOT_RUN
    for path in (("models", "fits_lacking_a_class"), ("detector", "second_detector_D")):
        shr.FROZEN_KINDS[path](values[path], ".".join(path))
    text = freeze.amendment_text(study, "frozen", values)  # type: ignore[arg-type]
    assert "- Power of the t >= 3 bar: not computed." in text
    assert "- Fits that lacked a class: none." in text
    assert "0 alarms over 0 days read (not run by stage C at this commit)." in text


def test_the_amendment_records_the_sha256_of_ds_results_files(tmp_path: Path) -> None:
    from trading_research.data.study_fetch import file_sha256
    from trading_research.pipeline import short_horizon_registration as shr

    for path in shr.D_RESULTS_FILES:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(f"# {path.name}\n", encoding="utf-8")
    lines = freeze.results_file_lines(SimpleNamespace(repo=tmp_path))  # type: ignore[arg-type]
    assert lines == [
        f"- {shr.results_file_line(p, file_sha256(tmp_path / p))}" for p in shr.D_RESULTS_FILES
    ]
    document = "# x\n\n## Amendments\n\n" + "\n".join(lines) + "\n"
    assert set(shr.recorded_results_files(document)) == {p.as_posix() for p in shr.D_RESULTS_FILES}
