"""The entry point end to end on a synthetic venue: every stage, then H, F and P through the ledger.

One dry run is shared by the module. It runs the entry point itself,
``python -m experiments.short_horizon_2025 --dry-run --keep <dir>``, in a fresh
interpreter: the network never shares a process with XGBoost, and this test
process has imported torch for the network's own tests. The dry run builds a
temporary repository holding the registration, answers every request from a
synthetic venue, runs stages 0 to E in order, then opens, commits, reads and
closes each block. The assertions read what it left behind.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trading_research.pipeline import short_horizon_registration as shr
from trading_research.pipeline.short_horizon import runner
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import DRY_RUN
from trading_research.pipeline.short_horizon.state import STAGES, StageOrderError
from trading_research.reporting.headlines import check_status

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("torch") is None, reason="the network's fits need 'deep'"
    ),
]
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dry(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    keep = tmp_path_factory.mktemp("dry_run")
    command = [sys.executable, "-m", "experiments.short_horizon_2025", "--dry-run",
               "--keep", str(keep), "--workers", "4"]  # fmt: skip
    done = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=1500, check=False
    )
    assert done.returncode == 0, done.stderr[-3000:]
    study = Study(keep / "repo", DRY_RUN)
    return {"report": json.loads(done.stdout), "study": study}


def tracked(repo: Path) -> set[str]:
    done = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True)
    return set(done.stdout.split())


def test_every_stage_completes_in_the_registered_order(dry: dict[str, Any]) -> None:
    study = dry["study"]
    assert dry["report"]["stages"] == list(STAGES)
    assert all(study.state.complete(stage) for stage in STAGES)
    assert study.state.complete("H") and study.state.complete("F")
    assert study.state.complete("P") or dry["report"]["reads"]["P"]["opened"] is False


def test_nothing_was_asked_of_the_network(dry: dict[str, Any]) -> None:
    assert dry["report"]["venue"]["requests"] > 0 and dry["report"]["venue"]["unknown"] == 0


def test_a_complete_stage_returns_its_record_and_computes_nothing(dry: dict[str, Any]) -> None:
    study = dry["study"]
    attempts = study.state.entries()["B"]["attempts"]
    assert runner.run_stage(study, "B") == study.state.record("B")
    assert study.state.entries()["B"]["attempts"] == attempts


def test_a_stage_refuses_to_run_before_the_stages_it_reads(dry: dict[str, Any]) -> None:
    other = Study(dry["study"].repo, DRY_RUN, run=2)
    with pytest.raises(StageOrderError, match="needs stage 0"):
        runner.run_stage(other, "A")
    assert not other.state.entries()


def test_stage_0_timed_the_code_and_passed_the_parity_gate(dry: dict[str, Any]) -> None:
    record = dry["study"].record("0")
    assert record["parity"]["ok"] and record["parity"]["runs"]
    assert record["catalogue_sha256"] == shr.REGISTERED_CATALOGUE_SHA256
    assert record["projected_hours"]["total"] > 0 and record["replay_memory"]["measured"]
    # the fold-level leakage test's CPU fits are projected, at the registered worker counts
    assert record["projected_hours"]["1_fold_leakage"] > 0
    assert record["projection_workers"]["general"] == 8 != record["workers"]


def test_stage_1_ran_the_fold_level_test_on_the_production_path(dry: dict[str, Any]) -> None:
    record = dry["study"].record("1")
    fold = record["leakage"]["fold_level"][0]
    for name in ("gbm/weights", "logistic/thresholds", "bound_ridge/weights", "gate_levels",
                 "tcn/state_sha256", "tcn/inner_ic", "tcn/checkpoint_epoch"):  # fmt: skip
        assert name in fold["artefacts"]
    assert all(e["rows_outside_day"] is not None for e in record["days"].values() if e["usable"])
    reader = record["reader_test"]["days"]
    assert len(reader) == 3 and all(e["passes"] for e in reader.values())


def test_the_trial_ledger_manifest_is_committed_and_its_parts_are_not(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon_ledger import (
        trial_ledger_parts,
        trial_ledger_sha256,
    )

    study = dry["study"]
    paths = trial_ledger_parts(study.repo)
    manifest = json.loads((study.repo / shr.TRIALS_MANIFEST_PATH).read_text(encoding="utf-8"))
    trials = study.record("E")["trials"]
    assert manifest["sha256"] == trial_ledger_sha256(paths) == trials["ledger_sha256"]
    assert [p.name for p in paths] == [part["file"] for part in manifest["parts"]]
    assert manifest["n_trials_taker"] == len(study.keys) == trials["N_trials_taker"]
    simulator = study.record("C")["trials"]["n_trials_simulator"]
    assert (
        simulator >= 4 and manifest["n_trials"] == len(study.keys) + simulator == trials["N_trials"]
    )
    assert manifest["runs"]["1"]["simulator"]["keys"] == simulator
    assert shr.TRIALS_MANIFEST_PATH.as_posix() in tracked(study.repo)
    assert not {p.relative_to(study.repo).as_posix() for p in paths} & tracked(study.repo)
    registration = shr.Registration.load(repo=study.repo).require_frozen()
    assert registration.value(("search", "result_trials"))["N_trials"] == trials["N_trials"]


def test_stage_d_chose_admitted_and_wrote_its_files(dry: dict[str, Any]) -> None:
    study = dry["study"]
    record = study.record("D")
    assert set(record["statuses_on_D"]) == {"S-C", "S-R"}
    for status in record["statuses_on_D"].values():
        assert status == "admitted" or check_status(status)
    for name in ("leaderboard.csv", "marginals.csv", "nulls.csv"):
        assert study.results(name).exists()


def test_the_amendment_freezes_every_placeholder(dry: dict[str, Any]) -> None:
    from trading_research.market_making.prereg import sha256_bytes

    study = dry["study"]
    record = study.record("E")
    assert record["dry_run"]["ran"] and record["dry_run"]["parity_repeated_ok"]
    # H's stand-in, then F's and P's on the reader-test days, each read end to end
    assert record["dry_run"]["read"] == dict.fromkeys(
        ("stand_in_H", "stand_in_F", "stand_in_P"), "read"
    )
    registration = shr.Registration.load(repo=study.repo).require_frozen()
    document = (study.repo / shr.DOCUMENT_PATH).read_text(encoding="utf-8")
    assert f"Frozen configuration sha256: `{sha256_bytes(registration.text.encode())}`" in document
    assert registration.weights_sha256 and registration.finalist_files_sha256
    reader = registration.value(("data", "ob200_reader_test_result"))
    assert set(reader["days"]) == {"2026-01-05", "2026-01-06", "2026-01-07"}
    assert "The ob200 reader test against" in document


def test_each_read_closes_its_session_with_registered_statuses(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon.reads import read_results
    from trading_research.pipeline.short_horizon_ledger import Ledger

    study = dry["study"]
    entries = Ledger(repo=study.repo, upstream=None).committed_entries()
    for block in ("H", "F"):
        entry = entries[block][-1]
        assert entry["read"] == "first read" and entry["fetch"] and entry["results_sha256"]
        results = read_results(study, block)
        assert set(results["statuses"]) == {"S25", "S-C", "S-R", "S-P", "S-M", "S25-Z"}
        for status in results["statuses"].values():
            check_status(status)
        assert results["stamp"] == "first read" and results["sentences"]


#: Every registered figure, each written in both themes.
FIGURES = ("placebo_search", "placebo_s25", "leaderboard", "marginals", "edge_ladder", "fill_rules",
           "regime_map_taker", "regime_map_entry", "regime_map_maker", "break_view")  # fmt: skip
#: The sentences the dry run fills (the regime-axis one from H's attribution of R*).
SENTENCES = ("search_on_D", "payoff_on_D", "slot_on_block", "S25", "passive_and_maker",
             "regime_axis", "regime_map", "break")  # fmt: skip


def test_the_pages_the_regime_map_and_every_figure_are_written(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon.page_figures import IMAGES, PREFIX
    from trading_research.pipeline.short_horizon.pages import PAGES_FILE, REGIME_MAP_FILE
    from trading_research.reporting import headlines as hl

    study, pages = dry["study"], dry["report"]["pages"]
    written = set(pages["written"])
    for name in (PAGES_FILE, REGIME_MAP_FILE):
        assert study.results(name).relative_to(study.repo).as_posix() in written
    for name in FIGURES:
        for theme in ("light", "dark"):
            path = f"{IMAGES.as_posix()}/{PREFIX}{name}_{theme}.png"
            assert path in written and (study.repo / path).exists(), path
    assert pages["pending"] == [] and pages["changed_on_rewrite"] == []
    assert set(SENTENCES) <= set(pages["sentences"])
    for name, rows in pages["sentences"].items():
        assert rows and all(hl.pattern(name).fullmatch(s) for s in rows), name
    assert {p.relative_to(study.repo).as_posix() for p in (study.repo / IMAGES).glob("*.png")} <= tracked(study.repo)  # fmt: skip


def test_the_pages_file_and_the_regime_map_carry_what_was_registered(dry: dict[str, Any]) -> None:
    import pandas as pd

    from trading_research.pipeline.short_horizon.pages import PAGES_FILE, REGIME_MAP_FILE
    from trading_research.reporting.regime_map import SEGMENT_COLUMNS

    study = dry["study"]
    text = study.results(PAGES_FILE).read_text(encoding="utf-8")
    for heading in ("## Headline sentences", "## The finalists on H, then F and P", "## S25",
                    "## The regime map", "## The break detector", "## The fill rule",
                    "## Rows for docs/findings.md", "## The README summary"):  # fmt: skip
        assert heading in text, heading
    assert "S25 ranks" in text and "on development data, selected" in text
    assert text.count("<picture>") == 2 * len(
        FIGURES
    )  # as the pages file and the README reach them
    assert 'src="../../docs/images/' in text and 'src="docs/images/' in text
    segments = pd.read_csv(study.results(REGIME_MAP_FILE), comment="#")
    assert list(segments.columns) == list(SEGMENT_COLUMNS)
    assert {"D", "H", "F"} <= set(segments["block"]) and segments["slot"].str.startswith("S").all()


def test_each_read_keeps_the_numbers_the_pages_show(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon.page_numbers import close_path
    from trading_research.pipeline.short_horizon.reads import read_results

    study = dry["study"]
    for block in ("H", "F"):
        results = read_results(study, block)
        numbers = results["pages"]
        assert "S25" in numbers["slots"] and set(numbers["slots"]) <= set(results["statuses"])
        for slot, found in numbers["slots"].items():
            assert len(found["values"]) == found["status_days"] <= found["window_days"]
            assert "break_even_per_side" in found
            assert len(found["daily"]["days"]) == len(found["daily"]["value"])
            assert (block == "H") == ("h_before_0801_mean" in found), slot
        assert close_path(study, block).exists()
        fill = numbers["fill_rules"]  # FILL is measured on H's streams alone
        assert set(fill) == ({"S-P", "S25-Z"} if block == "H" else set())
        for rows in fill.values():
            assert [r["rule"] for r in rows] == ["touch", "trade_through", "queue"]
            assert all(r["orders_posted"] >= r["orders_filled"] >= 0 for r in rows)


def test_r_star_is_the_network_admitted_on_d_and_read_on_h(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon.reads import read_results

    study = dry["study"]
    record = study.record("D")
    assert record["statuses_on_D"]["S-R"] == "admitted"
    assert "|R|tcn|" in record["choice"]["R*"]["cell"]
    assert any(
        k.startswith("control__") and k.endswith(".pt") for k in study.record("E")["weights_sha256"]
    )
    results = read_results(study, "H")
    assert results["statuses"]["S-R"] not in ("not assessed", "admitted") and not results[
        "statuses"
    ]["S-R"].startswith("not admitted")
    control = results["measurements"]["S-R"]["measured"]
    assert control["control_ic_label"]["days"] >= 1 and control["control_ic_move"]["days"] >= 1


def test_h_reports_every_registered_measurement_beside_the_statuses(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon.reads import read_results
    from trading_research.pipeline.short_horizon.statuses import S25_WINDOW_END

    results = read_results(dry["study"], "H")
    info, measurements = results["info"], results["measurements"]
    assert info["flags_in_block"] >= 1 and info["flag_sets"] == {"taker": 10, "simulator": 5}
    assert info["gap_days"] == [] and set(info["fill_rules"]) == {"S-P", "S25-Z"}
    guarded = [slot for slot, m in measurements.items() if m["regime_attribution"] is not None]
    assert guarded and all(
        m["regime_attribution"] is None for s, m in measurements.items() if s not in guarded
    )
    for slot in guarded:
        row = measurements[slot]["regime_attribution"]
        assert row["flags"] == info["flags_in_block"] and row["window_h"] in (4, 24, 120)
        assert row["sets"] == len(row["draws"]) == (5 if slot in ("S-P", "S-M") else 10)
        assert any(
            s.startswith(f"{slot} on H: its {row['window_h']} h guard")
            for s in results["sentences"]
        )
    for slot in ("S-P", "S25-Z"):
        fill = measurements[slot]["fill_rules"]
        assert [r["rule"] for r in fill["summary"]] == ["touch", "trade_through", "queue"]
        assert all(r["days"] >= 1 and 0 <= r["fill_ratio"] <= 1 for r in fill["summary"])
        assert set(fill["prediction"]) == {"touch_fill_ratio", "queue_fill_ratio", "holds"}
        assert measurements[slot]["measured"]["exit/passive"]["days"] >= 1
    assert {k for k in measurements["S-P"]["measured"] if k.startswith("regime/")} >= {
        "regime/pull_4h",
        "regime/pull_120h",
    }
    assert len([k for k in measurements["S-M"]["measured"] if k.startswith("regime/")]) >= 5
    window = len([d for d in info["usable_days"] if d <= S25_WINDOW_END.isoformat()])
    for slot, m in measurements.items():  # a taker slot counts its days with a trade
        back = m["gap_days_included"]
        assert back["gap_days"] == 0 and back["days"] <= m["all_of_H"]["days"], slot
    assert measurements["S25"]["gap_days_included"]["days"] <= window < len(info["usable_days"])


def test_f_puts_the_gap_excluded_day_back_as_a_measurement(dry: dict[str, Any]) -> None:
    from trading_research.pipeline.short_horizon.reads import read_results

    results = read_results(dry["study"], "F")
    info = results["info"]
    assert info["gap_days"] == ["2025-09-16"] and info["excluded"]["2025-09-16"] == [
        "sequence_gaps"
    ]
    assert "2025-09-16" not in info["usable_days"]
    for slot, m in results["measurements"].items():
        back, whole = m["gap_days_included"], m["all_of_F"]["days"]
        assert back["gap_days"] == 1, slot
        if slot in ("S-P", "S-M"):  # every usable day counts, zero-filled
            assert back["days"] == whole + 1 == len(info["usable_days"]) + 1, slot
        else:  # a taker slot's days with a trade; a guard may pull the gap day
            assert whole <= back["days"] <= whole + 1, slot
    assert results["split"]["b"] == "2025-09-15" and results["split"]["on_all_of_F"]


def test_the_reads_carry_the_measurements_beside_every_verdict_and_beside_r(
    dry: dict[str, Any],
) -> None:
    """Every slot on H reports the result at every tier, by side, its drawdown and
    counters; S25-Z its registered break-even and its base and cheapest tiers' nets;
    each read the second detector's span and the mean channel; stage D the power of
    the bar, stage C the second detector over D."""
    from trading_research.backtest.costs import tiers_for
    from trading_research.pipeline.short_horizon.reads import read_results

    study = dry["study"]
    results = read_results(study, "H")
    for slot, found in results["measurements"].items():
        beside = found["beside_verdict"]
        assert [r["tier"] for r in beside["tiers"]] == [t.name for t in tiers_for("g1")], slot
        assert set(beside["by_side"]) == {"long", "short"} and "max_drawdown" in beside
    numbers = results["pages"]
    if "S25-Z" in numbers["slots"]:
        assert {"net_base_tier_mean", "net_cheapest_tier_mean", "tiers_paying"} <= set(
            numbers["slots"]["S25-Z"]
        )
    for block in ("H", "F"):
        pages = read_results(study, block)["pages"]
        assert pages["second_detector"]["span"] == block and "mean_channel" in pages
    assert "power_at_bar" in study.record("D") and "fits_lacking_a_class" in study.record("D")
    assert study.record("C")["second_detector_D"]["span"] == "D"
