"""Tests for the report renderer.

A report is the only part of this project most readers will see, so the things
worth testing are the ones that decide whether it can mislead: that assumptions
appear before numbers, that synthetic output is labelled as such, that the
per-fold spread is shown rather than only the mean, and that the verdict
matches the arithmetic instead of being written optimistically by habit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lobml.reporting.report import ReportContext, summarise, to_markdown, write_report


def make_results(net_by_model: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for model, nets in net_by_model.items():
        for fold, net in enumerate(nets):
            trades = 0 if net == 0 else 100
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "block": "test",
                    "rows": 10_000,
                    "trades": trades,
                    "trade_rate": trades / 10_000,
                    "gross_bp": net + 50.0,
                    "cost_bp": 50.0,
                    "net_bp": net,
                    "net_per_trade_bp": net / trades if trades else 0.0,
                    "hit_rate": 0.51,
                    "min_confidence": 0.5,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def context() -> ReportContext:
    return ReportContext(
        symbol="BTCUSDT",
        source="Binance USD-M futures bookTicker",
        period="2024-02-01 .. 2024-03-09",
        grid="100 ms",
        horizon_obs=100,
        horizon_seconds=10.0,
        cost_round_trip_bp=11.02,
        cost_description={"execution": "taker", "fee_bp_per_side": 5.0, "slippage_bp": 0.5},
        label_threshold_bp=11.02,
        class_balance={"buy": 0.0054, "hold": 0.9889, "sell": 0.0056},
        perfect_foresight_share=0.0111,
        n_folds=7,
        fold_layout="14/7/7 days, step 1 day",
        features=["spread_bp", "queue_imbalance"],
    )


LOSING = {"always_hold": [0.0] * 7, "logistic": [-7000.0] * 7, "xgboost": [-24000.0] * 7}
WINNING = {"always_hold": [0.0] * 7, "xgboost": [500.0] * 7}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_counts_positive_folds() -> None:
    mixed = {"model_a": [10.0, -5.0, 3.0, -1.0, 8.0, -2.0, 4.0]}
    summary = summarise(make_results(mixed))
    assert summary.loc["model_a", "folds_positive"] == 4
    assert summary.loc["model_a", "folds"] == 7


def test_summary_reports_worst_and_best_fold() -> None:
    """A mean hides a fold that lost badly; the range does not."""
    summary = summarise(make_results({"m": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, -600.0]}))
    assert summary.loc["m", "net_bp_worst"] == -600.0
    assert summary.loc["m", "net_bp_best"] == 100.0


def test_summary_orders_by_net_result() -> None:
    summary = summarise(make_results(LOSING))
    assert summary.index[0] == "always_hold"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_assumptions_come_before_the_numbers(context: ReportContext) -> None:
    """A reader must meet the cost floor before meeting the result."""
    text = to_markdown(make_results(LOSING), context)
    assert text.index("What was assumed") < text.index("Test results")
    assert text.index("Round trip") < text.index("Test results")


def test_cost_floor_is_stated_with_the_ceiling(context: ReportContext) -> None:
    text = to_markdown(make_results(LOSING), context)
    assert "1.11%" in text
    assert "11.02 bp" in text
    assert "assumes the direction is predicted perfectly" in text


def test_every_fold_is_shown_not_just_the_mean(context: ReportContext) -> None:
    text = to_markdown(make_results(LOSING), context)
    assert "Per fold" in text

    # Every fold appears as its own row, so a single bad one cannot hide in a mean.
    per_fold_section = text.split("### Per fold")[1]
    for fold in range(7):
        assert f"| {fold} |" in per_fold_section

    # The overlap caveat has to travel with the table it qualifies, or a reader
    # will treat seven near-identical windows as seven independent trials.
    assert "96%" in per_fold_section
    assert "independent trials" in per_fold_section


def test_a_losing_result_is_reported_as_losing(context: ReportContext) -> None:
    """The verdict must follow the arithmetic, not the hope."""
    text = to_markdown(make_results(LOSING), context)
    assert "No model beat not trading" in text


def test_a_winning_result_is_reported_with_conditions(context: ReportContext) -> None:
    """A positive number is presented with the checks that would qualify it."""
    text = to_markdown(make_results(WINNING), context)
    assert "No model beat not trading" not in text
    assert "xgboost" in text
    assert "rescued by one" in text


def test_a_result_merely_equal_to_holding_is_not_called_a_win(context: ReportContext) -> None:
    """Matching 'do nothing' while paying costs is not an edge."""
    text = to_markdown(make_results({"always_hold": [0.0] * 7, "m": [0.0] * 7}), context)
    assert "No model beat not trading" in text


def test_synthetic_runs_are_labelled_at_the_top(context: ReportContext) -> None:
    """Demo output must not be quotable as a real result."""
    synthetic = ReportContext(**{**context.__dict__, "synthetic": True})
    text = to_markdown(make_results(LOSING), synthetic)
    assert "SYNTHETIC DATA" in text.split("## What was assumed")[0]


def test_limitations_are_always_present(context: ReportContext) -> None:
    text = to_markdown(make_results(WINNING), context)
    assert "not evidence that any strategy is or was profitable" in text
    assert "no market impact" in text


def test_features_are_listed(context: ReportContext) -> None:
    text = to_markdown(make_results(LOSING), context)
    assert "spread_bp" in text
    assert "queue_imbalance" in text


def test_report_writes_to_disk(tmp_path, context: ReportContext) -> None:
    target = write_report(make_results(LOSING), context, tmp_path / "sub" / "results.md")
    assert target.exists()
    assert "Walk-forward results" in target.read_text(encoding="utf-8")
