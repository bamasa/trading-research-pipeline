"""The information audit, and the two feature sources it was built to judge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.evaluation.information import AuditError, audit, linear_ceiling
from trading_research.features.cross import build_cross_features
from trading_research.features.flow import align_to_grid, build_flow_features

# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def _blocks(n: int = 8_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 3))
    x = pd.DataFrame(
        {
            "real_a": signal + rng.normal(0, 2, n),
            "real_b": signal + rng.normal(0, 2, n),
            "empty_a": noise[:, 0],
            "empty_b": noise[:, 1],
            "empty_c": noise[:, 2],
        }
    )
    y = pd.Series(signal * 10 + rng.normal(0, 30, n))
    return x, y


def test_the_ceiling_finds_signal_where_it_is_and_not_where_it_is_not() -> None:
    x, y = _blocks()
    train, test = slice(0, 6_000), slice(6_000, 8_000)
    real, _ = linear_ceiling(
        x.iloc[train][["real_a", "real_b"]],
        y.iloc[train],
        x.iloc[test][["real_a", "real_b"]],
        y.iloc[test],
    )
    empty, _ = linear_ceiling(
        x.iloc[train][["empty_a", "empty_b", "empty_c"]],
        y.iloc[train],
        x.iloc[test][["empty_a", "empty_b", "empty_c"]],
        y.iloc[test],
    )
    assert real > 0.15
    assert abs(empty) < 0.06


def test_a_duplicate_source_adds_nothing_however_well_it_scores_alone() -> None:
    """The column the audit exists for.

    A block that correlates well on its own but says the same thing as one
    already accepted must show a near-zero incremental contribution, or the
    audit would send someone off to build a source they already have.
    """
    rng = np.random.default_rng(1)
    n = 8_000
    signal = rng.normal(0, 1, n)
    x = pd.DataFrame(
        {
            "first_a": signal + rng.normal(0, 1, n),
            "first_b": signal + rng.normal(0, 1, n),
            "copy_a": signal + rng.normal(0, 1, n),
            "copy_b": signal + rng.normal(0, 1, n),
        }
    )
    y = pd.Series(signal * 10 + rng.normal(0, 20, n))
    train, test = slice(0, 6_000), slice(6_000, n)
    table = audit(
        {"first": ["first_a", "first_b"], "copy": ["copy_a", "copy_b"]},
        x.iloc[train],
        y.iloc[train],
        x.iloc[test],
        y.iloc[test],
        with_mutual_information=False,
    )
    second = table[table["order_accepted"] == 2].iloc[0]
    assert second["linear_ic"] > 0.15, "the duplicate scores well alone"
    assert abs(second["incremental_ic"]) < second["linear_ic"] / 2, "and adds little"


def test_a_genuinely_new_source_keeps_its_contribution() -> None:
    rng = np.random.default_rng(2)
    n = 8_000
    first, second = rng.normal(0, 1, n), rng.normal(0, 1, n)
    x = pd.DataFrame(
        {
            "first_a": first + rng.normal(0, 1, n),
            "second_a": second + rng.normal(0, 1, n),
        }
    )
    y = pd.Series((first + second) * 10 + rng.normal(0, 20, n))
    train, test = slice(0, 6_000), slice(6_000, n)
    table = audit(
        {"first": ["first_a"], "second": ["second_a"]},
        x.iloc[train],
        y.iloc[train],
        x.iloc[test],
        y.iloc[test],
        with_mutual_information=False,
    )
    later = table[table["order_accepted"] == 2].iloc[0]
    assert abs(later["incremental_ic"]) > 0.1


def test_the_strongest_source_is_accepted_first() -> None:
    x, y = _blocks()
    train, test = slice(0, 6_000), slice(6_000, 8_000)
    table = audit(
        {"real": ["real_a", "real_b"], "empty": ["empty_a", "empty_b", "empty_c"]},
        x.iloc[train],
        y.iloc[train],
        x.iloc[test],
        y.iloc[test],
        with_mutual_information=False,
    )
    assert table.iloc[0]["source"] == "real"


def test_the_best_single_column_is_named() -> None:
    """A block owing everything to one column should be visible as such."""
    x, y = _blocks()
    train, test = slice(0, 6_000), slice(6_000, 8_000)
    table = audit(
        {"mixed": ["real_a", "empty_a", "empty_b"]},
        x.iloc[train],
        y.iloc[train],
        x.iloc[test],
        y.iloc[test],
        with_mutual_information=False,
    )
    assert table.iloc[0]["best_column"] == "real_a"


def test_too_little_data_is_refused() -> None:
    x, y = _blocks(n=300)
    with pytest.raises(AuditError, match="usable training rows"):
        linear_ceiling(x, y, x, y)


# ---------------------------------------------------------------------------
# Trade flow
# ---------------------------------------------------------------------------


def _trades(stamps: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 500
    times = pd.to_datetime(rng.choice(stamps.to_numpy(), size=n))
    return (
        pd.DataFrame(
            {
                "timestamp": times,
                "price": 100 + rng.normal(0, 0.1, n),
                "size": rng.exponential(5, n),
                "aggressor": rng.choice([-1, 1], n),
            }
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def test_flow_imbalance_is_bounded_and_signed_correctly() -> None:
    stamps = pd.Series(pd.date_range("2024-02-01", periods=400, freq="5s", tz="UTC"))
    mid = np.full(len(stamps), 100.0)
    trades = _trades(stamps)
    features = build_flow_features(align_to_grid(trades, stamps, mid))
    for column in [c for c in features.columns if c.startswith("flow_imbalance")]:
        values = features[column].dropna()
        assert values.between(-1.0, 1.0).all()


def test_all_buying_gives_imbalance_one() -> None:
    """The sign convention, asserted rather than trusted."""
    stamps = pd.Series(pd.date_range("2024-02-01", periods=100, freq="5s", tz="UTC"))
    mid = np.full(len(stamps), 100.0)
    trades = pd.DataFrame(
        {
            "timestamp": stamps.iloc[10:60].to_numpy(),
            "price": 100.0,
            "size": 1.0,
            "aggressor": 1,
        }
    )
    features = build_flow_features(align_to_grid(trades, stamps, mid))
    assert features["flow_imbalance_6"].iloc[20] == pytest.approx(1.0)

    trades["aggressor"] = -1
    features = build_flow_features(align_to_grid(trades, stamps, mid))
    assert features["flow_imbalance_6"].iloc[20] == pytest.approx(-1.0)


def test_prints_outside_the_grid_are_dropped_not_misfiled() -> None:
    stamps = pd.Series(pd.date_range("2024-02-01", periods=50, freq="5s", tz="UTC"))
    mid = np.full(len(stamps), 100.0)
    early = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "price": [100.0],
            "size": [999.0],
            "aggressor": [1],
        }
    )
    buckets = align_to_grid(early, stamps, mid)
    assert buckets["buy_volume"].sum() == 0.0


# ---------------------------------------------------------------------------
# Cross-instrument
# ---------------------------------------------------------------------------


def _panel(n: int = 3_000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1e-3, n))
    return pd.DataFrame(
        {
            "LEAD": common,
            "A": common * 0.8 + np.cumsum(rng.normal(0, 5e-4, n)),
            "B": common * 1.2 + np.cumsum(rng.normal(0, 5e-4, n)),
            "C": np.cumsum(rng.normal(0, 1e-3, n)),
        }
    )


def test_the_index_excludes_the_instrument_being_described() -> None:
    """An index containing the target partly predicts it by construction."""
    panel = _panel()
    features = build_cross_features(panel, "A", leader="LEAD")
    others = panel[["LEAD", "B", "C"]].diff(24).mean(axis=1) * 1e4
    assert np.allclose(features["index_return_24"].dropna(), others.dropna(), atol=1e-9)


def test_catch_up_is_positive_when_the_leader_moved_and_the_name_did_not() -> None:
    panel = _panel()
    panel["A"] = panel["A"].iloc[0]  # this one goes nowhere
    panel["LEAD"] = panel["LEAD"].iloc[0] + np.linspace(0, 0.02, len(panel))
    features = build_cross_features(panel, "A", leader="LEAD")
    assert features["catch_up_leader_24"].dropna().mean() > 0


def test_beta_recovers_a_known_sensitivity() -> None:
    rng = np.random.default_rng(5)
    n = 6_000
    lead = np.cumsum(rng.normal(0, 1e-3, n))
    panel = pd.DataFrame(
        {
            "LEAD": lead,
            "HALF": 0.5 * lead,
            "OTHER": np.cumsum(rng.normal(0, 1e-3, n)),
        }
    )
    features = build_cross_features(panel, "HALF", leader="LEAD", beta_window=1_000)
    assert features["beta_to_leader"].dropna().median() == pytest.approx(0.5, abs=0.05)


def test_a_stale_book_is_flagged() -> None:
    panel = _panel()
    panel.iloc[-300:, panel.columns.get_loc("C")] = panel["C"].iloc[-301]
    features = build_cross_features(panel, "C", leader="LEAD")
    assert features["own_stale"].iloc[-1] == 1.0
    assert features["own_stale"].iloc[100] == 0.0


def test_a_panel_too_narrow_is_refused() -> None:
    panel = _panel()[["LEAD", "A"]]
    with pytest.raises(ValueError, match="at least three"):
        build_cross_features(panel, "A")


def test_an_unknown_symbol_is_named_in_the_error() -> None:
    with pytest.raises(KeyError, match="ZZZ"):
        build_cross_features(_panel(), "ZZZ")
