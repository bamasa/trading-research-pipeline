"""The spot study's plan: its rows and cells, D's folds and the purge, as registered."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

from trading_research.spot import access as sa
from trading_research.spot import plan as pl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "spot_2026.yaml").read_text(encoding="utf-8"))


def test_the_folds_are_the_registered_ones() -> None:
    fits = CONFIG["models"]["fits"]
    for name, (train, inner, test) in pl.FOLDS.items():
        assert list(train) == fits[name]["train"] and list(inner) == fits[name]["theta"]
        if test is not None:
            assert test[0].strftime("%Y-%m") == fits[name]["scored"]
    days = sa.block_days("D")
    found = pl.folds(days)
    assert len(found["june"].test) + len(found["july"].test) == CONFIG["blocks"]["D"]["test_days"]
    assert found["frozen"].cutoff_ns == pl.midnight_ns(date(2026, 8, 1))
    assert found["june"].inner_start == date(2026, 5, 25) and not found["frozen"].test


def test_every_row_runs_its_registered_cells_and_fits() -> None:
    per_slot = CONFIG["models"]["fits_per_slot"]
    for name, row in pl.ROWS.items():
        fits = len(row.fits) * 3 if row.fitted else 0
        assert fits == per_slot[name], name
        assert row.own == pl.Cell(50.0, 240.0, 0.0) and row.own in row.cells
    for name in ("S25-PF", "S25-BB", "S25-T"):
        hood = CONFIG["slots"][name]["neighbourhood"]
        assert {c.hold_s for c in pl.ROWS[name].cells} == set(hood["hold_s"])
        assert {c.rate for c in pl.ROWS[name].cells} == set(hood["trades_per_day"])
        assert {c.fee_bp for c in pl.ROWS[name].cells} == set(hood["label_fee_bp"])
        assert len(pl.ROWS[name].cells) == 12
    assert list(pl.ROWS["S25-PFT"].inputs) == [*CONFIG["features"]["print_members"],
                                               *CONFIG["features"]["touch_proxy_members"]]  # fmt: skip


def test_training_rows_respect_the_purge_and_the_label_reads() -> None:
    fold = pl.folds(sa.block_days("D"))["june"]
    start = fold.inner_start_ns
    hold = 50.0
    stamps = np.array([start - pl.purge_ns(hold) - 1, start - pl.purge_ns(hold), start - 10])
    reads = np.array([start - 5, start - 5, start - 5])
    assert fold.train_keep(stamps, reads, hold).tolist() == [True, False, False]
    late = np.array([start - pl.purge_ns(hold) - 1] * 3)
    assert fold.train_keep(late, np.array([start, start + 1, -1]), hold).tolist() == [
        True,
        False,
        False,
    ]


def test_inner_rows_end_before_the_cutoff_less_the_purge() -> None:
    fold = pl.folds(sa.block_days("D"))["june"]
    cutoff = fold.cutoff_ns
    stamps = np.array([fold.inner_start_ns - 1, fold.inner_start_ns, cutoff - pl.purge_ns(20) - 1,
                       cutoff - pl.purge_ns(20), cutoff + 5])  # fmt: skip
    assert fold.inner_keep(stamps, 20.0).tolist() == [False, True, True, False, False]


def test_the_dry_scale_plans_days_of_the_registered_blocks() -> None:
    for block in ("W", "D", "H", "P"):
        start, end = sa.BLOCKS[block]
        assert all(start <= d <= end for d in pl.DRY_RUN.days(block))
    assert pl.REGISTERED.days("H") == tuple(sa.block_days("H"))
    shifts = pl.REGISTERED.null_b_shifts(61)
    assert (shifts[0], shifts[-1]) == (7, 54)
    assert all(d - timedelta(days=1) for d in pl.DRY_RUN.days("D"))
