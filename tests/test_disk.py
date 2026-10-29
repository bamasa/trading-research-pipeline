"""The disk check before a stage, and the deletions that bound the peak."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_research.data.disk import (
    InsufficientDisk,
    StageBudget,
    check_disk,
    delete_archives,
    delete_days,
    delete_partials,
    free_gb,
    stage_budget,
)


def test_the_registered_stage_one_budget() -> None:
    """W and D: four workers, 212 replayed days, 15.3 GB of print archives,
    3 + 18 GB of features."""
    budget = stage_budget(book_days=212, print_archives_gb=15.3, workers=4, features_gb=21.0)
    assert budget.archives_in_flight_gb == pytest.approx(3.68)
    assert budget.books_gb == pytest.approx(12.72)
    assert budget.prints_gb == pytest.approx(3.825)
    assert budget.total_gb == pytest.approx(41.225)


def test_a_stage_refuses_unless_free_disk_exceeds_budget_and_margin(tmp_path: Path) -> None:
    budget = StageBudget(archives_in_flight_gb=4, books_gb=10, prints_gb=2, features_gb=0)
    assert check_disk(tmp_path, budget, free=26.5) == 26.5
    with pytest.raises(InsufficientDisk, match=r"needs more than 26\.0 GB"):
        check_disk(tmp_path, budget, free=26.0)
    with pytest.raises(InsufficientDisk):
        check_disk(tmp_path, budget, free=30.0, margin_gb=20.0)


def test_free_space_is_measured_for_a_directory_not_yet_made(tmp_path: Path) -> None:
    assert free_gb(tmp_path / "not" / "yet") == pytest.approx(free_gb(tmp_path), rel=0.01)


def test_nonsense_budgets_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one worker"):
        stage_budget(book_days=1, print_archives_gb=1, workers=0)
    with pytest.raises(ValueError, match="negative"):
        stage_budget(book_days=-1, print_archives_gb=1, workers=1)


def test_deleting_days_frees_their_files_only(tmp_path: Path) -> None:
    for d in (1, 2, 3):
        (tmp_path / f"2025-01-0{d}.parquet").write_bytes(b"x" * 10 * d)
    freed = delete_days(tmp_path, [date(2025, 1, 1), date(2025, 1, 3), date(2025, 1, 9)])
    assert freed == 40
    assert [p.name for p in tmp_path.iterdir()] == ["2025-01-02.parquet"]


def test_leftovers_of_an_interrupted_run_are_cleared(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    (archives / "deep").mkdir(parents=True)
    (archives / "BTCUSDT-2025-01-01-ob500.zip").write_bytes(b"z" * 5)
    (archives / "BTCUSDT2025-01-01.csv.gz").write_bytes(b"g" * 3)
    (archives / "deep" / "x.parquet.part").write_bytes(b"p" * 2)
    (archives / "notes.txt").write_text("kept")
    assert delete_archives(archives) == 10
    assert sorted(p.name for p in archives.rglob("*") if p.is_file()) == ["notes.txt"]
    assert delete_partials(tmp_path / "absent") == 0
    assert delete_archives(tmp_path / "absent") == 0
