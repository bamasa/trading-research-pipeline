"""Stage 1's ob200 reader test: each reader-test day loaded as a block day, beside D."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data import fetch_log as fl
from trading_research.data import study_access as sa
from trading_research.data import study_loader as sl
from trading_research.data.study_fetch import book_path
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import DRY_RUN
from trading_research.pipeline.short_horizon.stage1 import reader_test


def _book(day: date, gaps: int, step: float = 0.1) -> pd.DataFrame:
    start = datetime(day.year, day.month, day.day, 1, tzinfo=UTC)
    rows = 40
    data: dict[str, object] = {"timestamp": pd.date_range(start, periods=rows, freq="100ms")}
    mid = 97_000.0 + 0.1 * (np.arange(rows) % 5)
    for level in range(10):
        data[f"bid_price_{level}"] = np.round(mid - step * (level + 1), 1)
        data[f"bid_size_{level}"] = np.full(rows, 0.5)
        data[f"ask_price_{level}"] = np.round(mid + step * level, 1)
        data[f"ask_size_{level}"] = np.full(rows, 0.5)
    frame = pd.DataFrame(data)
    frame.attrs["sequence_gaps"] = gaps
    return frame


def test_each_reader_test_day_is_loaded_as_a_block_day_beside_d(tmp_path: Path) -> None:
    study = Study(tmp_path / "repo", DRY_RUN)
    first, second, third = sa.READER_TEST_DAYS
    log = fl.FetchLog(tmp_path / "log.json")
    for day, gaps in ((first, 0), (second, 9)):
        path = book_path(study.symbol, day, study.roots.book)
        path.parent.mkdir(parents=True, exist_ok=True)
        _book(day, gaps).to_parquet(path, index=False)
        record = log.get(sa.BOOK, day)
        record.status = fl.ON_DISK
        record.stats = {"messages": 100, "median_interval_ms": 10.0, "sequence_gaps": gaps}
    for k, day in enumerate(DRY_RUN.days("D")[:3]):
        record = log.get(sa.BOOK, day)
        record.status = fl.ON_DISK
        record.stats = {"messages": 200 + k, "median_interval_ms": 20.0, "sequence_gaps": k}
    spec = sl.StudyInstrument.from_spec(study.symbol, 0.1, 0.001)
    found = reader_test(study, log, spec, threshold=3, d_findings={"depth": {"info": 3}})
    days = found["days"]
    assert days[first.isoformat()]["passes"] and days[first.isoformat()]["exclusions"] == []
    assert days[first.isoformat()]["findings"]  # every finding by check and severity
    gapped = days[second.isoformat()]
    assert gapped["exclusions"] == [sl.SEQUENCE_GAPS] and gapped["fails_on_gaps_alone"]
    missing = days[third.isoformat()]
    assert not missing["passes"] and not missing["fails_on_gaps_alone"]
    assert missing["exclusions"] == [f"fetch: {fl.PENDING}"]
    assert found["D"]["stream"]["messages"] == {"days": 3, "min": 200.0, "median": 201.0,
                                                 "max": 202.0}  # fmt: skip
    assert found["D"]["findings"] == {"depth": {"info": 3}} and found["max_sequence_gaps"] == 3
