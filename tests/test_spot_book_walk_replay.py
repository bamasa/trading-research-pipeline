"""The 200-level walk replay keeps the ten-level grid: every row, its first ten levels
exactly, the deeper levels the archive holds, the missing ones empty and counted."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trading_research.backtest.walk import walk_book
from trading_research.features.timed import stamps_ns
from trading_research.spot.bybit_book import replay

BASE_MS = 1_786_000_000_000  # 2026-08-06


def _levels(side: str, n: int, size: str = "0.5") -> list[list[str]]:
    sign = -1 if side == "b" else 1
    return [[f"{60000.0 + sign * (0.5 + k):.2f}", size] for k in range(n)]


def write_archive(path: Path, ask_levels: int = 200) -> Path:
    msgs = [{"topic": "orderbook.200.BTCUSDT", "ts": BASE_MS, "type": "snapshot",
             "data": {"s": "BTCUSDT", "b": _levels("b", 200), "a": _levels("a", ask_levels),
                      "u": 1, "seq": 1}}]  # fmt: skip
    u = 1
    for k in range(1, 60):
        u += 1
        delta: dict = {"s": "BTCUSDT", "b": [], "a": [], "u": u, "seq": u}
        if k == 10:  # a level deleted, its replacement one interval later
            delta["a"] = [[f"{60000.0 + 199.5:.2f}", "0"]]
        elif k == 11:
            delta["a"] = [[f"{60000.0 + 200.5:.2f}", "0.5"]]
        elif k == 30:  # a thin moment: the bids keep nine levels for an interval
            delta["b"] = [[f"{60000.0 - (0.5 + j):.2f}", "0"] for j in range(9, 200)]
        elif k == 31:
            delta["b"] = _levels("b", 200)
        else:
            delta["b"] = [[f"{60000.0 - 0.5:.2f}", f"{0.5 + k / 1000:.3f}"]]
        msgs.append({"topic": "orderbook.200.BTCUSDT", "ts": BASE_MS + 100 * k, "type": "delta",
                     "data": delta})  # fmt: skip
    path.write_text("\n".join(json.dumps(m) for m in msgs) + "\n", encoding="utf-8")
    return path


def test_the_walk_replay_keeps_every_row_of_the_ten_level_grid(tmp_path: Path) -> None:
    archive = write_archive(tmp_path / "ob200.data")
    ten, deep = replay(archive, depth=10), replay(archive, depth=200)
    assert len(ten) == len(deep) == 59  # the nine-level interval has no row at either depth
    assert np.array_equal(stamps_ns(ten), stamps_ns(deep))
    assert ten["sequence_id"].tolist() == deep["sequence_id"].tolist()
    names = [c for c in ten.columns if c.startswith(("bid_", "ask_"))]
    assert np.array_equal(ten[names].to_numpy(np.float64), deep[names].to_numpy(np.float64))
    assert deep.attrs["deep_levels_missing_rows"] == 1  # the interval with 199 asks
    row = deep.iloc[int(np.flatnonzero(deep["ask_price_199"].isna())[0])]
    assert np.isfinite(row["ask_price_198"]) and np.isnan(row["ask_size_199"])
    asks = deep[[f"ask_price_{k}" for k in range(200)]].to_numpy()
    sizes = deep[[f"ask_size_{k}" for k in range(200)]].to_numpy()
    walked = walk_book(asks, sizes, 150.0)  # deeper than the archive: the empty level is none
    assert np.isfinite(walked.vwap).all() and (walked.beyond > 0).all()
    assert list(deep.columns[:4]) == ["timestamp", "symbol", "source", "sequence_id"]
