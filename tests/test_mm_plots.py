"""The market-making charts draw from tables shaped like the committed ones."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from trading_research.reporting import mm_plots
from trading_research.reporting.plots import both_themes


def test_every_chart_renders_in_both_themes(tmp_path: Path) -> None:
    markouts = pd.DataFrame(
        [
            {
                "strategy": s,
                "symbol": "BICOUSDT",
                "path": p,
                **{f"markout_bp_{h}": v for h in ("1s", "5s", "30s")},
            }
            for s, v in (("S0", -1.0), ("S1", 2.0))
            for p in ("queue", "through")
        ]
        + [
            {
                "strategy": "market-wide passive",
                "symbol": "BICOUSDT",
                "path": "benchmark",
                "markout_bp_1s": -0.5,
                "markout_bp_5s": -0.9,
                "markout_bp_30s": -0.6,
            }
        ]
    )
    parts = {
        f"{n}_usd_day": v
        for n, v in zip(
            ("spread", "adverse", "inventory", "fees", "funding", "net"),
            (3.0, -1.5, -0.4, 0.8, 0.01, 0.31),
            strict=True,
        )
    }
    strategies = pd.DataFrame(
        [{"strategy": s, "symbol": "BICOUSDT", **parts} for s in ("S0", "S1", "S4")]
    )
    inventory = pd.DataFrame(
        {
            "strategy": "S1",
            "symbol": "BICOUSDT",
            "time": pd.date_range("2024-02-26", periods=50, freq="5min").strftime("%Y-%m-%d %H:%M"),
            "position": [float(i % 7 - 3) for i in range(50)],
            "soft_limit": 6.0,
            "hard_limit": 7.0,
        }
    )
    placebos = pd.DataFrame(
        [
            {"placebo": name, "seed": k, "value": k * 0.1, "true_value": 2.0}
            for name in ("H2.1", "H3")
            for k in range(20)
        ]
    )
    ladder = pd.DataFrame(
        [
            {"symbol": s, "rung": r, "net_bp_turnover": r - 3.0}
            for s in ("BICOUSDT", "XRPUSDT")
            for r in range(7)
        ]
    )
    fees = pd.DataFrame(
        [
            {
                "strategy": "S0",
                "symbol": "BICOUSDT",
                "breakeven_maker_bp": 0.4,
                "best_published_maker_bp": 0.0,
                "mm_programme_maker_bp": -1.0,
            },
            {
                "strategy": "S1",
                "symbol": "BICOUSDT",
                "breakeven_maker_bp": math.inf,
                "best_published_maker_bp": 0.0,
                "mm_programme_maker_bp": -1.0,
            },
            {
                "strategy": "S4",
                "symbol": "XRPUSDT",
                "breakeven_maker_bp": -math.inf,
                "best_published_maker_bp": 0.0,
                "mm_programme_maker_bp": -1.0,
            },
        ]
    )
    charts = {
        "markouts": lambda p: mm_plots.markouts_by_path(markouts, p, symbol="BICOUSDT"),
        "decomposition": lambda p: mm_plots.decomposition(strategies, p, symbol="BICOUSDT"),
        "inventory": lambda p: mm_plots.inventory(inventory, p),
        "placebos": lambda p: mm_plots.placebos(placebos, p),
        "ladder": lambda p: mm_plots.ladder(ladder, p),
        "fees": lambda p: mm_plots.fee_breakeven(fees, p),
    }
    for name, draw in charts.items():
        for written in both_themes(draw, tmp_path / f"{name}.png"):
            assert written.exists() and written.stat().st_size > 1000
