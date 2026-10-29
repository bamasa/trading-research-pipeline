"""The README's first figure draws from frames shaped like its inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from trading_research.reporting import showcase
from trading_research.reporting.plots import both_themes


def test_the_overview_renders_in_both_themes(tmp_path: Path) -> None:
    trades = pd.DataFrame({"symbol": ["AUSDT", "BUSDT"] * 20, "net_bp": [3.0, -1.0] * 20})
    variants = pd.DataFrame(
        {
            "variant": list(showcase.VARIANTS),
            "net_per_trade_bp": [7.0, 14.0, 17.0, 18.0, 18.5],
            "two_se_bp": [3.0, 2.5, 2.7, 2.6, 2.7],
        }
    )
    quoter = pd.DataFrame(
        [
            {"day": f"2024-02-{26 + d}", "value": v, "hypothesis": name}
            for name, v in (("B3a", 0.8), ("B3b", 2.2), ("B1", -0.1))
            for d in range(3)
        ]
    )
    paths = both_themes(
        lambda path: showcase.overview(trades, variants, quoter, path), tmp_path / "s.png"
    )
    assert [p.name for p in paths] == ["s_light.png", "s_dark.png"]
    assert all(p.stat().st_size > 10_000 for p in paths)
