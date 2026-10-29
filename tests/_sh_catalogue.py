"""The 2025 search's catalogue and synthetic cell-by-day outcomes, for the
presentation tests.

The keys are generated from the registered axes (``configs/short_horizon_2025.yaml``,
``search.axes`` and ``search.catalogue``); a test checks their sha256 against
the registered one, so the generator cannot drift from the registration.
"""

from __future__ import annotations

import itertools

import numpy as np

from trading_research.validation.choice import CellDays

REGISTERED_SHA256 = "dba3b16a5ffb70ca03974c6a53a9529ba4009cdf796da92c2389b3822288538d"
GATES = ("open", "quiet_out", "tight_only")
HOLDS = (5, 20, 50, 120)
REGIMES = ("none", "pull_4h", "pull_24h", "pull_120h")


def _n(value: float) -> str:
    return f"{value:g}"


def catalogue_keys() -> list[str]:
    """Every taker cell key, C before R, in axis order (not byte order)."""
    keys = []
    for model, inputs, gate, hold, rate, fee, regime in itertools.product(
        ("gbm", "logistic"), ("micro", "micro+tensor"), GATES, HOLDS,
        (25, 50, 100, 240), (0, 5.5), REGIMES,
    ):  # fmt: skip
        keys.append(
            f"BTCUSDT|C|{model}|{inputs}|{gate}|hold_s={_n(hold)}|trades_per_day={_n(rate)}"
            f"|label_fee_bp={_n(fee)}|regime={regime}"
        )
    inputs_of = {"tcn": "tensor+micro", "bound_ridge": "micro"}
    for model, gate, hold, k, beta, regime in itertools.product(
        ("tcn", "bound_ridge"), GATES, HOLDS, (0, 0.5, 1), (0, 1, 2), REGIMES
    ):
        keys.append(
            f"BTCUSDT|R|{model}|{inputs_of[model]}|{gate}|hold_s={_n(hold)}|k={_n(k)}"
            f"|beta_bp={_n(beta)}|regime={regime}"
        )
    return keys


def synthetic_cells(
    cells: int, days: int = 30, *, seed: int = 0, edge: np.ndarray | None = None
) -> tuple[CellDays, np.ndarray]:
    """Random daily net, G and gross (gross = net + 11 bp) with some empty days.

    ``edge`` adds a per-cell drift to net and G. Returns the cells and gross.
    """
    rng = np.random.default_rng(seed)
    trades = rng.integers(0, 12, size=(cells, days))
    shift = np.zeros(cells) if edge is None else np.asarray(edge, dtype=float)
    net = rng.normal(-3.0, 4.0, size=(cells, days)) + shift[:, None]
    g = rng.normal(0.0, 2.0, size=(cells, days)) + shift[:, None]
    net[trades == 0] = np.nan
    g[trades == 0] = np.nan
    return CellDays(trades, net, g), net + 11.0
