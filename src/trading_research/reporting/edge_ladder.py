"""S25's edge ladder: one table, each step labelled with the one thing that changed.

1. A0 on O at 0/0, crossing and posted under the three fill rules;
2. S25's gross and net on D's walk-forward, at the base fee;
3. S25 on H before 2025-08-01 (its status window), then on all of H and on F,
   labelled inside its information set;
4. S25's fee break-even per side against every published tier for BTCUSDT;
5. S25-Z's net at 0/0, with FILL's fill ratios beside it.

The ladder is filled as the blocks are read: a step whose block has not been
read yet is written with no value and the note "not read yet", never left
out, so the table has the same rows at every stage. Values are in bp of
traded notional (the slot's ``net_bp_trade`` or ``gross_bp_trade``), the
break-even in bp per side, and the fill ratios as shares of posted orders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier, tiers_for
from trading_research.market_making.fill_rules import RULES

LADDER_COLUMNS: tuple[str, ...] = (
    "step",
    "row",
    "changed",
    "block",
    "value",
    "unit",
    "reference",
    "pays",
    "label",
)
#: The one thing each step changes.
CHANGED: dict[int, str] = {
    1: "start: S25's signal under the short-window protocol, fees 0/0",
    2: "protocol: D's walk-forward, at the base fee",
    3: "block: held out",
    4: "fee: every published tier",
    5: "execution: posted at the touch, fees 0/0 (S25-Z)",
}
BP = "bp of traded notional"
INFORMATION_SET = "inside S25's information set"
NOT_READ = "not read yet"
_RULE_NAMES = {"touch": "on touch", "trade_through": "trade-through", "queue": "queue"}


def _value(value: float | None) -> tuple[float, str]:
    return (float("nan"), NOT_READ) if value is None else (float(value), "")


def _row(
    step: int,
    row: str,
    block: str,
    value: float | None,
    unit: str = BP,
    *,
    reference: float = float("nan"),
    pays: object = None,
    label: str = "",
) -> dict[str, object]:
    number, note = _value(value)
    return {
        "step": step,
        "row": row,
        "changed": CHANGED[step],
        "block": block,
        "value": number,
        "unit": unit,
        "reference": reference,
        "pays": pays,
        "label": "; ".join(x for x in (label, note) if x),
    }


def edge_ladder(
    *,
    a0_zero_fee: Mapping[str, float] | None = None,
    s25_d: tuple[float, float] | None = None,
    s25_h_window: float | None = None,
    s25_h_all: float | None = None,
    s25_f: float | None = None,
    s25_break_even: float | None = None,
    break_even_block: str = "H before 2025-08-01",
    s25z_net_zero_fee: float | None = None,
    fill_ratios: Mapping[str, float] | None = None,
    fill_block: str = "O",
    tiers: Sequence[FeeTier] | None = None,
) -> pd.DataFrame:
    """The ladder. ``a0_zero_fee`` maps ``crossing`` and each fill rule
    (``touch``, ``trade_through``, ``queue``) to A0's net at 0/0; ``s25_d`` is
    S25's (gross, net) on D; ``fill_ratios`` maps each rule to FILL's share of
    posted orders with a fill. ``tiers`` defaults to BTCUSDT's (group g1)."""
    a0 = dict(a0_zero_fee or {})
    rules = [r.value for r in RULES]
    unknown = (set(a0) - {"crossing", *rules}) | (set(fill_ratios or {}) - set(rules))
    if unknown:
        raise ValueError(f"unknown A0 executions or fill rules: {sorted(unknown)}")
    rows = [_row(1, "A0, crossing", "O", a0.get("crossing"), label="measurement")]
    rows += [
        _row(1, f"A0, posted, {_RULE_NAMES[r]}", "O", a0.get(r), label="measurement") for r in rules
    ]
    gross, net = (None, None) if s25_d is None else s25_d
    dev = "on development data, never quoted as a result"
    rows += [_row(2, "S25, gross", "D", gross, label=dev), _row(2, "S25, net", "D", net, label=dev)]
    rows += [
        _row(3, "S25, net, status window", "H before 2025-08-01", s25_h_window),
        _row(3, "S25, net", "H", s25_h_all, label=INFORMATION_SET),
        _row(3, "S25, net", "F", s25_f, label=INFORMATION_SET),
    ]
    for tier in tiers if tiers is not None else tiers_for("g1"):
        pays = None if s25_break_even is None else bool(s25_break_even > tier.taker_bp)
        rows.append(
            _row(
                4,
                f"S25, break-even per side, {tier.name}",
                break_even_block,
                s25_break_even,
                "bp per side",
                reference=tier.taker_bp,
                pays=pays,
            )
        )
    rows.append(
        _row(
            5,
            "S25-Z, net at 0/0",
            "H before 2025-08-01",
            s25z_net_zero_fee,
            label="conditional on a zero-fee venue",
        )
    )
    ratios = dict(fill_ratios or {})
    rows += [
        _row(
            5,
            f"fill ratio, {_RULE_NAMES[r]}",
            fill_block,
            ratios.get(r) if fill_ratios else None,
            "share of posted orders",
            label="beside",
        )
        for r in rules
    ]
    table = pd.DataFrame(rows, columns=list(LADDER_COLUMNS))
    table["value"] = table["value"].astype(np.float64)
    return table
