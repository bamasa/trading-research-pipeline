"""The measurements a block read reports beside the statuses, from the per-day values.

None of them enters a status, a kill or Holm; each is a registered measurement:

* **regime attribution** (:func:`attribution`), for a slot with a regime window
  above 0: its mean minus its window-0 twin's, against the 95th percentile of
  the same difference under random flag sets (200 for a taker slot, 20 for a
  simulator slot; set ``k`` from ``default_rng(k)``, as many flags as the block
  has, at distinct usable hours). The flag sets are drawn over the block's
  usable days before any day is left out as stale, and every mean is taken on
  the slot's series as the block's statuses read it (its days with a trade, or
  every usable day for S-P and S-M);
* **FILL on H** (:func:`fill_rules`): S-P's and S25-Z's streams under the three
  fill rules, as round one's comparison tabulates them;
* the **regime measurements**, the **passive exit** and the **control network's
  IC** (:func:`measured_rows`), each as a daily series' mean, day-level t,
  Newey-West t and median day, or a mean over days.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

#: The fill rules in the comparison's order, with the variant each is read from.
FILL_VARIANTS: dict[str, str] = {
    "touch": "fill/touch",
    "trade_through": "fill/trade_through",
    "queue": "true",
}


def series_mean(s: Any, values: np.ndarray) -> float:
    """The mean of a per-day array as the slot's series reads it, over all its days."""
    from trading_research.pipeline.short_horizon.statuses import _series

    picked = _series(s, values, np.ones(len(s.days), dtype=bool))
    return float(picked.mean()) if len(picked) else float("nan")


def attribution(s: Any, flags_ns: np.ndarray, drawn: Sequence[date], n_flags: int,
                policy: str) -> dict[str, Any] | None:  # fmt: skip
    """The regime attribution of a slot with a window above 0, or None.

    ``s.extra`` holds the window-0 twin's per-day values (``twin``), one row of
    per-day values per random flag set (``flagsets``), drawn over ``drawn`` with
    ``n_flags`` flags each, in seed order, and, for a taker slot, the slot under
    the real flags from the same walk tables as its twin and the sets
    (``guarded``; a simulator slot's own series is already that run)."""
    from trading_research.pipeline.regime import random_flag_sets, regime_attribution

    if s.window_h <= 0 or "twin" not in s.extra or "flagsets" not in s.extra:
        return None
    rows = s.extra["flagsets"]
    sets = random_flag_sets(drawn, n_flags, sets=len(rows))
    real = np.asarray(flags_ns, dtype=np.int64)
    lookup = {f.tobytes(): series_mean(s, row) for f, row in zip(sets, rows, strict=True)}
    lookup[real.tobytes()] = series_mean(s, s.extra.get("guarded", s.value))
    found = regime_attribution(lambda f: lookup[np.asarray(f, dtype=np.int64).tobytes()], real,
                               series_mean(s, s.extra["twin"]), drawn, policy=policy,
                               sets=len(rows), n_flags=n_flags)  # fmt: skip
    return found.row() | {"window_h": s.window_h, "draws": [float(x) for x in found.draws],
                          "slot_mean": series_mean(s, s.value), "guarded_mean": lookup[real.tobytes()],
                          "twin_mean": series_mean(s, s.extra["twin"])}  # fmt: skip


def regime_sentence(slot: str, row: dict[str, Any]) -> str:
    """The registered regime-axis sentence of a slot's attribution on H."""
    from trading_research.reporting.headlines import regime_axis

    return regime_axis(slot, int(row["window_h"]), as_attribution(row))


def as_attribution(row: dict[str, Any]) -> Any:
    """An attribution row back as ``pipeline.regime.Attribution`` (for its sentence)."""
    from trading_research.pipeline.regime import Attribution

    return Attribution(row["policy"], row["difference"], row["random_p95"],
                       np.asarray(row["draws"], dtype=np.float64), int(row["flags"]))  # fmt: skip


def _fill_paths(texts: Sequence[str]) -> list[dict[str, Any]]:
    """Each fill path's fills, volume and volume-weighted markouts over the days."""
    total: dict[str, dict[str, float]] = {}
    for text in texts:
        for path, entry in json.loads(text or "{}").items():
            mine = total.setdefault(path, {})
            for name, value in entry.items():
                mine[name] = mine.get(name, 0.0) + float(value)
    out = []
    for path, entry in sorted(total.items()):
        row: dict[str, Any] = {"path": path, "fills": entry.get("fills", 0.0),
                               "volume": entry.get("volume", 0.0)}  # fmt: skip
        for name in [n for n in entry if n.startswith("markout_") and not n.endswith("_weight")]:
            weight = entry.get(name + "_weight", 0.0)
            row[name] = entry[name] / weight if weight > 0 else float("nan")
        out.append(row)
    return out


def fill_rules(rows: pd.DataFrame, slot: str, days: Sequence[date]) -> dict[str, Any]:
    """FILL on a passive slot's stream: per rule, the fill ratio (orders with any fill
    over orders posted), volume ratio, gross, fees and net, markouts by path; and the
    registered prediction read on it (touch at least 0.75, queue at most 0.50)."""
    from trading_research.market_making.fill_rules import BIAS, QUEUE_AT_MOST, TOUCH_AT_LEAST
    from trading_research.market_making.queue import FillRule

    kept = {d.isoformat() for d in days}
    mine = rows[
        (rows["run_slot"] == slot) & rows["day"].isin(kept) & ~rows["excluded"].astype(bool)
    ]
    summary: list[dict[str, Any]] = []
    for rule, variant in FILL_VARIANTS.items():
        part = mine[mine["variant"] == variant]
        sums = {name: float(part[name].sum()) if name in part else 0.0 for name in
                ("orders_posted", "orders_filled", "volume_posted", "volume_filled",
                 "gross_usdt", "fees_usdt", "net_usdt")}  # fmt: skip
        summary.append({"rule": rule, "bias": BIAS[FillRule(rule)], "days": len(part)} | sums | {
            "fill_ratio": sums["orders_filled"] / sums["orders_posted"] if sums["orders_posted"] else float("nan"),
            "volume_ratio": sums["volume_filled"] / sums["volume_posted"] if sums["volume_posted"] else float("nan"),
            "markouts": _fill_paths(list(part["fill_paths"]) if "fill_paths" in part else [])})  # fmt: skip
    ratio: dict[str, float] = {r["rule"]: float(r["fill_ratio"]) for r in summary}
    holds = ratio["touch"] >= TOUCH_AT_LEAST and ratio["queue"] <= QUEUE_AT_MOST
    return {"summary": summary, "prediction": {"touch_fill_ratio": ratio["touch"],
            "queue_fill_ratio": ratio["queue"], "holds": bool(holds)}}  # fmt: skip


def measured_series(s: Any, values: np.ndarray, trades: np.ndarray | None = None) -> dict[str, Any]:
    """A per-day array's numbers as the slot's series reads it: a measurement."""
    from dataclasses import replace

    from trading_research.pipeline.short_horizon.reads import measured

    found: dict[str, Any] = measured(
        replace(s, value=values, trades=s.trades if trades is None else trades)
    )
    return found


def measured_rows(s: Any) -> dict[str, Any]:
    """Every per-day measurement a slot carries: each regime measurement and the passive
    exit as a series, the control network's IC as a mean over days."""
    out: dict[str, Any] = {}
    for name, values in sorted(s.extra.items()):
        if name.startswith(("regime/", "exit/")) and not name.endswith("/trades"):
            trades = s.extra.get(f"{name}/trades")
            out[name] = measured_series(
                s, values, None if trades is None else trades.astype(np.int64)
            )
    for name in ("control_ic_label", "control_ic_move"):
        if name in s.extra:
            finite = s.extra[name][np.isfinite(s.extra[name])]
            out[name] = {
                "days": len(finite),
                "mean": float(finite.mean()) if len(finite) else None,
            }
    return out
