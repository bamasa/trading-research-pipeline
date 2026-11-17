"""A day's numbers from one run: the decomposition, ``y``, re-pricings, capacity, hours.

All money is USDT. Per passive fill at price ``p``, size ``q``, side ``s`` and
stamp ``t``, with ``mid_ref`` the mid proxy before ``t`` and ``mid(t + h)`` the
mid proxy at or before ``t + h``::

    spread = s (mid_ref - p) q        making = s (mid(t + 5 s) - p) q
    adverse = making - spread         rebate = -(the maker fees booked)
    inventory = net - making - rebate + taker fees - funding

``y`` is the day's net per 100 USDT of clip: ``100 net / (clip m)``, ``m`` the
mid proxy at 00:10. Every other per-day quantity is scaled the same way. The
re-pricings at another maker fee ``f`` are exact, since no quoter reads the fee:
``net - (f - maker) 1e-4 x passive turnover``. The flatten's cost is its price
against the mid proxy at its 23:59 decision plus its taker fee; the share of time
at the soft limit and the share of time a gate is open are over the quoting
seconds, 00:10 to 23:58.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit.engine import DayRun, QuoterInputs, RunSettings
from trading_research.mm_bybit.tape import DayTape

NS_PER_S = 1_000_000_000
HORIZONS_S: tuple[int, ...] = (1, 5, 30)
MAKING_HORIZON_S = 5
QUOTE_FROM_S = 600
REPRICED_BP: dict[str, float] = {
    "ex_rebate_y": 0.0,
    "negative_control_y": 7.5,
    "positive_control_y": -10.0,
}


def _soft_seconds(run: DayRun, soft: float, start_ns: int, end_ns: int) -> float:
    """Seconds between ``start_ns`` and ``end_ns`` with |position| at the soft limit."""
    if len(run.fill_ts) == 0:
        return 0.0
    position = np.cumsum(run.fill_side * run.fill_size)
    stamps = np.clip(np.append(run.fill_ts, end_ns), start_ns, end_ns)
    at_limit = np.abs(position) >= soft * (1 - 1e-9)
    return float(np.sum(np.diff(stamps)[at_limit])) / NS_PER_S


def gate_open_share(
    tape: DayTape, cell: q.Cell, inputs: QuoterInputs, first_s: int, last_s: int
) -> float:
    """The share of the quoting seconds at which the cell's gate is open (1 ungated)."""
    if not cell.gated:
        return 1.0
    spread = tape.sec_spread[first_s:last_s]
    vol = tape.sec_vol[first_s:last_s]
    if cell.gate == "spread":
        is_open = spread >= cell.param
    else:
        level = inputs.quantiles[cell.param]
        is_open = (vol <= level) if cell.gate == "vle" else (vol >= level)
        is_open &= np.isfinite(vol)
    return float(np.mean(is_open)) if len(is_open) else 0.0


def day_summary(
    tape: DayTape,
    run: DayRun,
    *,
    cell: q.Cell,
    inputs: QuoterInputs,
    settings: RunSettings,
) -> dict[str, Any]:
    """Every per-day number the study reports for one run."""
    clip, maker_bp, soft = settings.clip, settings.maker_bp, settings.soft
    maker = run.fill_maker
    side, price, size = run.fill_side[maker], run.fill_price[maker], run.fill_size[maker]
    ts = run.fill_ts[maker]
    notional = price * size
    turnover = float(notional.sum())
    mid_ref = tape.mid_at(ts, strictly_before=True)
    mid_ref = np.where(np.isfinite(mid_ref) & (mid_ref > 0), mid_ref, price)
    marks = {h: tape.mid_at(ts + h * NS_PER_S) for h in HORIZONS_S}
    markout = {
        h: float(np.sum(side * (np.where(np.isfinite(m), m, price) - price) * size))
        for h, m in marks.items()
    }
    making = markout[MAKING_HORIZON_S]
    spread = float(np.sum(side * (mid_ref - price) * size))
    rebate = -float(run.fill_fee[maker].sum())
    taker_fees = float(run.fill_fee[~maker].sum())
    inventory = run.net - making - rebate + taker_fees - run.funding
    # The flatten: its price against the mid proxy at its decision, plus its fee.
    taker = ~maker
    decided = tape.start_ns + settings.flatten_s * NS_PER_S
    flatten_mid = float(tape.mid_at(np.array([decided]))[0])
    flatten_price_cost = (
        float(np.sum(run.fill_side[taker] * (run.fill_price[taker] - flatten_mid)
                     * run.fill_size[taker]))
        if taker.any() and np.isfinite(flatten_mid) else 0.0
    )  # fmt: skip
    m = float(tape.sec_mid[QUOTE_FROM_S])
    if not m > 0:
        m = float(np.nanmedian(tape.sec_mid))
    scale = 100.0 / (clip * m)
    hourly = np.diff(np.concatenate([[0.0], run.hourly_equity]))
    sell_volume = float(tape.size[tape.agg < 0].sum())
    buy_volume = float(tape.size[tape.agg > 0].sum())
    bought = float(size[side > 0].sum())
    sold = float(size[side < 0].sum())
    capacity = max(
        bought / sell_volume if sell_volume > 0 else (np.inf if bought > 0 else 0.0),
        sold / buy_volume if buy_volume > 0 else (np.inf if sold > 0 else 0.0),
    )
    quote_from, quote_to = settings.warmup_s, settings.stop_s
    taker_turnover = float((run.fill_price[taker] * run.fill_size[taker]).sum())
    out: dict[str, Any] = {
        "y": run.net * scale,
        "making_y": making * scale,
        "spread_y": spread * scale,
        "adverse_y": (making - spread) * scale,
        "rebate_y": rebate * scale,
        "taker_fees_y": taker_fees * scale,
        "funding_y": run.funding * scale,
        "inventory_y": inventory * scale,
        "flatten_cost_y": (flatten_price_cost + taker_fees) * scale,
        **{
            name: (run.net - (bp - maker_bp) * 1e-4 * turnover) * scale
            for name, bp in REPRICED_BP.items()
        },
        "passive_fills": int(maker.sum()),
        "passive_turnover": turnover,
        "net_usdt": run.net,
        "making_usdt": making,
        "rebate_usdt": rebate,
        "markout_usdt": {str(h): v for h, v in markout.items()},
        "capacity": float(capacity),
        "max_abs_position": run.max_abs_position,
        "soft_limit_share": _soft_seconds(
            run, soft, tape.start_ns + quote_from * NS_PER_S, tape.start_ns + quote_to * NS_PER_S
        )
        / (quote_to - quote_from),
        "gate_open_share": gate_open_share(tape, cell, inputs, quote_from, quote_to),
        "taker_turnover": run.taker_turnover,
        "identity_ok": bool(run.identity_ok),
        "limit_breach": float(run.counters.get("limit_breach", 0.0)),
        "not_flat": float(run.counters.get("not_flat_at_end", 0.0)),
        "turnover_ok": bool(
            np.isclose(turnover, run.maker_turnover, rtol=1e-9, atol=1e-9)
            and np.isclose(taker_turnover, run.taker_turnover, rtol=1e-9, atol=1e-9)
        ),
        "clip_notional": clip * m,
        "hourly_y": (hourly * scale).tolist(),
        "counters": run.counters,
    }
    return out


def market_markout_bp(tape: DayTape, horizon_s: int = MAKING_HORIZON_S) -> float:
    """Every print scored from its resting side at ``horizon_s``, volume-weighted, bp."""
    later = tape.mid_at(tape.ts + horizon_s * NS_PER_S)
    resting = -tape.agg
    ok = np.isfinite(later)
    weights = tape.price[ok] * tape.size[ok]
    values = resting[ok] * (later[ok] - tape.price[ok]) / tape.price[ok] * 1e4
    return float(np.sum(values * weights) / np.sum(weights)) if weights.sum() > 0 else 0.0
