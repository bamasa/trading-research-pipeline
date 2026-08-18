"""Holding period and retraining schedule, searched together.

§10 searched the retraining schedule with the holding period fixed at the label
horizon. §12 searched the holding period with the schedule fixed. Each found
something, and neither result says what happens when both move: the best
schedule for a two-minute position is not necessarily the best one for a
thirty-second position, since a shorter position takes more trades and a
staler model hurts a busier strategy more.

This searches the product. Every combination of training window, apply window
and holding period is scored on validation; the best one is applied once to
test. The entry threshold moves with them — it is swept inside each training
window's tail, so it adapts to whatever holding period is being tried rather
than being carried over from a different one.

    uv run python -m experiments.joint_search --symbols XRPUSDT --model logistic
"""

from __future__ import annotations

import warnings

import pandas as pd

from experiments._common import COSTS, RESULTS, emit, parser
from trading_research.pipeline.retraining import (
    DayCache,
    positive_window_share,
    prepared_evaluator,
    trade_weighted,
)
from trading_research.pipeline.stages import load_selected
from trading_research.validation.retrain import (
    RetrainSchedule,
    ScheduleError,
    expand_grid,
    run_schedule,
    split_span,
)

#: Rows to hold. One row is 5 s, so this runs from 30 s to 4 min. The label
#: horizon is 24; §12 found shorter is better, and this checks whether that
#: survives when the schedule is free to move too.
HOLDS = (6, 12, 24, 48)

TRAIN_DAYS = (3, 7, 14, 21)
APPLY_DAYS = (1, 3, 7)

#: Round trip per instrument, used as the label threshold.
COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.68}


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="logistic")
    p.add_argument("--start", default="2024-02-11")
    p.add_argument("--min-windows", type=int, default=3)
    p.add_argument("--features-prefix", default="feat")
    p.add_argument("--max-train-rows", type=int, default=None)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    schedules = expand_grid(TRAIN_DAYS, APPLY_DAYS)
    rows: list[dict[str, object]] = []
    best: tuple[float, RetrainSchedule, int] | None = None

    for symbol in args.symbols:
        prepared = args.prepared_root / f"prepared_{symbol}"
        columns = load_selected(args.prepared_root / f"{args.features_prefix}_{symbol}")
        cost = COST_BP[symbol]

        available = DayCache(prepared).available()
        start = pd.Timestamp(args.start).date()
        span = [d for d in available if d >= start]
        history = [d for d in available if d < start]
        validation_days, test_days = split_span(span)
        print(
            f"{symbol} {args.model}: {len(schedules)} schedules x {len(HOLDS)} holds, "
            f"{len(validation_days)} validation days, {len(test_days)} test days",
            flush=True,
        )

        evaluators = {
            hold: prepared_evaluator(
                prepared,
                columns,
                threshold_bp=cost,
                hold_periods=hold,
                costs=COSTS,
                model=args.model,
                max_train_rows=args.max_train_rows,
            )
            for hold in HOLDS
        }

        scored_days = set(validation_days)
        for hold in HOLDS:
            for schedule in schedules:
                # Training may reach back into the days that chose the features;
                # only the validation days are ever scored.
                lead = history[-schedule.train_days :] if history else []
                try:
                    result = run_schedule(
                        schedule,
                        [*lead, *validation_days],
                        evaluators[hold],
                        apply_within=scored_days,
                        aggregate=trade_weighted,
                    )
                except ScheduleError:
                    continue
                if result.metrics["windows"] < args.min_windows:
                    continue

                row = {
                    "symbol": symbol,
                    "model": args.model,
                    "hold": hold,
                    "schedule": schedule.label,
                    **{
                        k: result.metrics.get(k)
                        for k in (
                            "windows",
                            "trades",
                            "hit_rate",
                            "gross_per_trade_bp",
                            "net_per_trade_bp",
                            "net_bp",
                        )
                    },
                }
                rows.append(row)
                value = float(result.metrics["net_per_trade_bp"])
                if result.metrics["trades"] >= 50 and (best is None or value > best[0]):
                    best = (value, schedule, hold)
            print(f"  hold {hold} done ({len(rows)} scored)", flush=True)

        if best is None:
            print("nothing scored", flush=True)
            continue

        _, schedule, hold = best
        print(f"\nbest on validation: {schedule.label}, hold {hold}", flush=True)

        lead = [*history, *validation_days][-schedule.train_days :]
        test = run_schedule(
            schedule,
            [*lead, *sorted(test_days)],
            evaluators[hold],
            apply_within=set(test_days),
            aggregate=trade_weighted,
        )
        print(
            f"test: {test.metrics['trades']:.0f} trades, "
            f"gross {test.metrics['gross_per_trade_bp']:.2f} bp, "
            f"net {test.metrics['net_per_trade_bp']:.2f} bp against {cost:.2f} cost, "
            f"{positive_window_share(test):.0%} of trading windows positive",
            flush=True,
        )
        rows.append(
            {
                "symbol": symbol,
                "model": args.model,
                "hold": hold,
                "schedule": f"TEST {schedule.label}",
                **{
                    k: test.metrics.get(k)
                    for k in (
                        "windows",
                        "trades",
                        "hit_rate",
                        "gross_per_trade_bp",
                        "net_per_trade_bp",
                        "net_bp",
                    )
                },
            }
        )

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(
        table.sort_values("net_per_trade_bp", ascending=False).head(20),
        f"joint_search_{args.model}",
    )


if __name__ == "__main__":
    main()
