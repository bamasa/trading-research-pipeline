"""Everything the strategy can choose, searched at once.

Earlier searches moved one thing at a time: the retraining schedule with the
holding period fixed, the holding period with the schedule fixed. This moves
all of it — when to trade at all, how much history to fit on, how often to
refit, how long to hold, how long to wait afterwards, and how selective to be.

    market gate     trade only when volatility clears a quantile of the
                    training window, and only when the spread is not unusually
                    wide. Applied to the fit as well as the trading, so the
                    model learns on the rows the strategy will act in.
    schedule        training window and how often the model is replaced.
    hold            how long a position stays open, and whether that scales
                    with how sure the model was at entry.
    cooldown        how long to wait after closing.
    selectivity     whether the entry threshold is swept for total profit or
                    for profit per trade. The second raises the threshold until
                    only the strongest signals clear it.

Everything is chosen on validation and the winner is applied once to test.

    uv run python -m experiments.full_search --symbols XRPUSDT --model ensemble
"""

from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pandas as pd

from experiments._common import COSTS, RESULTS, emit, parser
from trading_research.backtest.gating import GATE_GRID
from trading_research.pipeline.retraining import (
    DayCache,
    positive_window_share,
    prepared_evaluator,
    trade_weighted,
)
from trading_research.pipeline.stages import load_selected
from trading_research.validation.retrain import RetrainSchedule, run_schedule, split_span
from trading_research.validation.search import successive_halving

#: Kept deliberately short. The schedule grid alone was twelve candidates and
#: §13 showed what widening a search buys: a better validation figure and a
#: worse test one. These are the settings each earlier search actually favoured
#: plus their nearest alternatives, not a sweep of everything expressible.
SCHEDULES = (
    RetrainSchedule(train_days=7, apply_days=1),
    RetrainSchedule(train_days=14, apply_days=1),
    RetrainSchedule(train_days=21, apply_days=1),
)
HOLDS = (6, 24)
COOLDOWNS = (0, 24)
HOLD_SCALES = (0.0, 1.0)
OBJECTIVES = ("net_bp", "net_per_trade_bp")

COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.68}


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="logistic")
    p.add_argument("--start", default="2024-02-11")
    p.add_argument("--features-prefix", default="feat10")
    p.add_argument("--min-windows", type=int, default=3)
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument(
        "--candidates",
        type=int,
        default=48,
        help="Configurations to sample. 0 keeps the whole space.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    combinations = list(product(GATE_GRID, SCHEDULES, HOLDS, COOLDOWNS, HOLD_SCALES, OBJECTIVES))
    rows: list[dict[str, object]] = []

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
            f"{symbol} {args.model}: {len(combinations)} combinations, "
            f"{len(validation_days)} validation days, {len(test_days)} test days",
            flush=True,
        )

        def score(
            combination: tuple,
            budget: int,
            *,
            prepared=prepared,
            columns=columns,
            cost=cost,
            history=history,
            validation_days=validation_days,
        ) -> dict[str, float]:
            """Score one configuration over the first ``budget`` validation days.

            Budget is days rather than windows because that is what actually
            costs time: a schedule refits once per apply window, and truncating
            the span truncates the refits with it.
            """
            gate, schedule, hold, cooldown, scale, objective = combination
            evaluate = prepared_evaluator(
                prepared,
                columns,
                threshold_bp=cost,
                hold_periods=hold,
                cooldown_periods=cooldown,
                costs=COSTS,
                model=args.model,
                max_train_rows=args.max_train_rows,
                gate=gate,
                confidence_objective=objective,
                hold_scale_by_confidence=scale,
            )
            span_days = validation_days[:budget]
            lead = history[-schedule.train_days :] if history else []
            result = run_schedule(
                schedule,
                [*lead, *span_days],
                evaluate,
                apply_within=set(span_days),
                aggregate=trade_weighted,
            )
            return result.metrics

        def label(combination: tuple) -> str:
            gate, schedule, hold, cooldown, scale, objective = combination
            return (
                f"{gate.label} | {schedule.label} | hold{hold} | cd{cooldown} | "
                f"scale{scale:g} | {objective}"
            )

        sampled = combinations
        if args.candidates and args.candidates < len(combinations):
            rng = np.random.default_rng(args.seed)
            picked = rng.choice(len(combinations), size=args.candidates, replace=False)
            sampled = [combinations[int(i)] for i in picked]

        full = len(validation_days)
        budgets = tuple(sorted({max(6, full // 4), max(8, full // 2), full}))
        outcome = successive_halving(
            sampled,
            score,
            objective="net_per_trade_bp",
            budgets=budgets,
            minimum_trades=args.min_trades,
            label=label,
            on_rung=lambda r: print(
                f"  rung budget={r.budget}d: {r.candidates} candidates -> {r.survivors}",
                flush=True,
            ),
        )
        print(f"\n{outcome.summary()}", flush=True)
        rows.extend(
            {"symbol": symbol, "model": args.model, **row}
            for row in outcome.table.to_dict("records")
        )
        best = (outcome.best_metrics["net_per_trade_bp"], outcome.best)

        if best is None:
            print(f"{symbol}: nothing scored", flush=True)
            continue

        value, (gate, schedule, hold, cooldown, scale, objective) = best
        print(
            f"\nbest on validation ({value:.2f} bp/trade): gate={gate.label} "
            f"{schedule.label} hold={hold} cooldown={cooldown} scale={scale} "
            f"selectivity={objective}",
            flush=True,
        )

        evaluate = prepared_evaluator(
            prepared,
            columns,
            threshold_bp=cost,
            hold_periods=hold,
            cooldown_periods=cooldown,
            costs=COSTS,
            model=args.model,
            max_train_rows=args.max_train_rows,
            gate=gate,
            confidence_objective=objective,
            hold_scale_by_confidence=scale,
        )
        lead = [*history, *validation_days][-schedule.train_days :]
        test = run_schedule(
            schedule,
            [*lead, *sorted(test_days)],
            evaluate,
            apply_within=set(test_days),
            aggregate=trade_weighted,
        )
        print(
            f"TEST: {test.metrics['trades']:.0f} trades, "
            f"gross {test.metrics['gross_per_trade_bp']:.2f} bp, "
            f"net {test.metrics['net_per_trade_bp']:.2f} bp against {cost:.2f} cost, "
            f"{positive_window_share(test):.0%} of trading windows positive",
            flush=True,
        )
        rows.append(
            {
                "symbol": symbol,
                "model": args.model,
                "gate": f"TEST {gate.label}",
                "schedule": schedule.label,
                "hold": hold,
                "cooldown": cooldown,
                "hold_scale": scale,
                "selectivity": objective,
                **{
                    k: test.metrics.get(k)
                    for k in (
                        "trades",
                        "hit_rate",
                        "gross_per_trade_bp",
                        "net_per_trade_bp",
                        "net_bp",
                        "confidence",
                    )
                },
            }
        )

    table = pd.DataFrame(rows).sort_values("net_per_trade_bp", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(table.head(25), f"full_search_{args.model}_{'_'.join(args.symbols)}")
    table.to_csv(
        RESULTS / f"full_search_{args.model}_{'_'.join(args.symbols)}_all.csv", index=False
    )


if __name__ == "__main__":
    main()
