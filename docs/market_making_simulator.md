# The market-making simulator

An event-time simulator for a two-sided quoter on Bybit's public data: the
ten-level book photographed every 100 ms, and the tape of trade prints with the
aggressor's side. It exists to answer the questions pre-registered in
[`preregistration/market_making.md`](preregistration/market_making.md), and
nothing in this document is a result: no strategy has been run on any block.

This page states every rule the simulator follows, which way each rule biases a
result ("optimistic" flatters the market maker), and the test that pins it. The
code is under
[`src/trading_research/market_making/`](../src/trading_research/market_making/);
the tests are `tests/test_mm_*.py`, all on synthetic or hand-built streams.

## Why the rules look like this

The book on disk is a 100 ms conflated view, not a message feed. Within 100 ms
nobody can say who joined, who cancelled, or in what order. Measured while
designing (and disclosed in the pre-registration), most prints land at the touch
of the last snapshot before them, a fifth to a half land through it, and a few
per cent on the wrong side of it: the two clocks agree only to within the
conflation interval. So:

- **fills come from prints, never from snapshots** — a resting order fills only
  when a print reaches its price from the side that hits it;
- **snapshots only correct the queue estimate** — the visible size at the
  order's price says how much could still be ahead of it;
- **where the data cannot decide, a rule decides, in a stated direction**, and
  the rules that matter most come with brackets (pessimistic, optimistic) that
  are run as robustness checks, and in one case enter the verdict.

## The pieces

| Module | What it holds |
|---|---|
| [`events.py`](../src/trading_research/market_making/events.py) | `InstrumentSpec`, `DayEvents`, `assemble`, `load_day`, `derive_tick`, `derive_lot`, `to_ticks`, `sweep_order`, `off_book_eligible`, `OFF_BOOK_BP`, `DayUnavailable`, `OffGridPrice` |
| [`orders.py`](../src/trading_research/market_making/orders.py) | `OrderState`, `Order` |
| [`queue.py`](../src/trading_research/market_making/queue.py) | `CancelAttribution`, `ArrivalGrowth`, `QueuePriority`, `CrossedPolicy`, `on_arrival`, `on_print`, `on_snapshot` |
| [`accounting.py`](../src/trading_research/market_making/accounting.py) | `Account`, `IdentityError` |
| [`quoters.py`](../src/trading_research/market_making/quoters.py) | `MarketView`, `Quote`, `Quotes`, `Quoter`, `TouchQuoter` (S0) |
| [`signals.py`](../src/trading_research/market_making/signals.py) | `SignalTape` |
| [`simulator.py`](../src/trading_research/market_making/simulator.py) | `SimConfig`, `DayResult`, `simulate_day`, `run_days`, `source_fingerprint`, `OracleRefused` |
| [`analysis.py`](../src/trading_research/market_making/analysis.py) | `markouts`, `decompose`, `summarise_markouts`, `market_wide_markouts`, `mid_at`, `book_mid`, `clock_offset_profile`, `best_clock_offset_ms` |
| [`synthetic.py`](../src/trading_research/market_making/synthetic.py) | `known_answer_market`, `random_market`, `hand_built_market` |

Outside the package: `data/bybit_trades.load_day` and a stable sort that keeps
Bybit's match id; `data/bybit_funding.py` (the funding history from the public
REST endpoint) and `data/ensure.ensure_funding`; `data/grid.to_grid(label=...)`;
`backtest/costs.FeeTier`, `BYBIT_BASE`, `BYBIT_LINEAR_TIERS`, `fee_tier` and
`breakeven_maker_bp`.

## The event loop

```
pending = heap of (effective time, sequence, action)   # arrivals, cancels, stop, flatten
for each event in the day's merged stream, at time t:
    if no snapshot for 5 s: pull every order (cancels decided at the 5 s mark)
    apply every pending action effective strictly before t   # ties go to the market
    funding:  settle on the position held now, at the last mid
    print:    if on the book, offer it to own orders on the side it hits, best price
              first (each fill: book it, check the identity; past the soft limit,
              send a reduce-only taker order)
    snapshot: update the mark; execute the taker orders that have arrived, walking
              this snapshot; correct every resting order's queue estimate;
              if quoting is allowed, ask the quoter, pull a self-crossing pair
              apart, then reconcile: every cancel before any new order; keep an
              unchanged price unless the current clip or hard limit forbids it;
              cancel-and-new on a change; never resize in place
after the stream: apply what is still pending (stop at 23:58, flatten at 23:59),
                  close (flag the day if a taker had no fresh book), check the
                  identity, then markouts and the decomposition
```

`simulate_day(events, quoter, config, tapes=None)` runs one instrument-day and
returns a `DayResult`: fills, orders, one equity row a minute, crossed episodes,
counters, the decomposition and the day's flags. A flagged day (`excluded`) is
simulated and reported but must stay out of every verdict. `run_days` runs
independent days, optionally in worker processes and resumably against a cache,
and returns one row a day; a day that cannot be simulated is a row whose status
says why, and the run goes on.

## The rules

Each rule: what it does; the direction of its bias; the test that pins it.
Defaults are the pre-registered settings; every bracketed alternative is a
`SimConfig` field. The corrections made after code review, before any result,
are recorded as Amendment 1 of the
[pre-registration](preregistration/market_making.md#amendments).

### R1 — Ordering

One merged stream per instrument-day. Among market events at one timestamp:
funding, then prints, then snapshots. Own actions — an order arriving, a cancel
landing, a timed stop or flatten — fire only strictly before the next market
event, so at an equal timestamp every market event goes first: an order arriving
at a print's timestamp misses it, a cancel landing at a print's timestamp is
still filled by it, and an order arriving at a snapshot's timestamp joins the
queue that snapshot shows. Prints sharing a timestamp and an aggressor are one
sweep and are processed in sweep order: buys by ascending price, sells by
descending, larger prints first at one price, then file order. At one
timestamp, sells come before buys (arbitrary, fixed). Print timestamps are
rounded to the microsecond on loading, removing the float noise of the
archive's second-resolution format.

The two clocks are not the same clock. A book row carries the time the venue
generated it, a print the time it matched. `analysis.clock_offset_profile`
measures the offset as the shift of the prints that best explains the touch's
size changes; on the one development day it was run on (BICOUSDT, 2024-02-15)
the book lags the prints by about 3 ms (5 ms on a 5 ms grid; the residual is
flat within 0 to 10 ms). A lagging book is the direction that double-counts,
which R8 corrects; a leading book would be look-ahead, and was not found.

*Bias.* Ties going to the market is pessimistic by at most one print per order.
Sweep order is the only order consistent with price priority. Prints before a
snapshot stamped with the same time assumes that snapshot reflects them; the
measured lag says it may not yet, and R8's carry-forward covers that case.

*Tests.* `test_events_are_time_ordered_with_trades_before_books`,
`test_prints_sharing_a_timestamp_are_ordered_as_a_sweep`,
`test_the_assembled_stream_holds_prints_in_sweep_order` (`test_mm_events.py`);
`test_an_order_arriving_at_a_prints_timestamp_misses_it`,
`test_an_order_arriving_at_a_snapshots_timestamp_joins_that_snapshots_queue`
(`test_mm_latency.py`); `test_the_offset_diagnostic_recovers_a_known_lag`
(`test_mm_clock.py`).

### R2 — Information set

A quoter decides only at snapshots. It sees that snapshot (read-only views of
its ten levels), the mid, a one-minute volatility and the trailing one-hour
median touch (both computed per whole second from snapshots at or before it),
the clip and soft limit the simulator will apply, its own position, and
external signals by an as-of join on bin-end labels strictly before the
snapshot. It never sees fills, markouts or anything later. With
`feed_latency_ns` above zero, every decision is taken that much after its
snapshot.

*Bias.* `feed_latency_ns = 0` is optimistic (a live participant receives the
snapshot later); swept to 50 ms in the robustness block. Signals are held to the
same clock as the book: an external signal is read as of strictly before the
snapshot, not the decision.

*Tests.* `test_mutating_the_future_does_not_change_past_decisions` (ten streams
rewritten after a cutoff; every view, order and fill before it identical),
`test_grid_signals_are_joined_at_the_end_of_their_bin`,
`test_the_simulator_hands_quoters_only_past_signals` (`test_mm_lookahead.py`);
`test_feed_latency_delays_every_decision` (`test_mm_latency.py`);
`test_a_left_labelled_grid_is_five_seconds_ahead_when_joined_at_its_label`,
`test_a_right_labelled_grid_never_carries_a_value_from_after_its_label`
(`test_bybit.py`).

### R3 — Latency

A new order arrives `order_latency_ns` (10 ms) after its decision and cannot fill
before. A cancel lands `cancel_latency_ns` (10 ms) after its decision; until
then the order can still fill. A price change is a cancel and a new order, and
the new order joins the queue afresh; an unchanged price keeps the resting
order and its place. A cancel that would overtake its own order (cancel latency
shorter than order latency) is applied when the order arrives.

*Bias.* Neutral at 10 ms; swept to 1, 50, 100 and 250 ms. "A cancel in flight
does not protect" is pessimistic, and correct.

*Tests.* `test_an_order_cannot_fill_before_it_arrives`,
`test_a_cancel_in_flight_does_not_protect_the_order`,
`test_a_replace_loses_priority_and_a_kept_quote_keeps_it`,
`test_a_cancel_faster_than_its_order_waits_for_it` (`test_mm_latency.py`).

### R4 — Post-only

On arrival, an order at or through the opposite touch of the latest snapshot at
or before its arrival is rejected and counted (`rejected`). So is an order at or
through one of our own resting opposite orders (`rejected_self_cross`): our bid
and ask never meet. Before anything is sent, a quoter's bid at or above its own
ask is pulled apart around the middle, to at least one tick each side
(`self_cross_clamped`), and every cancel of a decision is sent before its new
orders, so at equal latency a new order never finds our own replaced order
still resting. Quotes never take liquidity.

*Bias.* Pessimistic: some rejected orders would have rested at a better price.

*Tests.* `test_post_only_rejects_an_order_that_would_cross_on_arrival`
(`test_mm_latency.py`); `test_own_quotes_never_lock_or_cross`,
`test_an_order_arriving_at_our_own_opposite_order_is_rejected`
(`test_mm_limits.py`).

### R5 — Queue on arrival

The size ahead is the visible size at the order's price in the latest snapshot
at or before its arrival: it joins the tail. A price better than its own side's touch
is a new level: nothing ahead. A price inside the visible levels but absent from
them is an empty level: nothing ahead. Growth of the level during the snapshot
interval in which the order arrived is split by time — the share of the
interval that had passed when it arrived is ahead (`pro_rata_time`), bracketed
by none of it (`none`) and all of it (`all_ahead`). A quote beyond the tenth
visible level is not placed; an order that arrives to find its price beyond the
tenth level (the book moved while it travelled) has an unknown queue, treated as
infinite until the level is visible, when it joins the tail of what is there.

*Bias.* Tail joining is correct for a new order. The snapshot can be up to
100 ms stale, which the growth rule partly corrects; the bracket is reported.
The unknown-queue case is pessimistic.

*Tests.* `test_joining_a_visible_level_starts_at_its_tail`,
`test_inside_the_spread_starts_at_the_front`,
`test_growth_in_the_arrival_interval_is_split_by_time`,
`test_an_order_beyond_the_visible_book_waits_for_the_level_to_appear`
(`test_mm_queue.py`); `test_quotes_beyond_the_visible_book_are_not_placed`
(`test_mm_limits.py`).

### R6 — Advance on prints at the price

A print at exactly the order's price, from the aggressor that hits it (a seller
for a bid), reduces the size ahead by its size; any excess fills the order, up
to what remains of it; the rest goes to orders behind. Prints from the other
side, or at prices that do not reach the order, do nothing. When several own
orders rest on one side (an old order whose cancel is in flight and its
replacement), a print is offered to them in price priority, and each sees what
the better-priced ones left of it.

**Before the first snapshot after arrival**, a print at the price for more than
the size ahead proves that others joined the level after the arrival snapshot,
and some may have joined before the order. The excess goes to those hidden
earlier joiners first, in the share the arrival-growth bracket gives them —
none (`none`), `(arrival - t_snap) / (print - t_snap)` (`pro_rata_time`), all
(`all_ahead`) — as a running total over the interval: after cumulative excess
`E` the order is entitled to `(1 - share) * E`, less what it has received, and
never to more than the print at hand brought. Without this, an order improving
the spread was filled first by every print at its price until the next
snapshot, whichever bracket was set: exactly the fills a faster competitor
reacting to the same snapshot would have taken.

**Off-book prints.** A print more than 25 bp outside the price range of the ten
visible levels of the latest snapshot at or before it could not have swept that
book (a negotiated block trade, or a bad record) and is offered to no own order;
it is counted (`off_book_prints`). On the development day checked (BICOUSDT,
2024-02-15), 19 of 15,506 prints lay beyond the tenth level, all on the side
they hit and at most three ticks (about 7 bp) out, none on the wrong side: deep
sweeps, which the rule keeps. None was excluded.

*Bias.* Neutral under price-time priority, which Bybit uses. The arrival-interval
share is bracketed with the growth rule, and the order of the bracket is pinned.
Excluding off-book prints removes fills that are usually adverse, so it is
mildly optimistic, and on the day checked it removed nothing.

*Tests.* `test_no_fill_without_a_print_at_or_through_the_price` (200 random
streams: every fill has a print at its timestamp, on the hitting side, at its
price for a queue fill or through it for a through fill),
`test_own_fills_never_exceed_the_print_volume_that_reached_them` (240 runs,
several own orders on a side at once),
`test_prints_on_the_wrong_side_do_not_advance_the_queue`,
`test_prints_at_other_prices_do_not_advance_the_queue`,
`test_partial_fills_sum_to_the_print_beyond_the_queue`,
`test_an_inside_order_shares_arrival_prints_with_hidden_earlier_joiners`,
`test_a_touch_order_shares_the_excess_in_its_arrival_interval`,
`test_arrival_prints_are_shared_as_a_running_total_and_never_exceed_the_print`,
`test_arrival_growth_orders_the_fills_of_one_order`,
`test_arrival_growth_brackets_the_fills_of_held_orders`,
`test_off_book_prints_never_fill` (`test_mm_queue.py`);
`test_prints_far_outside_the_visible_book_are_not_eligible_to_fill`
(`test_mm_events.py`).

### R7 — Trade-through

A print strictly through the order's price (a sale below the bid) means the
level was exhausted: nothing is ahead, and the order fills at its own limit
price, up to the size of that print. Summed over a sweep, that is the
through-volume of the sweep. Path `through`.

*Bias.* Realistic and adverse: these are the toxic fills, and every report
splits markouts by path. Taking the print sizes we see as the size that would
have reached the order is a lower bound on the fill: mildly pessimistic.

*Tests.* `test_a_print_through_the_price_fills_at_the_limit_and_clears_the_queue`,
`test_a_print_through_fills_at_the_order_price_in_the_simulator`.

### R8 — Cancellations ahead

At each snapshot, the shortfall of the level against what the prints at the
price should have left, `d = max(0, L_prev - traded_at_price - L_now)`, was
cancelled. `proportional` (default) removes it from ahead in proportion,
`queue_ahead -= d * queue_ahead / (L_prev - traded)`; `pessimistic` puts all of
it behind; `optimistic` all of it ahead. The estimate is then clamped to
`[0, L_now]`.

When a snapshot shows the level larger than the prints at the price allow, the
part of the growth those prints could explain, `min(traded, growth)`, is carried
into the next interval's traded volume: a print stamped just before a snapshot
that does not show it yet (the book lags, R1) is then not read as a
cancellation one snapshot later. If the growth was real, the carry masks at
most that much cancellation once.

*Bias.* Proportional is optimistic against the queue-reactive evidence that
later arrivals cancel more (Huang, Lehalle and Rosenbaum, 2015). Hence the
pre-registered kill condition K-pess: a positive verdict must also be positive
under `pessimistic`. The carry-forward is conservative: without it a lagging
book moved orders forward twice for one print; with it, a real growth after
prints delays one later cancellation.

*Tests.* `test_cancellation_attribution_brackets_on_one_snapshot`,
`test_cancellation_attribution_orders_the_fills_of_one_order` (200 streams,
exact), `test_cancellation_attribution_brackets` (through the simulator: exact
on every stream for orders held all day; for the touch quoter, which re-quotes
after each fill so that paths diverge, in total over forty streams),
`test_queue_ahead_is_never_negative_and_never_exceeds_the_level` (200 streams
under each of the nine rule pairs; the estimate also never rises except at the
first snapshot after arrival), `test_prints_at_the_price_are_not_counted_again_as_cancellations`
(`test_mm_queue.py`); `test_a_lagged_snapshot_does_not_count_a_print_twice`
(`test_mm_clock.py`).

### R9 — Growth behind

After the arrival interval, growth of the level is behind the order and never
moves it back.

*Bias.* Correct under FIFO.

*Test.* `test_orders_joining_after_us_do_not_move_us_back`.

### R10 — Invisible and crossed levels

A price inside the visible ten levels but absent is an empty level: nothing
ahead. A price beyond the tenth level is unobservable: the estimate is left
alone and the seconds are counted (`unobserved_level_s`). A snapshot showing the
opposite touch at or through a resting order (the historical book cannot contain
our order) fills nothing under `trade_tape`: a fill needs a print. Each such
episode is recorded with its time and mid. `assume_filled` fills the order in
full at its limit instead, and exists only as a diagnostic.

*Bias.* `trade_tape` is optimistic in profit: whoever posted through our price
might have traded with us, and those fills are adverse. Reported as a bracket,
with the count of crossed episodes and the markout after them.

*Tests.* `test_a_crossed_snapshot_alone_never_fills`,
`test_assume_filled_reports_more_fills_than_trade_tape`,
`test_an_empty_level_leaves_nothing_ahead_under_every_rule`.

### R11 — Own size and limits

At every decision the simulator sets the clip to
`min(clip_notional / mid, clip_touch_share * trailing one-hour median touch)`,
rounded down to the lot; below one lot the side is not quoted. The soft limit is
`soft_limit_clips * clip`: a side whose fill would take the position past it is
not quoted (and an order resting there is cancelled). The hard limit is the soft
limit plus one clip, and a new order is sent only if the position plus every
order still out on its side (cancels in flight included) plus the new clip stays
within it. The check is repeated at every decision for an order kept at its
price: if the clip has shrunk below its size, or the side's exposure exceeds the
current hard limit, it is cancelled and re-placed at the current clip
(`cancelled_for_limits`), losing its place. So the hard limit holds whatever the
latency, up to one cancel latency after the clip shrinks.

A fill that takes the position past the soft limit sends a taker order back to
it. Like every order it pays the order latency, and it walks the first snapshot
at or after its arrival — not the book the sweep had just consumed — with
`flatten_slippage_bp` (0.5) on top and the taker fee; size beyond the visible
book is charged at the tenth level and counted. It is reduce-only: if passive
fills shrank the position while it travelled, it trades only what is left and
never reverses the position.

Own impact on others is not modelled: an own order does not change the book any
other participant sees, and a second own order at another price reads the
volume the first one took as a cancellation at its level (a stated limitation,
pinned by a test). The mean clip over the touch is reported (`clip_over_touch`).

*Bias.* Ignoring own impact is optimistic, bounded by keeping the clip near a
tenth of the touch. Charging size beyond the visible book at the tenth level is
optimistic and counted (`flatten_beyond_visible_book`). The flatten's latency
and fresh book are the realistic pricing of an exit taken at a toxic moment;
priced at the pre-sweep book, as first written, it was optimistic.

*Tests.* `test_inventory_never_exceeds_the_hard_limit` (forty random streams,
three soft limits, latencies of 10 and 250 ms),
`test_the_growing_side_is_not_quoted_beyond_the_soft_limit`,
`test_a_sweep_past_the_soft_limit_is_flattened_back_to_it`,
`test_a_flatten_never_reverses_the_position`,
`test_the_hard_limit_holds_when_the_clip_shrinks`,
`test_a_clip_below_one_lot_is_not_quoted`,
`test_the_clip_adapts_to_the_trailing_touch` (`test_mm_limits.py`);
`test_a_second_own_order_reads_the_first_ones_fill_as_a_cancellation`
(`test_mm_queue.py`).

### R12 — Fees

The maker fee on passive fills, the taker fee on flattens, in basis points of
notional, by tier; a rebate is a negative fee. Base tier 2.0 / 5.5 bp. The
break-even maker fee of a quoter whose decisions do not read the fee (S0) is
`maker_bp + net / maker turnover * 1e4`, exact without a re-run.

*Bias.* Neutral; the tier is swept.

*Tests.* `test_maker_rebate_is_a_negative_fee`,
`test_fee_breakeven_matches_a_rerun_at_that_fee`,
`test_fee_tiers_are_looked_up_not_assumed` (`test_mm_accounting.py`).

*Not yet done.* Only the base tier is in `backtest/costs.BYBIT_LINEAR_TIERS`.
Bybit's fee page refused automated requests when this was written, and the
pre-registration requires the VIP and market-maker rows to be transcribed, not
assumed; `BYBIT_FEE_RETRIEVED` is `None` until they are. They are needed only by
the ladder's last rung and the fee break-even comparison.

### R13 — Mark

The mid of the latest snapshot.

*Bias.* Neutral. *Test.* Covered by the identity tests below.

### R14 — Funding

Settled at the funding events in the stream, on the position held at that
instant: `payment = -position * mark * rate`, with the mid as a proxy for
Bybit's mark price. The settlement interval is read from the fetched history and
checked against the instrument's (8 h); a settlement off that grid refuses the
day. Days start flat, so the 00:00 settlement never pays.

*Bias.* The mid for the mark is neutral at these spreads. Skipping 00:00 removes
a third of the funding exposure; its size can be read from the 08:00 and 16:00
payments, which are counted (`funding_settlements_with_position`).

*Tests.* `test_funding_sign_longs_pay_when_the_rate_is_positive`,
`test_flat_at_settlement_pays_nothing`,
`test_funding_is_applied_at_settlement_boundaries` (`test_mm_accounting.py`);
`test_a_funding_settlement_off_the_grid_is_refused`,
`test_funding_on_disk_joins_the_stream` (`test_mm_events.py`); the downloader in
`test_bybit_funding.py`.

### R15 — Accounting identity

Average cost; realised on reduction; a flip through zero realises the closed
part and opens the rest at the fill price. Cash is kept independently. After
every change to the account, and at the end of the day,
`realised + unrealised - fees + funding == cash + position * mark` must hold to
1e-9 of turnover, or the run stops with `IdentityError` at the event that broke
it.

*Bias.* None; a mismatch raises.

*Tests.* `test_pnl_identity_holds_over_random_fills` (500 fills with flips,
fees, rebates and funding), `test_a_broken_ledger_raises`,
`test_the_simulator_stops_at_the_event_that_breaks_the_ledger`,
`test_a_flip_through_zero_realises_the_closed_part`.

### R16 — The day

No quoting in the first `warmup_s` (10 minutes, the volatility warm-up) or from
`stop_quoting_at` (23:58), when every order is cancelled; flattened at
`flatten_at` (23:59) by a reduce-only taker order that, like the soft-limit
flatten, pays the order latency and walks the first snapshot at or after its
arrival — after a pause in the book, the first snapshot after the pause. If no
snapshot arrives for `suspend_after_pause_s` (5 s), every order is pulled at
that moment and nothing is quoted until the next snapshot.

A day that cannot be simulated — the book or the prints missing, a price off
the tick grid, a funding interval other than the one assumed — is reported with
its reason and skipped; the rest of the run goes on. A simulated day is
**flagged and excluded from every verdict** when:

- `sequence_gaps`: the stored book reports more sequence gaps than
  `max_sequence_gaps` (0: any gap). After a missed update the reconstructed book
  is wrong until the next full snapshot — a missed addition reads as an empty
  level, which empties every queue estimate there — and the stored book keeps
  only the day's count, not where the gaps fell, so the day cannot be
  suspended from the gap onward;
- `flatten_stale`: a taker order found no snapshot within the pause threshold of
  its arrival and had to walk an older book (the book ended early);
- `no_funding`: the day was simulated without a funding plane.

*Bias.* Conservative.

*Tests.* `test_days_start_and_end_flat`,
`test_the_day_end_flatten_crosses_the_spread_and_walks_the_depth`,
`test_a_day_end_flatten_with_no_fresh_book_is_flagged`,
`test_a_day_end_flatten_after_a_pause_walks_the_first_book_after_it`
(`test_mm_limits.py`), `test_quotes_are_pulled_across_a_book_gap`
(`test_mm_latency.py`), `test_a_day_missing_either_plane_is_reported_not_simulated`,
`test_a_day_with_bad_data_is_skipped_with_its_reason`,
`test_a_day_with_sequence_gaps_is_flagged_and_excluded`
(`test_mm_determinism.py`), `test_a_day_missing_a_plane_on_disk_is_reported`
(`test_mm_events.py`).

### R17 — Markouts

Per fill, `side * (mid(t + h) - p) / p * 1e4` for h = 1, 5 and 30 s, from the
last snapshot at or before `t + h`, computed after the loop. The market-wide
benchmark scores every print in the tape from its resting side, independently of
the simulator.

*Bias.* Neutral.

*Tests.* `test_markouts_have_the_sign_of_the_move`,
`test_the_market_wide_benchmark_scores_the_resting_side`
(`test_mm_known_answer.py`).

### R18 — Decomposition

Per fill, `spread = side * (mid_ref - p) * size` with `mid_ref` the last
snapshot mid at or before the fill; `adverse = side * (mid(t + 5 s) - mid_ref) *
size`; `inventory` is the gross trading profit less those two (the residual
mark-to-market of held inventory); then fees and funding.
`net = spread + adverse + inventory - fees + funding`, checked against cash at
the end of the day.

*Bias.* Only `spread + adverse` together (the 5 s markout) is free of the
100 ms staleness; the split between them is indicative, and reports say so.

*Test.* `test_decomposition_sums_to_net`.

### R19 — Determinism

No randomness in the simulator. Days are independent jobs; `run_days` sorts its
output by day, so the number of workers cannot change it. Its cache is keyed by
everything a day depends on: the quoter, the configuration, the data roots, the
path, size and modification time of every file the day reads (book, prints,
funding), and a hash of the simulator's source (`source_fingerprint`). New
data, newly fetched funding or a fix to the simulator therefore cannot be
answered from an old result.

*Tests.* `test_the_same_inputs_give_identical_results`,
`test_worker_count_does_not_change_results`,
`test_the_cache_is_keyed_by_the_data_it_read`,
`test_a_change_to_the_simulator_or_the_funding_root_invalidates_the_cache`
(`test_mm_determinism.py`).

### R20 — Oracle isolation

A quoter with `uses_future = True` (the ladder's forecast rungs) is refused by
`simulate_day` unless `allow_oracle=True` is passed, which only the advantage
ladder will do, and its result is stamped `oracle`. `run_days`, which produces
results tables, refuses it outright.

*Test.* `test_oracle_forecasts_are_refused_outside_the_ladder`.

## Known answers

[`synthetic.known_answer_market`](../src/trading_research/market_making/synthetic.py)
has a constant spread and prints that each sweep the touch, at least six
seconds apart:

- uninformed flow (the mid never moves): the touch quoter captures exactly the
  half-spread on every unit filled, with zero adverse selection and zero
  inventory result — `test_uninformed_flow_earns_the_half_spread`;
- fully informed flow (each print moves the mid to the price just traded, as in
  Glosten and Milgrom): spread captured and given back within five seconds, so
  spread plus adverse is exactly zero — `test_informed_flow_earns_nothing_before_fees`;
- flow that moves the mid a full spread: every fill's markout has the sign of
  the move — `test_markouts_have_the_sign_of_the_move`.

## What the data layer gained

- `data/grid.to_grid(..., label="right")` labels each bin by its end. The
  default stays `"left"`, which carries the bin's last value at its start: a
  five-second look-ahead when joined into event time at the label. The
  simulator's `SignalTape.from_grid` requires the label to be stated and moves a
  left-labelled grid to bin ends.
- `data/bybit_trades.parse` sorts stably and keeps `trdMatchID` as `match_id`;
  files already on disk keep their order, which the simulator does not rely on
  (R1).
- `data/bybit_trades.load_day` and `market_making.events.load_day` read one
  instrument-day without touching any other; `load` now warns about memory.
- `data/bybit_funding.py` and `data/ensure.ensure_funding` fetch the funding
  history from the public REST endpoint, a file per day.

## Speed

Pure Python over per-day arrays; numba is not installed and is not added. On
one development day already listed as read in the pre-registration's
disclosure (BICOUSDT, 2024-02-15: 249,105 snapshots and 15,506 prints), with the
touch quoter, the event loop ran at about 217,000 events a second (1.2 s for the
day) with a peak resident set of about 600 MB, most of it the Parquet read.
After the review fixes (re-checking kept orders against the limits at every
decision, the arrival-interval rule) it runs at about 186,000 events a second
(1.4 s). Neither run built a result: no markout, fill statistic or profit was
computed from them. Extrapolated at that rate, a BTCUSDT day of the held-out
block (about three million events) would take about 16 s; quoters richer than
the touch quoter will be slower.

## Not in this simulator yet

The quoters the study compares (S1–S4, X1), the reversion tape and the regime
flags, the admission screen, the pre-registration loader and the held-out
ledger, the advantage ladder and the fee-grid break-even, the pipeline stage and
the command-line entry points arrive in later pull requests, in the order the
pre-registration sets.
