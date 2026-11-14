# Registration: the automatic search for a spot–perpetual basis trade

| | |
|---|---|
| **Committed** | 15 November 2026, before any rule below was replayed on any month |
| **What it is** | `trp search basis`: twelve rules replayed on the development months, the best by monthly Sharpe chosen, the choice replayed once on the held-out months |
| **What kind of number** | A measurement on past data with the choice made on D alone and H read once. Not a forecast |
| **Searched** | The twelve rules of the grid below, on D only |
| **How it changes** | Only by a dated amendment appended at the end, which reports both numbers |

## Why

The funding carry pays because longs pay shorts to hold the perpetual. Now and
then the perpetual's price itself runs above spot as longs crowd it; the
premium comes back within hours. A carry opened at that moment earns the
premium's fall on top of the funding. The question is whether a rule that
opens only then, at Binance's base fees, makes money on months it was not
chosen on.

## The rule

[`strategies/basis.py`](../../src/trading_research/strategies/basis.py):

- **Basis.** `perp / spot - 1`, in bp, at each hourly close, for the
  seventeen coins of the carry search.
- **Deviation.** The basis minus its median over the previous `window_days`,
  the current hour excluded.
- **Open.** On an hour's close, a coin whose deviation is above `entry_bp`
  opens the carry (long spot, short the perpetual in equal quantity) at that
  close, when a slot is free; the largest deviations take the free slots.
  Four slots, each a quarter of the equity, half on each leg.
- **Close.** When the deviation is no longer above `exit_bp`, after 14 days
  held, or at the last hour before a month the coin is not readable in.
- **Costs and carry.** As `trp search carry`: spot 10 bp and perpetual 5 bp a
  side, 2 bp of slippage a leg and side (38 bp a round trip), funding at every
  settlement on the perpetual's quantity, the basis's change hour by hour.

## The grid, searched on D

`entry_bp` ∈ {30, 45, 60} × `exit_bp` ∈ {0, 15} × `window_days` ∈ {7, 30}:
twelve rules. The entries were set after looking at the deviation's spread on
the development months alone: its 99th percentile ran from 8 bp (BTC) to 27 bp
(AVAX), and only a few hundred of 22,000 hours per coin went above 40 bp. An
entry below the 38 bp round trip cannot pay on the premium's fall alone, so
the grid starts near it.

## Samples

- **D**, development: January 2021 to June 2023.
- **H**, held out: July 2023 to June 2025, read once, after the ledger is
  written.
- BTC and ETH are not read from December 2020 to June 2021 or from February
  to June 2024 (`funding_carry_2026` holds them out). Nothing after June 2025.
- **Seen before.** These prices were read by the carry replays (the basis's
  hour-by-hour change is part of the carry's result), but no rule opening on
  the basis's level was ever computed on them.

## Choice and verdict

- **Choice.** The highest monthly Sharpe ratio on D (ties by total return).
- **Placebos.** Nineteen replays of the chosen rule with each coin's deviation
  rolled by its own random offset over the span: as many signals, at the wrong
  hours.
- **Positive** when the chosen rule's total on H is above zero **and** above
  all nineteen placebos' totals. Otherwise **killed**, with the first failure.

## Expected outcome, stated now

**Killed, or positive and small.** The premium rarely runs past the round
trip's cost, so the rule will trade seldom: tens of times a year across
seventeen coins. Most of what it earns will be the funding while it waits,
which a placebo opening at any hour also earns, so beating all nineteen is
the hard part.
