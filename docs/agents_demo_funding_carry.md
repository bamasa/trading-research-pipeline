# The research loop on the funding-rate carry study: a plan

How the agent harness ([`agents/`](../agents/README.md)) would run the
roadmap's funding-rate carry study end to end, step by step, with what each
role would produce and what each gate would check. **Nothing here has been
run, fetched or registered**: no funding rate, perpetual price or basis of any
date has been read for this study, and the study has no registration, no
code and no entry point yet. The commands this plan names exist, except those
of the entry point to come, which are marked as such; a test
(`tests/test_agents_demo.py`) holds the plan to that, and checks each command's
outcome under the tool policy as the tables below state it.

The roadmap's item: long spot and short the perpetual in equal size, so the
price cancels and the position collects the funding that leveraged longs pay,
pre-registered, with held-out time.

## 0. The brief and the start

The owner writes the brief to a file (the roadmap's item, the venue, the
instruments, and "blocks in 2026") and starts the loop:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop run --study funding_carry_2026 --brief-file <brief.md>
```

The researcher and the registrar first read the protected calendar, which the
policy builds from every registration's blocks:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop calendar
```

| Range | Held by |
|---|---|
| 2024-02-26 to 2024-06-02 | the market-making studies: H, the buffer, F (unread), P and Q |
| 2025-07-01 to 2025-12-31 | the 2025 study: E, H, F and P, none read yet |
| 2026-01-05 to 2026-01-07 | the 2025 study's reader-test days, in no block of any study |
| 2026-08-01 to 2026-12-31 | the spot study: E, H (read), the P embargo and P |

A replication on another local branch reads Bybit ETHUSDT days of late July and
August 2026 (its plan, read with `git show`, never checked out); the registrar keeps
the carry's blocks clear of them as well.

## 1. Propose (researcher)

**The question.** Does a delta-neutral carry on Binance (long spot, short the
USD-margined perpetual, the same quantity of the coin on both legs) earn more
funding than it pays in fees and loses in basis, on months nobody has read,
at the fees an account without volume pays?

**Hypotheses**, a family of three:

- **C1, always on, BTCUSDT.** The hedged position is opened at the block's
  first hour and held to its last, one clip, never re-timed.
- **C2, gated, BTCUSDT.** The position is held only while the mean of the last
  21 settled funding rates (seven days at three settlements a day), each
  settled before the decision, exceeds a threshold θ set on D; it opens and
  closes on the hour after a settlement.
- **C3, gated, ETHUSDT.** C2's rule on ETHUSDT, with its own θ set on D.

**Metric.** Each calendar day's net in bp of the hedged notional: funding
received on the short leg, minus fees on both legs, plus the day's change in
the basis (perpetual minus spot) on the hedged quantity. A position is held
across days, so the days are not independent: the status reads the Newey-West
t (lag 7) of the daily series, never a plain day-level t.

**Kill conditions**, in the order they are read (void conditions first:
K-acct, the accounting identity; K-units, a fee or a price through a
normalised column; K-cap, more than one clip):

| Kill | Fires when |
|---|---|
| K1 | the mean of the daily series on the block is at or below 0 |
| K-fund | funding received over the block is at or below the fees paid: the carry does not pay for its own trades |
| K-basis | the series without its basis term is at or below 0: the result is basis, not carry |
| K-liq | the perpetual leg, at the registered margin, would have been liquidated at any hour's high |
| K-placebo (C2, C3) | the true mean does not exceed the 95th percentile of 200 random gates (seeds 0 to 199) with the same holding periods placed at random settlements, or the gate shifted by one week |
| K-nbhd (C2, C3) | the median over the neighbourhood (θ times 0.5, 0.75, 1.25 and 1.5; windows of 9, 21 and 42 settlements) is at or below 0 |
| K-delay | a positive result is at or below 0 with every entry and exit an hour later |
| K-days | half or fewer of the days with a position are positive |
| K-fee | the fee break-even per side is at or below the verdict tier's taker fee (reported for how far each slot was from paying) |

**Expected outcome, with the arithmetic.** A round trip costs two taker fills
on each leg; at the verdict tier's fees f_spot and f_perp (from the sourced
table) it costs 2(f_spot + f_perp) of notional, and a funding rate of r per
settlement pays 3r a day, so C1 breaks even after 2(f_spot + f_perp) / 3r
days. C1 trades once per block, so its fees are spread over every day of it,
and its daily series is dominated by the basis term: **C1 is predicted
inconclusive** (the Newey-West t short of the bar on a block of a few months,
whatever the sign). A gate on trailing funding re-times a payment the
always-on position collects anyway and pays an extra round trip each time it
re-opens: **C2 and C3 are predicted killed by K-placebo**. Not predicted: the
sign of the basis term on H.

**The information set.** No funding rate, perpetual price or basis of any date
has been read for this study. Known before it: the market-making studies read
Bybit funding for their 2024 instruments; the 2025 study fetched Bybit BTCUSDT
funding for its W and D (December 2024 to June 2025); the spot study read
Binance BTC/FDUSD prints and Bybit spot BTCUSDT books from March to October
2026, so the 2026 path of BTC's price is inside the information set; the
author's 2025 research worked on BTC/FDUSD in 2025. The owner confirms, as an
amendment before any fetch, the last date they looked at funding rates of
either coin; without it, H is inside the information set and only P confirms.

## 2. Register (registrar)

The registrar writes `docs/preregistration/funding_carry_2026.md` and
`configs/funding_carry_2026.yaml` on the spot study's pattern, section by
section, and refuses if the proposal leaves a decision open. What it would
register beyond the proposal:

**Blocks**, whole months throughout, because Binance publishes `fundingRate`
by the month only and the policy counts a month archive as every day of it:

| Block | Dates (UTC, inclusive) | Use | Why its edges |
|---|---|---|---|
| D | 2020-01-01 to 2024-01-31 and 2024-07-01 to 2025-06-30 | θ for C2 and C3, the neighbourhood, σ of daily results, the minimum detectable effect | February to June 2024 hold the market-making studies' days |
| E | 2025-07-01 to 2026-03-31 | never fetched | holds the 2025 study's E, H, F and P, its reader-test days and the spot study's reader-test days |
| H | 2026-04-01 to 2026-06-30 | the three statuses, read once after the D amendment | 91 days, the last whole months before the spot study's and the replication's days |
| E2 | 2026-07-01 to 2026-12-31 | never fetched | holds the spot study's E, H and P and the replication's days |
| P | 2027-01-01 to 2027-03-31 | confirmation of a candidate, read once, after March 2027 is published | outside every information set |

**Fees.** Binance spot and USD-margined futures, the Regular tier as the
verdict tier, each value with its source and the date it was read, from fee
pages on a host the owner lists for the run (`--web-host`); every other tier
and zero fees as measurements. A day no sourced interval covers is excluded.

**Settings.** One clip; the perpetual leg at isolated margin of one times its
notional (so K-liq fires only on a move that doubles the price within the
hour's high); fills at the open of the hour after the decision, taker on both
legs, slippage bracketed at 0, 1 and 2 bp a side with the verdict at 2; the
funding paid at each settlement on the short leg's notional at that hour's
open; capital counted as both legs' notional, its annual return a measurement.

**Status rules.** The family {C1, C2, C3}, Holm's m = 3 on H and on P; a slot
passes when no kill fires, its one-sided p passes Holm at 5%, its Newey-West t
is at least 3, its median day is positive and it held a position on at least 60
days of H (30 of P); the precedence and the statuses (`void`, `inconclusive`,
`killed`, `candidate`, `confirmed`, `not assessed`) as in the spot study.

**Headline sentences**, fixed now, brackets filled by the entry point:
"[slot] on H, Binance [pair]: [status]; first kill [K-x] at [value] against
[reference]." and "C1 on H: funding [x] bp a day against fees [y] and basis
[z]; [status]."

**The null-placeholder sha256.** θ for C2 and C3, σ_D, the minimum detectable
effects and the compute record are `null` placeholders commented `# frozen by
the D amendment`; the registrar computes the hash and writes it into the
document:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop null-hash configs/funding_carry_2026.yaml
```

**The gate.** The loop stops here. The owner reviews and commits both files;
`run --resume` then finds them committed, unchanged, and the document holding
the YAML's hash, and only then starts the run step. From that commit on, the
study's own E, H, E2 and P are in the protected calendar.

### The code to come

No role writes study code. The owner writes, in new files of about 120 lines
with a test each: the fee table and its exclusion, the hedged-position
accounting with its identity, the funding and basis ledger, the gate and its
placebos, the statuses, the ledger on the spot study's pattern, the pages, and
a synthetic venue for the dry run; then Bybit's funding as a measurement,
through the existing `data/bybit_funding.py`. The skeptic reviews the code in
its three lenses; every choice the code makes is written into the registration
before any day of a block is fetched; the code-freeze note
`docs/funding_carry_2026_code_freeze.md` names the commit.

### The entry point to come

`experiments/funding_carry_2026.py` does not exist. Once it does, and its
code-freeze note is committed, the executor would run, in order:

```bash
python -m experiments.funding_carry_2026 --dry-run
python -m experiments.funding_carry_2026 0
python -m experiments.funding_carry_2026 1
python -m experiments.funding_carry_2026 D
python -m experiments.funding_carry_2026 amend
python -m experiments.funding_carry_2026 open H
python -m experiments.funding_carry_2026 read H
python -m experiments.funding_carry_2026 status
python -m experiments.funding_carry_2026 pages
```

`open H` and `read H` run only on the owner's grant for the block,
`--grant funding_carry_2026:H`, after the owner has committed the D amendment,
and the first `read H` stops after the fetch for its records to be committed.

## 3. Run (executor)

Until the entry point exists, the executor runs only existing commands, and
only once the registration is committed. The policy decides each; the table
gives its answer today and once the registration's blocks have joined the
calendar:

| Command | Today | Registered |
|---|---|---|
| `trading-research download --start 2020-01-01 --end 2024-01-31 --symbol BTCUSDT --kind fundingRate --market futures-um --output data/funding_carry_2026/raw` | allowed | allowed |
| `trading-research download --start 2024-07-01 --end 2025-06-30 --symbol ETHUSDT --kind klines --interval 1h --market spot --output data/funding_carry_2026/raw` | allowed | allowed |
| `trading-research download --start 2024-02-01 --end 2024-02-25 --symbol BTCUSDT --kind fundingRate --market futures-um --output data/funding_carry_2026/raw` | refused | refused |
| `trading-research download --start 2025-01-01 --end 2025-07-31 --symbol BTCUSDT --kind klines --interval 1h --market futures-um --output data/funding_carry_2026/raw` | refused | refused |
| `trading-research download --start 2026-04-01 --end 2026-06-30 --symbol BTCUSDT --kind fundingRate --market futures-um --output data/funding_carry_2026/raw` | allowed | refused |
| `trading-research download --start 2026-08-01 --end 2026-08-31 --symbol BTCUSDT --kind klines --interval 1h --market spot --output data/funding_carry_2026/raw` | refused | refused |
| `trading-research download --start 2020-01-01 --end 2024-01-31 --symbol BTCUSDT --kind fundingRate --market futures-um` | refused | refused |
| `python -m experiments.spot_2026 open H` | refused | refused |

The third is refused because February 2024's archive holds the market-making
studies' H; the fourth because July 2025 is the 2025 study's E and H; the fifth
is allowed today only because no registration protects H yet, which is why the
run step waits for the committed registration; the seventh writes to
`data/raw`, outside the study's root. The fetch itself belongs to the entry
point's stage 1, after the code freeze, with its records and checksums.

## 4. Review (skeptic)

In its three lenses, each finding with a `file:line`:

- **Leakage and look-ahead.** The gate reads only rates settled before the
  decision: the rate shown during a funding period is a forecast, and the
  archive's rate at a settlement is known only at it. θ is fitted on D alone.
  No path under `data/funding_carry_2026/` names a month of H, E, E2 or P
  before the ledger opens it.
- **Fidelity to the registration.** The null-placeholder sha256, recomputed
  with `python -m agents.workflows.research_loop null-hash configs/funding_carry_2026.yaml`,
  equals the document's; the frozen configuration's sha256 equals the D
  amendment's line; `git diff` since the code-freeze commit touches no study
  code; the fee intervals cover every day of H; Holm runs with m = 3; the kill
  order and the sentences are the registered ones.
- **Execution realism and money.** Fees on both legs, at entry and exit, at the
  verdict tier; the funding's sign (a positive rate pays the short leg) and its
  notional at the settlement hour; the basis on the hedged quantity; K-liq at
  each hour's high; capital as both legs; the funding part and the basis part
  reported apart, so that a result carried by basis is not read as carry.

A blocking finding stops the loop before the verdict.

## 5. Verdict (code)

The orchestrator reads `experiments/results/funding_carry_2026_H.json`, checks
its sha256 against the ledger's first read of H, and copies each slot's status
and first kill and the page's headline sentences into `verdict.json`. No role
writes or changes a status; D numbers are not results.

## 6. Record (scribe)

A dated entry in `docs/funding_carry_2026_run_log.md`; the study's row in
`docs/findings.md` (`pre-registered` from the registration's commit, then its
status after H, killed hypotheses with the kill that fired); the README's
roadmap row; each sentence one of the verdict's, verbatim.

## What waits on the owner

The brief and the start; the information-set date; the hosts of the fee
pages; the commit of the registration; the study's code, its reviews and its
code freeze; the commit of the D amendment; the grant of H and the commit of
its open entry and fetch records; the commit of the results and the records;
and, for a candidate only, P in 2027. Nothing is pushed at any step.
