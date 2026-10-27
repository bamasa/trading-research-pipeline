# Pre-registration: the synthetic carry study (`synth_carry`)

Registered <date> by the registrar of the harness's demonstration, in a scratch
repository, before any of the study's series was drawn.

Null-placeholder sha256: `<null-sha256>`

**Synthetic.** Every daily series this study reads is drawn from the seeded
generator `configs/synth_carry.yaml` registers, by `experiments/synth_carry.py`.
No market's data is read, and no number the study produces is evidence about a
market. It exists so that a reviewer can watch the research loop run end to end
in minutes, gates and refusals included.

## Why the study

The loop needs a study whose every step runs on a laptop without data: a
registration with a null placeholder, a development stage that fills it, a
ledger whose open entry is committed before the held-out block is read, and a
page whose headline sentences the scribe copies.

## What was known before this was written

Nothing: no draw of the generator had been made for any block.

## Expected outcome, stated in advance

C1 is killed by K1 on H. The generator's drift is -1.0 bp a day, net of fees,
against a standard deviation of 4.0 bp, so over H's 92 days the mean is about
-1.0 bp with a standard error of about 4.0 / sqrt(92) = 0.42 bp, and K1 (the
mean at or below 0.00 bp) fires with a probability above 99%.

## Blocks

| Block | Dates (inclusive) | Use |
|---|---|---|
| D | 2031-01-01 to 2031-06-30 | `sigma_d_bp`, frozen by the D amendment |
| E | 2031-07-01 to 2031-07-31 | never read |
| H | 2031-08-01 to 2031-10-31 | C1's status, read once, through the ledger |

## Slots and metric

C1, always on, one clip. The metric is each day's net in bp of notional after
the registered 2.0 bp taker fee per side; the status reads the mean of the
daily series and its day-level t.

## Status rules

The family is {C1}, Holm's m = 1. A slot is `killed` when a kill fires,
`candidate` when none fires and its t is at least 3.0, and `inconclusive`
otherwise.

## Kill conditions, in the order they are read

| Kill | Fires when |
|---|---|
| K1 | the mean of H's daily series is at or below 0.00 bp |
| K-days | half or fewer of H's days are positive |

## Placebos

None: one always-on slot has no timing to shuffle, and the generator has no
signal to destroy.

## Headline sentences

- When a kill fires: "C1 on H, synthetic venue: [status]; first kill [K-x] at
  [value] bp against [reference] bp."
- Otherwise: "C1 on H, synthetic venue: [status]; no kill fired; mean [value]
  bp a day, t [t]."

## What the study cannot establish

Anything about a market.

## Amendments
