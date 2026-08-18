# Strategy review rubric

A checklist for judging whether a systematic trading result means anything. It
is written to be applied by someone — or something — that did not build the
strategy, and it is deliberately hard to score well on.

The rubric exists because the failure mode in this field is not bad models. It
is good-looking numbers produced by a process that could not have produced
anything else. Most of what follows checks the process, not the result.

## How to score

Eight dimensions, each 0–3. A dimension scores 0 when the question cannot be
answered from what was provided — "not stated" is not a pass.

| Score | Meaning |
|---|---|
| 0 | Not addressed, or cannot be determined from what was given |
| 1 | Addressed but with a defect that could plausibly reverse the result |
| 2 | Addressed adequately; residual risk is stated |
| 3 | Addressed thoroughly, with evidence that the check would have caught a failure |

**A result with a 0 or 1 on leakage, costs or selection should not be believed
regardless of its total.** Those three can each turn a loss into a profit on
their own, so a high score elsewhere is not compensation. Say so explicitly in
the verdict rather than averaging it away.

## The dimensions

### 1. Leakage
Could information from after the decision have reached the features or the
labels? Look for centred windows, backward fills, normalisation fitted over the
whole sample, and labels whose horizon overlaps the next training block. Ask
whether anything mechanically checks this, or whether it is asserted.

### 2. Costs
Is there a cost model, is it applied consistently at labelling, at the decision
and in the profit and loss, and does it match what the venue actually charges?
Check whether the spread is charged once per round trip rather than twice or
not at all. Check whether results are quoted gross or net, and whether that is
made unmissable.

### 3. Selection
Was anything chosen after seeing the data it is reported on — a threshold, a
horizon, an instrument, a date range, a model? Ask how many configurations were
tried in total, not how many are shown. A search over forty candidates reported
as one result is forty chances to fit the validation block.

### 4. Sample size and statistical power
How many trades, over how long, and what is the dispersion per trade? An edge
of 1 bp against per-trade dispersion of 8 bp needs thousands of trades before it
can be distinguished from nothing. A strategy trading a few times a month cannot
be evaluated on a year of data whichever way the number comes out.

### 5. Robustness
Does the result reproduce on a second instrument, a second period, a second
model, a different random seed? A result that appears once is a description of
the sample it appeared in. Look for the author testing this rather than
asserting it.

### 6. Path and risk
Are drawdown, its duration, and the distribution of outcomes reported, or only
the average? A strategy is run by someone who sees the path. Look for maximum
drawdown, return over drawdown, and the share of periods that lost.

### 7. Capacity and execution realism
Trades per day, position size, and what the fills assume. Does it model queue
position, partial fills, market impact? Would the strategy still work at ten
times the size? A backtest that assumes it is the only participant is a
different strategy from the one that would run.

### 8. Reproducibility
Can the numbers be regenerated from what was provided? Is there code for each
published figure, are seeds fixed, are data sources and versions recorded? A
result nobody else can produce is a claim, not a finding.

## What to write

A verdict of at most a paragraph, then the eight scores with one line of
evidence each, then the two or three things that would most change the
conclusion if they were wrong.

Two things to avoid. Do not reward effort — a thorough study of a strategy that
does not work scores well on process and badly on result, and both should be
said. And do not soften a low score because the author was candid about the
weakness; candour is what makes the score knowable, not what makes it higher.
