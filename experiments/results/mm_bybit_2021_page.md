# Market making on Bybit BTCUSDT at the 2021 fees, from prints only: results

The study registered in
[`docs/preregistration/mm_bybit_2021.md`](../../docs/preregistration/mm_bybit_2021.md):
a passive quoter at the touch proxies of Bybit's BTCUSDT linear perpetual,
filled only when a print trades through its price, at the fees every account
paid in 2021 (maker −2.5 bp, a rebate; taker +7.5 bp). H, 8 April to 30 June
2021, was read once through the ledger at 21:13 UTC on 27 October 2026 and
closed with the results' sha256 `b19a334a…`. Every number below comes from
[`mm_bybit_2021_H.json`](mm_bybit_2021_H.json),
[`mm_bybit_2021_H_table.csv`](mm_bybit_2021_H_table.csv) or
[`mm_bybit_2021_H_gross.json`](mm_bybit_2021_H_gross.json) (sums of the read's
row file; nothing re-run). Money is USDT; "per 100 USDT of clip a day" is the
registered daily series `y`, the day's net over the notional of one 0.010 BTC
clip at 00:10 (a median 421 USDT on H).

**Uncalibrated fill rule.** The registration carries no calibration of the
prints-only fill rule against a real book. Every verdict on this page carries
that label. The calibration is left to a follow-up study: it was asked for after
H was opened, and nothing may change after the read.

## Verdicts (uncalibrated fill rule)

| Slot | Cell | Status | First kill | Net, bp per fill | Net, per 100 USDT of clip a day | Net, USDT a day | Days positive |
|---|---|---|---|---:|---:|---:|---:|
| S0 | `l0\|none` | **killed** | K1 at −227.27 against 0.00 | −2.69 | −227.27 | −964.50 | 0 of 84 |
| F | `l2\|vle0.33` | **not admitted on D (i, iv)**; killed by the precedence | K1 at −39.13 against 0.00 | −2.64 | −39.13 | −182.10 | 0 of 84 |

F, the cell D chose, failed both admission conditions on D (its S, −42.63, is
not positive, and p_A = p_B = 1.000), so it is a labelled measurement with Holm
p = 1, not a primary verdict. No slot is a candidate; Holm (m = 2) has nothing
to pass. Nothing was voided: the accounting identity and the turnover check held
on every day of every run (K-acct), and |position| never went past the soft
limit plus a clip (K-limit). No day of W, D or H was excluded.

### The registered sentences

- S0 (`l0|none`) on Bybit BTCUSDT, 8 April to 30 June 2021, at −2.5 / +7.5 bp:
  killed; first kill K1 at −227.27 USDT per 100 USDT of clip a day against 0.00
  USDT per 100 USDT of clip a day. *(Uncalibrated fill rule.)*
- At the fees every account paid in 2021, a touch quoter filled only on
  trade-throughs lost 2.69 bp per fill (−227.27 USDT per 100 USDT of clip a
  day); without the rebate it lost 5.19 bp per fill, of which the making part
  was −3.75 bp. *(S0.)*
- S0: not paid with the rebate; not paid ex-rebate.
- F (`l2|vle0.33`) on Bybit BTCUSDT, 8 April to 30 June 2021, at −2.5 / +7.5 bp:
  not admitted on D (i, iv); first kill K1 at −39.13 USDT per 100 USDT of clip a
  day against 0.00 USDT per 100 USDT of clip a day. *(Uncalibrated fill rule.)*
- At the fees every account paid in 2021, a touch quoter filled only on
  trade-throughs lost 2.64 bp per fill (−39.13 USDT per 100 USDT of clip a day);
  without the rebate it lost 5.14 bp per fill, of which the making part was
  −2.54 bp. *(F.)*
- F: not paid with the rebate; not paid ex-rebate.

## Net at the fees charged, with the zero-fee row beside it

| Slot | Row | bp per fill | Per 100 USDT of clip a day | USDT a day | Days positive |
|---|---|---:|---:|---:|---:|
| S0 | **net at −2.5 / +7.5 bp** | **−2.69** | **−227.27** | **−964.50** | **0** |
| S0 | gross, zero fees (reference only) | −5.19 | −441.62 | −1,858.93 | 0 |
| F | **net at −2.5 / +7.5 bp** | **−2.64** | **−39.13** | **−182.10** | **0** |
| F | gross, zero fees (reference only) | −5.13 | −76.78 | −354.39 | 0 |

The rebate credited exactly 2.50 bp of passive turnover; the 23:59 taker flatten
cost 0.004 (S0) and 0.007 (F) bp per fill in fees. Funding is under 0.003 bp
per fill.

### Where the loss is, bp of passive turnover

| Slot | Spread | Adverse (5 s) | Making part | Rebate | Taker fees | Funding | Inventory | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | +0.71 | −4.46 | −3.75 | +2.50 | −0.00 | −0.00 | −1.44 | −2.69 |
| F | +0.22 | −2.76 | −2.54 | +2.50 | −0.01 | −0.00 | −2.59 | −2.64 |

The making part (spread plus the five-second markout, before fees) is below
−2.5 bp for both slots, so the rebate does not cover it, and the position held
after the five seconds loses a further 1.4 (S0) to 2.6 (F) bp.

## Fills, inventory and markouts

| Slot | Fills a day | Passive turnover a day (USDT) | Max \|position\| (BTC) | Mean daily max \|position\| | Time at the soft limit | Gate open | Capacity (median day) |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 13,979 | 3,582,879 | 0.060 | 0.060 | 2.9% | 100% | 0.23% |
| F | 2,544 | 690,995 | 0.060 | 0.054 | 0.2% | 29.1% | 0.04% |

| Markout of the passive fills (bp) | 1 s | 5 s | 30 s |
|---|---:|---:|---:|
| S0 | −1.82 | −3.75 | −4.97 |
| F | −1.21 | −2.54 | −4.34 |
| Every print from its resting side (market-wide) | −0.85 | −2.19 | −2.67 |

## Kills, value against reference

`y` and its variants in USDT per 100 USDT of clip a day; K-days in share of
days positive. The conditional kills (K-delay, K-latency, K-fund, K-drift) fire
only on a positive result.

| Kill | S0 | F |
|---|---|---|
| K1 | **fires**: −227.27 vs 0.00 | **fires**: −39.13 vs 0.00 |
| K-dir | **fires**: −328.20 vs 0.00 | **fires**: −39.50 vs 0.00 |
| K-delay | −118.71 (not positive) | −25.97 (not positive) |
| K-latency | −211.97 at 150 ms (not positive) | −38.25 at 150 ms (not positive) |
| K-fund | −227.26 (not positive) | −39.10 (not positive) |
| K-drift | inventory −113.41 vs −113.64 (not positive) | inventory −37.25 vs −19.57 (not positive) |
| K-days | **fires**: 0% vs 50% | **fires**: 0% vs 50% |
| K-nbhd | **fires**: median −212.99 over {`l0`, `l0.5`} | **fires**: median −55.15 over F's neighbourhood |
| K-placebo | **fires**: −227.27 vs 95th percentile 14.55 | **fires**: −39.13 vs 95th percentile 2.26; shifted-state mean −44.61 |

t, Newey–West t and median day: S0 −11.46, −6.40, −182.34; F −18.74, −9.58,
−43.37. MDE: 61.46 (S0) and 6.70 (F).

## Measurements, each labelled with what changed (not results)

| Slot | Touch bracket (optimistic) | One further print | 150 ms | 10 ms | Ex-rebate (0 bp maker) | Negative control (+7.5 bp maker) | Positive control (−10 bp maker) |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | +324.35 | −118.71 | −211.97 | −228.19 | −441.90 | −1,085.78 | +416.60 |
| F | +60.66 | −25.97 | −38.25 | −38.87 | −76.87 | −190.08 | +74.08 |

Per 100 USDT of clip a day. The touch bracket fills a whole order at the first
print at or through its price, whatever was queued ahead: an upper bound on
fills, which is why both slots carry the label "paid at the touch only". It is
not a result. The negative control is negative and the positive control
positive, as the pipeline should give.

- **The stress day 2021-05-19**: S0 −1,204.39 (making part −1,517.03, inventory
  −384.33, 44,125 fills); H's mean without it −215.50. F's calm gate kept it out
  of the market that day (no fill, 0.00); its mean without it −39.60.
- **By month** (`y`): S0 April −129.35, May −290.35, June −237.17; F April
  −45.71, May −24.71, June −48.99.

## What D showed (development, selected, biased upward)

Every one of the 40 cells lost on D, from −2.31 to −2.63 bp per fill at the
rebate. F was the cell with the highest neighbourhood median and the outright
peak, and was not admitted. These numbers chose F and are not results.

## What happened to the registration

- Before the code freeze, stage 0 found that the registered 120 s print-gap
  rule excluded both reader-test days for a quiet market, not a hole in the
  archive; a dated amendment raised the threshold to 600 s (null-placeholder
  sha256 `17fca5f8…`). No day of W, D or H was excluded under it.
- The D amendment froze the configuration at sha256 `219b5aa0…`; the code was
  frozen at `e1f9961` and unchanged when H was opened.
- No calibration of the fill rule against a book was registered; hence the
  label above.

## What this cannot establish

What a real queue position would have earned (the rule assumes an infinite
queue ahead: a lower bound on fills, an upper bound on their toxicity); where the
touch really was (the proxies are the last prints of each side); anything about
today, other instruments, venues or years, or latency below 50 ms.
