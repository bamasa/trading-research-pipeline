# Run log of the 2026 funding-rate carry study

One entry per day the study moves (times UTC). The registration is
[`preregistration/funding_carry_2026.md`](preregistration/funding_carry_2026.md);
the loop that wrote it is described in [`agents.md`](agents.md).

## 2026-11-02

- Nothing fetched: no funding rate, perpetual price or basis of any date has
  been read for this study, and no block has been granted.
- The research loop's first live run, on the owner's brief. 16:33, the
  researcher's first turn did not reach the model (the SDK had no
  credentials); taken again after the owner logged the command-line client
  in. 16:49, the researcher proposed C1 to C4; the registrar refused, naming
  nine decisions the brief left open (the fee source and its dates, the BNB
  discount, the clip, the price of each mark, the funding notional, K-liq's
  series, the meaning of an unstated information set, P's deadline).
- 19:52, after the owner's answers, the registrar wrote the document and the
  YAML; the loop stopped the turn because prose in the document that
  describes the D amendment's line read as the line itself and froze the
  registrar's files. A fault of the harness, corrected with a test before the
  step was taken again.
- 19:58, the registrar finished the registration, null-placeholder sha256
  `5ce30410bd3e6928ea92ce11db0ed68c206a52dd14912501813678ea63930500`, and the
  owner committed it.
- 20:12, the executor's first command, the entry point's `status`, was
  refused: the entry point and the study's code do not exist. The next step is
  the owner's: write the code, have it reviewed and freeze it.

## 2026-11-03

- Nothing fetched. The first live run recorded in the harness's page.
