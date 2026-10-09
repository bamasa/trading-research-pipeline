"""The 2025 short-horizon study's stages, composed for its single entry point.

``experiments/short_horizon_2025.py`` is the study's one entry point. It drives
the registered stages in order, each a subcommand that refuses to run before
the stages it reads from are complete, and that is idempotent and resumable:

* **0** (:mod:`.stage0`): the catalogue's hash, the code timed on the two 2024
  days already on disk, D's projected wall time against the ceiling with the
  reduction order, the walk-table parity test (a gate for stage B) and the
  replay's peak memory;
* **1** (:mod:`.stage1`): the catalogue's hash again, the code-freeze note,
  the fetch of W, D and the reader-test days through
  :func:`trading_research.data.fetch_stage.run_stage` under the development
  access, day validation, the fitted values, the feature stores, selection on
  January and the leakage tests;
* **A** (:mod:`.fits`): the walk-forward fits, predictions and candidate-row tapes;
* **B** (:mod:`.scoring`): every cell on every test day into the trial ledger;
* **D** (:mod:`.search`): the choice, Null A, Null B, the refit, carry-forward
  and admission;
* **C** (:mod:`.passive`): the simulator for S-P, S-M and S25-Z;
* **E** (:mod:`.freeze`): the frozen fits and files, and the D amendment;
* **H, F, P** (:mod:`.reads`): each block read in one session through the
  study's ledger, computing every status and kill.

:mod:`.synthetic` serves a synthetic venue for the dry run, which runs every
stage end to end in a temporary repository without touching the network.
"""
