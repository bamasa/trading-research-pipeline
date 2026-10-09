"""The search engine of the 2025 short-horizon study.

The pieces the single entry point composes for stages 0 to D: the catalogue
of taker cells and its hash (:mod:`.catalogue`), the trial ledger
(:mod:`.ledger`), the market gate's inputs (:mod:`.market_gate`), walk tables
at every 100 ms row (:mod:`.walk_tables`), candidate-row tapes
(:mod:`.tapes`), the per-cell driver (:mod:`.cells`) and Null B's walk-table
route (:mod:`.null_b`).
"""
