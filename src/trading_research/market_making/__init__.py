"""Market making on public Bybit data: an event-time simulator and what it needs.

The modules, in the order an instrument-day passes through them:

* :mod:`.events` — one instrument-day as a single time-ordered stream of
  funding settlements, trade prints and book snapshots, prices in integer ticks;
* :mod:`.orders` and :mod:`.queue` — an order's life and its estimated place in
  the queue at its price, advanced only by prints;
* :mod:`.accounting` — average-cost position, cash, fees and funding, with an
  identity checked at every change;
* :mod:`.quoters` — what a quoting rule may see and what it returns, and the
  rules the study compares (S0-S4, X1);
* :mod:`.signals` — external signals joined at the end of their bin, and the
  reversion tape, its thresholds, triggers and taker twin;
* :mod:`.flags` — regime flags from both detectors, effective when known;
* :mod:`.simulator` — the event loop, latency, limits, flattening and the
  per-day result, and the runners over many days and configurations;
* :mod:`.analysis` — markouts and the profit decomposition, after the loop;
* :mod:`.screen` — instrument admission, computed on the development block;
* :mod:`.prereg` — the pre-registration's YAML, its blocks, and the ledger
  that guards the held-out reads;
* :mod:`.verdicts` — the registered statistics: day-level t, Holm, placebo
  percentiles, fee break-evens, and the status rule;
* :mod:`.heldout` — the held-out runner: per-day tapes built from the loaded
  day, fill placement, and the jobs that keep every simulated day's outputs;
* :mod:`.synthetic` — known-answer markets for the tests.

The rules the simulator follows, each with the direction it biases a result
and the test that pins it, are in ``docs/market_making_simulator.md``. The study
they serve is pre-registered in ``docs/preregistration/market_making.md``.
"""
