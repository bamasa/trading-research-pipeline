"""Market making on public Bybit data: an event-time simulator and what it needs.

The modules, in the order an instrument-day passes through them:

* :mod:`.events` — one instrument-day as a single time-ordered stream of
  funding settlements, trade prints and book snapshots, prices in integer ticks;
* :mod:`.orders` and :mod:`.queue` — an order's life and its estimated place in
  the queue at its price, advanced only by prints;
* :mod:`.accounting` — average-cost position, cash, fees and funding, with an
  identity checked at every change;
* :mod:`.quoters` — what a quoting rule may see and what it returns;
* :mod:`.signals` — external signals joined at the end of their bin;
* :mod:`.simulator` — the event loop, latency, limits, flattening and the
  per-day result;
* :mod:`.analysis` — markouts and the profit decomposition, after the loop;
* :mod:`.synthetic` — known-answer markets for the tests.

The rules the simulator follows, each with the direction it biases a result
and the test that pins it, are in ``docs/market_making_simulator.md``. The study
they serve is pre-registered in ``docs/preregistration/market_making.md``.
"""
