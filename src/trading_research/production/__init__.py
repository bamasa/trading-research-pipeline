"""Production: frozen strategies run as one supervised, balanced, observable portfolio.

:mod:`.spec` reads a deployment and pins its models; :mod:`.allocator` balances
capital across strategies; :mod:`.monitors` watch each one for drift from its
backtest, a regime break and a dead feed; :mod:`.supervisor` turns their alarms
into states and position limits; :mod:`.health` serves health, readiness and
metrics; :mod:`.runner` puts them together over paper sessions. The broker is
the paper ledger: nothing here sends an order to a venue.
"""
