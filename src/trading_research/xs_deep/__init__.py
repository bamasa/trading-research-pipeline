"""Machine learning and deep learning on the cross-section of 26 perpetuals.

The study registered in ``docs/preregistration/xs_deep_forward_2026_12.md``:
six rankers of the instruments' next ten minutes, one rule, two classical
models and three networks, the last of which reads all instruments at once.

- :mod:`.panel`: one-minute bars of the recorded top of book;
- :mod:`.features`: the twelve features, the four series, the target, the costs;
- :mod:`.rankers`: the six rankers, their settings fixed by the registration;
- :mod:`.evaluate`: the walk-forward replay, the legs and the daily numbers.
"""
