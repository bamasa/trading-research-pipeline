"""Version 1 of the benchmark, frozen: what a card must match to be compared.

A card states the version it was scored under and, for each track, the
fingerprint of the data it read (:meth:`trading_research.bench.tracks.TrackData.fingerprint`).
A card whose fingerprint matches the one here was scored on the same numbers
as every other version-1 card; one that does not is reported as such and is
not comparable. Changing a track, a seed, the scoring rule or the verdict makes
a version 2, with its own fingerprints; version 1 stays as written.
"""

from __future__ import annotations

VERSION = "1"

#: The date version 1 was frozen.
FROZEN = "2026-11-04"

#: sha256 of each track's data, as :meth:`TrackData.fingerprint` computes it.
FINGERPRINTS: dict[str, str] = {
    "planted-book": "b0c0866966a68715d513ca0b0691b95db997f27852e270b8fc8a7343b0189aad",
    "null-book": "65803b0480263c8587e4e0ccb253d24a7ed245686c426ef34745f2ee19b022bf",
    "planted-reversion": "515da5c2ff25bb5cd770aac65c864378984b86096bd762005b1a748562282986",
    "null-reversion": "d0b5418e848ed25d32e45af9e9b6d604e843e5f6d5c6d71ae31fdd1682fcf5fa",
    "bybit-2024-02": "3a348047e711e2478b78dcd92a039a2066fdef0d65b112f21f0c52d6c5d19d44",
    "bybit-2024-04": "ddb5b3eba54d46e2a626b4e220042b913d0df261a11a596075dc4158e4731c94",
}

#: The forward track, registered before its first scored day. Its data has no
#: fingerprint yet: the days had not happened. The first read records one.
FORWARD_TRACK = "bybit-2026-11"
FORWARD_REGISTERED = "2026-11-05"

#: The strategies the forward track is read for, frozen at registration: the
#: version-1 board's twelve, each at the sha256 of its plugin's source module
#: (``PluginSpec.fingerprint``) with its default parameters. A card read on the
#: forward track whose source differs is a new strategy, not a forward result.
FORWARD_STRATEGIES: dict[tuple[str, str], str] = {
    (
        "book_imbalance",
        "strategy",
    ): "02bdac939e7a6cfa12ba4cf3509eee8399382b78f01c31c4b3fead369546a2c5",
    ("ev_gate", "strategy"): "0eb9ae99255c2aea9929721fda74e6a7aca0a00acdd3b570b82eee439688ac98",
    ("hold_first", "strategy"): "0eb9ae99255c2aea9929721fda74e6a7aca0a00acdd3b570b82eee439688ac98",
    ("relative", "strategy"): "0eb9ae99255c2aea9929721fda74e6a7aca0a00acdd3b570b82eee439688ac98",
    ("reversion", "strategy"): "0eb9ae99255c2aea9929721fda74e6a7aca0a00acdd3b570b82eee439688ac98",
    ("spread", "strategy"): "0eb9ae99255c2aea9929721fda74e6a7aca0a00acdd3b570b82eee439688ac98",
    ("breakout", "model"): "a2328577914a8a8e14f32263d376ab784188ebaba691d76026fee313148cc475",
    ("logistic", "model"): "589473a780207da94ea461136ed809bea7159a358c49aabc0642b6f7582ea75f",
    ("mean_reversion", "model"): "a2328577914a8a8e14f32263d376ab784188ebaba691d76026fee313148cc475",
    ("momentum", "model"): "a2328577914a8a8e14f32263d376ab784188ebaba691d76026fee313148cc475",
    ("order_flow", "model"): "a2328577914a8a8e14f32263d376ab784188ebaba691d76026fee313148cc475",
    ("xgboost", "model"): "7c77c93096121ab29cda77ce9dcb6db4e1afa7e9f8ce6d04f71c2c3aa4ab137f",
}

#: sha256 of the forward track's fitted days (9 October to 5 November 2026) as
#: :func:`trading_research.bench.tracks.fitted_window` grids them, recorded
#: before the track opens; empty until recorded. Those days ended before the
#: registration, so they are not sealed; the scored days are not downloaded.
FORWARD_FIT_FINGERPRINT = ""
