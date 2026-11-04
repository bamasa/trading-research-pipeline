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
    "planted-reversion": "7d96db8982524fefc2b9a396aa85d355fa7466a6cd3999e927e5239dc4e4030c",
    "null-reversion": "db93168e8fc9fc286730cb544892781fffb3bcd7a194b970df1c37bd291dff50",
    "bybit-2024-02": "3a348047e711e2478b78dcd92a039a2066fdef0d65b112f21f0c52d6c5d19d44",
    "bybit-2024-04": "ddb5b3eba54d46e2a626b4e220042b913d0df261a11a596075dc4158e4731c94",
}
