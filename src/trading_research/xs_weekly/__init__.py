"""ML and DL ranking a hundred Binance perpetuals a week: the study ``xs_weekly_2026_11``.

- :mod:`.panel`: daily bars and funding of every symbol as arrays of day by coin;
- :mod:`.features`: the universe each day, the twelve features, the four series, the target;
- :mod:`.rankers`: the eight rankers;
- :mod:`.portfolio`: the weekly long-short book with fees and funding.
"""

#: Left out: held out by funding_carry_2026, or a stablecoin.
EXCLUDED = frozenset(
    {"BTCUSDT", "ETHUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "DAIUSDT"}
)
