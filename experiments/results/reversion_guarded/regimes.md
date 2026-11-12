# Regime suite

Fitted on the two weeks before the week before each break, frozen, and traded through the week before and the two weeks after, at Bybit's base taker fee, on Binance's one-minute bars; the kill switch is Page's CUSUM on the strategy's hourly results ([`bench/regimes.py`](../../src/trading_research/bench/regimes.py)).

| strategy | scored | holds | breaks | never paid | kill switch helped | mean kept by the switch, % | stops before the break |
|---|--:|--:|--:|--:|--:|--:|--:|
| reversion | 10 of 10 | 3 | 4 | 3 | 7 | +7.46 | 2 |
| reversion_guarded | 10 of 10 | 1 | 5 | 4 | 3 | +0.23 | 2 |

## reversion

| episode | status | trades before / after | net bp before | net bp after | left running, % | with the kill switch, % | stop, hours after the break |
|---|---|--:|--:|--:|--:|--:|--:|
| china-ban-2021 | never paid | 287 / 1,582 | -9.63 | -22.64 | -32.1 | -24.9 | +13 |
| top-2021 | breaks | 61 / 361 | +31.86 | -40.17 | -10.5 | +0.1 | +22 |
| luna-2022 | breaks | 119 / 2,226 | +14.39 | -9.03 | -15.3 | +0.1 | +17 |
| celsius-2022 | breaks | 233 / 1,320 | +25.40 | -13.68 | -10.1 | +1.9 | +6 |
| ftx-2022 | holds | 220 / 2,036 | -4.75 | +0.48 | -0.1 | -2.0 | +4 |
| svb-2023 | never paid | 154 / 978 | -22.85 | -3.92 | -6.1 | -2.6 | +71 |
| etf-2024 | holds | 181 / 186 | +43.22 | +12.88 | +8.5 | +4.7 | +16 |
| ath-2024 | holds | 47 / 84 | +83.33 | +56.61 | +7.2 | +7.2 | - |
| yen-carry-2024 | breaks | 315 / 1,261 | +9.22 | -5.81 | -3.7 | +1.2 | -57 |
| election-2024 | never paid | 181 / 2,478 | -26.36 | -12.17 | -29.1 | -2.5 | -69 |

## reversion_guarded

| episode | status | trades before / after | net bp before | net bp after | left running, % | with the kill switch, % | stop, hours after the break |
|---|---|--:|--:|--:|--:|--:|--:|
| china-ban-2021 | never paid | 275 / 95 | -12.79 | -152.28 | -15.0 | -15.0 | +13 |
| top-2021 | breaks | 61 / 59 | +31.86 | -47.78 | -0.7 | +0.4 | +22 |
| luna-2022 | breaks | 119 / 86 | +14.39 | -14.40 | +0.4 | +0.2 | +17 |
| celsius-2022 | breaks | 195 / 71 | +21.78 | -32.02 | +1.6 | +1.6 | - |
| ftx-2022 | never paid | 200 / 29 | -2.62 | -47.25 | -1.6 | -1.1 | +4 |
| svb-2023 | never paid | 151 / 170 | -24.03 | -3.69 | -3.5 | -3.5 | - |
| etf-2024 | breaks | 159 / 59 | +43.99 | -0.43 | +5.8 | +5.8 | - |
| ath-2024 | holds | 47 / 79 | +83.33 | +56.11 | +7.0 | +7.0 | - |
| yen-carry-2024 | breaks | 126 / 0 | +11.90 | +nan | +1.2 | +1.2 | -57 |
| election-2024 | never paid | 126 / 7 | -28.92 | -78.60 | -3.5 | -2.5 | -69 |
