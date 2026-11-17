# Regime suite

Fitted on the two weeks before the week before each break, frozen, and traded through the week before and the two weeks after, at Bybit's base taker fee, on Binance's one-minute bars; the kill switch is Page's CUSUM on the strategy's hourly results ([`bench/regimes.py`](../../src/trading_research/bench/regimes.py)).

| strategy | scored | holds | breaks | never paid | kill switch helped | mean kept by the switch, % | stops before the break |
|---|--:|--:|--:|--:|--:|--:|--:|
| relative | 10 of 10 | 0 | 0 | 10 | 10 | +32.12 | 10 |
| reversion | 10 of 10 | 3 | 4 | 3 | 7 | +7.46 | 2 |
| spread | 10 of 10 | 0 | 0 | 10 | 0 | +0.00 | 10 |

## relative

| episode | status | trades before / after | net bp before | net bp after | left running, % | with the kill switch, % | stop, hours after the break |
|---|---|--:|--:|--:|--:|--:|--:|
| china-ban-2021 | never paid | 14,774 / 30,512 | -12.20 | -8.84 | -100.0 | -64.4 | -94 |
| top-2021 | never paid | 14,530 / 29,406 | -12.82 | -12.40 | -100.0 | -67.9 | -95 |
| luna-2022 | never paid | 14,978 / 30,235 | -13.18 | -11.88 | -100.0 | -73.1 | -95 |
| celsius-2022 | never paid | 15,175 / 30,248 | -12.02 | -11.88 | -100.0 | -68.4 | -94 |
| ftx-2022 | never paid | 14,730 / 29,894 | -12.11 | -11.38 | -100.0 | -64.8 | -94 |
| svb-2023 | never paid | 15,117 / 30,215 | -12.63 | -12.21 | -100.0 | -68.3 | -95 |
| etf-2024 | never paid | 15,134 / 30,025 | -12.46 | -12.31 | -100.0 | -67.1 | -95 |
| ath-2024 | never paid | 14,610 / 29,259 | -12.84 | -12.62 | -100.0 | -68.7 | -95 |
| yen-carry-2024 | never paid | 15,053 / 30,238 | -12.72 | -12.44 | -100.0 | -69.0 | -95 |
| election-2024 | never paid | 14,974 / 29,288 | -12.28 | -12.34 | -100.0 | -67.2 | -95 |

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

## spread

| episode | status | trades before / after | net bp before | net bp after | left running, % | with the kill switch, % | stop, hours after the break |
|---|---|--:|--:|--:|--:|--:|--:|
| china-ban-2021 | never paid | 24,054 / 48,379 | -13.23 | -13.22 | -100.0 | -100.0 | -89 |
| top-2021 | never paid | 23,858 / 47,837 | -13.27 | -12.72 | -100.0 | -100.0 | -93 |
| luna-2022 | never paid | 23,933 / 48,278 | -13.37 | -13.23 | -100.0 | -100.0 | -94 |
| celsius-2022 | never paid | 24,016 / 48,314 | -13.64 | -12.75 | -100.0 | -100.0 | -94 |
| ftx-2022 | never paid | 23,724 / 48,098 | -13.01 | -13.05 | -100.0 | -100.0 | -94 |
| svb-2023 | never paid | 23,872 / 47,965 | -13.26 | -12.60 | -100.0 | -100.0 | -94 |
| etf-2024 | never paid | 23,673 / 47,459 | -12.96 | -12.91 | -100.0 | -100.0 | -94 |
| ath-2024 | never paid | 23,480 / 48,203 | -13.25 | -13.10 | -100.0 | -100.0 | -94 |
| yen-carry-2024 | never paid | 23,786 / 48,257 | -13.51 | -12.97 | -100.0 | -100.0 | -95 |
| election-2024 | never paid | 24,057 / 48,239 | -13.10 | -12.73 | -100.0 | -100.0 | -94 |
