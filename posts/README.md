# Posts

Short pieces for social media, each built from this repository's code and
data and kept here so every number in them can be traced and rebuilt. A post
quotes backtests on recorded data: measurements, never a study's result, and
each says so.

Each post's folder holds the script that computes its numbers from the
archives (`make_data.py`), the numbers themselves (`numbers.json`) and the
series it draws (`data.json`), the animation (`animation.html`) and the script
that renders it to video (`render.mjs`, written under `artifacts/posts/`,
which is not committed). [`tests/test_posts.py`](../tests/test_posts.py) checks
that each post's text and animation quote its `numbers.json`.

| Post | What it shows |
|---|---|
| [`btc_ath_2024`](btc_ath_2024/README.md) | bitcoin's March 2024 high, and the reversion strategy fitted on February giving back everything within two days |
