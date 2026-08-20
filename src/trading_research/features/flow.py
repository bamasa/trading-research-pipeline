"""What was actually traded, as opposed to what was offered.

Every feature this project has used until now comes from the order book: sizes
resting at prices, and how they move. That describes intent. It does not
describe action, and the two differ in exactly the way that matters — a book can
be rebuilt and cancelled without anything changing hands, while a print is
somebody committing.

The distinction has a name in the literature. Order-flow imbalance — signed
aggressive volume — is the quantity Cont, Kukanov and Stoikov relate linearly to
price change, and it is a different measurement from queue imbalance, which is
what this project has been using. Queue imbalance says who is *waiting*;
order-flow imbalance says who is *crossing*. The second is the one prices move
with.

Bybit's public prints carry the aggressor side, so all of this is computable:
:mod:`trading_research.data.bybit_trades` parses them, and everything here is a
trailing summary aligned to the same grid as the book.

Scale invariance
----------------
Every feature is a ratio, a share, or a count normalised by its own trailing
level. An instrument trading a thousand lots a second and one trading three must
produce comparable numbers, or a model fitted on one says nothing about the
other and a cross-sectional comparison is a comparison of tick sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trailing windows, in rows. On a five-second grid these are half a minute,
#: two minutes and ten.
WINDOWS = (6, 24, 120)


def align_to_grid(trades: pd.DataFrame, timestamps: pd.Series, mid: np.ndarray) -> pd.DataFrame:
    """Bucket prints onto the book's grid, keeping the aggressor's sign.

    A print is assigned to the interval containing it, so a row summarises the
    trades that happened *during* it — strictly in the past relative to the next
    row, which is what makes the features below usable at decision time.
    """
    edges = timestamps.to_numpy()
    slot = np.searchsorted(edges, trades["timestamp"].to_numpy(), side="right") - 1
    inside = (slot >= 0) & (slot < len(edges))
    slot = slot[inside]
    aggressor = trades["aggressor"].to_numpy()[inside]
    size = trades["size"].to_numpy()[inside]
    price = trades["price"].to_numpy()[inside]

    n = len(edges)
    out = pd.DataFrame(
        {
            "buy_volume": np.zeros(n),
            "sell_volume": np.zeros(n),
            "trade_count": np.zeros(n),
            "signed_notional": np.zeros(n),
            "traded_notional": np.zeros(n),
            "largest_trade": np.zeros(n),
        }
    )
    buying = aggressor == 1
    np.add.at(out["buy_volume"].to_numpy(), slot[buying], size[buying])
    np.add.at(out["sell_volume"].to_numpy(), slot[~buying], size[~buying])
    np.add.at(out["trade_count"].to_numpy(), slot, np.ones(len(slot)))
    np.add.at(out["signed_notional"].to_numpy(), slot, aggressor * size * price)
    np.add.at(out["traded_notional"].to_numpy(), slot, size * price)
    np.maximum.at(out["largest_trade"].to_numpy(), slot, size * price)

    # Where the trades happened relative to the mid, so a row knows whether the
    # crossing was into the offer or into the bid.
    out["vwap_deviation_bp"] = 0.0
    traded = out["traded_notional"].to_numpy()
    weighted = np.zeros(n)
    np.add.at(weighted, slot, size * price)
    volume = np.zeros(n)
    np.add.at(volume, slot, size)
    active = volume > 0
    vwap = np.where(active, weighted / np.maximum(volume, 1e-12), np.nan)
    with np.errstate(invalid="ignore"):
        out.loc[active, "vwap_deviation_bp"] = (vwap[active] / mid[active] - 1.0) * 1e4
    out.loc[~active, "vwap_deviation_bp"] = 0.0
    del traded
    return out


def build_flow_features(buckets: pd.DataFrame) -> pd.DataFrame:
    """Trailing summaries of aggressive trading, all scale-free."""
    out = pd.DataFrame(index=buckets.index)
    buy = buckets["buy_volume"]
    sell = buckets["sell_volume"]

    for window in WINDOWS:
        rolled_buy = buy.rolling(window, min_periods=1).sum()
        rolled_sell = sell.rolling(window, min_periods=1).sum()
        total = rolled_buy + rolled_sell
        # The Cont-Kukanov-Stoikov quantity: net aggressive volume as a share of
        # aggressive volume. Bounded in [-1, 1] and comparable across
        # instruments by construction.
        out[f"flow_imbalance_{window}"] = np.where(
            total > 0, (rolled_buy - rolled_sell) / total, 0.0
        )
        out[f"trade_intensity_{window}"] = (
            buckets["trade_count"].rolling(window, min_periods=1).mean()
        )

    # Is the market busier or quieter than it has been? A ratio of a short
    # window to a long one, so the level drops out.
    short = buckets["traded_notional"].rolling(WINDOWS[0], min_periods=1).sum()
    long = buckets["traded_notional"].rolling(WINDOWS[-1], min_periods=1).sum()
    out["volume_surge"] = np.where(long > 0, short / (long * WINDOWS[0] / WINDOWS[-1] + 1e-12), 1.0)

    # How concentrated the flow is. One large print and a hundred small ones
    # move a book differently even at identical volume.
    traded = buckets["traded_notional"].rolling(WINDOWS[1], min_periods=1).sum()
    largest = buckets["largest_trade"].rolling(WINDOWS[1], min_periods=1).max()
    out["trade_concentration"] = np.where(traded > 0, largest / traded, 0.0)

    # Average aggressive trade size relative to its own trailing level: a jump
    # means larger participants arrived.
    mean_size = buckets["traded_notional"] / buckets["trade_count"].replace(0, np.nan)
    out["trade_size_shock"] = (
        mean_size / mean_size.rolling(WINDOWS[-1], min_periods=10).median()
    ).fillna(1.0)

    out["vwap_deviation_bp"] = buckets["vwap_deviation_bp"]
    # Sustained pressure: the share of recent rows where buyers dominated.
    signed = np.sign(buckets["signed_notional"])
    out["flow_persistence_24"] = signed.rolling(24, min_periods=1).mean()

    return out.replace([np.inf, -np.inf], np.nan)
