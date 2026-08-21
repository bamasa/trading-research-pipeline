"""Where the information is, and at what horizon — before any model is chosen.

The project's shortfall is now a number: an information coefficient of about
0.08 to 0.10 is needed to pay for a taker round trip at two minutes, and the
best of everything tried reaches 0.023. What that number does not say is
*which* of the missing four fifths is findable and where to look for it.

This script asks the question source by source, using
:mod:`trading_research.evaluation.information`. Five blocks:

* **touch** — the top of the book, which is what every earlier section used.
* **depth** — ten levels, added in §24 and never audited on its own.
* **flow** — aggressive trade prints, which the project never had at all.
* **cross** — what the other twenty-five instruments were doing.
* **calendar** — hour of day and day of week, present only as a control. If a
  calendar block scores like a real source, the measurement is picking up
  seasonality and the other numbers need discounting.

and three horizons: two minutes, one hour, one day. The horizon axis is here
because the cost of a round trip does not grow with it while the size of the
move does — roughly as its square root — so the coefficient needed to break even
falls the further out one looks. Two minutes needs 0.08; a day needs something
closer to 0.01, which is *below what the project already achieves*. Whether that
survives contact with a real forecast is exactly what wants measuring.

Read the ``incremental_ic`` column, not ``linear_ic``. The first says whether a
source is worth building; the second says whether it correlates, which a source
that merely restates one already in hand also does.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from trading_research.data.bybit_trades import load as load_trades
from trading_research.data.ensure import ensure_book, ensure_trades
from trading_research.evaluation.information import audit
from trading_research.features.cross import build_cross_features
from trading_research.features.depth import build_depth_features
from trading_research.features.flow import align_to_grid, build_flow_features
from trading_research.features.normalise import RollingNormaliser

GRID_SECONDS = 5
ROWS_PER_DAY = 86_400 // GRID_SECONDS

#: Two minutes, one hour, one day — in rows of the five-second grid.
HORIZONS = {"2min": 24, "1h": 720, "1day": 17_280}

#: Ordered by liquidity; the first is the lead-lag leader.
LEADERS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def to_grid(book: pd.DataFrame, seconds: int = GRID_SECONDS) -> pd.DataFrame:
    frame = book.set_index("timestamp").sort_index()
    parts = [
        day.resample(f"{seconds}s").last().ffill()
        for _, day in frame.groupby(frame.index.date, sort=True)
    ]
    grid = pd.concat(parts)
    return grid.dropna(subset=["bid_price_0", "ask_price_0"]).reset_index()


def touch_features(book: pd.DataFrame) -> pd.DataFrame:
    bid, ask = book["bid_price_0"].to_numpy(), book["ask_price_0"].to_numpy()
    bid_size, ask_size = book["bid_size_0"].to_numpy(), book["ask_size_0"].to_numpy()
    mid = (bid + ask) / 2.0
    log_mid = pd.Series(np.log(mid))
    out = pd.DataFrame(index=book.index)
    out["spread_bp"] = (ask - bid) / mid * 1e4
    out["queue_imbalance"] = (bid_size - ask_size) / (bid_size + ask_size)
    for span in (10, 20, 50, 120):
        out[f"ret{span}"] = log_mid.diff(span) * 1e4
    for span in (50, 200):
        out[f"vol{span}"] = log_mid.diff().rolling(span).std() * 1e4
    out["imbalance_change_10"] = out["queue_imbalance"].diff(10)
    out["quote_intensity_20"] = (log_mid.diff() != 0).rolling(20).mean().to_numpy()
    return out


def calendar_features(stamps: pd.Series) -> pd.DataFrame:
    """A control block. Real sources must beat the clock."""
    hour = stamps.dt.hour + stamps.dt.minute / 60.0
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday": stamps.dt.dayofweek.astype(float).to_numpy(),
        },
        index=stamps.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BICOUSDT")
    parser.add_argument("--book", type=Path, default=Path("data/book"))
    parser.add_argument("--trades", type=Path, default=Path("data/trades"))
    parser.add_argument("--universe", type=Path, default=Path("data/universe"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    span = (date(2024, 2, 1), date(2024, 3, 10))
    ensure_book(args.symbol, *span, root=args.book, depth=10)
    ensure_trades(args.symbol, *span, root=args.trades)

    files = sorted((args.book / args.symbol).glob("*.parquet"))
    book = to_grid(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
    stamps = book["timestamp"]
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
    print(f"{args.symbol}: {len(book):,} rows on a {GRID_SECONDS}s grid")

    blocks: dict[str, list[str]] = {}
    frame = pd.DataFrame(index=book.index)

    touch = touch_features(book)
    blocks["touch"] = list(touch.columns)
    frame = pd.concat([frame, touch], axis=1)

    depth = build_depth_features(book)
    blocks["depth"] = list(depth.columns)
    frame = pd.concat([frame, depth.set_index(frame.index)], axis=1)

    trades = load_trades(args.symbol, args.trades)
    flow = build_flow_features(align_to_grid(trades, stamps, mid))
    blocks["flow"] = list(flow.columns)
    frame = pd.concat([frame, flow.set_index(frame.index)], axis=1)
    print(f"  {len(trades):,} prints aligned")

    # The cross-sectional panel, resampled onto the same grid.
    panel: dict[str, pd.Series] = {}
    for directory in sorted(p for p in args.universe.iterdir() if p.is_dir()):
        universe_files = sorted(directory.glob("*.parquet"))
        if len(universe_files) < 20:
            continue
        other = pd.concat(
            [
                pd.read_parquet(f, columns=["timestamp", "bid_price_0", "ask_price_0"])
                for f in universe_files
            ],
            ignore_index=True,
        )
        other = to_grid(other)
        series = pd.Series(
            np.log(((other["bid_price_0"] + other["ask_price_0"]) / 2).to_numpy()),
            index=pd.DatetimeIndex(other["timestamp"]),
        )
        panel[directory.name] = series[~series.index.duplicated()]

    if args.symbol not in panel:
        panel[args.symbol] = pd.Series(np.log(mid), index=pd.DatetimeIndex(stamps))
    ordered = [s for s in LEADERS if s in panel] + [s for s in panel if s not in LEADERS]
    aligned = pd.DataFrame({name: panel[name] for name in ordered}).ffill()
    aligned = aligned.reindex(pd.DatetimeIndex(stamps)).ffill()
    print(f"  {aligned.shape[1]} instruments in the panel")

    cross = build_cross_features(aligned, args.symbol, leader=ordered[0])
    cross.index = frame.index
    blocks["cross"] = list(cross.columns)
    frame = pd.concat([frame, cross], axis=1)

    clock = calendar_features(stamps)
    blocks["calendar"] = list(clock.columns)
    frame = pd.concat([frame, clock], axis=1)

    # Rolling normalisation, as everywhere else, so scales do not decide the
    # ceiling. Bounded columns are left alone.
    bounded = {
        c for c in frame.columns if "imbalance" in c or "rank" in c or "sin" in c or "cos" in c
    }
    scaled = [c for c in frame.columns if c not in bounded]
    frame = RollingNormaliser(window=4000, exclude=frozenset(bounded)).transform(frame, scaled)

    cut = int(len(frame) * 0.65)
    tables = []
    for name, horizon in HORIZONS.items():
        forward = np.full(len(mid), np.nan)
        forward[:-horizon] = (mid[horizon:] / mid[:-horizon] - 1.0) * 1e4
        target = pd.Series(forward, index=frame.index)
        # Purge the horizon out of the training block so the label cannot read
        # into the rows being scored.
        train = slice(0, max(0, cut - horizon))
        test = slice(cut, len(frame))
        if train.stop < 20_000:
            print(f"  {name}: training block too short after purging, skipped")
            continue
        print(f"  auditing {name} (horizon {horizon} rows)...", flush=True)
        table = audit(
            blocks,
            frame.iloc[train],
            target.iloc[train],
            frame.iloc[test],
            target.iloc[test],
            with_mutual_information=(horizon <= 720),
        )
        table.insert(0, "horizon", name)
        table.insert(1, "move_sd_bp", float(np.nanstd(forward[test])))
        tables.append(table)

    if not tables:
        raise SystemExit("no horizon could be audited")
    out = pd.concat(tables, ignore_index=True)
    out.insert(0, "symbol", args.symbol)
    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(out, f"information_audit_{args.symbol}")


if __name__ == "__main__":
    main()
