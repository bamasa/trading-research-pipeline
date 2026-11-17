"""Funding carry, always on (slot C1), measured on the development block of funding_carry_2026.

``docs/preregistration/funding_carry_2026.md`` registers a delta-neutral carry
on Binance (long spot, short the USD-M perpetual, equal quantity, one clip of
10,000 USDT a leg) and reads its verdict on blocks H and P. Its development
block D (2020-01 to 2020-11, 2021-07 to 2024-01, 2024-07 to 2025-06) is open
to the study by its own registration. This entry point measures slot C1 on
those months with the registration's fee table and slippage, every month whole,
nothing searched: **a measurement on development data, not the study's
result**, which only P can give.

Accounting, per coin and per D range: open both legs at the range's first hour
as a taker (spot 10 bp, perpetual 5 bp, 2 bp of slippage a leg), receive the
funding the short perpetual is paid at every settlement (the rate times the
quantity times the perpetual's price at the settlement hour; the registration
uses the mark price, here the last price's open), carry the basis (spot's move
less the perpetual's, on the same quantity), close both legs at the range's
last hour the same way. Writes experiments/results/funding_carry_dev.md.

    PYTHONPATH=src python experiments/funding_carry_dev.py --data data
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from trading_research.data.binance import ArchiveSpec, download_range

CLIP = 10_000.0
FEE_BP = {"spot": 10.0, "perp": 5.0}
SLIP_BP = 2.0
RANGES = ((date(2020, 1, 1), date(2020, 11, 30)), (date(2021, 7, 1), date(2024, 1, 31)),
          (date(2024, 7, 1), date(2025, 6, 30)))  # fmt: skip
COINS = ("BTCUSDT", "ETHUSDT")
OUT = Path(__file__).parent / "results" / "funding_carry_dev.md"


def _load(root: Path, spec: ArchiveSpec, start: date, end: date) -> pd.DataFrame:
    download_range(spec, start, end, root)
    files = sorted((root / spec.symbol / spec.directory).glob("*.parquet"))
    keep = [f for f in files if start <= date.fromisoformat(f.stem) <= end]
    return pd.concat([pd.read_parquet(f) for f in keep], ignore_index=True)


def carry(root: Path, coin: str, start: date, end: date) -> dict[str, float]:
    # each market in a folder of its own: spot and perpetual klines share a directory name
    kline = "klines"
    spot = _load(root / "spot", ArchiveSpec("spot", kline, coin, "1h"), start, end)
    perp = _load(root / "um", ArchiveSpec("futures-um", kline, coin, "1h"), start, end)
    rates = _load(root / "um", ArchiveSpec("futures-um", "fundingRate", coin), start, end)
    s = spot.set_index("open_time")["open"].astype(float).sort_index()
    f = perp.set_index("open_time")["open"].astype(float).sort_index()
    s_close = spot.set_index("open_time")["close"].astype(float).sort_index()
    f_close = perp.set_index("open_time")["close"].astype(float).sort_index()
    q = CLIP / float(s.iloc[0])
    fees = sum(2 * CLIP * (FEE_BP[leg] + SLIP_BP) / 1e4 for leg in ("spot", "perp"))
    hours = rates["timestamp"].dt.floor("1h")
    price = f.reindex(hours.to_numpy(), method="ffill").to_numpy()
    funding = float((rates["funding_rate"].to_numpy() * q * price).sum())
    basis = q * (float(s_close.iloc[-1]) - float(s.iloc[0])) - q * (
        float(f_close.iloc[-1]) - float(f.iloc[0])
    )
    days = (end - start).days + 1
    net = funding + basis - fees
    return {"days": days, "settlements": len(rates), "funding": funding, "basis": basis,
            "fees": fees, "net": net, "net_bp_a_day": net / CLIP * 1e4 / days,
            "apr_pct": net / (2 * CLIP) * 100 * 365 / days,
            "mean_rate_bp": float(rates["funding_rate"].mean() * 1e4),
            "share_positive": float((rates["funding_rate"] > 0).mean())}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    root = parser.parse_args().data / "funding_carry_dev_markets"
    rows = [(coin, a, b, carry(root, coin, a, b)) for coin in COINS for a, b in RANGES]
    lines = ["# Funding carry C1 on funding_carry_2026's development block", "",
             "A measurement on development data, not the study's result. Regular fees "
             "(spot 10 bp, perpetual 5 bp a side), 2 bp slippage a leg and side, one "
             "10,000 USDT clip a leg, always on, opened and closed once per range; capital is "
             "both legs, 20,000 USDT, the perpetual unlevered.", "",
             "| coin | range | days | settlements | mean rate, bp | rate > 0 | funding, USDT "
             "| basis, USDT | fees, USDT | net, USDT | net a year, % of capital |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]  # fmt: skip
    for coin, a, b, r in rows:
        lines.append(f"| {coin} | {a} to {b} | {r['days']} | {r['settlements']} | "
                     f"{r['mean_rate_bp']:+.2f} | {r['share_positive']:.0%} | {r['funding']:+,.0f} | "
                     f"{r['basis']:+,.0f} | {r['fees']:,.0f} | {r['net']:+,.0f} | {r['apr_pct']:+.1f} |")  # fmt: skip
        print(lines[-1], flush=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
