"""xs_weekly_2026_11: fetch, the development replay, and the read of W1 and W2 on H.

PYTHONPATH=src python -m experiments.xs_weekly fetch-dev --data data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_research.data import binance_monthly
from trading_research.xs_weekly import EXCLUDED

D_MONTHS = [str(m) for m in pd.period_range("2020-01", "2024-12", freq="M")]
H_MONTHS = [str(m) for m in pd.period_range("2025-01", "2026-10", freq="M")]


def symbols() -> list[str]:
    every = binance_monthly.list_symbols()
    return sorted(s for s in every if s.endswith("USDT") and "_" not in s and s not in EXCLUDED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("fetch-dev",))
    parser.add_argument("--data", type=Path, default=Path("data"))
    a = parser.parse_args()
    root = a.data / "xs_weekly"
    syms = symbols()
    (root).mkdir(parents=True, exist_ok=True)
    (root / "symbols.txt").write_text("\n".join(syms) + "\n")
    print(len(syms), "symbols", flush=True)
    print(binance_monthly.download(root, syms, D_MONTHS), flush=True)


if __name__ == "__main__":
    main()
