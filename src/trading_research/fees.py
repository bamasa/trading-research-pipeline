"""Every fee table the repository holds, read where it lives and listed in one place.

Fees are written down in several places, each by the study or module that
uses them, and each stays the authority for its own numbers:

* ``backtest/costs.py``: Bybit's linear-perpetual schedule
  (:data:`~trading_research.backtest.costs.BYBIT_LINEAR_TIERS`, its source and
  the date it was transcribed), the market-maker programme's rebate, and the
  Binance USD-M taker fee that :class:`~trading_research.backtest.costs.TakerCosts`
  defaults to;
* ``backtest/maker.py``: the defaults of
  :class:`~trading_research.backtest.maker.MakerCosts`;
* ``configs/mm_prereg.yaml`` and ``configs/mm_prereg_round2.yaml``: the
  market-making rounds' fee blocks;
* ``configs/short_horizon_2025.yaml``: the 2025 study's verdict tier and its
  zero-fee measurement;
* ``configs/spot_2026.yaml``: the spot study's dated fee intervals per venue
  and tier, and its undated cells;
* ``configs/demo.yaml``: the synthetic demo configuration's fee.

This module only reads them. It changes nothing anywhere, chooses nothing and
holds no number of its own: a row is ``(venue, tier, period, maker_bp,
taker_bp, source)``, ``source`` naming the file the row was read from and, where
that file records one, the page it was transcribed from. A taker or maker rate
the source does not state is NaN, never filled in. ``trp fees`` prints the
table; :func:`tier` looks a row up for a run that states the fee it used.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

#: The venue name of Bybit's linear-perpetual schedule in ``backtest/costs.py``.
BYBIT_LINEAR = "bybit_linear_perpetual"


@dataclass(frozen=True)
class FeeRow:
    """One fee as a source states it, in basis points of notional per side."""

    venue: str
    tier: str
    period: str
    maker_bp: float
    taker_bp: float
    source: str
    group: str = "all"
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _repo() -> Path:
    """The repository holding ``configs/``: the working directory, or the package's checkout."""
    here = Path.cwd()
    if (here / "configs").is_dir():
        return here
    return Path(__file__).resolve().parents[2]


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _span(blocks: Any) -> str:
    """The first and last date a registration's blocks name, as the fee's period."""
    found: list[date] = []

    def walk(value: Any) -> None:
        if isinstance(value, date):
            found.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(blocks)
    return f"{min(found)}..{max(found)} (the registration's blocks)" if found else "as registered"


def _day(stamp: Any) -> str:
    """An interval bound as a day when it is midnight UTC, else as written."""
    text = str(stamp)
    return text.removesuffix("T00:00:00Z")


def _yaml(repo: Path, name: str) -> dict[str, Any] | None:
    path = repo / "configs" / name
    if not path.exists():
        return None
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def code_rows() -> list[FeeRow]:
    """The rows held in code: ``backtest/costs.py`` and ``backtest/maker.py``."""
    from trading_research.backtest import costs
    from trading_research.backtest.maker import MakerCosts

    period = f"2026 schedule, read {costs.BYBIT_FEE_RETRIEVED}"
    source = f"backtest/costs.py BYBIT_LINEAR_TIERS; {costs.BYBIT_FEE_SOURCE}"
    rows = [
        FeeRow(BYBIT_LINEAR, t.name, period, t.maker_bp, t.taker_bp, source, t.group, t.requires)
        for t in costs.BYBIT_LINEAR_TIERS
    ]
    rows.append(
        FeeRow(
            BYBIT_LINEAR,
            "mm_programme",
            period,
            costs.BYBIT_MM_PROGRAMME_MAKER_BP,
            math.nan,
            f"backtest/costs.py BYBIT_MM_PROGRAMME_MAKER_BP; {costs.BYBIT_FEE_SOURCE}",
            note="a bound on the maker fee; no taker rate is published for it",
        )
    )
    rows.append(
        FeeRow(
            "binance_usdm_futures",
            "standard",
            "as the module states it",
            math.nan,
            costs.BINANCE_UM_TAKER_BP,
            "backtest/costs.py BINANCE_UM_TAKER_BP (TakerCosts' default)",
            note="the maker rate is not a constant of the module",
        )
    )
    defaults = MakerCosts()
    rows.append(
        FeeRow(
            BYBIT_LINEAR,
            "maker_costs_default",
            "as the module states it",
            defaults.fee_bp_per_side,
            defaults.taker_fee_bp_per_side,
            "backtest/maker.py MakerCosts defaults",
        )
    )
    return rows


def config_rows(repo: Path | None = None) -> list[FeeRow]:
    """The rows the registrations' and the demo's YAML files hold."""
    root = repo or _repo()
    rows: list[FeeRow] = []
    for name in ("mm_prereg.yaml", "mm_prereg_round2.yaml"):
        config = _yaml(root, name)
        if not config:
            continue
        period = _span(config.get("blocks"))
        for tier, values in dict(config.get("fees") or {}).items():
            if isinstance(values, dict) and "maker_bp" in values:
                maker, taker = _number(values["maker_bp"]), _number(values.get("taker_bp"))
                rows.append(
                    FeeRow(BYBIT_LINEAR, str(tier), period, maker, taker, f"configs/{name} fees")
                )
    config = _yaml(root, "short_horizon_2025.yaml")
    if config:
        period = _span(config.get("blocks"))
        fees = dict(dict(config.get("common") or {}).get("fees") or {})
        for key in ("verdict_tier", "zero_fee"):
            values = fees.get(key)
            if isinstance(values, dict):
                rows.append(
                    FeeRow(
                        BYBIT_LINEAR,
                        str(values.get("name", key)),
                        period,
                        _number(values.get("maker_bp")),
                        _number(values.get("taker_bp")),
                        f"configs/short_horizon_2025.yaml common.fees.{key}",
                    )
                )
    config = _yaml(root, "spot_2026.yaml")
    if config:
        fees = dict(config.get("fees") or {})
        for venue, intervals in dict(fees.get("intervals") or {}).items():
            for interval in intervals or []:
                period = f"{_day(interval.get('start'))}..{_day(interval.get('end'))}"
                sources = ", ".join(map(str, interval.get("sources") or []))
                for tier, pair in dict(interval.get("tiers") or {}).items():
                    rows.append(
                        FeeRow(
                            str(venue),
                            str(tier),
                            period,
                            _number(pair[0]),
                            _number(pair[1]),
                            f"configs/spot_2026.yaml fees.intervals ({sources})",
                        )
                    )
        for venue, cell in dict(fees.get("undated_cells") or {}).items():
            sources = ", ".join(map(str, cell.get("sources") or []))
            for tier, pair in dict(cell.get("tiers") or {}).items():
                rows.append(
                    FeeRow(
                        str(venue),
                        str(tier),
                        f"undated, read {cell.get('read')}",
                        _number(pair[0]),
                        _number(pair[1]),
                        f"configs/spot_2026.yaml fees.undated_cells ({sources})",
                        note=str(cell.get("covers", "")),
                    )
                )
    config = _yaml(root, "demo.yaml")
    if config:
        fee = dict(config.get("costs") or {}).get("fee_bp_per_side")
        if fee is not None:
            rows.append(
                FeeRow(
                    "synthetic",
                    "demo_config",
                    "synthetic data only",
                    math.nan,
                    _number(fee),
                    "configs/demo.yaml costs.fee_bp_per_side",
                    note="a fee per side for every crossing",
                )
            )
    return rows


def rows(repo: Path | None = None) -> list[FeeRow]:
    """Every row: the code's tables first, then the configurations'."""
    return [*code_rows(), *config_rows(repo)]


def table(repo: Path | None = None, *, venue: str | None = None) -> pd.DataFrame:
    """The rows as a frame, optionally only one venue's."""
    found = [r.as_row() for r in rows(repo) if venue is None or r.venue == venue]
    return pd.DataFrame(found, columns=[*FeeRow.__dataclass_fields__])


def tier(
    name: str, venue: str = BYBIT_LINEAR, candidates: Iterable[FeeRow] | None = None
) -> FeeRow:
    """The row of ``venue`` named ``name``, from ``backtest/costs.py`` for Bybit's
    linear schedule; refused when no source states it."""
    pool = list(candidates) if candidates is not None else code_rows()
    for row in pool:
        if row.venue == venue and row.tier == name:
            return row
    known = sorted({r.tier for r in pool if r.venue == venue})
    raise KeyError(f"no fee tier {name!r} for {venue}; the sources hold {known}")


def ladder(group: str = "altcoin") -> list[FeeRow]:
    """Bybit's linear rows that apply to an instrument of ``group``, in the venue's order."""
    from trading_research.backtest.costs import tiers_for

    names = {t.name for t in tiers_for(group)}
    return [r for r in code_rows() if r.venue == BYBIT_LINEAR and r.tier in names]
