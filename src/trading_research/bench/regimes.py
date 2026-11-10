"""The regime suite: how a strategy behaves across known market breaks, and whether it is caught.

Version 1 of the benchmark scores a strategy on fixed spans. A strategy that
pays on one span can still break when the market's regime changes under it
(``posts/btc_ath_2024``: the reversion rule, fitted on February 2024, gave
back everything within two days of bitcoin's March high). This suite asks the
question directly, on ten dated, widely reported breaks in crypto markets:

1. **Does the strategy survive the break?** It is fitted on the three weeks
   before the week before the break, frozen, and traded through the week before
   and the two weeks after; its net per trade before and after is reported.
2. **Is the break caught?** The pipeline's kill switch
   (:mod:`trading_research.validation.kill_switch`, Page's CUSUM at its textbook
   values) runs on the strategy's own hourly results: when it stops, and what
   stopping keeps against running on.

It is outside version 1's frozen rule and does not change any version-1
verdict; a card reports it in a section of its own.

The data is Binance's public one-minute USD-M futures bars, not a book: the
order-book archives of these dates run to several hundred megabytes an
instrument-day, the one-minute bars to a few kilobytes. Each bar's close is
held for the minute on the benchmark's five-second grid, and the touch is set
around it at a fixed half-spread per instrument, the median measured on the
recorded tracks (:data:`HALF_SPREAD_BP`). A strategy that needs the book's
sizes is therefore not applicable here and says so; one that needs only
prices runs as it does on the recorded tracks, with every row inside a minute
priced at that minute's close.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.bench.tracks import Panel

#: The instruments every episode trades: liquid USDT perpetuals listed on
#: Binance's USD-M market since 2020, so all ten episodes have all twelve.
PANEL: tuple[str, ...] = (
    "ADAUSDT", "ATOMUSDT", "AVAXUSDT", "BCHUSDT", "BTCUSDT", "DOGEUSDT",
    "DOTUSDT", "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT",
)  # fmt: skip

#: Days fitted, scored before the break, and scored after it.
FIT_DAYS, BEFORE_DAYS, AFTER_DAYS = 14, 7, 14


@dataclass(frozen=True)
class Episode:
    """A dated break: the day the market's behaviour is widely reported to have changed."""

    name: str
    breaks: datetime
    what: str

    @property
    def fit_start(self) -> date:
        return (self.breaks - timedelta(days=FIT_DAYS + BEFORE_DAYS)).date()

    @property
    def scored_start(self) -> date:
        return (self.breaks - timedelta(days=BEFORE_DAYS)).date()

    @property
    def last_day(self) -> date:
        return (self.breaks + timedelta(days=AFTER_DAYS - 1)).date()


def _at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(tzinfo=UTC)


#: Ten breaks, dated to the day they began, before any data of them was read
#: by this suite. None falls in a block a study of this repository holds out
#: (``docs/preregistration``): the 2025 short-horizon study's BTCUSDT blocks
#: from July to December 2025, the market-making studies' pristine weeks of
#: April to June 2024, and the benchmark's forward fortnight of November 2026.
EPISODES: tuple[Episode, ...] = (
    Episode("china-ban-2021", _at("2021-05-19"), "China's mining and trading crackdown; a 30% intraday crash"),
    Episode("top-2021", _at("2021-11-10"), "bitcoin's 2021 all-time high and the start of the bear market"),
    Episode("luna-2022", _at("2022-05-09"), "TerraUSD loses its peg; LUNA collapses within the week"),
    Episode("celsius-2022", _at("2022-06-13"), "Celsius freezes withdrawals; the 3AC liquidation cascade"),
    Episode("ftx-2022", _at("2022-11-08"), "FTX halts withdrawals and collapses"),
    Episode("svb-2023", _at("2023-03-10"), "Silicon Valley Bank fails; USDC loses its peg over the weekend"),
    Episode("etf-2024", _at("2024-01-11"), "the US spot bitcoin ETFs start trading; sell the news"),
    Episode("ath-2024", _at("2024-03-14"), "bitcoin's March 2024 all-time high (posts/btc_ath_2024)"),
    Episode("yen-carry-2024", _at("2024-08-05"), "the yen carry-trade unwind; a global risk-off crash"),
    Episode("election-2024", _at("2024-11-06"), "the US election result; a momentum rally"),
)  # fmt: skip


def episode(name: str) -> Episode:
    found = [e for e in EPISODES if e.name == name]
    if not found:
        raise ValueError(f"unknown episode {name!r}; known: {', '.join(e.name for e in EPISODES)}")
    return found[0]


# --- the market: one-minute closes on the benchmark's five-second grid ------------

#: Median half-spread in bp of each panel instrument on the bybit-2024-02 track's
#: recorded top of book (measured with :func:`measured_half_spreads`).
HALF_SPREAD_BP: dict[str, float] = {
    "ADAUSDT": 0.84, "ATOMUSDT": 0.49, "AVAXUSDT": 0.63, "BCHUSDT": 0.93, "BTCUSDT": 0.01,
    "DOGEUSDT": 0.59, "DOTUSDT": 0.65, "ETHUSDT": 0.02, "LINKUSDT": 0.26, "LTCUSDT": 0.71,
    "SOLUSDT": 0.05, "XRPUSDT": 0.91,
}  # fmt: skip

GRID = "5s"
BARS_ROOT_NAME = "regimes"


def bar_grid(bars: pd.DataFrame, symbol: str, start: date, last: date) -> pd.DataFrame:
    """A five-second top of book from one-minute bars, causal by construction.

    Each row, labelled at the end of its five seconds, carries the close of the
    last bar that had closed by then; the touch is that close plus and minus
    the instrument's half-spread. The book's sizes are not known and are NaN.
    """
    closes = pd.Series(
        bars["close"].to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(bars["timestamp"]).tz_convert("UTC"),
    ).sort_index()
    begin = pd.Timestamp(start.isoformat(), tz="UTC") + timedelta(seconds=5)
    end = pd.Timestamp((last + timedelta(days=1)).isoformat(), tz="UTC")
    labels = pd.date_range(begin, end, freq=GRID)
    mid = closes.reindex(closes.index.union(labels)).ffill().reindex(labels)
    half = HALF_SPREAD_BP[symbol] * 1e-4
    frame = pd.DataFrame({"timestamp": labels, "bid_price_0": mid * (1 - half),
                          "bid_size_0": np.nan, "ask_price_0": mid * (1 + half),
                          "ask_size_0": np.nan})  # fmt: skip
    return frame.dropna(subset=["bid_price_0"]).reset_index(drop=True)


def episode_panel(ep: Episode, data_root: Path) -> Panel:
    """The episode's instruments on one grid, its scored span starting a week before the break."""
    from trading_research.data.ensure import load_bars

    grids = {}
    for symbol in PANEL:
        bars = load_bars(symbol, "1m", root=data_root / BARS_ROOT_NAME,
                         start=ep.fit_start, end=ep.last_day)  # fmt: skip
        grids[symbol] = bar_grid(bars, symbol, ep.fit_start, ep.last_day)
    start = pd.Timestamp(ep.scored_start.isoformat(), tz="UTC")
    return Panel(grids, f"binance futures-um 1m bars, {ep.name}", start=start)


# --- the score --------------------------------------------------------------------


@dataclass
class EpisodeScore:
    """One strategy on one episode."""

    episode: str
    status: str  # "holds", "breaks", "never paid", "no trades" or "n/a"
    reason: str = ""
    trades_before: int = 0
    trades_after: int = 0
    net_bp_before: float = math.nan
    net_bp_after: float = math.nan
    equity_left_running_pct: float = math.nan
    equity_kill_switch_pct: float = math.nan
    stop_hours_after_break: float | None = None

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()}


def ruined(equity: pd.Series) -> float:
    """The last equity, in %, of an account that stops for good once it has lost everything."""
    path = equity.to_numpy(dtype=np.float64)
    gone = np.flatnonzero(path <= -100.0)
    return -100.0 if len(gone) else float(path[-1])


def needs_sizes(strategy: object) -> bool:
    return any("size" in pattern for pattern in tuple(getattr(strategy, "inputs", ()) or ()))


def score_episode(build: Callable[[], object], ep: Episode, panel: Panel) -> EpisodeScore:
    """Fit before the scored span, freeze, trade through it, and run the kill switch on the result."""
    from trading_research.bench.score import _taker, missing_inputs
    from trading_research.validation.kill_switch import kill_switch

    if needs_sizes(build()):
        return EpisodeScore(ep.name, "n/a", "needs the book's sizes; this suite has bars only")
    rows, n = [], len(panel.grids)
    for symbol in panel.grids:
        strategy: Any = build()
        market = panel.market(symbol)
        missing = missing_inputs(strategy, market.frame)
        if missing:
            return EpisodeScore(ep.name, "n/a", f"no column for {', '.join(missing)}")
        start = int(pd.DatetimeIndex(market.frame["timestamp"]).searchsorted(panel.start))
        strategy.fit(market.frame.iloc[:start])
        decisions = np.asarray(strategy.decide(market.frame), dtype=np.int64).copy()
        decisions[:start] = 0
        hold = int(strategy.horizon)
        filled = _taker(decisions, market, hold, int(getattr(strategy, "cooldown", 0) or 0))
        stamps = pd.DatetimeIndex(pd.to_datetime(market.frame["timestamp"], utc=True))
        rows.append(pd.DataFrame({
            "entry": pd.to_datetime(filled["ts"], utc=True).to_numpy(),
            "exit": stamps[filled["exit_index"].to_numpy(dtype=np.int64)],
            "net_bp": filled["net_bp"].to_numpy(),
        }))  # fmt: skip
    trades = pd.concat(rows, ignore_index=True)
    if trades.empty:
        return EpisodeScore(ep.name, "no trades")
    breaks = pd.Timestamp(ep.breaks)
    before, after = trades[trades["exit"] < breaks], trades[trades["exit"] >= breaks]
    last_hour = pd.Timestamp(ep.last_day.isoformat(), tz="UTC") + timedelta(hours=24)
    hours = pd.date_range(panel.start + timedelta(hours=1), last_hour, freq="1h")

    # Hours are labelled by their end, in % of capital split equally over the
    # instruments. The switch decides on the trades that closed in each hour,
    # what was known at its end; a stop forbids new entries and lets the open
    # positions run out, so it is credited with the trades opened while live.
    def by_hour(stamp: str) -> pd.Series:
        pnl = trades.set_index(stamp)["net_bp"].resample("1h", label="right", closed="right").sum()
        return pnl.reindex(hours, fill_value=0.0) / n / 100.0

    hourly, opened = by_hour("exit"), by_hour("entry")
    switch = kill_switch(hourly, warm_up=72, booked=opened)
    stop = switch.first_stop
    out = EpisodeScore(
        ep.name, "", trades_before=len(before), trades_after=len(after),
        net_bp_before=float(before["net_bp"].mean()) if len(before) else math.nan,
        net_bp_after=float(after["net_bp"].mean()) if len(after) else math.nan,
        equity_left_running_pct=ruined(hourly.cumsum()),
        equity_kill_switch_pct=ruined(switch.equity),
        stop_hours_after_break=None if stop is None
        else round((hours[stop.position] - breaks).total_seconds() / 3600, 1),
    )  # fmt: skip
    paid_before = out.net_bp_before > 0
    out.status = "holds" if out.net_bp_after > 0 else ("breaks" if paid_before else "never paid")
    return out


def summary(scores: list[EpisodeScore]) -> dict[str, object]:
    """The suite on one line: what survived, what broke, what the kill switch kept."""
    scored = [s for s in scores if s.status not in ("n/a", "no trades")]
    saved = [s.equity_kill_switch_pct - s.equity_left_running_pct for s in scored]
    early = [
        s for s in scored if s.stop_hours_after_break is not None and s.stop_hours_after_break < 0
    ]
    return {
        "episodes": len(scores), "scored": len(scored),
        "holds": sum(s.status == "holds" for s in scored),
        "breaks": sum(s.status == "breaks" for s in scored),
        "never_paid": sum(s.status == "never paid" for s in scored),
        "kill_switch_saved_pct_mean": round(float(np.mean(saved)), 2) if saved else None,
        "kill_switch_helped": sum(v > 0 for v in saved),
        "stops_before_break": len(early),
    }  # fmt: skip


def measured_half_spreads(data_root: Path) -> dict[str, float]:
    """The median half-spread, in bp, of each panel instrument on the bybit-2024-02 track."""
    from trading_research.bench.tracks import TRACKS

    part = TRACKS["bybit-2024-02"].markets(data_root).parts[0]
    if not isinstance(part, Panel):
        raise TypeError("the recorded track is a panel of instruments")
    out = {}
    for symbol in PANEL:
        bid, ask = part.grids[symbol]["bid_price_0"], part.grids[symbol]["ask_price_0"]
        out[symbol] = round(float(((ask - bid) / ((ask + bid) / 2) * 1e4 / 2).median()), 2)
    return out


def run_suite(
    specs: list[Any],
    data_root: Path,
    episodes: tuple[Episode, ...] = EPISODES,
    on_score: Callable[[str, EpisodeScore], None] | None = None,
) -> dict[str, list[EpisodeScore]]:
    """Every strategy on every episode, one episode's market in memory at a time."""
    from trading_research.bench import builder

    out: dict[str, list[EpisodeScore]] = {s.name: [] for s in specs}
    for ep in episodes:
        try:
            panel = episode_panel(ep, data_root)
        except FileNotFoundError as exc:
            for spec in specs:
                out[spec.name].append(EpisodeScore(ep.name, "n/a", f"no data: {exc}"))
            continue
        for spec in specs:
            score = score_episode(builder(spec, {}), ep, panel)
            out[spec.name].append(score)
            if on_score:
                on_score(spec.name, score)
    return out


def page(results: dict[str, list[EpisodeScore]]) -> str:
    """The suite as a Markdown page: a summary row per strategy, then each episode."""
    lines = ["# Regime suite", "",
             "Fitted on the two weeks before the week before each break, frozen, and traded "
             "through the week before and the two weeks after, at Bybit's base taker fee, on "
             "Binance's one-minute bars; the kill switch is Page's CUSUM on the strategy's hourly "
             "results ([`bench/regimes.py`](../../src/trading_research/bench/regimes.py)).", "",
             "| strategy | scored | holds | breaks | never paid | kill switch helped "
             "| mean kept by the switch, % | stops before the break |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]  # fmt: skip
    for name, scores in results.items():
        s = summary(scores)
        kept = (
            "-"
            if s["kill_switch_saved_pct_mean"] is None
            else f"{s['kill_switch_saved_pct_mean']:+.2f}"
        )
        lines.append(f"| {name} | {s['scored']} of {s['episodes']} | {s['holds']} | {s['breaks']} | "
                     f"{s['never_paid']} | {s['kill_switch_helped']} | {kept} | {s['stops_before_break']} |")  # fmt: skip
    for name, scores in results.items():
        lines += ["", f"## {name}", "",
                  "| episode | status | trades before / after | net bp before | net bp after "
                  "| left running, % | with the kill switch, % | stop, hours after the break |",
                  "|---|---|--:|--:|--:|--:|--:|--:|"]  # fmt: skip
        for sc in scores:
            if sc.status in ("n/a", "no trades"):
                lines.append(f"| {sc.episode} | {sc.status} {sc.reason} | | | | | | |")
                continue
            stop = "-" if sc.stop_hours_after_break is None else f"{sc.stop_hours_after_break:+.0f}"
            lines.append(
                f"| {sc.episode} | {sc.status} | {sc.trades_before:,} / {sc.trades_after:,} | "
                f"{sc.net_bp_before:+.2f} | {sc.net_bp_after:+.2f} | {sc.equity_left_running_pct:+.1f} | "
                f"{sc.equity_kill_switch_pct:+.1f} | {stop} |"
            )
    return "\n".join(lines) + "\n"
