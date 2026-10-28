"""The catalogue: everything the repository can run, read from where it is defined.

``trading-research catalog`` writes ``docs/catalog.md``. Nothing in it is typed
twice: plugins come from the plugin registry, single features from the feature
registry, commands and their help from the command tree, fees from the fee
tables, studies from the study registry and test cases from ``tests/``. The few
tables that have no registry to read (data sources, feature filters, the
evaluation measures, the regime detectors) name the function or command that
implements each row, and a test imports every one of them, so a row cannot
outlive its code.

A part added as a plugin (``strategies new NAME --kind ...``) appears here the
next time the page is written, and ``tests/test_catalog.py`` fails until it is.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tables with no registry to read: each row names the code that implements it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One public data source: what it holds and the command that fetches it."""

    venue: str
    instrument_type: str
    datasets: str
    book: str
    url: str
    fetch: str
    implemented_by: str


SOURCES: tuple[Source, ...] = (
    Source(
        "Binance",
        "spot (e.g. BTCUSDT, BTCFDUSD)",
        "aggTrades, klines (bars, any interval)",
        "none",
        "https://data.binance.vision",
        "trp data download --market spot --kind aggTrades --symbol BTCFDUSD --start 2025-06-01 --end 2025-06-07",
        "trading_research.data.binance:download_range",
    ),
    Source(
        "Binance",
        "USD-M perpetual (linear, e.g. BTCUSDT, XRPUSDT)",
        "aggTrades, bookTicker, bookDepth, metrics, fundingRate, klines",
        "the touch (bookTicker)",
        "https://data.binance.vision",
        "trp data download --market futures-um --kind bookTicker --grid 100ms --symbol BTCUSDT --start 2024-02-01 --end 2024-02-07",
        "trading_research.data.binance:download_range",
    ),
    Source(
        "Binance",
        "COIN-M futures (inverse, e.g. BTCUSD_PERP)",
        "aggTrades, bookTicker, klines",
        "the touch (bookTicker)",
        "https://data.binance.vision",
        "trp data download --market futures-cm --kind klines --interval 1h --symbol BTCUSD_PERP --start 2024-01-01 --end 2024-01-31",
        "trading_research.data.binance:download_range",
    ),
    Source(
        "Bybit",
        "linear perpetual (e.g. BTCUSDT, ETHUSDT, any listed contract)",
        "order book replayed from snapshots and deltas",
        "500 levels to 2025-08-20, 200 levels after",
        "https://quote-saver.bycsi.com/orderbook/linear/",
        "trp data download-book --symbol ETHUSDT --depth 10 --grid-ms 100 --start 2024-03-01 --end 2024-03-03",
        "trading_research.data.bybit:download_range",
    ),
    Source(
        "Bybit",
        "linear perpetual prints",
        "every trade with its aggressor",
        "none",
        "https://public.bybit.com/trading/",
        "python: trading_research.data.bybit_trades.download_range",
        "trading_research.data.bybit_trades:download_range",
    ),
    Source(
        "Bybit",
        "linear perpetual funding",
        "every funding settlement",
        "none",
        "https://api.bybit.com/v5/market/funding/history",
        "python: trading_research.data.bybit_funding.download_range",
        "trading_research.data.bybit_funding:download_range",
    ),
    Source(
        "Bybit",
        "spot (e.g. BTCUSDT)",
        "order book replayed from snapshots and deltas",
        "the archive's levels",
        "https://quote-saver.bycsi.com/orderbook/spot/",
        "python: trading_research.spot.bybit_book.fetch_day",
        "trading_research.spot.bybit_book:fetch_day",
    ),
    Source(
        "the venues' websockets",
        "Binance and Bybit, spot and perpetual, live",
        "the touch and prints, recorded going forward",
        "the touch, or the depth the stream carries",
        "public streams, no key",
        "trp paper record --venue bybit --market linear --symbol BTCUSDT --minutes 60",
        "trading_research.live.stream",
    ),
)

#: Venues and instrument types the repository does not fetch yet; listed so the
#: catalogue does not suggest otherwise.
NOT_YET: tuple[tuple[str, str], ...] = (
    (
        "Bybit inverse contracts (e.g. XRPUSD)",
        "no fetcher; the linear archive's layout is the model for one",
    ),
    ("Bitfinex spot", "no fetcher and no fee table"),
    ("Options, equities, FX", "out of scope; every contract above is crypto"),
)


@dataclass(frozen=True)
class Implemented:
    """A named method and the function or class that implements it."""

    name: str
    what: str
    implemented_by: str


FEATURE_FILTERS: tuple[Implemented, ...] = (
    Implemented(
        "variance filter",
        "drops columns that barely move, fitted on training rows only",
        "trading_research.features.selection:FeatureSelector",
    ),
    Implemented(
        "correlation filter",
        "keeps one of each group of near-duplicate columns",
        "trading_research.features.selection:FeatureSelector",
    ),
    Implemented(
        "mutual information ranking",
        "keeps the columns that say most about the label",
        "trading_research.features.selection:FeatureSelector",
    ),
    Implemented(
        "rolling normalisation",
        "z-scores on a trailing window, no statistic from the future",
        "trading_research.features.normalise:RollingNormaliser",
    ),
    Implemented(
        "pooled normalisation",
        "one scale across instruments, floors fitted on training days",
        "trading_research.features.normalise:normalise_pooled",
    ),
    Implemented(
        "look-ahead check",
        "truncates the data and requires that no earlier value moves",
        "trading_research.validation.leakage",
    ),
    Implemented(
        "information audit",
        "the linear ceiling and mutual information of a source against the move",
        "trading_research.evaluation.information:audit",
    ),
)

EVALUATION: tuple[Implemented, ...] = (
    Implemented(
        "net per trade, gross, cost, hit rate, drawdown",
        "the backtest's summary of a trade list",
        "trading_research.backtest.metrics:summarise",
    ),
    Implemented(
        "day-clustered t",
        "each day one observation, so one day's many trades are not many proofs",
        "trading_research.evaluation.significance:assess",
    ),
    Implemented(
        "Newey-West t",
        "a t robust to autocorrelation in a daily series",
        "trading_research.evaluation.significance:newey_west_t",
    ),
    Implemented(
        "effective count",
        "instruments that move together counted as fewer",
        "trading_research.evaluation.significance:effective_count",
    ),
    Implemented(
        "minimum detectable effect and power",
        "what a block of this length could have found",
        "trading_research.evaluation.significance:minimum_detectable_effect",
    ),
    Implemented(
        "deflated Sharpe ratio",
        "the Sharpe a search of this many trials would reach by chance",
        "trading_research.evaluation.overfitting:deflated_sharpe_ratio",
    ),
    Implemented(
        "probability of backtest overfitting",
        "combinatorially symmetric cross-validation",
        "trading_research.evaluation.overfitting:cscv_pbo",
    ),
    Implemented(
        "Holm's procedure",
        "a family of claims tested together",
        "trading_research.evaluation.family:family_holm",
    ),
    Implemented(
        "placebos",
        "the same decisions moved in time; the result must beat their 95th percentile",
        "trading_research.walkthrough.verdict:placebo_values",
    ),
    Implemented(
        "kill conditions",
        "written before the read, applied in order",
        "trading_research.walkthrough.verdict:KILLS",
    ),
    Implemented(
        "fee ladder",
        "the same trades re-costed at every published tier",
        "trading_research.walkthrough.execution:at_fee",
    ),
    Implemented(
        "headroom",
        "the cost, or the share of trades a forecast must get right, at which the net is zero",
        "trading_research.walkthrough.verdict:headroom",
    ),
    Implemented(
        "bias-variance decomposition",
        "how much of a model's error is the model and how much the data",
        "trading_research.evaluation.bias_variance:decompose",
    ),
    Implemented(
        "scorecard",
        "every measure of one result on one page",
        "trading_research.evaluation.scorecard:build_scorecard",
    ),
)

DETECTORS: tuple[Implemented, ...] = (
    Implemented(
        "CUSUM",
        "a persistent drift in a standardised statistic, in either direction",
        "trading_research.validation.changepoint:detect",
    ),
    Implemented(
        "structural-break monitors",
        "the statistics of a break-detection library on a whitened stream, calibrated on a null",
        "trading_research.validation.structural_breaks:detect_breaks",
    ),
)

#: Command groups whose subcommands the catalogue lists with their help.
COMMAND_SECTIONS: tuple[tuple[str, str], ...] = (
    ("data", "Data"),
    ("synthetic", "Synthetic markets: a planted answer, and its null"),
    ("verify", "Checks of the code against planted answers"),
    ("backtest", "Backtests"),
    ("search", "The automatic search"),
    ("paper", "Paper trading"),
)

#: How a model's module places it in a family.
FAMILIES: dict[str, str] = {
    "trading_research.models.base": "baseline",
    "trading_research.models.classical": "rule (fitted to nothing)",
    "trading_research.models.linear": "linear",
    "trading_research.models.regression": "regression",
    "trading_research.models.gbm": "gradient-boosted trees",
    "trading_research.models.forests": "tree ensemble",
    "trading_research.models.tcn": "neural network",
}

ADD: dict[str, str] = {
    "strategy": "trp strategies new NAME",
    "model": "trp strategies new NAME --kind model",
    "features": "trp strategies new NAME --kind features",
    "label": "trp strategies new NAME --kind label",
    "execution": "trp strategies new NAME --kind execution",
}


def resolve(path: str) -> Any:
    """The object ``module:attribute`` (or the module) names; raises if it is gone."""
    module, _, attribute = path.partition(":")
    found = importlib.import_module(module)
    return getattr(found, attribute) if attribute else found


def implemented_paths() -> list[str]:
    """Every ``implemented_by`` the hand-written tables name."""
    rows: Iterable[Any] = (*SOURCES, *FEATURE_FILTERS, *EVALUATION, *DETECTORS)
    return [row.implemented_by for row in rows]


# ---------------------------------------------------------------------------
# What is read from registries.
# ---------------------------------------------------------------------------


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _first_sentence(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split("\n\n")[0].split())


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def plugin_specs(kind: str) -> list[Any]:
    from trading_research.plugins import registry

    registry.discover()
    return registry.specs(kind)


def model_family(spec: Any) -> str:
    """A model plugin's family, from its stage table entry or its module."""
    from trading_research.pipeline import stages

    name = spec.name
    if name in stages.ENSEMBLES:
        return "ensemble (average)"
    if name in stages.COMPOSITES:
        return "composite (agreement or gate)"
    if name in stages.META:
        return "meta-labelled"
    if name in stages.UNCERTAIN:
        return "confidence bound"
    if name in stages.MODELS:
        return FAMILIES.get(stages.MODELS[name].split(":")[0], "other")
    return f"plugin ({spec.origin})"


def commands(group: str) -> list[tuple[str, str]]:
    """A command group's subcommands and the first line of each one's help."""
    import typer.main

    from trading_research.cli import app

    root = typer.main.get_command(app)
    found = root.commands.get(group)  # type: ignore[attr-defined]
    if found is None:
        return []
    sub = getattr(found, "commands", None)
    if not sub:
        return [(group, _first_sentence(found.help))]
    return [(f"{group} {name}", _first_sentence(cmd.help)) for name, cmd in sub.items()]


def single_features() -> list[dict[str, Any]]:
    import pkgutil

    import trading_research.features as package
    from trading_research.features.registry import REGISTRY

    for module in pkgutil.iter_modules(package.__path__):
        try:
            importlib.import_module(f"trading_research.features.{module.name}")
        except ImportError:  # an optional dependency absent: its features are not listed
            continue
    return REGISTRY.describe().to_dict("records")


def test_files(repo: Path) -> list[tuple[str, int]]:
    """Each test file and how many test functions it defines."""
    pattern = re.compile(r"^(?:async )?def (test_\w+)", re.MULTILINE)
    out = []
    for path in sorted((repo / "tests").glob("test_*.py")):
        out.append((path.stem, len(pattern.findall(path.read_text(encoding="utf-8")))))
    return out


def _area(stem: str) -> str:
    return stem.removeprefix("test_").split("_")[0]


# ---------------------------------------------------------------------------
# The page.
# ---------------------------------------------------------------------------


def sections(repo: Path) -> list[tuple[str, list[str]]]:
    """The catalogue as (title, lines) pairs, in reading order."""
    from trading_research import fees
    from trading_research.studies.registry import studies

    out: list[tuple[str, list[str]]] = []

    lines = _table(
        ["venue", "instrument type", "datasets", "book", "where", "fetch"],
        [(s.venue, s.instrument_type, s.datasets, s.book, s.url, f"`{s.fetch}`") for s in SOURCES],
    )
    lines += ["", "Not fetched yet:", ""]
    lines += [f"- **{what}**: {why}." for what, why in NOT_YET]
    lines += ["", "Every command of the data step:", ""]
    lines += _table(["command", "what it does"], [(f"`trp {c}`", h) for c, h in commands("data")])
    out.append(("Instruments and data: where to get them", lines))

    lines = ["Planted answers to test a strategy on before any real day: each has a null with "
             "the effect removed.", ""]  # fmt: skip
    lines += _table(
        ["command", "what it writes"], [(f"`trp {c}`", h) for c, h in commands("synthetic")]
    )
    out.append(("Synthetic and simulated markets", lines))

    specs = plugin_specs("strategy")
    lines = _table(
        ["strategy", "origin", "what it does"],
        [(f"`{s.name}`", s.origin, s.summary) for s in specs],
    )
    pairs = commands("pairs")
    if pairs:
        lines += ["", "Pairs trading:", ""]
        lines += _table(["command", "what it does"], [(f"`trp {c}`", h) for c, h in pairs])
    lines += ["", f"Add one: `{ADD['strategy']}`, then `trp strategies check NAME`."]
    out.append(("Strategies", lines))

    specs = sorted(plugin_specs("model"), key=lambda s: (model_family(s), s.name))
    lines = _table(
        ["model", "family", "what it is"],
        [(f"`{s.name}`", model_family(s), s.summary) for s in specs],
    )
    lines += [
        "",
        f"Add one: `{ADD['model']}`; the search then takes it with `trp search register --models logistic,NAME`.",
    ]
    out.append(("Models", lines))

    specs = plugin_specs("features")
    lines = _table(["feature set", "what it holds"], [(f"`{s.name}`", s.summary) for s in specs])
    lines += ["", "Single registered features, each with the look-back it declares and the "
              "look-ahead check enforces:", ""]  # fmt: skip
    lines += _table(["feature", "plane", "look-back rows", "what it measures"],
                    [(f"`{f['name']}`", f["plane"], f["lookback"], f["description"]) for f in single_features()])  # fmt: skip
    lines += ["", f"Add a set: `{ADD['features']}`."]
    out.append(("Features", lines))

    lines = _table(["filter", "what it does", "implemented by"],
                   [(f.name, f.what, f"`{f.implemented_by}`") for f in FEATURE_FILTERS])  # fmt: skip
    out.append(("Feature filters and checks", lines))

    specs = plugin_specs("label")
    lines = _table(
        ["label", "what it asks the model to predict"], [(f"`{s.name}`", s.summary) for s in specs]
    )
    lines += ["", f"Add one: `{ADD['label']}`."]
    out.append(("Labels", lines))

    specs = plugin_specs("execution")
    lines = _table(
        ["execution", "how a decision becomes fills"], [(f"`{s.name}`", s.summary) for s in specs]
    )
    lines += ["", "Backtest commands:", ""]
    lines += _table(
        ["command", "what it does"], [(f"`trp {c}`", h) for c, h in commands("backtest")]
    )
    lines += ["", f"Add an execution: `{ADD['execution']}`."]
    out.append(("Executions and backtests", lines))

    lines = _table(["measure", "what it answers", "implemented by"],
                   [(e.name, e.what, f"`{e.implemented_by}`") for e in EVALUATION])  # fmt: skip
    out.append(("Evaluation: how a result is judged", lines))

    lines = _table(["detector", "what it finds", "implemented by"],
                   [(d.name, d.what, f"`{d.implemented_by}`") for d in DETECTORS])  # fmt: skip
    lines += ["", "Run: `trp regimes breaks`."]
    out.append(("Regime and structural-break detectors", lines))

    lines = _table(["command", "what it does"], [(f"`trp {c}`", h) for c, h in commands("search")])
    out.append(("The automatic search", lines))

    lines = _table(
        ["command", "what it checks"], [(f"`trp {c}`", h) for c, h in commands("verify")]
    )
    out.append(("Checks against planted answers", lines))

    table = fees.table()
    venues = (
        table.groupby("venue")["tier"].apply(lambda t: ", ".join(dict.fromkeys(t))).reset_index()
    )
    lines = _table(["venue", "tiers"], venues.itertuples(index=False))
    lines += ["", "Every row with its period and source: `trp fees`."]
    out.append(("Fee tables", lines))

    found = studies(repo)
    lines = _table(["study", "what it registered"],
                   [(f"`{name}`", _first_sentence(s.description)) for name, s in found.items()])  # fmt: skip
    lines += ["", "Start one: `trp study new NAME`."]
    out.append(("Registered studies", lines))

    lines = _table(["command", "what it does"], [(f"`trp {c}`", h) for c, h in commands("paper")])
    out.append(("Paper trading", lines))

    files = test_files(repo)
    areas: dict[str, tuple[int, int]] = {}
    for stem, count in files:
        n_files, n_tests = areas.get(_area(stem), (0, 0))
        areas[_area(stem)] = (n_files + 1, n_tests + count)
    lines = [f"{sum(c for _, c in files)} test functions in {len(files)} files, by area "
             "(the first word of the file name):", ""]  # fmt: skip
    lines += _table(["area", "files", "tests"], [(a, f, t) for a, (f, t) in sorted(areas.items())])
    out.append(("Test cases", lines))
    return out


def page(repo: Path) -> str:
    """``docs/catalog.md`` as text."""
    body = sections(repo)
    lines = [
        "# Catalogue",
        "",
        "Everything the repository can run, written by `trp catalog --write` from where each "
        "part is defined. Do not edit by hand.",
        "",
    ]
    lines += [f"- [{title}](#{_anchor(title)})" for title, _ in body]
    for title, content in body:
        lines += ["", f"## {title}", "", *content]
    return "\n".join(lines) + "\n"


def _anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-")


def summary(repo: Path) -> list[tuple[str, int, str]]:
    """The README's short table: each catalogue section, how many entries, how to add one."""
    from trading_research.studies.registry import studies

    return [
        ("Data sources", len(SOURCES), "a fetcher under `data/`"),
        ("Synthetic markets", len(commands("synthetic")), "a generator under `synthetic/`"),
        ("Strategies", len(plugin_specs("strategy")), f"`{ADD['strategy']}`"),
        ("Models", len(plugin_specs("model")), f"`{ADD['model']}`"),
        ("Feature sets", len(plugin_specs("features")), f"`{ADD['features']}`"),
        ("Feature filters and checks", len(FEATURE_FILTERS), "a step in `features/selection.py`"),
        ("Labels", len(plugin_specs("label")), f"`{ADD['label']}`"),
        ("Executions", len(plugin_specs("execution")), f"`{ADD['execution']}`"),
        ("Evaluation measures", len(EVALUATION), "a function under `evaluation/`"),
        ("Regime detectors", len(DETECTORS), "a detector under `validation/`"),
        ("Registered studies", len(studies(repo)), "`trp study new NAME`"),
    ]
