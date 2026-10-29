"""The studies registered on this branch, and any a ``trp study new`` has added.

Each is an :class:`~trading_research.studies.base.EntryPointStudy`: its stage
names come from the study's own state module (``spot/state.py``,
``pipeline/short_horizon/state.py``), or, for a study created from the
template, from the ``STAGES`` and ``READS`` its entry point declares, read
from its source without importing it.
"""

from __future__ import annotations

from pathlib import Path

from trading_research.studies.base import EntryPointStudy, declared


def repository(start: Path | None = None) -> Path:
    """The repository the studies live in: ``start`` when given; else the working
    directory when it holds ``experiments/``, else the checkout this package was
    loaded from."""
    if start is not None:
        return start.resolve()
    if (Path.cwd() / "experiments").is_dir():
        return Path.cwd().resolve()
    return Path(__file__).resolve().parents[3]


def builtin(repo: Path) -> dict[str, EntryPointStudy]:
    """The two studies whose entry points this branch holds."""
    from trading_research.pipeline.short_horizon import state as short_horizon
    from trading_research.spot import state as spot

    return {
        "short_horizon_2025": EntryPointStudy(
            "short_horizon_2025",
            repo,
            short_horizon.STAGES,
            short_horizon.READS,
            options=("--workers", "--run", "--keep"),
            description="BTCUSDT perpetual, 2025: taker and passive cells searched on D, "
            "the finalists read on H, F and P through the ledger",
        ),
        "spot_2026": EntryPointStudy(
            "spot_2026",
            repo,
            spot.STAGES,
            spot.READS,
            options=("--workers", "--run", "--frozen-2025", "--keep"),
            description="S25 on spot, Binance BTC/FDUSD and Bybit BTCUSDT, 2026: nothing "
            "searched, H read once, P for a candidate",
        ),
    }


def from_template(repo: Path, known: set[str]) -> dict[str, EntryPointStudy]:
    """Entry points with a registration document that declare their stages and reads."""
    found: dict[str, EntryPointStudy] = {}
    for path in sorted((repo / "experiments").glob("*.py")):
        name = path.stem
        if name in known or name.startswith("_"):
            continue
        if not (repo / "docs" / "preregistration" / f"{name}.md").exists():
            continue
        names = declared(path)
        if "STAGES" in names and "READS" in names:
            found[name] = EntryPointStudy(
                name, repo, names["STAGES"], names["READS"], options=("--workers",),
                description="registered from the template; its stages refuse until its code exists",
            )  # fmt: skip
    return found


def studies(repo: Path | None = None) -> dict[str, EntryPointStudy]:
    """Every registered study in ``repo``, the built-in ones first."""
    root = repository(repo)
    found = builtin(root)
    found.update(from_template(root, set(found)))
    return found


def get(name: str, repo: Path | None = None) -> EntryPointStudy:
    """One study by name, or a ``KeyError`` naming the ones that exist."""
    found = studies(repo)
    if name not in found:
        raise KeyError(f"no registered study {name!r}; registered: {', '.join(found)}")
    return found[name]
