"""Experiment configuration.

Every assumption that changes a result is named here and comes from a YAML file:
the symbol, the date ranges, the label horizon, the fee, the assumed slippage,
the seed. Nothing that affects a number is allowed to live as a default buried
in a function, because a result is only interpretable next to the assumptions
that produced it.

The loaded config is written verbatim into each run manifest. That is the point
of the round trip: given a report, the exact configuration that produced it can
be recovered, and given a configuration, the report can be reproduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    """Reject unknown keys everywhere.

    A silently ignored typo in a config file is the worst kind of bug in
    research code: the run succeeds, the number changes, and nothing points at
    the cause.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Base):
    """Where the data comes from and which slice of it to use."""

    source: Literal["synthetic", "binance"] = "synthetic"
    symbol: str = "DEMOUSDT"
    plane: Literal["trades", "book"] = "book"
    path: Path | None = Field(
        default=None,
        description="Dataset directory. Required for real data; synthetic runs may generate on the fly.",
    )
    depth: int = Field(default=10, ge=1, le=50)


class SyntheticSection(_Base):
    """Overrides passed to the synthetic generator.

    Kept as a free-form mapping validated by the generator's own config class
    rather than duplicated field by field, so the two cannot drift apart.
    """

    n_steps: int = Field(default=20_000, ge=2)
    seed: int = 20260101
    overrides: dict[str, Any] = Field(default_factory=dict)


class SplitConfig(_Base):
    """Chronological split boundaries and the leakage guard around them.

    ``embargo`` is the number of observations dropped on each side of a split
    boundary. Its correct value is driven by the label horizon: a label at time
    *t* is computed from data up to *t + horizon*, so the last ``horizon``
    observations of a training block overlap the start of the next block.
    Leaving it at ``null`` makes the pipeline derive it from the label, which is
    the behaviour to prefer — an embargo chosen by hand is usually chosen too
    small.
    """

    train_fraction: float = Field(default=0.6, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    embargo: int | None = Field(default=None, ge=0)
    n_folds: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _fractions_leave_room_for_test(self) -> SplitConfig:
        used = self.train_fraction + self.validation_fraction
        if used >= 1.0:
            raise ValueError(
                f"train_fraction + validation_fraction must leave room for a test block, got {used}"
            )
        return self


class LabelConfig(_Base):
    """How the target is defined."""

    kind: Literal["directional"] = "directional"
    horizon: int = Field(default=20, ge=1, description="Observations ahead the label looks.")
    threshold_bp: float = Field(
        default=1.0,
        ge=0.0,
        description="Move in basis points above which a direction is labelled rather than HOLD.",
    )


class CostConfig(_Base):
    """Trading costs, all in basis points of notional.

    Stated per side and applied consistently in labelling, in the decision rule
    and in the backtest. A cost model that differs between the three is the most
    common way an otherwise careful backtest overstates its result.
    """

    fee_bp_per_side: float = Field(default=2.0, ge=0.0)
    slippage_bp: float = Field(default=0.5, ge=0.0)
    cross_spread: bool = Field(
        default=True,
        description="Whether an entry pays the quoted spread by crossing it.",
    )


class ModelConfig(_Base):
    name: Literal["naive", "logistic", "xgboost"] = "logistic"
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(_Base):
    """A complete, self-describing experiment."""

    name: str = "demo"
    seed: int = 42
    output: Path = Path("runs")
    data: DataConfig = Field(default_factory=DataConfig)
    synthetic: SyntheticSection = Field(default_factory=SyntheticSection)
    features: list[str] = Field(default_factory=list)
    label: LabelConfig = Field(default_factory=LabelConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @model_validator(mode="after")
    def _real_data_needs_a_path(self) -> RunConfig:
        if self.data.source != "synthetic" and self.data.path is None:
            raise ValueError(f"data.path is required when data.source is {self.data.source!r}")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the run manifest, with paths as strings."""
        return self.model_dump(mode="json")


def load_config(path: Path | str) -> RunConfig:
    """Load and validate a YAML experiment configuration."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"config not found: {file}")
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: expected a mapping at the top level, got {type(raw).__name__}")
    return RunConfig.model_validate(raw)


def dump_config(config: RunConfig, path: Path | str) -> Path:
    """Write a configuration back out, so a run can be replayed exactly."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return file
