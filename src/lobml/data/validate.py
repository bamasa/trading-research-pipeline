"""Dataset quality checks.

:mod:`lobml.data.schema` answers "is this frame the right shape?". This module
answers "is this data usable?", which is a different and more interesting
question. Market data arrives with crossed books, repeated sequence numbers,
clock jumps and silent gaps, and every one of those breaks a downstream result
in a way that is hard to notice from the result alone.

The checks therefore return a report rather than raising on the first problem.
A dataset with a handful of crossed snapshots out of a million is usually worth
keeping with the bad rows dropped; a dataset where the timestamps are not
monotone is not worth modelling at all. Only the caller can make that call, so
the checks grade findings by severity and leave the decision explicit:

    report = validate_book(df)
    report.raise_if_failed()   # opt in to strictness where it matters

Gaps are reported rather than repaired. Interpolating across a gap invents
observations, and invented observations are indistinguishable from real ones by
the time they reach a model.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import pandas as pd

from lobml.data.schema import (
    TRADE_SCHEMA,
    Schema,
    SchemaError,
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
    book_schema,
    infer_depth,
)


class Severity(StrEnum):
    """How much a finding matters.

    ``ERROR``
        The data violates an invariant the pipeline relies on. Modelling on it
        produces results that cannot be interpreted.
    ``WARNING``
        Real and worth knowing about, but survivable: usually a small number of
        bad rows that can be dropped, or a gap that can be excluded.
    ``INFO``
        Descriptive. Recorded so that a report is self-contained.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One observation about a dataset."""

    check: str
    severity: Severity
    message: str
    count: int = 0
    examples: tuple[object, ...] = ()

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}] {self.check}: {self.message}"
        if self.examples:
            shown = ", ".join(str(e) for e in self.examples)
            return f"{prefix} (e.g. {shown})"
        return prefix


@dataclass
class ValidationReport:
    """The outcome of validating one dataset."""

    plane: str
    rows: int
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        check: str,
        severity: Severity,
        message: str,
        *,
        count: int = 0,
        examples: tuple[object, ...] = (),
    ) -> None:
        self.findings.append(
            Finding(check=check, severity=severity, message=message, count=count, examples=examples)
        )

    def of(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity is severity]

    @property
    def errors(self) -> list[Finding]:
        return self.of(Severity.ERROR)

    @property
    def warnings(self) -> list[Finding]:
        return self.of(Severity.WARNING)

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found. Warnings do not fail a dataset."""
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        detail = "\n".join(f"  {f}" for f in self.errors)
        raise DataQualityError(f"{self.plane}: {len(self.errors)} blocking problem(s):\n{detail}")

    def summary(self) -> str:
        head = (
            f"{self.plane}: {self.rows:,} rows, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        if not self.findings:
            return head
        body = "\n".join(f"  {f}" for f in self.findings)
        return f"{head}\n{body}"

    def to_dict(self) -> dict[str, object]:
        return {
            "plane": self.plane,
            "rows": self.rows,
            "ok": self.ok,
            "findings": [
                {
                    "check": f.check,
                    "severity": f.severity.value,
                    "message": f.message,
                    "count": f.count,
                }
                for f in self.findings
            ],
        }

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)


class DataQualityError(ValueError):
    """A dataset failed a blocking quality check."""


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------


def _check_schema(df: pd.DataFrame, schema: Schema, report: ValidationReport) -> bool:
    try:
        schema.validate(df)
    except SchemaError as exc:
        report.add("schema", Severity.ERROR, str(exc))
        return False
    return True


def _check_empty(df: pd.DataFrame, report: ValidationReport) -> bool:
    if df.empty:
        report.add("non_empty", Severity.ERROR, "dataset is empty")
        return False
    return True


def _check_timestamps(df: pd.DataFrame, report: ValidationReport) -> None:
    """Timestamps must be UTC and non-decreasing within each symbol.

    Non-decreasing rather than strictly increasing: several trades can share a
    millisecond, and forcing uniqueness here would reject valid data. What
    matters for the no-look-ahead guarantee is that the ordering never goes
    backwards, because every rolling window in the project trusts row order.
    """
    ts = df["timestamp"]
    if ts.isna().any():
        report.add(
            "timestamp_null", Severity.ERROR, "timestamp contains nulls", count=int(ts.isna().sum())
        )
        return

    if ts.dt.tz is None:
        report.add("timestamp_tz", Severity.ERROR, "timestamp is timezone-naive; UTC is required")
        return

    for symbol, group in df.groupby("symbol", sort=False):
        order = group["timestamp"]
        backwards = order.diff() < pd.Timedelta(0)
        n_back = int(backwards.sum())
        if n_back:
            first = order.index[backwards][:3]
            report.add(
                "timestamp_monotonic",
                Severity.ERROR,
                f"{symbol}: timestamp decreases at {n_back} row(s); row order cannot be trusted",
                count=n_back,
                examples=tuple(first),
            )

    span = ts.max() - ts.min()
    report.add(
        "coverage",
        Severity.INFO,
        f"spans {span} from {ts.min().isoformat()} to {ts.max().isoformat()}",
    )


def _check_positive(
    df: pd.DataFrame,
    columns: list[str],
    report: ValidationReport,
    *,
    check: str,
    allow_zero: bool = False,
) -> None:
    """Prices must be positive; sizes may be zero at empty levels."""
    for col in columns:
        values = df[col]
        bad = values.isna() | (values < 0 if allow_zero else values <= 0)
        n_bad = int(bad.sum())
        if n_bad:
            report.add(
                check,
                Severity.ERROR,
                f"{col}: {n_bad} non-positive or null value(s)",
                count=n_bad,
                examples=tuple(df.index[bad][:3]),
            )


# ---------------------------------------------------------------------------
# Trade plane
# ---------------------------------------------------------------------------


def validate_trades(df: pd.DataFrame) -> ValidationReport:
    """Check a trade frame against the trade contract and for usability."""
    report = ValidationReport(plane="trades", rows=len(df))
    if not _check_empty(df, report) or not _check_schema(df, TRADE_SCHEMA, report):
        return report

    _check_timestamps(df, report)
    _check_positive(df, ["price", "quantity"], report, check="positive_values")

    for symbol, group in df.groupby("symbol", sort=False):
        ids = group["trade_id"]

        duplicates = int(ids.duplicated().sum())
        if duplicates:
            report.add(
                "trade_id_unique",
                Severity.ERROR,
                f"{symbol}: {duplicates} duplicate trade_id(s); rows are double counted",
                count=duplicates,
                examples=tuple(ids[ids.duplicated()].head(3)),
            )

        step = ids.diff().dropna()
        # A step of exactly 1 means nothing was missed. Aggregated trades roll
        # up several matches into one row, so larger steps are normal there and
        # are reported rather than treated as loss.
        gaps = step[step > 1]
        if not gaps.empty:
            missing = int(gaps.sum() - len(gaps))
            report.add(
                "trade_id_gaps",
                Severity.INFO,
                f"{symbol}: trade_id advances by more than 1 at {len(gaps)} point(s), "
                f"{missing} id(s) not present; expected for aggregated trades, "
                f"a sign of dropped data otherwise",
                count=len(gaps),
            )

        if (step < 0).any():
            n_back = int((step < 0).sum())
            report.add(
                "trade_id_monotonic",
                Severity.ERROR,
                f"{symbol}: trade_id decreases at {n_back} row(s); the frame is not in exchange order",
                count=n_back,
            )

    _report_time_gaps(df, report)
    return report


# ---------------------------------------------------------------------------
# Book plane
# ---------------------------------------------------------------------------


def validate_book(
    df: pd.DataFrame,
    *,
    depth: int | None = None,
    sampled: bool = False,
) -> ValidationReport:
    """Check an order-book frame against the book contract and for usability.

    Set ``sampled=True`` when the frame is a subsample of the update stream —
    a resampled book, or snapshots taken on a grid. It only affects the
    sequence-continuity check, and it matters: in a sample, skipped
    ``sequence_id`` values are the intended result, not evidence of a lost
    update. Reported as gaps they would produce one warning per row, and a real
    gap would be invisible in the noise.
    """
    report = ValidationReport(plane="book", rows=len(df))
    if not _check_empty(df, report):
        return report

    resolved = depth if depth is not None else infer_depth(df)
    if resolved < 1:
        report.add("schema", Severity.ERROR, "no complete price/size level found")
        return report

    if not _check_schema(df, book_schema(resolved), report):
        return report

    report.add("depth", Severity.INFO, f"{resolved} level(s) per side")

    _check_timestamps(df, report)
    _check_positive(
        df,
        [bid_price_col(i) for i in range(resolved)] + [ask_price_col(i) for i in range(resolved)],
        report,
        check="positive_prices",
    )
    _check_positive(
        df,
        [bid_size_col(i) for i in range(resolved)] + [ask_size_col(i) for i in range(resolved)],
        report,
        check="non_negative_sizes",
        allow_zero=True,
    )

    _check_not_crossed(df, report)
    _check_level_ordering(df, resolved, report)
    _check_sequence_ids(df, report, sampled=sampled)
    _report_time_gaps(df, report)
    return report


def _check_not_crossed(df: pd.DataFrame, report: ValidationReport) -> None:
    """The best bid must sit strictly below the best ask.

    A crossed book is either a genuine exchange artefact lasting microseconds or,
    far more often, a reconstruction bug. Either way every spread-based feature
    and every cost calculation goes negative on those rows, so they cannot be
    modelled through.
    """
    bid = df[bid_price_col(0)]
    ask = df[ask_price_col(0)]

    crossed = bid > ask
    n_crossed = int(crossed.sum())
    if n_crossed:
        report.add(
            "not_crossed",
            Severity.ERROR,
            f"{n_crossed} snapshot(s) with best bid above best ask",
            count=n_crossed,
            examples=tuple(df.index[crossed][:3]),
        )

    locked = bid == ask
    n_locked = int(locked.sum())
    if n_locked:
        report.add(
            "not_locked",
            Severity.WARNING,
            f"{n_locked} snapshot(s) with zero spread; costs and imbalance are degenerate there",
            count=n_locked,
            examples=tuple(df.index[locked][:3]),
        )


def _check_level_ordering(df: pd.DataFrame, depth: int, report: ValidationReport) -> None:
    """Bid prices must decrease and ask prices increase away from the touch.

    Out-of-order levels usually mean the sides were mixed up or a snapshot was
    assembled from an unsorted update, which silently corrupts every depth and
    slope feature.
    """
    for level in range(depth - 1):
        bad_bid = df[bid_price_col(level)] <= df[bid_price_col(level + 1)]
        n_bid = int(bad_bid.sum())
        if n_bid:
            report.add(
                "bid_level_ordering",
                Severity.ERROR,
                f"bid level {level} is not above level {level + 1} in {n_bid} snapshot(s)",
                count=n_bid,
                examples=tuple(df.index[bad_bid][:3]),
            )

        bad_ask = df[ask_price_col(level)] >= df[ask_price_col(level + 1)]
        n_ask = int(bad_ask.sum())
        if n_ask:
            report.add(
                "ask_level_ordering",
                Severity.ERROR,
                f"ask level {level} is not below level {level + 1} in {n_ask} snapshot(s)",
                count=n_ask,
                examples=tuple(df.index[bad_ask][:3]),
            )


def _check_sequence_ids(
    df: pd.DataFrame, report: ValidationReport, *, sampled: bool = False
) -> None:
    """Sequence ids must strictly increase; a jump marks an unreliable segment.

    This is the check the future LOB collector depends on. A local book is built
    by applying deltas in order, so a missed update means every later snapshot is
    wrong until the next resynchronisation — and nothing about the resulting
    prices looks wrong on its own.

    Uniqueness and ordering are checked either way; only continuity is
    conditional. In a sampled frame a skipped id carries no information, so the
    gap count is reported once as context rather than as a finding per row.
    """
    for symbol, group in df.groupby("symbol", sort=False):
        seq = group["sequence_id"]

        duplicates = int(seq.duplicated().sum())
        if duplicates:
            report.add(
                "sequence_unique",
                Severity.ERROR,
                f"{symbol}: {duplicates} duplicate sequence_id(s)",
                count=duplicates,
            )

        step = seq.diff().dropna()
        if (step <= 0).any():
            n_back = int((step <= 0).sum())
            report.add(
                "sequence_monotonic",
                Severity.ERROR,
                f"{symbol}: sequence_id does not strictly increase at {n_back} row(s)",
                count=n_back,
            )

        gaps = step[step > 1]
        if gaps.empty:
            continue

        missing = int(gaps.sum() - len(gaps))
        if sampled:
            report.add(
                "sequence_sampled",
                Severity.INFO,
                f"{symbol}: {len(gaps)} skipped sequence range(s) covering {missing} update(s), "
                f"as expected for a sampled frame; continuity is not checked",
                count=len(gaps),
            )
        else:
            report.add(
                "sequence_gaps",
                Severity.WARNING,
                f"{symbol}: {len(gaps)} sequence gap(s) totalling {missing} missed update(s); "
                f"book state is unreliable after each gap until resynchronisation",
                count=len(gaps),
            )


# ---------------------------------------------------------------------------
# Gaps in time
# ---------------------------------------------------------------------------


#: A pause shorter than this is never interesting, whatever the instrument's
#: usual rate. Well below any outage worth knowing about, well above the
#: ordinary quiet spells in event data.
MIN_INTERESTING_GAP: Final = pd.Timedelta(30, unit="s")

#: On a sample too short for the absolute floor to mean anything, a pause is
#: judged against the period instead: swallowing this share of the data matters
#: regardless of how few seconds it is.
GAP_SHARE_OF_SPAN: Final = 0.05


def _report_time_gaps(df: pd.DataFrame, report: ValidationReport) -> None:
    """Flag pauses long enough to distort a window, and describe the rest.

    A gap is not an error — exchanges halt, collectors restart, activity dries
    up overnight. It matters because rolling windows silently span it: a
    sixty-observation window covering a two-hour outage measures something quite
    different from the same window in normal trading.

    Picking the threshold is the whole problem. An earlier version used a
    multiple of the median spacing, which fails badly on event data: trades
    arrive in bursts, so the median is tiny and any multiple of it flags routine
    quiet spells. On one day of real BTCUSDT trades that produced over a hundred
    thousand warnings — noise that would hide the single outage worth finding.

    So the test is absolute and scale-aware instead: a pause is reported when it
    is long in human terms *or* covers a meaningful share of the sample. Shorter
    pauses are summarised in one informational line, because the distribution of
    spacing is worth knowing even when nothing is wrong.
    """
    for symbol, group in df.groupby("symbol", sort=False):
        if len(group) < 3:
            continue

        delta = group["timestamp"].diff().dropna()
        if delta.empty:
            continue

        # Whichever bar is easier to clear. On a long sample the absolute floor
        # binds and routine quiet spells are ignored; on a sample so short that
        # thirty seconds would cover most of it, the share binds instead. Taking
        # the smaller of the two is what keeps both ends honest — an earlier
        # version paired the floor with a 0.1% share, which on a six-minute demo
        # set the bar at 0.4s and flagged 82% of the period.
        span = group["timestamp"].iloc[-1] - group["timestamp"].iloc[0]
        threshold = min(MIN_INTERESTING_GAP, span * GAP_SHARE_OF_SPAN)
        threshold = max(threshold, pd.Timedelta(1, unit="ms"))

        gaps = delta[delta > threshold]
        longest = pd.Timedelta(delta.max())
        median = pd.Timedelta(delta.median())

        if gaps.empty:
            report.add(
                "spacing",
                Severity.INFO,
                f"{symbol}: median spacing {median}, longest pause {longest}; no gap above {threshold}",
            )
            continue

        lost = pd.Timedelta(gaps.sum())
        share = lost / span if span > pd.Timedelta(0) else 0.0
        biggest = gaps.nlargest(3)
        report.add(
            "time_gaps",
            Severity.WARNING,
            f"{symbol}: {len(gaps)} pause(s) over {threshold}, {lost} total "
            f"({share:.1%} of the period); longest {longest}. Rolling windows span these.",
            count=len(gaps),
            examples=tuple(group["timestamp"][biggest.index].astype(str)),
        )


def validate(
    df: pd.DataFrame,
    plane: str,
    *,
    depth: int | None = None,
    sampled: bool = False,
) -> ValidationReport:
    """Dispatch to the right validator for a plane."""
    if plane == "trades":
        return validate_trades(df)
    if plane == "book":
        return validate_book(df, depth=depth, sampled=sampled)
    raise ValueError(f"unknown plane {plane!r}; expected 'trades' or 'book'")
