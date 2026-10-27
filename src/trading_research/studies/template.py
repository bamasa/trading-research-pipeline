"""A new study's registration skeleton: the document, its YAML and an entry point that refuses until both are filled.

``trp study new <name> --template taker|passive|maker`` writes three files:

* ``docs/preregistration/<name>.md``: the sections every registration here
  has, in the order of ``docs/preregistration/spot_2026.md``, each with the
  rule it must state in one line and ``<...>`` where the author's values go;
* ``configs/<name>.yaml``: the same values in the form code reads them, with
  the D amendment's values as ``null`` placeholders commented
  "frozen by the D amendment", so that the null-placeholder sha256 is defined;
* ``experiments/<name>.py``: the study's single entry point, declaring its
  stages and blocks. It answers ``status``, and refuses every other step while
  a ``<...>`` remains in either file, while the document does not hold the
  YAML's null-placeholder sha256, and, after that, until the study's own code
  is written; it never reads data.

The three templates differ only in the slot they register and what that
execution needs: a taker crosses the spread on a signal, a passive entry rests
at the touch and fills only when the queue ahead has traded, a market maker
quotes both sides in event time. Nothing here chooses a value: the author does,
before the registration is committed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from trading_research.market_making.prereg import PLACEHOLDER_MARK, sha256_bytes, unfrozen_text

TEMPLATES: tuple[str, ...] = ("taker", "passive", "maker")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
#: An author's placeholder: anything between angle brackets on one line.
AUTHOR_PLACEHOLDER = re.compile(r"<[^<>\n]+>")
#: Angle brackets that show the form of a line the code writes later, not a value to fill.
EXAMPLES = frozenset({"<sha>", "<hash>"})

#: The slot each template registers, as YAML lines under ``slots``.
SLOTS: dict[str, str] = {
    "taker": """  S1:
    execution: taker  # cross on the signal, cross back after the hold (pipeline/execution.py)
    signal: <a rule or a model, as trp signal train --model names it>
    hold_s: <seconds>
    entry: the decision rule's threshold, chosen on D
    costs: {fee: fees.verdict_tier, slippage_bp: <bp per side>}
""",
    "passive": """  S1:
    execution: passive_entry  # rest at the touch, fill once the queue ahead traded (backtest/maker.py)
    signal: <a rule or a model, as trp signal train --model names it>
    hold_s: <seconds>
    timeout_s: <seconds a resting entry waits before it is cancelled>
    exit: <taker or passive>
    costs: {maker: fees.verdict_tier.maker_bp, taker: fees.verdict_tier.taker_bp}
""",
    "maker": """  S1:
    execution: market_maker  # both sides in event time, fills only from prints (market_making/simulator.py)
    quoter: <S0, S1, S2 or S3 (pipeline/market_making.py)>
    clip_notional: null  # frozen by the D amendment
    sigma_ref: null  # frozen by the D amendment
    params: null  # frozen by the D amendment
    soft_limit_clips: <clips>
    latency_ms: <ms>
    cancel_model: <pessimistic, proportional or optimistic>
""",
}

#: Kill conditions every template has, then the template's own.
KILLS: dict[str, str] = {
    "common": """  K1: {test: "the mean of H's daily net at or below 0 bp", outcome: killed}
  K-days: {test: "fewer than statuses.min_days_H days with a trade on H", outcome: inconclusive}
  K-placebo: {test: "H's mean not strictly above the placebos' 95th percentile", outcome: killed}
""",
    "taker": "",
    "passive": """  K-fill: {test: "a fill rate on H below <share>", outcome: inconclusive}
""",
    "maker": """  K-dir: {test: "the making part (spread + adverse - fees) not positive while the net is", outcome: killed}
  K-pess: {test: "positive under proportional cancels, not under pessimistic", outcome: killed}
""",
}


@dataclass(frozen=True)
class NewStudy:
    """The three files a new study starts from."""

    name: str
    template: str
    document: Path
    config: Path
    entry_point: Path


def config_text(name: str, template: str) -> str:
    """The YAML: the author's values as ``<...>``, the D amendment's as null placeholders."""
    mark = PLACEHOLDER_MARK
    return f"""# {name}: the registration's values, in the form the code reads them.
# Registered in docs/preregistration/{name}.md; the two must agree. Every <...>
# is the author's to fill before the registration is committed; every value
# commented "{mark}" stays null until that amendment.
name: {name}
template: {template}
registered: <YYYY-MM-DD, the commit's date>
document: docs/preregistration/{name}.md
venue: <venue and product>
instruments: [<SYMBOL>]
hypothesis:
  claim: <one sentence on what pays, where, net of the verdict fee>
  metric: each day's net in bp of notional; its mean and its day-level t
  expected: <the outcome expected, stated before any data is read>
blocks:  # whole days, UTC, inclusive; clear of every range `python -m agents.workflows.research_loop calendar` prints
  D: {{role: development, start: <YYYY-MM-DD>, end: <YYYY-MM-DD>, reads: free}}
  E: {{role: embargo, start: <YYYY-MM-DD>, end: <YYYY-MM-DD>, reads: never}}
  H: {{role: held_out, start: <YYYY-MM-DD>, end: <YYYY-MM-DD>, reads: "once, after the D amendment and H's committed open entry"}}
fees:  # rows of `trp fees`, named by venue and tier
  verdict_tier: {{venue: <venue>, tier: <tier>, maker_bp: <bp>, taker_bp: <bp>}}
  ladder: <the tiers reported beside the verdict, never instead of it>
slots:
{SLOTS[template]}family: {{slots: [S1], m: 1}}  # Holm's m is fixed here and never depends on an outcome
statuses:
  alpha: 0.05  # Holm's procedure across the family
  t_bar: 3.0  # day-level t
  newey_west_bar: 2.0
  min_days_H: <days with a trade>
kill_conditions:  # applied in this order; the first that fires decides
{KILLS["common"]}{KILLS[template]}placebos: {{kind: shuffled-state circular shift of the decisions, draws: <n>, seed: <n>, percentile: 95}}
frozen:  # what the D amendment writes, and nothing else
  thresholds: {mark_null(mark)}
  fitted_sha256: {mark_null(mark)}
  sigma_d_bp: {mark_null(mark)}
  minimum_detectable_effect_bp: {mark_null(mark)}
code_freeze:
  note: docs/{name}_code_freeze.md  # a line "Code-freeze commit: `<sha>`", committed before the first fetch of a block
  covers: [src/, experiments/{name}.py, configs/{name}.yaml, pyproject.toml, uv.lock]
access:
  ledger: experiments/results/{name}_ledger.json
  open_entry: committed before the block's fetch begins; a read without it is refused
  second_read: refused unless forced, and then stamped "second read"
amendment:
  protocol: dated entries appended to the document; none cites a number seen on the data as its reason, except a fault it names
  d_amendment: writes the frozen values into their placeholders and records "Frozen configuration sha256" on one line
"""


def mark_null(mark: str) -> str:
    return f"null  # {mark}"


def document_text(name: str, template: str) -> str:
    """The registration document: every section, its rule, and ``<...>`` for the author."""
    execution = {"taker": "taker", "passive": "passive entry", "maker": "market maker"}[template]
    return f"""# Pre-registration: {name}, <instrument>, <venue>, <period>

This document registers a study before any of its data is read. It is
committed together with [`configs/{name}.yaml`](../../configs/{name}.yaml),
which holds the same values in the form the code reads, and the two must agree.
Written from the {execution} template by `trp study new`; every `<...>` is the
author's to fill before the registration is committed.

| | |
|---|---|
| **Committed** | <date>, on the local branch <branch>, before any day of a block is fetched |
| **What it registers** | <the hypothesis in one sentence>; the blocks, the fee, the slot, its kill conditions and placebos, the status rules under Holm with m = 1, the code to come and the amendment protocol |
| **Searched** | <what D chooses, and how many configurations it tries; "nothing" if it chooses nothing> |
| **Computed so far** | <nothing; or exactly what was read, when, and why> |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end |

## Why this study

<What question it answers, and why that question now.>

## What was known before this was written

<Every result, file and fact the author knew that bears on the outcome, with its date. The protected
ranges of `python -m agents.workflows.research_loop calendar` are not touched by any block below.>

## Expected outcome, stated in advance

<The outcome expected on H, as numbers, written before any day of a block is read.>

## Data

<Venue, product, planes (book, prints, funding), their archives, and the loader that reads them through
the study's access.>

## Fees

The verdict is made at one tier, a row of `trp fees` named here: <venue>, <tier>, maker <bp>, taker
<bp>, sourced <where, when read>. Other tiers are a ladder reported beside the verdict, never instead
of it. A fee is never changed after the freeze.

## Blocks

| Block | Dates (UTC, inclusive) | Use | Read |
|---|---|---|---|
| **D** development | <start> to <end> | fits, thresholds, sigma of daily results | freely |
| **E** embargo | <start> to <end> | a gap, so that H tests transfer | never fetched |
| **H** held out | <start> to <end> | the slot's status | once, after the D amendment is committed, through the ledger |

## The slots

**S1**, {execution}: <signal, hold, entry rule, exit rule, costs>, as `configs/{name}.yaml` `slots.S1`.

## Metrics

Each day's net in bp of notional; its mean, its day-level t (`market_making/verdicts.day_t`), its
Newey-West t, its median and its count of days with a trade.

## Status rules

**The family** is **{{S1}}, with m = 1**. A slot passes when no kill condition fires, its one-sided
day-level p-value passes Holm's procedure at 5% with m = 1, the day-level t is at least 3, the
Newey-West t at least 2, the median day positive, and it traded on at least <n> days of H.

### Kill conditions

Applied in the order of `kill_conditions` in the YAML; the first that fires decides. Written before
any day of H is read; none is added, removed or reordered after the freeze.

### Placebos

<The placebo, its draws and its seed>: the result must lie strictly above the placebos' 95th percentile.

## Who may read what, and the code that will enforce it

### The code to come

The study's code is written between registration and the code freeze, which a note
`docs/{name}_code_freeze.md` records with a line ``Code-freeze commit: `<sha>` `` before the first
fetch of a block. It covers `src/`, `experiments/{name}.py`, the configuration, `pyproject.toml` and
`uv.lock`; a change to any of them after the freeze blocks the read.

### Access

H is opened once, through `experiments/results/{name}_ledger.json`: the open entry is written by the
entry point and committed before the block's fetch begins, and a read without it is refused. A second
read is refused unless forced, and then stamped "second read".

## Amendment protocol

Amendments are dated and appended under [Amendments](#amendments). None cites a number seen on the data
as its reason, except a fault it names. **The D amendment**, after D and before H is read, writes the
values commented "{PLACEHOLDER_MARK}" into their placeholders and nothing else, and records the
configuration's sha256 on a line of the form ``Frozen configuration sha256: `<hash>` ``.

## Artefacts and headline sentences

<The results file, the page and the headline sentences, one per status, fixed here and copied verbatim
by whoever reports the result.>

## What this study cannot establish

<Its limits, stated before the result.>

## Amendments

The configuration as registered, every placeholder null, has the sha256 below (`trp study null-hash
{name}`), recomputed and written here when the author's values are filled in:

Null-placeholder sha256: `<null-placeholder sha256>`
"""


def entry_point_text(name: str, template: str) -> str:
    """``experiments/<name>.py``: the stages declared, every step refused until the registration is filled."""
    return f'''"""{name}'s single entry point, written from the {template} template by ``trp study new``.

Registered in ``docs/preregistration/{name}.md`` with ``configs/{name}.yaml``.
Its subcommands are the stages in the registered order, then the open and the
read of H through the ledger::

    python -m experiments.{name} status
    python -m experiments.{name} 0
    python -m experiments.{name} D
    python -m experiments.{name} amend
    python -m experiments.{name} open H
    python -m experiments.{name} read H

``status`` always answers. Every other step is refused while the registration
holds a ``<...>``, while the document does not hold the YAML's null-placeholder
sha256, and then until the study's code is written here; nothing reads data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAME = "{name}"
CONFIG = Path("configs") / f"{{NAME}}.yaml"
DOCUMENT = Path("docs") / "preregistration" / f"{{NAME}}.md"
STAGES = ("0", "D", "amend")
READS = ("H",)
PLACEHOLDER = re.compile(r"<[^<>\\n]+>")


class Refused(RuntimeError):
    """A step the registration does not allow yet."""


def problems(repo: Path) -> list[str]:
    """What stands between the registration and a run, in the order it must be fixed."""
    from trading_research.studies.template import EXAMPLES, null_sha256

    found = []
    for path in (CONFIG, DOCUMENT):
        text = (repo / path).read_text(encoding="utf-8") if (repo / path).exists() else ""
        if not text:
            found.append(f"{{path}} does not exist")
            continue
        unfilled = sorted(set(PLACEHOLDER.findall(text)) - EXAMPLES)
        if unfilled:
            found.append(f"{{path}} still holds {{len(unfilled)}} placeholder(s): {{', '.join(unfilled[:6])}}")
    if not found:
        expected = null_sha256((repo / CONFIG).read_text(encoding="utf-8"))
        if expected not in (repo / DOCUMENT).read_text(encoding="utf-8"):
            found.append(f"{{DOCUMENT}} does not hold the null-placeholder sha256 {{expected}}")
    return found


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    found.add_argument("command", nargs="?", choices=[*STAGES, "open", "read", "status", "pages"])
    found.add_argument("block", nargs="?", choices=READS, help="the block to open or read")
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--dry-run", action="store_true", help="every stage on synthetic data")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "status":
        print(json.dumps({{step: "not run" for step in (*STAGES, *READS)}}, indent=2))
        return 0
    if args.command is None and not args.dry_run:
        parser().error("name a stage, open, read, status or pages (or pass --dry-run)")
    found = problems(REPO)
    if found:
        raise Refused("the registration is not complete: " + "; ".join(found))
    step = "--dry-run" if args.dry_run else " ".join(filter(None, (args.command, args.block)))
    raise Refused(
        f"{{step}}: the study's code is not written yet; write it, freeze it as the registration "
        "says, and replace this refusal with the stage it runs"
    )


def entry() -> int:
    """``main``, with a refusal as a message."""
    try:
        return main()
    except Refused as exc:
        print(f"refused: {{exc}}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(entry())
'''


def null_sha256(config_text: str) -> str:
    """The YAML's sha256 with every placeholder put back to null, by the repository's
    own helpers (``market_making/prereg.py``), as the agent harness computes it."""
    return sha256_bytes(unfrozen_text(config_text).encode())


def create(repo: Path, name: str, template: str, *, overwrite: bool = False) -> NewStudy:
    """Write the three files of a new study under ``repo``; refuse to overwrite any."""
    if not NAME.match(name):
        raise ValueError(f"a study is named by lowercase letters, digits and underscores: {name!r}")
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
    study = NewStudy(
        name,
        template,
        repo / "docs" / "preregistration" / f"{name}.md",
        repo / "configs" / f"{name}.yaml",
        repo / "experiments" / f"{name}.py",
    )
    existing = [p for p in (study.document, study.config, study.entry_point) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"already registered: {', '.join(str(p) for p in existing)}")
    for path, text in (
        (study.document, document_text(name, template)),
        (study.config, config_text(name, template)),
        (study.entry_point, entry_point_text(name, template)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return study


def unfilled(path: Path) -> list[str]:
    """The author's placeholders still in a file."""
    return sorted(set(AUTHOR_PLACEHOLDER.findall(path.read_text(encoding="utf-8"))) - EXAMPLES)
