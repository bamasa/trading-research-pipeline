# Skills: how to use them, with examples

The agent harness ([`agents.md`](agents.md)) runs the research loop
unattended, one role per step. The skills under
[`../.claude/skills/`](../.claude/skills/) are the same discipline for an
interactive session: a person works in Claude Code, and the session loads the
skill whose description matches the request. Each skill is one `SKILL.md`: what
to read first, the steps, the commands, what the model must not do, and what it
hands back. [`../CLAUDE.md`](../CLAUDE.md) binds every one of them.

## Calling a skill

There are two ways:

- **By name.** Type `/` and the skill's name in a Claude Code session opened at
  the repository's root, for example `/propose-hypotheses`.
- **By asking.** Describe the task. Each description ends with a "Use when"
  sentence, and the session loads the skill whose sentence fits. "What should
  we research next?" loads `propose-hypotheses`; "check this study's code for
  look-ahead" loads `pre-freeze-review` or `skeptic-review`.

A skill gives the model no new permission. A held-out block is still opened
only through its study's ledger, and only on the owner's word; a commit is
still the owner's.

## Which skill, in the order of a study

| Stage | Skill | Ask it, for example | It hands back |
|---|---|---|---|
| idea | `propose-hypotheses` | "Three hypotheses that build on the 10-minute reversion, cheapest to refute first" | three to five hypotheses, each with a mechanism, a kill condition and the arithmetic against the cost floor |
| scoring an idea | `benchmark-strategy` | "Score my_rule on the benchmark and tell me at what fee it would pay" | the card: a mark per track, net at each fee, the break-even fee, and what a pass still needs |
| registration | `preregister-study` | "Register the second one: blocks, slots, kills" | a document under `docs/preregistration/` and its YAML under `configs/`, with the null-placeholder sha256 |
| before the freeze | `pre-freeze-review` | "Review the carry study's code before we freeze it" | each defect with `file:line`, the input that exposes it and the amendment it needs |
| forward block | `news-screen` | "Screen the instruments of P for delistings and unlocks" | a dated table of exclusions and flags with their sources, never an addition |
| held-out read | `held-out-read` | "Is H open yet? Read it" | the ledger's state, then one read after the owner's grant |
| after a run | `skeptic-review` | "What would refute this result?" | findings under three lenses (leakage, fidelity, execution), each marked as blocking or not |
| result | `record-result` | "Write up H's result" | the run log entry, the findings row and the README line, in the registered sentences |
| commit | `check-before-commit` | "Check before I push" | the checks CI runs, their output, and a commit message in the repository's style |
| the whole loop | `research-loop` | "Start the loop on this brief" or "resume the stopped run" | the harness's run directory and the step it stopped at |

## Where a language model earns its place

Three skills do work that is reading and judgement. A model is good at that,
and it is also where it can leak.

**Proposing hypotheses.** The model reads
[`findings.md`](findings.md), every idea tried and how each was killed, and
names a mechanism for a new one: who is on the other side and why they pay.
It does not propose a killed idea again without saying what changed. It never
quotes a number about a market from memory, since its training data reaches
into the periods held out here. Every number in a proposal comes with the path
of the file it was read from.

**Reviewing code before a freeze.** The checklist is the 21 defects that two
reviews found in the 2025 study's code before its first held-out read, dated
in the amendments of 2026-10-28 in
[`preregistration/short_horizon_2025.md`](preregistration/short_horizon_2025.md):
day boundaries, a cache shared between blocks, a fee from the wrong tier, a
Holm family counted from the slots that ran instead of the slots registered. A
finding counts only with the smallest input that exposes it and what the code
returned, so the review cannot pad itself with suspicions. Before it starts it
runs the study's tests:

```bash
PYTHONPATH=src python -m pytest tests/ -q -k <study>
```

**Screening by news.** A model reads a delisting notice better than a keyword
filter does. In a backtest that is a leak, since news about a past date was
not known at that date. So the screen runs only forward, for a block whose
first day is after today, and only if the study's registration named it as a
rule before the block started. It can only remove an instrument, never add
one, so it cannot select for a good outcome. Whether a block is still ahead:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop calendar
```

## What was done with them so far

- **The funding-rate carry study** (2 to 3 November 2026) went from a brief to
  a committed registration through the harness's researcher and registrar,
  which follow the same steps as `propose-hypotheses` and
  `preregister-study`. The run log is
  [`funding_carry_2026_run_log.md`](funding_carry_2026_run_log.md).
- **The 2025 study's H read** follows `held-out-read`: the open entry was
  committed and pushed before any day of H was fetched, then the fetch records
  were committed before any of H was computed.

Neither of these is a result about a market. A skill makes a step repeatable
and checkable. Whether a strategy earns money is decided by the data, after a
registration.

## Writing a new skill

Make a directory under `.claude/skills/` with a `SKILL.md`. Its frontmatter
`name` must be the directory's name, and its `description` must say what the
skill does and contain "Use when". The body names `CLAUDE.md` or the role it
follows under `agents/roles/`. Every command it shows must exist with the
options it uses. The tests check all of this:

```bash
PYTHONPATH=src python -m pytest tests/test_agents_prompts.py -q
```
