# The research loop

The loop an agent follows in this repository, and the discipline the registered
studies already follow by hand: **propose, register, run, review, verdict,
record**. Each step produces one artefact and passes one gate before the next
starts. Five steps belong to a role ([`roles/`](roles/)); the verdict belongs
to the code. The orchestrator
([`workflows/research_loop.py`](workflows/research_loop.py)) runs the steps in
this order, checks the gates it can check mechanically, and stops at every gate
the owner holds. The rules in [`../CLAUDE.md`](../CLAUDE.md) bind every step.

| Step | By | Artefact | Gate before the next step | Where the repository does it |
|---|---|---|---|---|
| propose | researcher | hypotheses, each with metric, kills, placebos and expected outcome | every hypothesis has a kill condition and an expected outcome | "Expected outcome, stated in advance" in each registration; `pre-registered` in `docs/findings.md` |
| register | registrar | `configs/<study>.yaml` and `docs/preregistration/<study>.md`, the null-placeholder sha256 | both committed and clean; the document holds the YAML's null-placeholder sha256 | `unfrozen_text`, `sha256_bytes` in `market_making/prereg.py`; each study's `REGISTERED_SHA256` |
| (code to come) | the owner | new files, a test each, reviews, the code-freeze note | `docs/<study>_code_freeze.md` committed | `spot/code_freeze.py`, `docs/spot_2026_code_freeze.md` |
| run | executor | the entry point's records, verbatim; after a read, the results file and page | the stage order, the ledger, the protected calendar, the owner's grant | `StageOrderError`, `HeldOutLocked`, `spot/ledger.py`, `spot/runlog.py` |
| review | skeptic | findings with `file:line`, severity and lens | no blocking finding | `docs/evaluation/rubric.md`; the three code reviews each study had |
| verdict | code | `verdict.json`: statuses, kills, the results file's sha256, the headline sentences | the results file is the one the ledger recorded | `spot/statuses.py`, `results_sha256` in the ledger, `spot/pages.py` |
| record | scribe | run log entry, findings row, README row, the registered sentences | every sentence is one of the page's headline sentences | `docs/<study>_run_log.md`, `docs/findings.md`, the README roadmap |

## 1. Propose (researcher)

**Produces** a proposal: the question; each hypothesis as a claim with its
metric (the daily series and its unit), its kill conditions in the order they
are read, its placebos, its neighbourhood, and its expected outcome with the
arithmetic behind it; the information set (what the owner and this repository
have already seen of the data, with dates); and the blocks it wants, checked
against the protected calendar
(`python -m agents.workflows.research_loop calendar`).

**Gate.** Every hypothesis carries at least one kill condition and an expected
outcome, stated before anything is computed. A proposal that cannot say what
would refute it is declined, not sent on.

**In the repository.** The registrations' "Expected outcome, stated in advance"
sections, the predictions per slot of `docs/preregistration/spot_2026.md`, and
the `pre-registered` status of `docs/findings.md`.

## 2. Register (registrar)

**Produces** the document and the YAML, which agree: why the study, what was
known before it was written, the expected outcome, data, fees (sourced and
dated), blocks, common settings, slots, metrics, status rules with the family
and Holm's m, kill conditions per slot, placebos, measurements without a
hypothesis, data budget, who may read what and the code to come, the
amendment protocol, the artefacts and the headline sentences, what the study
cannot establish, and the choices with their reasons. Values the development
block sets are `null` placeholders commented "frozen by the D amendment". The
registrar computes the null-placeholder sha256 with
`python -m agents.workflows.research_loop null-hash configs/<study>.yaml` and
writes it into the document.

**Gate**, checked by the orchestrator before the run step: both files exist,
are committed and unchanged since, and the document holds the YAML's
null-placeholder sha256. The registrar refuses an underspecified plan (a
hypothesis without a kill, a block on a protected day, a fee without a source
and date, a family without its m) and says what is missing.

**In the repository.** `market_making/prereg.py` (`placeholder_lines`,
`unfrozen_text`, `sha256_bytes`, `freeze_text`); each study's loader with its
`REGISTERED_SHA256` (`pipeline/short_horizon_registration.py`, and
`spot/registration.py`, which also refuses its `SUPERSEDED_SHA256`); the
amendment lines
``Null-placeholder sha256: `<hash>` `` and ``Frozen configuration sha256: `<hash>` ``.

### Between register and run: the code to come

A new study needs code: an entry point `experiments/<study>.py` and a package
of new modules, as each registration's "The code to come" lists them. No role
of this harness writes study code. The owner writes it, or directs a separate
session to, in pieces of about 120 lines with a test each; the skeptic reviews
it in its three lenses; every choice the code makes is written into the
registration before any day of a block is fetched; and the code-freeze note
`docs/<study>_code_freeze.md` names the commit the ledger will hold the code
to. Until that note is committed, the executor runs only the entry point's
`--dry-run` and the stages the registration allows before the freeze.

## 3. Run (executor)

**Produces** the records the commands print, verbatim (each entry point prints
its record as JSON), and, after a read, the paths and sha256 of the results
file, the ledger and the page the entry point wrote. Nothing is summarised or
rounded.

**Gate.** Only the commands of the [command table](../CLAUDE.md#commands-and-what-each-may-read)
run, one at a time. The entry point refuses a stage out of order
(`StageOrderError`) and a block the ledger has not opened (`HeldOutLocked`);
the harness's tool policy refuses a fetch that touches a protected day, and
`open`, `read` and `--allow-heldout` without the owner's grant. A refusal is
reported as it came and ends the step; it is never worked around.

**In the repository.** `spot/state.py` (`STAGES`, `READS`, the stage machine),
`spot/ledger.py` (`Ledger.open`, `Ledger.session`: a clean tree, the frozen
code, the environment fingerprint), `spot/runlog.py` (every computation on a
development test day on record before the held-out read).

## 4. Review (skeptic)

**Produces** findings, each with a lens, a severity, a `file:line` and the
attempted refutation. The three lenses are the ones the studies' code reviews
used: **leakage and look-ahead** (anything after the decision reaching a
feature, a label, a threshold or a fold; a file of a protected block on disk);
**fidelity to the registration** (every coordinate, fee, block, kill and
sentence as registered; the null-placeholder and frozen hashes; the trial
count); **execution realism and money** (fills, fees per side on notional,
latency, capacity, the accounting identity, gross against net).

**Gate.** A finding of severity `blocking` stops the loop: nothing is recorded
until the owner has read it. `major` and `minor` findings are recorded beside
the result. After a read, a defect can only lower a status: it is written to
`docs/<study>_defects.md` with the mechanism by which it could have raised the
result, and the status becomes `void (defect found after the <block> read)`.

**In the repository.** `docs/evaluation/rubric.md` and `docs/evaluation/prompt.md`
(a reviewer that did not build the strategy, "a positive result is a claim to
be checked"); the README's record of three reviews per study, each applied
before any data was read.

## 5. Verdict (code)

**Produces** `verdict.json`: each slot's status and first kill exactly as the
results file holds them, the results file's sha256, and the headline sentences
of the block's section of the page. No role writes a verdict, and none can
change one: the statuses were computed by the study's code under its
precedence and hashed into its ledger.

**Gate.** The results file's sha256 equals the `results_sha256` the ledger
recorded for the block's first read. A run that read no held-out block has no
verdict, and says so: development numbers are never results.

## 6. Record (scribe)

**Produces** a dated entry in `docs/<study>_run_log.md`, the study's row in
`docs/findings.md` with its status and the kill that fired, the README's
roadmap row, and nothing else.

**Gate.** Every sentence the scribe hands back is one of the verdict's headline
sentences, verbatim; the orchestrator checks it.

## The owner's gates

The loop stops, and resumes only when the owner acts, at: the commit of a
registration; the code freeze; the grant of a held-out block for a run
(`--grant <config>:<block>`), and the commit of its open entry before its
fetch; the commit of results; and a blocking finding. Publishing is never a
step of this loop.

## The handback block

Every role ends its reply with one fenced block tagged `handback`, in YAML.
Common keys: `role`, `step`, `status`. The rest by role:

| Role | `status` | Further keys |
|---|---|---|
| researcher | `proposed` or `declined` | `hypotheses`: each with `id`, `claim`, `metric`, `kills`, `placebos`, `expected`; `known`; `blocks` |
| registrar | `registered` or `refused` | `files`; `null_sha256`; `missing` when refused |
| executor | `ran`, `refused` or `failed` | `commands`; `records`; `results`, `ledger`, `page` and `block` after a read |
| skeptic | `pass` or `block` | `findings`: each with `lens`, `severity` (`blocking`, `major`, `minor`), `where`, `finding`, `refutation` |
| scribe | `recorded` | `files`; `sentences` |

## The run directory

Each run writes `agents/runs/<UTC time>-<study>/` (ignored by git, so a ledger's
clean-tree check never sees it): `run.json` (the study, the brief, the grants,
the model's name, the commit), and per step the prompt, the reply, the
handback, and `events.jsonl` with every tool call and the policy's decision;
then `verdict.json`, and a line in `stops.jsonl` each time a gate stopped the loop.
