# The agent harness

The layer that lets an LLM agent run this repository's research loop under the
discipline the registered studies used by hand: a hypothesis and its kill
condition written before a run, a registration committed before any of its
data is read, held-out blocks opened once through a ledger, statuses computed
by code, and results reported only in sentences the registration fixed. It
adds no study code and changes none: it runs the commands that exist.

| File | What it is |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | the rules every agent follows here, by hand or through this harness |
| [`protocol.md`](protocol.md) | the loop: propose, register, run, review, verdict, record; each step's artefact and gate |
| [`roles/`](roles/) | one prompt per role: researcher, registrar, executor, skeptic, scribe |
| [`workflows/policy.py`](workflows/policy.py) | the tool policy: what each role may run, read and write |
| [`workflows/prompts.py`](workflows/prompts.py) | each role's system prompt and step prompt, and the handback it must end with |
| [`workflows/research_loop.py`](workflows/research_loop.py) | the orchestrator, its gates, the code's verdict, and the helpers |
| [`../docs/agents_demo_funding_carry.md`](../docs/agents_demo_funding_carry.md) | how the loop would run the roadmap's funding-rate carry study |

## Running it

The SDK's Python package, `claude-agent-sdk`, is an optional dependency: it is
not in `pyproject.toml`, and it is installed only into the environment that
runs a loop, with what its documentation requires. The model is named by an
environment variable and the SDK reads its credentials from the environment
itself; the harness never reads, logs or stores a key.

```bash
export TRP_AGENT_MODEL=<model name>            # TRP_AGENT_MODEL_<ROLE> overrides it per role
PYTHONPATH=src python -m agents.workflows.research_loop run --study funding_carry_2026 --brief-file <brief.md>
```

`TRP_AGENT_PYTHON` and `TRP_AGENT_CLI` name the interpreter and the CLI when
they are not `python` and `trading-research` on the path (a study's own
environment, say); `TRP_AGENT_MAX_TURNS` bounds a role's turn (40 by default).

A run stops at every gate the owner holds and says why. The owner acts (commits
the registration, commits an open entry, reads a blocking finding) and resumes
it, granting a block if the next step reads one:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop run --resume agents/runs/<run> --grant funding_carry_2026:H
```

The helpers need no model and no SDK:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop calendar
PYTHONPATH=src python -m agents.workflows.research_loop null-hash configs/spot_2026.yaml
PYTHONPATH=src python -m agents.workflows.research_loop check --role executor "git push"
```

`calendar` prints every range a registration protects; `null-hash` prints a
YAML's null-placeholder sha256 by the repository's own `unfrozen_text` and
`sha256_bytes`; `check` says whether the policy would let a role run a
command, and why.

## What the policy enforces

Every tool call passes `ToolPolicy.decide` twice: as a hook before the call,
which no allow rule or mode can bypass, and as the SDK's tool callback. Project
and user settings are not loaded, so nothing on the machine widens it.

- **Commands**: one at a time, never chained, piped, redirected or
  substituted: `trading-research` and its subcommands, the study entry points,
  `pytest`, read-only `git` (`log`, `diff`, `status`, `show`) and the helpers
  above, each role only its own (the table in `workflows/policy.py`).
- **Never**: `git push`, `git remote`, any git that writes; `--second-read`;
  starting a loop from inside one; a command in the background.
- **Protected days**: a command whose dates touch a range of the calendar is
  refused, an archive published by the month counting as its whole month;
  `open`, `read` and `--allow-heldout` need the owner's grant for that block.
- **Files**: nothing under `data/`, `models/` or `.git/` is read by hand;
  nothing outside the repository either. The registrar writes its own study's
  registration until the code freeze, then only appends; the scribe writes the
  README, the findings register and appends to the study's run log; no role
  writes `src/`, `experiments/`, `tests/`, another registration, a result, a
  ledger or the harness. A fetch writes under `data/` only into `data/<study>/`.
- **Web**: no search; the registrar alone reads a fee page, from a host the
  owner listed (`--web-host`), never a host the downloaders read data from.

## Checks

```bash
ruff format --check agents tests
ruff check agents tests
MYPYPATH=src mypy --explicit-package-bases agents/workflows tests/test_agents_policy.py tests/test_agents_loop.py tests/test_agents_prompts.py tests/test_agents_demo.py
PYTHONPATH=src python -m pytest tests/test_agents_policy.py tests/test_agents_loop.py tests/test_agents_prompts.py tests/test_agents_demo.py
```

The tests run from the repository's root (`python -m pytest` puts it on the
path, as the entry points' tests need). Before any model is called, a run also
checks that every command its documents name exists, with its options, and
stops if one does not.

No test reaches a model or the network: the loop's tests answer each role with
a fake client, and the verdict's test reads the spot study's committed results,
ledger and page.

## What it does not do

It writes no study code: a new study's entry point and modules are the owner's,
written between registration and run, reviewed and frozen as the registrations
describe. It does not commit, push or publish. It does not decide a status: the
study's code does, and the harness only checks that what is recorded is what
the code wrote.
