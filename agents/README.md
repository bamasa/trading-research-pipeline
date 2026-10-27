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
| [`workflows/research_loop.py`](workflows/research_loop.py) | the orchestrator, its gates, the code's verdict, the SDK runner and the helpers |
| [`workflows/demo.py`](workflows/demo.py) and [`demo/`](demo/) | the demonstration on a synthetic study, with a scripted client in place of the model |
| [`../docs/agents.md`](../docs/agents.md) | how to run it, the roles, each rule with its test, what is logged and published |
| [`../docs/agents_demo_funding_carry.md`](../docs/agents_demo_funding_carry.md) | how the loop would run the roadmap's funding-rate carry study |

## Running it

[`../docs/agents.md`](../docs/agents.md) has the exact commands: the
demonstration a reviewer re-runs on synthetic data in seconds, the helpers,
and a live run. In short:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop demo
PYTHONPATH=src python -m agents.workflows.research_loop calendar
PYTHONPATH=src python -m agents.workflows.research_loop check --role executor "git push"
PYTHONPATH=src python -m agents.workflows.research_loop null-hash configs/spot_2026.yaml
PYTHONPATH=src python -m agents.workflows.research_loop prompts
```

A live run needs the model's name in `TRP_AGENT_MODEL` and the optional SDK,
`claude-agent-sdk`, installed in a small environment of its own
(`agents/.venv-agents/`, ignored), never in the study's: the ledgers
fingerprint every distribution installed where an entry point runs, and an
extra one would block a held-out read. The loop runs from that environment
and the agent's commands run in the study's (`TRP_AGENT_PYTHON`,
`TRP_AGENT_CLI`). The SDK reads its credentials from the environment itself;
the harness never reads, logs or stores a key.

A run stops at every gate the owner holds and says why. The owner acts and
resumes it, granting a block if the next step reads one; the step that stopped
is taken again, its earlier attempt kept under `superseded/`:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop run --resume agents/runs/<run> --grant funding_carry_2026:H
```

## What the policy enforces

A role is given only its own tools (`TOOLS_BY_ROLE` in
`workflows/research_loop.py`), and every tool call passes `ToolPolicy.decide`
twice: as a hook before the call, which no allow rule or mode can bypass, and
as the SDK's tool callback. Project and user settings are not loaded, so
nothing on the machine widens it. After each turn the orchestrator also reads
from git what the turn changed and stops the loop if it is more than the role
may change, whatever command changed it.

- **Commands**: one at a time, never chained, piped, redirected or
  substituted: `trading-research` and its subcommands, the study entry points,
  `pytest`, read-only `git` (`log`, `diff`, `status`, `show`) and the helpers
  above, each role only its own (the table in `workflows/policy.py`).
- **Never**: `git push`, `git remote`, `git commit` or any git that writes;
  `--second-read`; starting a loop from inside one; a command in the
  background; a subagent; a web search.
- **Protected days**: a command whose dates touch a range of the calendar is
  refused, an archive published by the month counting as its whole month; a
  plain `download` or `download-book` fetches free days only, whatever was
  granted; `open`, `read` and `--allow-heldout` need the owner's grant for that
  block, and `read` also needs the block's open entry committed in
  `experiments/results/<study>_ledger.json` with its read unfinished; another
  study's entry point runs only `status`.
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
MYPYPATH=src mypy --explicit-package-bases agents/workflows agents/demo tests/test_agents_policy.py tests/test_agents_loop.py tests/test_agents_prompts.py tests/test_agents_demo.py tests/test_agents_demo_run.py tests/test_agents_sdk.py
PYTHONPATH=src python -m pytest tests/test_agents_policy.py tests/test_agents_loop.py tests/test_agents_prompts.py tests/test_agents_demo.py tests/test_agents_demo_run.py tests/test_agents_sdk.py
```

The tests run from the repository's root (`python -m pytest` puts it on the
path, as the entry points' tests need). Before any model is called, a run also
checks that every command its documents name exists, with its options, and
stops if one does not.

No test reaches a model or the network. The loop's tests answer each role
with a fake client in a scratch repository on the synthetic study; the
demonstration's test runs the whole loop with the scripted client; the
verdict's test reads the spot study's committed results, ledger and page; the
SDK's test is skipped unless the SDK is installed.

## What it does not do

It writes no study code: a new study's entry point and modules are the owner's,
written between registration and run, reviewed and frozen as the registrations
describe. It does not commit, push or publish. It does not decide a status: the
study's code does, and the harness only checks that what is recorded is what
the code wrote.
