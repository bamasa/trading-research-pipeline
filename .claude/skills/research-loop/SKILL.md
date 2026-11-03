---
name: research-loop
description: Run, resume or inspect this repository's agent research loop (propose, register, run, review, verdict, record). Use when asked to start a study with the harness, resume a stopped run, grant a block, see the protected calendar, check whether a role may run a command, or audit a run directory.
---

# The research loop

The loop is `agents/workflows/research_loop.py`; its steps, artefacts and gates
are in `agents/protocol.md`, and the exact commands in `docs/agents.md`. Read
`CLAUDE.md` first: its rules bind this skill.

## Before a run

1. Print the protected calendar, so no block of the new study touches a
   protected day:

   ```bash
   PYTHONPATH=src python -m agents.workflows.research_loop calendar
   ```

2. Check that every command the agents' documents name exists:

   ```bash
   PYTHONPATH=src python -m agents.workflows.research_loop prompts
   ```

3. Ask the owner for the brief: the question, the venue, the instruments, the
   years, and the last date they looked at the data of those instruments.
   Write it to a file outside the repository's tracked paths.

## A run

The loop runs from its own environment, `agents/.venv-agents/`, never the
study's: an extra package in the study's environment changes the ledger's
fingerprint and blocks a held-out read. `docs/agents.md`, "A live run", has
the setup. Then:

```bash
PYTHONPATH=src agents/.venv-agents/bin/python -m agents.workflows.research_loop run --study <study> --brief-file <brief.md>
```

The run stops at every gate the owner holds and prints why. Report the stop to
the owner in one sentence and wait: committing the registration, writing and
freezing the code, the D amendment and granting a block are the owner's.
After the owner has acted:

```bash
PYTHONPATH=src agents/.venv-agents/bin/python -m agents.workflows.research_loop run --resume <run> --grant <study>:H
```

Pass `--grant` only for a block the owner granted in so many words.

## After a run

Audit the run directory before anyone shares it:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop audit <run>
```

Read `stops.jsonl`, then each step's `prompt.md`, `reply.md` and
`handback.yaml`, then `verdict.json`. The run directory is ignored by git; what
may be committed is listed in `docs/agents.md`, "What is published".

## A question about the policy

```bash
PYTHONPATH=src python -m agents.workflows.research_loop check --role <role> "<command>"
```

## Without a model

The demonstration runs the whole loop on a synthetic study with a scripted
client, in seconds, in a scratch repository:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop demo
```
