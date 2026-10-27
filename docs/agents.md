# The agent harness

The harness lets an LLM agent run this repository's research loop under the
discipline the registered studies followed by hand. A hypothesis and its kill
conditions are written before any data is read; the registration is committed
before the run; a held-out block is opened once, through its ledger; the
statuses are computed by the study's code; and a result is reported only in a
sentence the registration fixed. The harness adds no study code and changes
none. It runs the commands that exist, decides every tool call through a
policy, checks every gate it can check mechanically, stops at every gate the
owner holds, and logs every step.

Its parts:

| File | What it is |
|---|---|
| [`CLAUDE.md`](../CLAUDE.md) | the binding rules for any agent in this repository, read first by every role |
| [`agents/protocol.md`](../agents/protocol.md) | the loop: propose, register, run, review, verdict, record; each step's artefact and gate |
| [`agents/roles/`](../agents/roles/) | one prompt per role |
| [`agents/workflows/policy.py`](../agents/workflows/policy.py) | the tool policy: what each role may run, read and write |
| [`agents/workflows/prompts.py`](../agents/workflows/prompts.py) | each role's system and step prompts, and the check of its handback |
| [`agents/workflows/research_loop.py`](../agents/workflows/research_loop.py) | the orchestrator, its gates, the code's verdict, the SDK runner and the command line |
| [`agents/workflows/demo.py`](../agents/workflows/demo.py) | the demonstration: a scratch repository, a scripted client in place of the model, the owner's actions |
| [`agents/demo/`](../agents/demo/) | the synthetic study the demonstration registers and runs |
| [`docs/agents_demo_funding_carry.md`](agents_demo_funding_carry.md) | how the loop would run the roadmap's funding-rate carry study; nothing of it has been run |

## The loop

| Step | By | Produces | Checked by the orchestrator | Held by the owner |
|---|---|---|---|---|
| propose | researcher | hypotheses, each with metric, kills, placebos, expected outcome; the information set; the blocks | the handback has every key; the turn changed nothing | the brief |
| register | registrar | `configs/<study>.yaml`, `docs/preregistration/<study>.md`, the null-placeholder sha256 | the handback's hash; the turn changed only those two files | the commit of the registration |
| (code) | the owner | the entry point, its modules and tests, the code-freeze note | no role writes code | the code and its freeze |
| run | executor | the entry point's records, verbatim; after a read, the results file, ledger and page | before it: both files committed, unchanged, the document holding the YAML's null hash; after it: the tree changed only where an entry point writes | the D amendment, the grant of a block, its open entry, the fetch records |
| review | skeptic | three lenses, each with checks and a refutation; findings with `file:line` | all three lenses named; a blocking finding stops the loop | reading a blocking finding |
| verdict | code | `verdict.json`: statuses and first kills from the results file | the results file's sha256 is the one the ledger recorded for the first read | |
| record | scribe | run log entry, findings row, README row | every sentence is a registered headline sentence and stands verbatim in a file the scribe changed; the turn changed only the scribe's files | the commit of the record |

A stop is written to the run directory with its reason. The owner acts and
resumes the run; the step that stopped is taken again, and its earlier attempt
is kept whole under `superseded/`. Nothing is published at any step.

## Re-running it on synthetic data, in minutes

No model, no key, no network and no market data are needed. From the
repository's root, in the study's environment:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop demo
```

The command builds a scratch git repository under `agents/runs/demo-<UTC
time>/` (ignored by git; `--into <empty dir>` puts it elsewhere). It holds a
copy of every registration under `configs/`, so the protected calendar is the
real one, and the synthetic carry study of [`agents/demo/`](../agents/demo/),
whose daily series are drawn from seeds its registration names. A scripted
client stands in for the model: it acts each role as the role's prompt says,
and every tool call it makes goes through that role's policy, as a model's
would. The study's entry point, the helpers and read-only git run for real;
the allowed writes are made. Each role also tries what it must never do: a
push, a remote, a commit, a write to `src/trading_research/spot/` or
`src/trading_research/pipeline/short_horizon/`, a fetch of a day of
`spot_2026:H` and of the study's own H, `open H` before the grant and
`read H` before the open entry is committed. The policy decides and logs each
of these, and none is carried out. Where the loop stops for the owner, the
demonstration acts as the owner: it commits the registration with the entry
point and its code freeze, then the D amendment with the grant of H, then H's
open entry, and resumes.

It prints a summary in JSON: `status` (`complete`), the three stops and what
the owner did at each, the verdict, every refused attempt with its reason
(`allowed_but_never` must be empty), and the run directory's files. The run
directory is `<scratch>/agents/runs/<UTC time>-synth_carry/`; read
`stops.jsonl`, then each step's `prompt.md`, `reply.md`, `events.jsonl` and
`handback.yaml`, then `verdict.json`. The scratch repository's `git log` holds
the owner's four commits and nothing else.

The tests run the same loop:

```bash
PYTHONPATH=src python -m pytest tests/test_agents_policy.py tests/test_agents_loop.py tests/test_agents_prompts.py tests/test_agents_demo.py tests/test_agents_demo_run.py tests/test_agents_sdk.py
```

`tests/test_agents_demo_run.py` runs the demonstration end to end and checks
each refusal; `tests/test_agents_loop.py` drives the orchestrator with a fake
client through every stop; `tests/test_agents_sdk.py` is skipped unless the
optional SDK is installed (see below).

## The helpers

They need no model and no SDK, and answer for the repository the command runs
in:

```bash
PYTHONPATH=src python -m agents.workflows.research_loop calendar
PYTHONPATH=src python -m agents.workflows.research_loop check --role executor "git push"
PYTHONPATH=src python -m agents.workflows.research_loop null-hash configs/spot_2026.yaml
PYTHONPATH=src python -m agents.workflows.research_loop prompts
```

`calendar` prints every range a registration protects; `check` says whether
the policy would let a role run a command, and why; `null-hash` prints a
YAML's null-placeholder sha256 by the repository's own `unfrozen_text` and
`sha256_bytes`; `prompts` checks that every command the agent's documents name
exists, with its options.

## A live run

The SDK (`claude-agent-sdk`) is kept out of the study's environment on
purpose. Each study's ledger fingerprints every distribution installed in the
environment that runs its entry point, and a read session refuses unless the
fingerprint equals the one its block was opened with: one extra package
installed into the study's environment would block a held-out read. The loop
therefore runs from a small environment of its own, and the agent's commands
run in the study's:

```bash
uv venv agents/.venv-agents --python 3.12
uv pip install --python agents/.venv-agents/bin/python claude-agent-sdk pyyaml
export TRP_AGENT_MODEL=<model name>
export TRP_AGENT_PYTHON=.venv/bin/python
export TRP_AGENT_CLI=.venv/bin/trading-research
PYTHONPATH=src agents/.venv-agents/bin/python -m agents.workflows.research_loop run --study funding_carry_2026 --brief-file <brief.md>
```

`agents/.venv-agents/` is ignored by git. Call its interpreter by its path;
do not activate it, so that a bare `python` never resolves to it.
`TRP_AGENT_MODEL_<ROLE>` overrides the model for one role,
`TRP_AGENT_MAX_TURNS` bounds a role's turn (40 by default) and
`TRP_AGENT_MAX_BUDGET_USD` its spend. The SDK reads its credentials from the
environment itself; the harness never reads, logs or stores a key. Before any
model is called, the run checks its documents' commands with
`TRP_AGENT_PYTHON`.

The run stops at every gate the owner holds and prints why. The owner acts
(commits the registration, writes and freezes the code, commits the D
amendment, commits an open entry or fetch records, reads a blocking finding)
and resumes, granting a block when the next step reads one:

```bash
PYTHONPATH=src agents/.venv-agents/bin/python -m agents.workflows.research_loop run --resume agents/runs/<run> --grant funding_carry_2026:H
```

`tests/test_agents_sdk.py` checks the runner against the installed SDK
without starting a session (its options, and the policy as its hook and its
tool callback). It runs in the loop's environment, where the study's
`tests/conftest.py` cannot be imported:

```bash
uv pip install --python agents/.venv-agents/bin/python pytest
PYTHONPATH=src agents/.venv-agents/bin/python -m pytest tests/test_agents_sdk.py --noconftest -p no:cacheprovider
```

## The roles

Each role's system prompt is `CLAUDE.md`, then `agents/protocol.md`, then its
own file; its step prompt carries the brief, the registration's paths, the
grants, the interpreter to use, the earlier steps' handbacks, the gate it must
pass and, on a resumed step, why the earlier attempt stopped. A reply counts
only through its last fenced `handback` block, which the orchestrator checks.

| Role | Prompt | Runs | Writes | Its handback must carry |
|---|---|---|---|---|
| researcher | [`researcher.md`](../agents/roles/researcher.md) | `describe-schema`, an entry point's `status`, read-only git, the helpers | nothing | per hypothesis `id`, `claim`, `metric`, `kills`, `placebos`, `expected`; `known`; `blocks`; a `reason` when declining |
| registrar | [`registrar.md`](../agents/roles/registrar.md) | as the researcher; a fee page from a host the owner listed | its study's YAML and document until the code freeze, then appends only | `null_sha256`, or `missing` when refusing |
| executor | [`executor.md`](../agents/roles/executor.md) | the study's entry point in order, the CLI on free days, `pytest`, read-only git | nothing by hand; its commands write their outputs | `commands`, `records`; `waiting_on` when waiting; `results`, `ledger`, `page`, `block` after a read |
| skeptic | [`skeptic.md`](../agents/roles/skeptic.md) | `pytest`, `review`, `validate-data`, `status`, `--dry-run`, read-only git, the helpers | nothing | `lenses` (leakage, fidelity, execution, each with `checks` and `refutation`); `findings`, each with `refutation` |
| scribe | [`scribe.md`](../agents/roles/scribe.md) | `describe-schema`, an entry point's `status`, read-only git, the helpers | `README.md`, `docs/findings.md`; appends to the study's run log and defects | `sentences`, each a registered headline sentence it wrote |

Every role may read the repository outside `data/`, `models/` and `.git/`,
and nothing outside it. No role commits, pushes, opens a subagent or searches
the web; a role is given only its own tools (`TOOLS_BY_ROLE`).

## Each rule, how it is enforced, and its test

The rules are [`CLAUDE.md`](../CLAUDE.md)'s. A rule the harness cannot check
mechanically is marked as the owner's review.

| Rule | Enforced by | Tests |
|---|---|---|
| Nothing is published; a role never commits | the policy allows only `git log`, `diff`, `status`, `show`; the SDK's disallowed tools | `test_no_role_pushes_touches_a_remote_or_moves_a_branch`, `test_a_turns_options_load_no_settings_and_give_the_role_only_its_tools`, `test_the_scratch_repository_has_only_the_owners_commits_and_no_remote` |
| Frozen study code is never edited | the policy's writable files per role; the orchestrator's check of what each turn changed | `test_no_role_writes_study_code_results_data_or_the_harness`, `test_no_command_writes_study_code_either`, `test_a_turn_that_changes_what_its_role_may_not_stops_the_loop` |
| A held-out day is reached only through the ledger | the protected calendar; downloads of free days only; the grant for `open`, `read`, `--allow-heldout`; `read` only with the open entry committed; no hand reads of `data/` | `test_a_fetch_of_any_day_of_any_registered_block_is_refused_even_under_a_grant`, `test_a_fetch_that_touches_a_protected_day_is_refused`, `test_open_read_and_allow_heldout_need_the_owners_grant_for_that_block`, `test_reading_a_held_out_block_needs_its_open_entry_committed`, `test_reads_stay_inside_the_repository_and_out_of_the_data`, `test_the_calendar_holds_every_unread_or_spent_block_and_no_development_block` |
| Nothing that decides a status changes after a freeze | the registrar appends only after the freeze; the run waits for the registration committed and unchanged with its null hash | `test_the_registrar_writes_its_own_registration_until_the_freeze_then_only_appends`, `test_a_registration_changed_since_its_commit_holds_the_run`, `test_the_run_waits_for_the_owner_to_commit_the_registration_and_resumes` |
| Provenance | the owner's review; the scribe writes only registered sentences | `test_claude_md_states_each_rule` |
| Language and attribution | no role commits; no trailer or credit in any commit or harness file; the rest is the owner's review | `test_no_commit_and_no_harness_file_credits_a_co_author_or_a_tool` |
| Secrets | the model's name from the environment; the key is the SDK's to read; nothing of either in the run directory | `test_the_sdk_is_optional_and_the_model_comes_from_the_environment`, `test_the_run_directory_is_readable_and_logs_every_decision` |
| Size and checks | the owner's (no role writes code) | `test_claude_md_states_each_rule` |
| Hypothesis and kill condition first | the researcher's handback check; the researcher reads no data | `test_a_handback_must_carry_its_roles_keys`, `test_each_role_prompt_states_the_duties_its_gate_enforces`, `test_each_role_runs_only_its_own_commands` |
| Every run is logged | the run directory | `test_the_run_directory_is_readable_and_logs_every_decision`, `test_the_steps_run_in_the_registered_order_and_the_verdict_is_the_codes` |
| Amendments are dated and appended | append-only writes for the registrar after the freeze and for the run log | `test_the_scribe_appends_to_the_run_log_and_edits_only_the_registers` |
| No re-ranking after a read | a finished read is never run again; the verdict is the results file's, checked against the ledger | `test_reading_a_held_out_block_needs_its_open_entry_committed`, `test_the_verdict_on_the_spot_studys_committed_read_is_the_codes` |
| Results with their kills, placebos and neighbourhoods | the skeptic's three lenses with checks and refutations | `test_a_handback_must_carry_its_roles_keys`, `test_a_blocking_finding_stops_the_loop_before_the_verdict` |
| Negative results are results | the scribe must record a read block's sentences | `test_a_refusal_an_unregistered_sentence_or_no_handback_stops_the_loop`, `test_a_sentence_handed_back_but_not_written_stops_the_loop` |
| One command at a time, never chained | the policy's parser | `test_one_known_command_at_a_time_and_nothing_else` |
| No market data from the web | no web search; the downloaders' hosts closed; fee pages only for the registrar, from listed hosts | `test_the_web_is_closed_but_for_the_owners_fee_pages_to_the_registrar`, `test_the_downloaders_hosts_are_all_closed_to_the_web_tool` |

`tests/test_agents_prompts.py` holds this table to the tests: every test it
names exists.

## What is logged

Each run writes `agents/runs/<UTC time>-<study>/`:

- `run.json`: the study, the brief, the grants, the model's name, the commit,
  and each resumption with its grants;
- `system-<role>.md`: each role's system prompt;
- per step, `NN-<step>.prompt.md`, `.reply.md`, `.handback.yaml`,
  `.events.jsonl` (every tool call, the policy's decision and its reason) and
  `.changed.json` (the paths the turn changed, as git sees them);
- `verdict.json`; `stops.jsonl`, one line per stop with its reason;
- `superseded/<UTC time>-NN-<step>/`: every earlier attempt of a step taken
  again.

## What is published, and what is not

The harness publishes nothing: no role pushes or commits, and the run
directories are ignored by git. What the owner may commit after reading a run
is what the studies already commit: the registration and its amendments, the
code and its freeze, the results file, the ledger, the page, the run log and
the findings row. The harness's own files (`CLAUDE.md`, `agents/`, this page
and the tests) carry no key, no model output and no data.

Never published:

- **keys**: never read, logged or stored by the harness; the SDK reads its own
  from the environment;
- **the run directories**: the model's replies and every tool call, with
  absolute paths; they stay on the machine that ran them;
- **data**: nothing under `data/` or `models/` (both ignored), and no
  held-out day, which no role can fetch or read except through its study's
  ledger, on the owner's grant;
- **the loop's environment**, `agents/.venv-agents/` (ignored).

## What a live run still needs

- **A live model run has not been made.** Every test answers the roles with a
  fake or a scripted client. A first run needs the model's name, the SDK's
  credentials in the environment and the SDK installed as above; the options
  were checked against `claude-agent-sdk` 0.2.174 without starting a session.
- **The study's code.** No role writes study code. A new study, such as the
  funding-rate carry study, needs its entry point and modules written,
  reviewed and frozen by the owner between registration and run.
- **The owner at every gate**: the brief, the commits, the grants, and the
  reading of every blocking finding.
- **Limits of the policy.** It decides commands as written; it cannot see what
  a Python entry point does inside, which the entry point's own stage order,
  access and ledger govern. `mm-backtest --allow-heldout` is lifted by the
  grant alone, as the market-making studies' own ledgers then decide.
  Provenance and language are the owner's review.
