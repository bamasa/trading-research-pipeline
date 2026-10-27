# CLAUDE.md: rules for any agent in this repository

A research pipeline whose studies are registered before their data is read and
whose held-out blocks are opened once, through a ledger. These rules bind any
agent, run by hand or through the harness in [`agents/`](agents/README.md),
whose loop is [`agents/protocol.md`](agents/protocol.md).

Before anything else, read: the README's status and roadmap,
[`docs/findings.md`](docs/findings.md), the registration of the study at hand
under `docs/preregistration/` with its YAML under `configs/`, the study's run
log `docs/<study>_run_log.md`, and [`agents/protocol.md`](agents/protocol.md).

## Binding rules

1. **Nothing is published.** No `git push`, no `git remote`, no pull request,
   nothing that changes a remote. `main` is never checked out, merged into or
   edited. Work happens on a local branch in its own worktree; a branch held by
   the main checkout or by another worktree is never checked out or edited
   (reading it with `git show <branch>:<path>` is allowed). A role of the
   harness never commits: the owner reviews and commits what it hands back.
2. **Frozen study code is never edited.** Existing files under `src/`,
   `experiments/`, `configs/` and `docs/preregistration/`, and `pyproject.toml`
   and `uv.lock`, belong to studies whose ledgers check the tree: a change to
   any of them blocks a held-out read. New work goes into new files, and no
   dependency is added.
3. **A held-out day is reached only through the ledger.** No day of a block
   that is held out, boundary, pristine, prospective, confirmation, excluded or
   embargo is fetched, opened or computed on by any route unless the owner
   granted that block for the run and its open entry is committed. A month
   archive that holds such a day counts as that day.
4. **Nothing that decides a status changes after a freeze.** Fees, thresholds,
   fill rules, kill conditions, blocks, the family and its m are fixed by the
   registration and its D amendment; after the code freeze the study code is
   the frozen commit's, and a correction is a dated amendment made before the
   read.
5. **Provenance.** The owner's earlier work is cited only as "the author's 2025
   research". No file or commit names another organisation, or anyone's role,
   contract or sign-off.
6. **Language and attribution.** English. No emoji. No `Co-Authored-By` or
   other trailer line in a commit message, and no file, commit, comment or
   note credits an assistant, a model or a tool as an author.
7. **Secrets.** API keys and model names come only from environment variables,
   read at run time. Nothing secret is committed, logged or written to a run
   directory, and no absolute local path, email address or IP address is
   committed either (`scripts/audit_repo.py` checks).
8. **Size and checks.** One piece at a time, edits under about 120 lines, a
   commit after each piece. `ruff format`, `ruff check`, `mypy` and the tests
   pass before handing back. A foreground command finishes in about five
   minutes; a longer stage is launched by the owner and followed with `status`.

## The research protocol

- **Registration before data.** A study is a document under
  `docs/preregistration/` and a YAML under `configs/` that agree. Values set
  on the development block are `null` placeholders commented "frozen by the D
  amendment"; the document records the null-placeholder sha256.
- **Hypothesis and kill condition first.** Each hypothesis is written with its
  metric, kill conditions, placebos and expected outcome, and committed, before
  the run that tests it. A plan without them is refused, not completed.
- **Every run is logged.** The study's run log gets a dated entry; the trial
  ledger counts every configuration tried; the harness keeps a run directory.
- **Amendments are dated and appended.** None cites a number seen on the data
  as its reason, except a fault it names; a correction before a read re-runs
  what it could change.
- **No re-ranking after a read.** The family, Holm's m, the order of the kills
  and the headline sentences are fixed in advance; a defect found after a read
  can only lower a status; a corrected run is a successor registration.
- **Results come with their kills, placebos and neighbourhoods.** A number at
  any tier but the verdict tier is a measurement, never the strategy working;
  a development number is never quoted as a result.
- **Negative results are results.** A killed hypothesis is recorded in
  `docs/findings.md` with the same care as a candidate.

## Commands, and what each may read

| Command | Reads | Rule |
|---|---|---|
| `trading-research describe-schema` | nothing | prints a data contract |
| `trading-research generate-demo-data` | nothing; writes synthetic data to `data/demo` | synthetic, never evidence about a market |
| `trading-research synthetic` | nothing; writes generated data and its `truth.json` under `data/synthetic/` | synthetic, never evidence about a market |
| `trading-research verify` | nothing; generates its own data, writes pages under `artifacts/verify/` | checks code against planted answers, never a market |
| `trading-research validate-data`, `screen`, `review` | files already on disk | no network |
| `trading-research prepare`, `select`, `train`, `predict`, `backtest`, `retrain-search`, `exit-search` | the pipeline's staged files | the window each is given, nothing else |
| `trading-research download`, `download-book` | public archives over the network | free days only, written under `data/<study>/`; `fundingRate` and `klines` come as whole months |
| `trading-research breaks` | the `--prices` file it is given | fetching bars would write the shared `data/bars` |
| `trading-research discover` | the venue's current listing | no historical data |
| `trading-research reproduce --no-fetch` | the panel already on disk | fetching is the owner's call |
| `trading-research mm-backtest` | the market-making study's book, prints and funding | `--allow-heldout` only on the owner's grant; `--second-read` never |
| `data`, `regimes`, `instruments`, `signal`, `search`, `backtest`, `study`, `agent`, `demo`, `fees` | as the commands they group | the `trp` command tree: the commands above grouped by step, under the same names; a role runs the flat names, which the policy knows |
| `trading-research paper` | public market-data streams (`run`, `record`) or a stored day (`replay`, `parity`, `status`) | public endpoints only, no key and no order; recordings under `data/paper/`, runs under `runs/paper/` |
| `python -m experiments.short_horizon_2025` | its blocks, through its access and ledger | stages in order; `open` and `read` only on the owner's grant |
| `python -m experiments.spot_2026` | its blocks, through its access and ledger | as above; `status`, `pages` and `--dry-run` freely |
| `python -m pytest` | synthetic data and fixtures | no network |
| `git log`, `git diff`, `git status`, `git show` | the history | read-only |
| `python -m agents.workflows.research_loop` | the registrations | `calendar`, `null-hash`, `check`, `prompts`, `audit`, `run`; `demo` on synthetic data in a scratch repository |

`python -m agents.workflows.research_loop calendar` prints every protected
range the registrations hold; `check` says whether a command would be allowed
for a role, and why.

## Never

- Edit a frozen file; write by hand under `data/` or `models/`; rewrite a
  results file, a ledger, a run log entry or a page an entry point wrote.
- Fetch, open or read a protected day; run `open` or `read`, or pass
  `--allow-heldout`, without the owner's grant; pass `--second-read`.
- Change a fee, a threshold, a fill rule, a block or a kill after a freeze;
  re-run a block under another rule; choose among results after a read.
- Push, add or change a remote, merge into `main`, rewrite history.
- Chain commands with `;`, `&&` or a pipe: a step that depends on a check runs
  only after the check has exited 0, as its own command.
- Look up market data (prices, funding, volumes) for any date on the web; a
  source is read for fees and formats only, and the date it was read is noted.
- Describe a measurement as the strategy working, or round a number up.

## Handing back

A step hands back a file the code wrote and a sentence the registration fixed,
never a prose claim alone:

1. the path and sha256 of each result file, run record or page, as the entry
   point wrote it;
2. one fenced block tagged `handback`, in YAML, with the keys
   [`agents/protocol.md`](agents/protocol.md) gives for the role; a headline is
   a registered sentence copied verbatim from the page the entry point wrote.

A commit message names the study, the stage, the reading and the verdict.
