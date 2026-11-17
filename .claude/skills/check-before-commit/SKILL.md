---
name: check-before-commit
description: Run the checks this repository's CI runs, locally, before a commit or a pull request, and write the commit message in the repository's style. Use when about to commit, push or open a pull request, or when CI failed.
---

# Checks before a commit

CI (`.github/workflows/ci.yml`) runs four jobs: lint and types, the tests, the
synthetic demo and the repository audit. Run the same locally first, so a pull
request is green on its first run.

## The checks

```bash
make lint
```

```bash
make typecheck
```

```bash
make audit
```

```bash
make demo
```

The whole suite takes about forty minutes; run it in the background and read
its last lines:

```bash
make test
```

While working, run only the tests of what changed:

```bash
PYTHONPATH=src python -m pytest tests/test_agents_prompts.py -q
```

`make fmt` applies the formatter. A file that `make lint` reports as
unformatted is formatted, never excluded.

## Differences from CI

CI runs on Linux with the Python in `.python-version`. A test that passes on
macOS can still fail there: peak memory (`ru_maxrss` survives `exec` on
Linux), exact float comparisons (compare with `pytest.approx`), and timeouts
on a slower runner.

## A failing test

The tests were written for this repository, so a test that is out of date may
be updated. A test that guards a rule is not loosened to make CI green: the
seal on held-out data, the code freeze, and the ban on attribution change only
when the guard itself is wrong, and the commit says why.

## The commit message

One sentence in English that names what changed and why, in the style of
`git log`. No `Co-Authored-By`, `Signed-off-by` or `Generated with` line, and
no mention of an assistant or a tool (`CLAUDE.md`, rule 6): a test reads
every commit message.
