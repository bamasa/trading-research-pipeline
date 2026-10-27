# Code freeze for the 2026 spot maker study

Date: 2026-10-28.

Code-freeze commit: `4b5d7d3bf902619da68322c31e97e0e517fedff5`

Every piece of study code the registration names — the entry point
`experiments/spot_maker_2026.py` and the package under
`src/trading_research/spot_maker/`, with the modules it reuses by import — is
at that commit, together with `configs/spot_maker_2026.yaml` (null-placeholder
sha256 `aa367b64…`, recorded under the registration's amendments),
`pyproject.toml` and `uv.lock`. No day file of BTC/FDUSD has been copied,
opened or computed on by this study, and no number of the fill rule's
calibration has been computed. Any change to that code before H is read is a
correction under the registration's amendment protocol; the ledger refuses to
open a block on code that differs from it.

Before the freeze, at the same code:

- the package's tests (69, the dry run among them) passed in the spot study's
  environment;
- the dry run (`python -m experiments.spot_maker_2026 --dry-run`) ran every
  stage and H's read end to end on stand-ins (the spot study's synthetic venue
  converted by its parser, a stand-in model in place of S25-PF), in a temporary
  repository holding the registration as registered: stage 0's checks passed
  (about 31,000 resting orders re-filled per stand-in day, no violation, no
  void condition), the amendment froze the configuration it records, H was
  opened, copied, committed, computed and closed through the ledger, and every
  gate refused a second open, a closed session and P without a candidate. Its
  numbers are a test of the code and are discarded.
