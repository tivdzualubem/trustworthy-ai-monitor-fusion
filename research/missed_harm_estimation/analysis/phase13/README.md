# Trustworthy AI — Phase 13 audited method fidelity and CI repair

**Date:** 2026-11-12  
**Status:** Code validation complete; **NOT** a full official published-method replication; **NOT** verified latent missed-harm outcome analysis.

## Contents

- `phase11_reference_unmodified.py`: byte-identical Phase 11 benchmark implementation. Never edit old results.
- `phase13_methods.py`: parameter support CI clipping, exploratory normal intervals, convex restricted-MultiPPI allocation certificate, fixed-policy published AM-PPI Eq.(1) Bernoulli estimator and unbiased variance identity.
- `phase13_targeted_interval_recheck.py`: focused recheck of the *existing synthetic comparison* with the same original estimators, annotating original and clipped intervals; checks restricted convex solver optimality gap on each MultiPPI run; writes aggregate JSON/MD only.
- `phase13_reference_validity_gate.py`: **read-only** source-schema/rubric provenance check on private review study, with counts/hash only; does not inspect texts, export IDs or reclassify votes.
- `test_phase13.py` and `test_phase13_reference.py`: ten automated tests.
- `METHOD_FIDELITY_AUDIT.md`, `INTERVAL_VALIDATION.md`: paper references, estimator derivations, assumptions, and do-not-claim limits.
- `MANIFEST.json`: frozen source SHA256 hashes.

## Fast tests

```bash
PY="$HOME/trustworthy-ai-monitor-fusion/.venv/bin/python"
"$PY" -m unittest -q test_phase13.py test_phase13_reference.py
```

## Run once against the private time database (WSL Ubuntu)

Use the original *observed human active seconds* by supplying the database path. Phase 13 does **not** read human label values, case IDs, prompts, or responses during simulation; `phase13_reference_validity_gate.py` reads study counts and a local hash of the original policy snapshot only.

```bash
PY="$HOME/trustworthy-ai-monitor-fusion/.venv/bin/python"
DB="$PRIVATE_DIR/cases_private.sqlite"
BASE="$PRIVATE_DIR/phase13_20261112_v1"

"$PY" -m unittest -q test_phase13.py test_phase13_reference.py

"$PY" phase13_reference_validity_gate.py \
  --db "$DB" --out "$BASE/reference_validity_gate.json"

"$PY" phase13_targeted_interval_recheck.py \
  --db "$DB" --geometry development_1687 \
  --populations 8 --repetitions 25 --budget 4800 \
  --out-dir "$BASE/development"

"$PY" phase13_targeted_interval_recheck.py \
  --db "$DB" --geometry phase6_5000 \
  --populations 8 --repetitions 25 --budget 4800 \
  --out-dir "$BASE/phase6"
```

The default proxy cost values `0.04565729060156249` and `1.5975605696953123` are historical measured *seconds* from Phase 12B, converted into the same assumed-normalized budget using **rho = 1**, not observed monetary prices. They must not be called real economic parity.

**Only upload these aggregate reports:**
- `reference_validity_gate.json`
- `development/phase13_interval_summary.json` and `.md`
- `phase6/phase13_interval_summary.json` and `.md`

Never upload the private DB, complete rubric, individual reviewer durations, local private casebook, or free-text answers. All old code/data are preserved, and GitHub is untouched. Existing output directories are not overwritten.

## Report decision

Once the interval check is verified, we can report restricted-method outcomes as such, with calibrated claims about reviewer-defined endpoints. A fully scientifically warranted latent `A_Y` estimate would require external reference-label validation and representative target-stream sampling, which cannot be manufactured from existing data. No 9D R2-only study is reopened.
