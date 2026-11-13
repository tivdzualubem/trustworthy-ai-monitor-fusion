# Trustworthy AI — Phase 14 closure and Phase 15 independent reference study

**Date:** 2026-11-13  
**Status:** Phase 14 numerical optimizer issue CLOSED on aggregate evidence. Phase 15 prospective reference review INFRASTRUCTURE READY, **no new human reviews or native-model outcomes have been generated**.

## Why this is the next decisive task

The restricted MultiPPI allocator is now numerically certified. The simulation says nothing more about real assistant-response harmfulness. The primary unsolved issue is that the real 96-case, 336-opinion development study does not provide independently verified response-level Y. Rather than reopen R2-only cases or run another synthetic sweep, this package prepares a **fresh probability-sampled, independent, blinded reference study**. It does not imply that the historical pool is prospective data.

## Contents

- `phase14_verified_closure.json`: read-only evidence from the two **uploaded** aggregate Phase 14 files; 1,200 certified repaired solutions, zero budget violations, no meaningful estimator RMSE gain. Only aggregates and file hashes.
- `close_phase14.py`: verify both Phase14 aggregate JSONs without private data. Already executed in preparing this package.
- `phase15_reference_study.py`: separate `template`, `prepare`, `adjudication`, `summarize` operations. Uses only Python standard library and the SciPy interval function below.
- `exact_finite_population_interval.py`: Phase14's mathematically unchanged exact hypergeometric SRSWOR reference-endpoint interval function.
- `test_phase15.py`: nine fixture-only tests; no private data accessed during tests.
- `METHOD_FIDELITY_AND_CLAIMS.md`: what methods/estimands we can, and cannot, claim.

## What must exist before preparing human packets

1. A **complete and permission-approved** fixed CSV frame of native prompt/assistant-response/monitor decisions, collected and frozen **before** sampling. Required columns: `case_id,prompt,response,monitor_decision,source`. `monitor_decision=0` means the monitor **allowed** the response, `1` means it blocked. The frame cannot contain only cases selected for human review. The operational target is the response in prompt context. Never upload this CSV or raw texts to ChatGPT, GitHub or unsecured cloud.
2. The original complete `POLICY_FOR_REVIEWERS.json` beside the private reviewer database, including `target_policy.system_policy` and C1–C8 ontology. The output policy hash is checked against the frame freeze manifest.
3. A frozen, truthful metadata JSON with `frame_type=prospective_new` only if a genuinely new native target-assistant/monitor stream exists, otherwise `historical_pilot`. It must record exact source scope, assistant and native-monitor versions, thresholds, date window and reviewer permissions. `authorized_review=true` must be **genuinely authorized**; do not set it merely to pass a script check.
4. **Two actually independent qualified human reviewers** with local authorized access, plus a qualified third adjudicator for every disagreement/Uncertain and a 10% pre-randomized quality-audit subset. They must not see historic decisions or each other's labels. New automation cannot substitute for their judgments.

The 96 historical study cases, stored in the 96-case private `cases_private.sqlite`, are **not** a full 1,687-response sampling frame. Do not convert that database into the prospective CSV. A new stream must be acquired and frozen using the legitimate target assistant and native monitor if the claim concerns deployment.

## Minimal usage on WSL after ZIP extraction

From the extracted folder, first run tests:

```bash
PY="$HOME/trustworthy-ai-monitor-fusion/.venv/bin/python"
"$PY" -m unittest -q test_phase15.py
```

If you already have a candidate full frame, inspect **aggregate structure only** without creating any sampling or exposing private content:

```bash
"$PY" phase15_reference_study.py inspect \
  --frame "$PRIVATE_DIR/native_prospective_frame_PRIVATE.csv"
```

This command reports N, allowed M, duplicate count and file hash only. It cannot prove that a frame is new/prospective.

Create the private freeze **template** (does not sample cases or generate any reviewer packets):

```bash
PY="$HOME/trustworthy-ai-monitor-fusion/.venv/bin/python"
"$PY" phase15_reference_study.py template \
  --policy "$PRIVATE_DIR/POLICY_FOR_REVIEWERS.json" \
  --out "$PRIVATE_DIR/phase15_freeze_template_PRIVATE.json"
```

Edit the template locally once a genuinely frozen frame has been collected. Replace all `FILL_` fields, choose the actual `frame_type`, and attest authorization only with real permission. Do not upload the completed template if it contains private provenance.

Then prepare an **independently selected** sample of **200 unique monitor-allowed responses** (only after all prerequisites are met):

```bash
"$PY" phase15_reference_study.py prepare \
  --frame "$PRIVATE_DIR/native_prospective_frame_PRIVATE.csv" \
  --freeze "$PRIVATE_DIR/phase15_freeze_PRIVATE.json" \
  --policy "$PRIVATE_DIR/POLICY_FOR_REVIEWERS.json" \
  --sample-n 200 --audit-fraction 0.10 \
  --out "$PRIVATE_DIR/phase15_reference_PRIVATE"
```

Give each qualified reviewer **only** their corresponding `reviewer_A_PRIVATE` or `reviewer_B_PRIVATE` folder, using authorized local access. They complete `answers_PRIVATE.csv` without knowledge of prior judgments. The organizer must keep `organizer_ONLY_PRIVATE` secret. For Harmful, require C1–C8 and evidence specific to the assistant response. Uncertain stays explicit.

After the independent reviewers finish:

```bash
"$PY" phase15_reference_study.py adjudication \
  --private-dir "$PRIVATE_DIR/phase15_reference_PRIVATE"
```

Only the qualified third adjudicator gets `reviewer_C_PRIVATE`. After that reviewer finishes, compute the **aggregate-only** output:

```bash
"$PY" phase15_reference_study.py summarize \
  --private-dir "$PRIVATE_DIR/phase15_reference_PRIVATE" \
  --out "$PRIVATE_DIR/phase15_reference_AGGREGATE.json"
```

**Only `phase15_reference_AGGREGATE.json` is eligible for upload** (after local inspection). All casebooks, mapping, sample seed, original IDs, explanations, reviewer answers and source texts stay private. If the stream, permission or independent reviewers do not exist, stop at the template and report Phase15 as awaiting external evidence.

## Statistical target and costs

For a frozen frame of `N` responses with `M` allowed and a probability SRSWOR review sample of size `n` from the `M`, each allowed case has sampling probability `pi=n/M`. For a **fixed, fully adjudicated** operational binary protocol endpoint `Z*`, the missed-harm prevalence is `A_Z*=M/N × (K/M)=K/N`, where `K` is the total Harmful count among the allowed frame. Exact hypergeometric inversion yields a conservative 95% interval for `K/N`. If `u` selected review outcomes remain Uncertain, report the wider lower/upper endpoints obtained by assigning all `u` Safe for the lower and Harmful for the upper. The interval **does not validate reviewer accuracy or automatically identify latent true Y**.

Human review is charged in actual active seconds from both independent reviewers and the adjudicator. Training, organizer coordination, response generation, and model inference remain separate ledger components; elapsed GPU seconds cannot be equated with wages without explicit economic assumptions. With the historical mean ~47.16 seconds per judgment, 200 dual reviews imply ~5.24 hours of active review **before adjudication**, merely for planning; no human review has occurred here.

## Known protocol limitations (do not hide)

- The code verifies frozen file hashes, case IDs, consistency and exclusion of raw texts from exported aggregates. It cannot verify that `prospective_new` is truthful, that reviewers are qualified/independent, that policy is normatively correct, or that all population sources were included. Those are external evidence requirements.
- A predefined 10% audit does not establish reviewer sensitivity or specificity. Two matching judgments can both be wrong; third-party disagreement handling is an **operational definition**, not gold truth.
- The exact SRSWOR interval is for a pre-frozen fixed reference endpoint, not for human responses whose values would change if reviewers or instructions changed. This stability assumption requires explicit reporting.
- The tool does **not** evaluate the official MultiPPI/AM-PPI algorithms; do not claim author-equivalent benchmarks from restricted Phase11/14 adaptations.
- Duplicate response templates and source clustering still matter for source/deployment generalization even if fixed-frame SRSWOR yields a valid finite-frame estimator. Declare these dependencies before any transport claim.
- No GitHub changes or research review contact are performed by the toolkit.
