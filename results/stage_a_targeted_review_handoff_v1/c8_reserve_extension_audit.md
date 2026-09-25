# Controlled C8 reserve extension audit

## Result

Both approved extensions validate successfully: `selected_pending_downstream_reviews`, 0 validation errors, 104 selected base intents, exactly 13 per category, seed 20260921. Accepted counts: C1=17, C2=17, C3=16, C4=15, C5=16, C6=19, C7=18, C8=14.

Every selected row has `selection_status=selected_pending_downstream_reviews`. All eight authorization flags remain false; severity, formulation/pair generation and semantic-equivalence review remain pending. No downstream experimental work performed.

## Starting state and preservation

The requested branch was correct. Actual starting HEAD was `3f0e6a4922da2ed4d23c373d03e366798385aacd` (complete frozen queue review), a descendant of the supplied `72318fc318936a50724447715604be004d80a7bd`. Its scientific counts matched the request exactly. HEAD and branch were left unchanged.

The full completed 200-row worksheet, its C3 content, the frozen queue, both existing C5 artifacts, and the old validation outputs were preserved. The only modified tracked implementation file is the validator. The pre-existing stale-test modification remains untouched. No staging, commit, push, reset, checkout, cleanup, or stash was performed.

Frozen queue canonical fingerprint remains `f1704482b35fbe6e5c79304733e948d2fbd2cefdf76a498a125d9a45c47b0937`. Existing C3-screened record fingerprint remains `f417e31d44a442f36c4c4f6605c06b56801d35e70e849a0d85423782ac7ad155`. The C8 contract additionally pins all 200 reviewed records and the original worksheet bytes.

Before/after SHA256 checks (each pair identical):

- `results/stage_a_targeted_review_v1/review_queue.csv`
  - Before: `e2b0c230710b32ccc90e26313ec43ac27d1d89ecf8b33c7d801c7bf3d92851e7`
  - After: `e2b0c230710b32ccc90e26313ec43ac27d1d89ecf8b33c7d801c7bf3d92851e7`
- `results/stage_a_targeted_review_handoff_v1/reviewer_worksheet.csv`
  - Before: `1dadb3aa7a8df73a60ec152acf3a5c75eb739d32e2d3604023f5887cdea4eb88`
  - After: `1dadb3aa7a8df73a60ec152acf3a5c75eb739d32e2d3604023f5887cdea4eb88`
- `configs/stage_a_c5_reserve_review_extension_v1.json`
  - Before: `ffaff10c573230aeb57104b314f47e24e9bc50ed219267a5965702fcde1eeb07`
  - After: `ffaff10c573230aeb57104b314f47e24e9bc50ed219267a5965702fcde1eeb07`
- `results/stage_a_targeted_review_handoff_v1/c5_reserve_reviewer_worksheet.csv`
  - Before: `86c4c7fd357159e5bd895747c35cae247eea08778524d241605f06770cac8820`
  - After: `86c4c7fd357159e5bd895747c35cae247eea08778524d241605f06770cac8820`
- `tests/test_external_validation_cluster_design_envelope_screen.py`
  - Before: `004bf5d7d2cc41388b2e3c2094c3104e125b7f757232b680380f2314bb6027b5`
  - After: `004bf5d7d2cc41388b2e3c2094c3104e125b7f757232b680380f2314bb6027b5`

## Approved C8 batch and provenance

Exactly four separately recorded C8 reserve rows, all reviewed by `project_author`:

| Candidate | Approved independent intent group |
|---|---|
| wildguard:357 | c8-unauthorized-private-communications |
| jbb:76 | c8-nonconsensual-location-tracking |
| wildguard:1477 | c8-private-personal-banking-history |
| wildguard:906 | c8-bulk-customer-financial-identity-exposure |

All four IDs were verified in `audit.json` at `by_category.C8.reserve_candidate_ids` and outside the frozen queue before decisions were written. Exact texts, labels, normalized text hashes and source revisions were checked against the existing `results/stage_a_public_selection_v1/review_template.csv`. Source provenance matches the existing source audit and the reserve audit. The new contract pins the candidate snapshot table and source audit separately.

Pinned JBB revision: `886acc352a31533ffbcf4ef22c744658688086fc`. Pinned WildGuard revision: `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`. These are public/development base-intent candidates; human formulation provenance is not established. No monitor outcomes were retrieved or used. All other C8 reserve rows remain unreviewed.

## Implementation and validation

Shared contract validation now handles multiple approved batches; source-specific checks retain the BeaverTails restriction and validate C8 against the pinned JBB/WildGuard snapshot. Single-Path C5 API calls and single C5 CLI usage remain supported. Repeated contracts and overlapping candidate IDs fail closed. Altered/incomplete extensions, changed reviews or inputs remove stale selection output. C5 artifacts are also pinned by the C8 contract.

C5 only: C5=16, C8=10, selection blocked. C8 only: C5=7, C8=14, selection blocked. Both: 104 selected with the accepted counts above. Repeated validation and reversed contract order produced byte-identical selection output.

```sh
.venv/bin/python -m pytest -q tests/test_stage_a_targeted_review_handoff.py tests/test_stage_a_targeted_review_queue.py tests/test_stage_a_targeted_review_c5_extension.py tests/test_stage_a_targeted_review_c8_extension.py
# 49 passed in 2.95s

git diff --check
# passed

.venv/bin/python scripts/validate_stage_a_targeted_review.py validate \
  --extension-contract configs/stage_a_c5_reserve_review_extension_v1.json \
  --extension-contract configs/stage_a_c8_reserve_review_extension_v1.json \
  --out-dir results/stage_a_targeted_review_handoff_v1/c8_extension_validation
# exit 0; selected_pending_downstream_reviews; selected_count=104; errors=0
```

New tests cover exact reserve IDs/provenance, queue/worksheet/C3 preservation, C5 compatibility, single/multiple extension coverage, deterministic selection, repeated CLI arguments, tampered/missing/incomplete extensions, stale-selection removal, duplicated/relabelled intents, source integrity, and all closed gates/pending statuses. The prior 35 tests are unchanged and pass alongside 14 new tests.

## Exact intended changed/new files

- `scripts/validate_stage_a_targeted_review.py`
- `configs/stage_a_c8_reserve_review_extension_v1.json`
- `results/stage_a_targeted_review_handoff_v1/c8_reserve_reviewer_worksheet.csv`
- `tests/test_stage_a_targeted_review_c8_extension.py`
- `results/stage_a_targeted_review_handoff_v1/c8_extension_validation/validation.json`
- `results/stage_a_targeted_review_handoff_v1/c8_extension_validation/validation.txt`
- `results/stage_a_targeted_review_handoff_v1/c8_extension_validation/selected_base_intents.csv`
- `results/stage_a_targeted_review_handoff_v1/c8_reserve_extension_audit.md`

The package `c8-reserve-extension.zip` contains exactly these eight files, preserving repository-relative paths. It excludes existing C5 artifacts, the completed original worksheet, old validation results, unrelated tests, and all other repository files.
