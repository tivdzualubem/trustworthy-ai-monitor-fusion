# Targeted-review handoff

The immutable input is the 200-row `results/stage_a_targeted_review_v1/review_queue.csv` from commit `5563b49d9603a57cc24b8ff93eaea6ffd994fcc6`. The handoff contract pins its exact table values and the frozen ontology/crosswalk. Only CSV record line-ending differences are ignored by the table fingerprint. The original queue must not be edited or rebuilt for this handoff.

## Reviewer workflow

1. Use the separate, blank `results/stage_a_targeted_review_handoff_v1/reviewer_worksheet.csv`. To create another working copy, run:

   ```bash
   python scripts/validate_stage_a_targeted_review.py prepare --worksheet /path/to/my_reviewer_worksheet.csv
   ```

   Preparation never overwrites an existing file. All original columns, including provenance, text, hash, screening category and screening reason, are preserved. The script performs no review and fills no review fields.

2. Review only these 200 rows against `configs/external_validation_common_ontology_v1.json`. Edit only the five review fields:

   - `review_decision`: exactly `include` or `exclude` for every row.
   - `reviewed_category`: for an included row, exactly one of `C1` through `C8`, based on explicit primary-category review rather than the screening hint.
   - `review_rationale`: the included row's text-specific reason for satisfying that category.
   - `reviewer`: the included row's reviewer identifier.
   - `independent_intent_group`: the included row's underlying-intent identifier. Check overlap across all sources and categories. Include at most one row per underlying intent; exclude duplicates rather than assigning artificial distinct IDs.

   Exclude mixed, ambiguous, benign, defensive, or otherwise ineligible cases. For excluded rows the final category and group may stay blank; recording a rationale and reviewer is recommended. Do not turn an exclusion into an inclusion to meet a quota. Do not edit source/provenance fields, prompt text, hashes, IDs or screening evidence. Save as UTF-8 CSV; reordering rows/columns is allowed but deleting/adding them is not. Quoted commas and embedded text must remain intact.

   Differences between `reviewed_category` and `proposed_screen_category` are logged explicitly and both values are preserved. Included counts use the submitted reviewed category. The frozen BeaverTails eligibility rule remains in force: an other-primary-category BeaverTails case must be excluded, not reassigned into an accepted non-C5 pool. No category is inferred by the validator.

3. Validate the completed worksheet:

   ```bash
   python scripts/validate_stage_a_targeted_review.py validate
   ```

   For a different working copy, add `--worksheet /path/to/my_reviewer_worksheet.csv`. Reports go to `results/stage_a_targeted_review_handoff_v1/validation/` by default; `--out-dir` changes that location.

4. Read `validation.txt` and `validation.json`. The JSON contains validation errors, category differences, counts and shortfalls by reviewed category, and source/worksheet fingerprints. Fix errors through explicit review. A shortage requires more review or a future authorized screening extension, never fabricated decisions. C3 currently has no reserve beyond its 25 queued candidates.

## Validation and outputs

The validator requires the exact 200 frozen IDs and unchanged non-review cell values. Included cases require all specified review metadata. Independent-group identifiers are compared after case folding and whitespace normalization; any repeated included group blocks selection. The validator checks submitted metadata and structural consistency, not the scientific correctness of a human judgment or semantic independence of differently named groups.

- **Exit 2:** malformed/incomplete review, altered input, duplicate included intent, or fewer than 13 eligible independent intents in any category. No selection is emitted. Any previous `selected_base_intents.csv` in the specified output directory is removed before validation, preventing stale success from surviving a failed rerun.
- **Exit 0:** every category has at least 13 valid included intents. `selected_base_intents.csv` contains exactly 13 per reviewed category (104 total), ordered by the existing SHA-256 seed/group convention with seed 20260921. The selected rows retain the complete review, provenance and original screening category. `validation.json` records the selection hash and IDs.

The selection is a development-data candidate selection pending severity review, formulation/pair generation, and semantic-equivalence review. Each selected row and the report keep all downstream gates false: no pair generation, severity/equivalence review, guard scoring, q estimation, W0, fresh confirmation, or prospective design freeze is authorized. This workflow does none of those activities.

The 699 candidates outside the queue remain unreviewed and receive no decisions. Do not use the old full-pool `select_stage_a_public_base_intents.py --decisions` interface for this worksheet. Do not create dummy exclusions for the rest of the pool.

## Verification

```bash
python -m unittest discover -s tests -p 'test_stage_a_targeted_review*.py'
```

Success-path tests use synthetic 200-row fixtures, not scientific reviews of the actual queue. The real worksheet is delivered with all five review fields blank and therefore correctly fails validation until the explicit review is completed.
