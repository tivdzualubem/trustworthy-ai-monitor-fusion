# Completed label-audit summary

The current worksheet contains an assistant-assisted first-pass adjudication.
It is not treated as final human ground truth until the review-required subset
has been checked by the project author.

## Counts

- Total audited rows: 200
- Proposed label corrections: 19
- Review-required rows: 40
- GCG negative rows: 16

## By audit group

- GCG_negative: n=16, corrections=4, correction_rate=0.2500, audited_positive=4
- JailbreakBench: n=184, corrections=15, correction_rate=0.0815, audited_positive=115

## Confidence

- high: 180
- medium: 20

## Required author review

Review the local file `data/audit_private/label_validity_review_required.csv` before applying the proposed
corrections to the canonical dataset. This file is private and ignored by Git.

The tracked public artifacts contain IDs and decisions only; raw prompt and
response text remains private.
