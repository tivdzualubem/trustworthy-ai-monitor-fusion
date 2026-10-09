# Phase 3C hard-benign lexical pilot: researcher approval and factual result

**Date:** 2026-10-09. **Analysis:** exploratory, case-role proxy labels, five frozen pair-ID folds, 38 harmful anchors and 38 new same-topic hard benign candidates; no new native guard GPU scores.

The researcher explicitly reported opening and reviewing all 38 candidate texts, approving them without item-level corrections. This approves the **direct-form candidate wording only**, not the 304 per-guard native-policy judgments or 76 direct/O2 meaning-and-policy equivalence judgments. Independent multi-rater annotation has not occurred. The source pilot ZIP and all its original files are included unchanged in this archive.

## Within-representation OOF balanced accuracy

| Task | Easy: word | Hard: word | Easy: char | Hard: char |
|---|---:|---:|---:|---:|
| direct to direct | 1.0000 | 0.6711 | 1.0000 | 0.7237 |
| O2 to O2 | 1.0000 | 0.6711 | 1.0000 | 0.7237 |
| direct to O2 | 0.5921 | 0.4868 | 0.7895 | 0.5921 |

All 38 new benign texts improve paired word overlap and paired character-trigram overlap over the previously length-matched easy controls. This illustrates how easy negatives can inflate lexical proxy-label accuracy, but low lexical accuracy on the new draft cannot establish a hidden-state-probe advantage or native safety understanding. Stronger hard-negative validity checks remain needed.

## Scientific next gate

Preserve the candidate wordings; use native-policy-specific review on each direct candidate (and transformed variants if ultimately compared), explicitly retain `uncertain`/`out_of_scope`, and adjudicate O2 equivalence. Reuse pre-existing approved labels for identical historical anchor prompts where join and policy match can be proven. Only consider additional guard scoring after this design/label gate and a narrow prospective question. Maintain the October 7 original failed 0/4 screen.

**Source evidence:** the contained `lexical_summary_metrics.csv`, `lexical_oof_predictions.csv`, `hard_benign_researcher_review.csv`, and signed-off addendum. Do not treat this exploratory finding as a confirmatory paper result.
