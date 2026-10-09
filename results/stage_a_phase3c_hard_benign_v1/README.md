# Phase 3C hard-benign candidate pilot (2026-10-09)

**Exploratory, unreviewed, text-only.** New candidates are an AI-authored proposed pool, not independent/native policy labels. The 38 original harmful-source prompts remain unchanged and the 38 old easy negatives remain frozen.

## Method
- 38 same-topic drafted benign prompts replacing ONLY the negative examples in this separate pilot.
- Two lexical baselines: word unigram/bigram TF-IDF and character 2–5 gram TF-IDF, each L2 logistic regression C=1, liblinear.
- Same five original pair-ID folds with the harmful and candidate benign pair together; train-only vectorizer and model.
- Three train/test representation tasks: direct-to-direct, direct-to-O2, O2-to-O2.
- Original easy dataset rerun in exactly the same CPU script for apples-to-apples comparison.
- Only case-role proxy labels; no native scoring, GPU hidden-state probe, or response FNR.

## Candidate overlap audit
- Word-overlap Jaccard strictly higher than old easy negative for 38/38 pairs.
- Character 3-gram Jaccard strictly higher than old easy negative for 38/38 pairs.
- Overlap is only a surface diagnostic and is NOT proof the benign meaning/policy is matched.

## Required researcher review before meaningful native-policy conclusions
1. `hard_benign_researcher_review.csv`: accept/uncertain/reject each same-topic control; edit neither frozen input nor original ratings.
2. `native_policy_review_pending.csv`: guard-specific direct policy labels remain blank (304 entries, including source anchors for conservative comparison).
3. `O2_equivalence_review_pending.csv`: direct vs transformed wording/label preservation remains blank (76 entries).
4. Drop or revise invalid controls only in a **new draft version**; freeze a reviewed version before new guard scoring.

## Restrictions
Original October 7 readout-transfer screen remains **0/4 passed**. A high text baseline on these prompts would support shortcut risk; poor text accuracy does not establish safety semantics in a hidden-state probe. This is not a prospective sample, independent annotation, causal study, or evidence of response-level guard FNR. The professor's preferred primary direction remains cost-efficient auditing of missed harmful assistant responses.
