# Development-Only Decision-Value Diagnostic

This protocol is frozen before fitting any real-data value model. It does not
retune the previous low/high Qwen routing thresholds.

## Development boundary

Only `policy_train`, `policy_selection`, and `calibration` are used: 1,687
rows, comprising 1,396 negatives and 291 positives. `final_test` and
`held_out_shift` remain excluded from feature design, model fitting, threshold
selection, and milestone decisions.

## Optional monitors

Two acquisitions are evaluated separately:

1. compact moderation after the rule filter;
2. Qwen prompt-response classification after rule and compact outputs.

For compact acquisition, no compact output or post-run metadata is available
to the router. For Qwen acquisition, no Qwen output or post-run metadata is
available to the router. Prompt-only and response-only Qwen modes are not
added in this first diagnostic.

## Cross-fitted target

For each optional monitor and each development example,

\[
V_i=L(Y_i,\widehat a_{\mathrm{base},i})
-L(Y_i,\widehat a_{\mathrm{augmented},i}).
\]

Both actions are out-of-fold. With zero-one loss, value is 1 when acquisition
corrects a decision, 0 when it does not change loss, and -1 when it worsens
loss.

Five outer stratified group folds provide unseen development evaluation. Four
inner stratified group folds create downstream out-of-fold scores, choose
thresholds, and construct value targets inside each outer-training partition.
Grouping uses nonempty `group_id`, then `pair_id`, then `example_id`.

## Operating risk

The common target is FPR 0.05. Each final mixed-score threshold is selected
from outer-training inner-OOF predictions only, maximizing recall among
thresholds with empirical FPR at most 0.05. Outer-fold FPR, recall, precision,
decision loss, and a one-sided 95% FPR upper bound are reported.

This is a development diagnostic, not a test-set or shift guarantee.

## Predictor families

The required primary comparisons are:

- current cheap monitor features;
- frozen prompt-response embeddings;
- legitimate runtime metadata.

The frozen representation is
`sentence-transformers/all-MiniLM-L6-v2`, requested revision `1110a24`.
The generation step must resolve and record the complete revision, use a fixed
prompt-response template, normalize the 384-dimensional vectors, record
truncation, and measure embedding runtime. PCA is fit inside each
outer-training fold only.

Dataset source, attack family, split identity, labels, audit fields, and
optional-monitor outputs are prohibited router inputs.

## Matched budgets

Random, ordinary uncertainty, learned decision value, and an oracle diagnostic
are compared at identical acquired-example counts. Never-acquire and
always-acquire endpoints are included. Primary budgets are 0%, 5%, 10%, 20%,
30%, 40%, 50%, 60%, 75%, and 100%.

The primary evidence is matched-budget downstream decision loss, FPR, recall,
and measured cost. AUC and value-prediction correlations are secondary.

## Milestone

The milestone passes only if development-only cross-fitted evidence shows:

1. complementary value from at least one heterogeneous optional monitor;
2. learned value predicts useful acquisition better than ordinary uncertainty
   across the matched-budget curve;
3. a selective point improves the safety-cost frontier at common FPR 0.05;
4. no excluded row influenced the result.

Until then, the experiment is not scaled and no new routing method is claimed.
