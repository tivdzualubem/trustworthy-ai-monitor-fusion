# Supervisor feedback alignment checkpoint (2026-10-09)

**Verbatim source:** user-uploaded `EDITS(5).txt` (October 8 feedback). This is a checkpoint, not a report of finished research.

| Professor requirement | Evidence/status | Remaining action |
|---|---|---|
| 1. Lexical shortcuts; hard same-topic benign cases; same folds | Original easy text baselines 100%; new harder matched 38-control pilot word 67.11%, char 72.37% direct-to-direct. New 38 cases signed off by one researcher. | Native guard/probe comparison **on hard examples not yet measured**. Keep exploratory. |
| 2. Native guards vs post-hoc probes vs causes | Results explicitly separated. | Do not infer causation or native understanding from a supervised probe. |
| 3. Inspect source-probe scores; cutoff-only calibration | Foldwise O2 source-probe cutoff-only diagnostic done: Llama .500→.645; Shield .500→.789; Granite .763→.947; Qwen .934→.934. Source scaler/weights held fixed; labeled O2 calibration used. Mean shift mathematically only intercept. | Avoid interpretation of threshold repair as new geometry; no post-hoc rewrite of original screen. |
| 4. Harmful recall and benign false alarms; simple decoding/normalization | Both rates in prompt-proxy metrics, O3 indiscriminate Llama blocking documented. | **Open:** native formatting normalization/decoding comparison. Known paired source decoding replay is not a general deployable algorithm. |
| 5. Task labels, native policy scope, transformation equivalence | Old 608 policy labels and 304 transformations AI-assisted reviewed; new 304 hard-benign policy assessments and 38 O2-equivalence reviewed by one researcher. New 18 uncertain policy rows preserved. | No independent raters; no fresh response-level outcomes; exclude ambiguity from definitive native-policy error estimates. |
| 6. Missing original results, preserve failed screen | Original scores/hidden states/OOF restored at `1f441df...`; 0/4 formal screen fixed. | Never rerun original scoring or overwrite frozen result files. |
| **Main path: cheaper statistically reliable measurement of missed harmful responses** | Strategic preference, **not yet empirically compared**. | PRIORITY: select response-level estimand; compare uniform/stratified/probability-based active audits and honest uncertainty/monitor acquisition costs. |
| Backup mechanism study | Still conditional. | Explore only if transparent decoding/normalization/thresholds plus hard negatives leave meaningful policy-aligned native failure. |
| Stop/go principle | Phase 3 cheap falsifiers completed, but new guard scoring not executed. | No new complex adapters, theoretical claims or repeated Kaggle sweeps. |

## Provenance and signoff

A user-reported single-researcher signoff accepted every drafted row of the Phase 3C policy review. This does **not** establish independent human adjudication or change any `uncertain` entry into a negative class. Prior approval of 38 drafted prompt texts is separate. The new review is exploratory **prompt moderation**, not harmful-response ground truth.
