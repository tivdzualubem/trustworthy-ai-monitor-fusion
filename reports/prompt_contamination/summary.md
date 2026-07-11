# Prompt-contamination diagnostic

This report compares official Qwen3Guard generation and parsing under prompt-only, response-only, and prompt-response inputs.

## Overall unsafe-only metrics

- prompt_only: ROC-AUC=0.8270, AP=0.4121, recall=0.8883, FPR=0.2635, precision=0.4294
- response_only: ROC-AUC=0.8980, AP=0.7228, recall=0.6701, FPR=0.0261, precision=0.8516
- prompt_response: ROC-AUC=0.9200, AP=0.7413, recall=0.8071, FPR=0.0402, precision=0.8175

## GCG negatives

- prompt_only: 13/16 flagged Unsafe (FPR=0.8125)
- response_only: 6/16 flagged Unsafe (FPR=0.3750)
- prompt_response: 6/16 flagged Unsafe (FPR=0.3750)

## Prompt-contamination flag

Among Y=0 examples, prompt-only was Unsafe while response-only was Safe for 382/1765 examples (0.2164).

These are diagnostics and do not establish causality or robustness.

## Timing note

Timing fields are retained as provenance only. The three modes were run in different sessions and batch-size conditions, so a separate controlled batch-1 and tail-latency benchmark is still required.
