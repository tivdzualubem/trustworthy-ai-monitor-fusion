# Stop/go decision

## Decision: PROVISIONAL NO-GO FOR A ROUTING-PERFORMANCE PAPER

The selective-acquisition policy reduces expensive-monitor calls, but the
current evidence does not show that it preserves the intended risk constraint.
Therefore, the project should pivot toward a measurement-validity paper rather
than make a routing-performance claim.

This decision is provisional because the label audit is assistant-assisted and
still requires author review, no formal Neyman–Pearson or Learn-then-Test
certificate has been produced, and the existing final-test split previously
influenced development.

## Target FPR 0.05 comparison

- Calibration: full recall=0.7083, FPR=0.0257; selective recall=0.7083, FPR=0.0400, expensive-call reduction=0.2417.
- Final test: full recall=0.7500, FPR=0.0343; selective recall=0.7500, FPR=0.0457, expensive-call reduction=0.2583.
- Held-out shift: full recall=0.9118, FPR=0.3750; selective recall=0.8824, FPR=0.4375, expensive-call reduction=0.1600.

On held-out shift, the selective policy has lower recall and higher FPR than
the full-information predictor. Its FPR is also far above the nominal 5%
target. This fails the stated stop/go requirement that routing retain a cost
advantage while satisfying the risk constraint.

## Measurement-validity pivot

The revised paper should center on:

1. Invalid first-token Qwen scoring versus official generated classification.
2. Prompt contamination across prompt-only, response-only, and prompt-response inputs.
3. Label-proxy errors in JailbreakBench and GCG evaluation.
4. Complete model serialization and reproducible score-cache provenance.
5. Timing-validity requirements: synchronization, warm-up, batch comparability,
   generation latency, tail latency, and end-to-end policy timing.
6. Failure of nominal FPR control under source/family shift.

## Remaining work before final claims

- Author-review the 40 flagged audit rows.
- Lock corrected labels and regenerate the cache, serialized models, and comparisons.
- Run a controlled timing benchmark on fixed hardware and batch size.
- Use a new locked test or nested leave-source/leave-family-out evaluation.
- Apply formal risk control only on an untouched risk-control split.
