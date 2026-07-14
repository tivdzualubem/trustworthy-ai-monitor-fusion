# Stop/go decision v3

## Decision: NO-GO FOR A ROUTING-PERFORMANCE PAPER UNDER THE CURRENT AUDITED-LABEL EVIDENCE

The selective-acquisition policy has a measurable cost advantage, but it does
not satisfy the required risk gate across the evaluation splits. Therefore,
the project should continue as a **measurement-validity paper**, not as a
routing-performance paper.

This decision uses the final author-reviewed labels, the regenerated monitor
caches, the complete serialized v3 fusion pipelines, the controlled T4 timing
measurements, and the corrected v3 policy comparison.

## Prespecified decision rule

At the common selection target FPR of `0.05`, a routing-performance
GO requires both:

1. a positive paired 95% lower confidence bound for estimated cost reduction;
2. selective-policy FPR at or below `0.05`, with its one-sided 95%
   upper bound also at or below `0.05`, in every evaluated split.

This stop/go gate is deliberately conservative and is not a substitute for
the later formal Neyman-Pearson or Learn-then-Test certificate.

## Cost result

The cost requirement passes descriptively:

- Calibration: estimated reduction
  `0.2857`,
  paired 95% CI
  `[0.2442,
  0.3317]`.
- Final test: estimated reduction
  `0.3018`,
  paired 95% CI
  `[0.2580,
  0.3433]`.
- Held-out shift: estimated reduction
  `0.1556`,
  paired 95% CI
  `[0.0778,
  0.2528]`.

## Risk result

The risk requirement fails:

- Calibration selective FPR:
  `0.0260`;
  one-sided 95% upper bound:
  `0.0450`.
- Final-test selective FPR:
  `0.0402`;
  one-sided 95% upper bound:
  `0.0622`.
- Held-out-shift selective FPR:
  `0.2500`;
  one-sided 95% upper bound:
  `0.5273`.

The final-test upper bound exceeds 5%, and the held-out-shift observed FPR is
far above 5%. Selective acquisition also has lower point recall and higher
point FPR than full-information fusion on all three evaluated splits.

## Measurement-validity paper direction

The evidence supports a paper centered on:

1. invalid first-token versus official generated Qwen3Guard classification;
2. prompt contamination across prompt-only, response-only, and
   prompt-response inputs;
3. label-proxy errors and the effect of author-reviewed corrections;
4. complete model serialization and reproducible score-cache provenance;
5. controlled timing validity, including synchronization, warm-up, generation,
   tail latency, and end-to-end policy timing;
6. failure of nominal FPR control under attack-family shift.

## Remaining professor-required stages

This decision does not complete the later requirements for:

- a new locked test or nested leave-source/leave-family-out evaluation;
- formal risk control on an untouched risk-control split.

Those stages may refine the strength of the final claims, but the current v3
evidence does not justify a routing-performance claim.
